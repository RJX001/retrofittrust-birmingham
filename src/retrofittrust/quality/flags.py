"""Quarantine / flag helpers — never silently delete anomalous EPC records.

Downstream contract for the ranking model (Program 1):
- Flagged rows stay in the dataset.
- LightGBM training should pass ``sample_weight`` (down-weight, do not drop).
- At inference, surface ``inference_caveat`` on low-confidence records.

Silent deletion would systematically exclude unusual-but-real stock
(flats / maisonettes are disproportionately flagged in EPC error research) —
an equity problem in a fuel-poverty setting.

Literature sanity band for the operational flagged rate (CURSOR_BUILD_SPEC §4):
- roughly 27% of records show at least one quality flag
- true error rate estimated 36–62% in the wider literature
If the screen's flagged rate sits well outside ~27–60%, investigate before
trusting the flags. This is a sanity check, not a target to overfit.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from retrofittrust.config import SEED

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None  # type: ignore[assignment]

# Policy: quarantine only. Callers must not drop on these flags.
NEVER_SILENTLY_DELETE = True
QUARANTINE_LABEL = "quarantine"
OK_LABEL = "ok"

# Down-weight flagged rows in LightGBM rather than excluding them.
FLAGGED_SAMPLE_WEIGHT = 0.35
CLEAN_SAMPLE_WEIGHT = 1.0

# Documented EPC quality-flag / error-rate band used as a sanity check.
LITERATURE_FLAG_RATE_LOW = 0.27
LITERATURE_FLAG_RATE_HIGH = 0.60
LITERATURE_FLAG_RATE_RANGE = (LITERATURE_FLAG_RATE_LOW, LITERATURE_FLAG_RATE_HIGH)

CONFIDENCE_FLAG_COL = "data_quality_flag"
CONFIDENCE_SCORE_COL = "data_quality_confidence"
CONFIDENCE_LABEL_COL = "data_quality_label"


def sanity_check_flag_rate(flag_rate: float) -> dict[str, Any]:
    """Compare an operational flagged rate to the EPC error literature band."""
    rate = float(flag_rate)
    ok = LITERATURE_FLAG_RATE_LOW <= rate <= LITERATURE_FLAG_RATE_HIGH
    if ok:
        message = (
            f"Flagged rate {rate:.1%} sits inside the documented EPC quality-flag "
            f"band (~27–60%; ~27% of records show at least one quality flag, "
            f"true error rate estimated 36–62%)."
        )
    elif rate < LITERATURE_FLAG_RATE_LOW:
        message = (
            f"Flagged rate {rate:.1%} is below the ~27–60% EPC literature band. "
            "Investigate under-detection (threshold too harsh / model underfit) "
            "before trusting the screen."
        )
    else:
        message = (
            f"Flagged rate {rate:.1%} is above the ~27–60% EPC literature band. "
            "Investigate over-flagging (threshold too lenient / unstandardised "
            "inputs) before trusting the screen. Do not silently delete the extra rows."
        )
    return {
        "ok": ok,
        "flag_rate": rate,
        "literature_band": LITERATURE_FLAG_RATE_RANGE,
        "message": message,
    }


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-z))


def _confidence_from_scores(
    consensus_score: np.ndarray,
    consensus_threshold: Optional[float],
) -> np.ndarray:
    """Map consensus score to (0, 1] confidence; high score → low confidence."""
    scores = np.asarray(consensus_score, dtype=float)
    if consensus_threshold is None or not np.isfinite(consensus_threshold):
        lo, hi = np.quantile(scores, 0.05), np.quantile(scores, 0.95)
        scale = max(hi - lo, 1e-6)
        centred = (scores - lo) / scale
        return np.clip(1.0 - centred, 0.05, 1.0)
    # logistic centred on the consensus threshold
    scale = max(float(np.std(scores)), 1e-6)
    return np.clip(1.0 - _sigmoid((scores - float(consensus_threshold)) / scale), 0.05, 1.0)


def inference_caveat(
    flagged: bool,
    top_feature: Optional[str] = None,
    confidence: Optional[float] = None,
) -> str:
    """Low-confidence note for dashboard / ``/rank`` responses. Empty if clean."""
    if not flagged:
        return ""
    feature_bit = f" Top implausible field: {top_feature}." if top_feature else ""
    conf_bit = f" Confidence={confidence:.2f}." if confidence is not None else ""
    return (
        "Low data-quality confidence: this record was quarantined by the "
        "AutoEncoder + Isolation Forest screen (flagged, not deleted)."
        f"{feature_bit}{conf_bit} Treat the retrofit ranking with caution."
    )


def sample_weights_for_lightgbm(
    flagged: np.ndarray,
    flagged_weight: float = FLAGGED_SAMPLE_WEIGHT,
    clean_weight: float = CLEAN_SAMPLE_WEIGHT,
) -> np.ndarray:
    """Vector to pass as LightGBM ``weight`` / ``sample_weight``. Never zeros out a row."""
    flags = np.asarray(flagged, dtype=bool)
    weights = np.full(flags.shape[0], float(clean_weight), dtype=float)
    weights[flags] = float(flagged_weight)
    if np.any(weights <= 0):
        raise ValueError("Sample weights must stay strictly positive — never drop via zero weight.")
    return weights


def attach_quality_flags(
    records,
    score_frame,
    *,
    flag_mode: str = "union",
    flagged_sample_weight: float = FLAGGED_SAMPLE_WEIGHT,
    consensus_threshold: Optional[float] = None,
):
    """Copy ``records`` and attach quarantine columns. Length is preserved.

    Parameters
    ----------
    records
        Original (preprocessed) feature table. Not mutated.
    score_frame
        Output of :meth:`AnomalyEnsemble.score_frame` aligned by position or index.
    flag_mode
        ``union`` (high recall; default operational flag) or ``consensus``
        (high precision). Both binary columns are always attached.
    """
    if pd is None:
        raise ImportError("pandas is required to attach quality flags.")
    if flag_mode not in {"union", "consensus"}:
        raise ValueError("flag_mode must be 'union' or 'consensus'.")
    if not NEVER_SILENTLY_DELETE:
        raise RuntimeError("NEVER_SILENTLY_DELETE is a hard policy flag and must stay True.")

    out = records.copy()
    n = len(out)
    if len(score_frame) != n:
        raise ValueError(
            f"score_frame has {len(score_frame)} rows but records has {n}. "
            "Refusing to attach flags (cannot drop or recycle rows)."
        )

    aligned = score_frame
    if not out.index.equals(score_frame.index):
        aligned = score_frame.reset_index(drop=True)
        aligned.index = out.index

    flag_col = "flagged_union" if flag_mode == "union" else "flagged_consensus"
    flagged = np.asarray(aligned[flag_col], dtype=bool)
    consensus = np.asarray(aligned["consensus_score"], dtype=float)
    confidence = _confidence_from_scores(consensus, consensus_threshold)
    weights = sample_weights_for_lightgbm(flagged, flagged_weight=flagged_sample_weight)
    top_feat = (
        aligned["top_implausible_feature"].astype(str).tolist()
        if "top_implausible_feature" in aligned.columns
        else [None] * n
    )
    caveats = [
        inference_caveat(bool(flagged[i]), top_feat[i], float(confidence[i]))
        for i in range(n)
    ]

    extra: dict[str, Any] = {}
    passthrough = [
        "ae_score",
        "iforest_score",
        "ae_score_z",
        "iforest_score_z",
        "consensus_score",
        "union_score",
        "flagged_ae",
        "flagged_iforest",
        "flagged_consensus",
        "flagged_union",
        "top_implausible_feature",
        "top_implausible_error",
    ]
    for col in passthrough:
        if col in aligned.columns:
            extra[col] = aligned[col].to_numpy()
    for col in aligned.columns:
        if col.startswith("recon_err__"):
            extra[col] = aligned[col].to_numpy()

    extra[CONFIDENCE_FLAG_COL] = flagged.astype(int)
    extra[CONFIDENCE_SCORE_COL] = confidence
    extra[CONFIDENCE_LABEL_COL] = np.where(flagged, QUARANTINE_LABEL, OK_LABEL)
    extra["sample_weight"] = weights
    extra["inference_caveat"] = caveats
    extra["data_quality_flag_mode"] = flag_mode
    orig_index = out.index
    overlap = [c for c in extra if c in out.columns]
    if overlap:
        out = out.drop(columns=overlap)
    out = pd.concat([out.reset_index(drop=True), pd.DataFrame(extra)], axis=1)
    out.index = orig_index

    if len(out) != n:
        raise RuntimeError("attach_quality_flags changed row count — deletion is forbidden.")
    return out
