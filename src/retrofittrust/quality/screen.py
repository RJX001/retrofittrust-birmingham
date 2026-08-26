"""Script-facing quality-screen runner (checkpoint 2).

Wraps :class:`DataQualityScreen` so ``scripts/02_train_quality_screen.py`` can
call :func:`run_quality_screen` and so Program 1 / the API can load
``quality_flagged.parquet``.

Internal AE features are a preprocessed (imputed, scaled, one-hot) copy.
Flags are attached back to the **original** merged rows. Nothing is deleted.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd

from retrofittrust.config import DATA_PROCESSED, MODELS_DIR, SEED
from retrofittrust.quality.ensemble import DataQualityScreen
from retrofittrust.quality.evaluation import evaluate_recall
from retrofittrust.quality.flags import (
    CONFIDENCE_FLAG_COL,
    CONFIDENCE_SCORE_COL,
    LITERATURE_FLAG_RATE_RANGE,
    sanity_check_flag_rate,
)

logger = logging.getLogger(__name__)

# Column aliases expected by modeling.features / scripts/02 / the API.
CONSENSUS_FLAG_COL = "quality_flag"
UNION_FLAG_COL = "quality_flag_union"
CONFIDENCE_COL = "quality_confidence"
FLAGGED_PARQUET = "quality_flagged.parquet"
SCREEN_JOBLIB = "quality_screen.joblib"


def _numeric_fallback(df: pd.DataFrame) -> pd.DataFrame:
    exclude = {
        "lsoa21cd",
        "retrofit_priority_score",
        CONSENSUS_FLAG_COL,
        UNION_FLAG_COL,
        CONFIDENCE_COL,
        CONFIDENCE_FLAG_COL,
        CONFIDENCE_SCORE_COL,
    }
    num = df.select_dtypes(include=[np.number]).copy()
    cols = [
        c
        for c in num.columns
        if c not in exclude and not str(c).startswith("quality_") and not str(c).startswith("recon_err")
    ]
    X = num[cols].replace([np.inf, -np.inf], np.nan)
    return X.fillna(X.median(numeric_only=True)).fillna(0.0)


def _feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, Any]:
    """One-hot + standardised copy for the autoencoder; same index as ``df``."""
    try:
        from retrofittrust.data.preprocess import preprocess

        X, state = preprocess(df, fit=True)
        if X.shape[1] < 2:
            raise ValueError("preprocess produced too few columns")
        return X, state
    except Exception as exc:
        logger.warning("Falling back to numeric columns for the AE (%s)", exc)
        return _numeric_fallback(df), None


def run_quality_screen(
    *,
    merged_path: Path,
    interim_dir: Path,
    processed_dir: Path,
    models_dir: Path,
    seed: int = SEED,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Train AE + IForest, flag anomalies, write parquet + joblib. Never deletes rows."""
    np.random.seed(seed)
    merged_path = Path(merged_path)
    processed_dir = Path(processed_dir)
    models_dir = Path(models_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    Path(interim_dir).mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(merged_path) if merged_path.suffix == ".parquet" else pd.read_csv(merged_path)
    n_in = len(df)
    X, prep_state = _feature_matrix(df)
    if X.shape[1] < 2:
        raise ValueError("Insufficient features for anomaly screening.")

    screen = DataQualityScreen(
        random_state=seed,
        flag_mode="union",
        prefer_evt=True,
        target_flag_rate=0.35,
        k=0.75,
        ae_kwargs={
            "epoch_num": 30,
            "batch_size": min(256, max(16, len(X))),
            "patience": 6,
            "verbose": 0,
            "preprocessing": False,  # Program 1 preprocess already standardised
        },
    )
    flagged = screen.fit_transform(X, feature_names=list(X.columns))
    if len(flagged) != n_in:
        raise RuntimeError("Quality screen changed row count — deletion is forbidden.")

    out = df.copy()
    passthrough = [
        "ae_score",
        "iforest_score",
        "consensus_score",
        "union_score",
        "flagged_ae",
        "flagged_iforest",
        "flagged_consensus",
        "flagged_union",
        "top_implausible_feature",
        "top_implausible_error",
        "sample_weight",
        "inference_caveat",
        CONFIDENCE_FLAG_COL,
        CONFIDENCE_SCORE_COL,
        "data_quality_label",
    ]
    for col in passthrough:
        if col in flagged.columns:
            out[col] = flagged[col].to_numpy()
    for col in flagged.columns:
        if col.startswith("recon_err__"):
            out[col] = flagged[col].to_numpy()

    # Aliases for modeling.features / scripts/02 / the FastAPI rank payload.
    out[CONSENSUS_FLAG_COL] = flagged["flagged_consensus"].astype(int).to_numpy()
    out[UNION_FLAG_COL] = flagged["flagged_union"].astype(int).to_numpy()
    out[CONFIDENCE_COL] = flagged[CONFIDENCE_SCORE_COL].to_numpy()
    out["low_confidence_caveat"] = flagged["inference_caveat"].to_numpy()

    if len(out) != n_in:
        raise RuntimeError("Flag attach changed row count — deletion is forbidden.")

    flagged_path = processed_dir / FLAGGED_PARQUET
    out.to_parquet(flagged_path, index=False)

    artefact = {
        "screen": screen,
        "preprocess_state": prep_state,
        "feature_names": list(X.columns),
        "ae_threshold": screen.ensemble.ae_threshold_,
        "iforest_threshold": screen.ensemble.iforest_threshold_,
        "consensus_threshold": screen.ensemble.consensus_threshold_,
        "note": "Quarantine/flag only — never silently delete.",
    }
    model_path = models_dir / SCREEN_JOBLIB
    joblib.dump(artefact, model_path)

    report = screen.flag_rate_report(X)
    recall = evaluate_recall(screen, X, seed=seed)
    lit = sanity_check_flag_rate(report["operational_flag_rate"])
    logger.info(lit["message"])

    metrics: dict[str, Any] = {
        "input_rows": n_in,
        "output_rows": len(out),
        "n_features": int(X.shape[1]),
        "flagged_rate": report["operational_flag_rate"],
        "flagged_rate_consensus": report["flag_rate_consensus"],
        "flagged_rate_union": report["flag_rate_union"],
        "synthetic_injection_recall": recall.recall,
        "synthetic_beats_chance": recall.beats_chance,
        "literature_band": LITERATURE_FLAG_RATE_RANGE,
        "literature_ok": lit["ok"],
        "ae_threshold_method": (
            screen.ensemble.ae_threshold_.method if screen.ensemble.ae_threshold_ else None
        ),
        "output_path": str(flagged_path),
        "model_path": str(model_path),
    }
    logger.info(
        "Quality screen saved to %s (union flagged %.1f%%, consensus %.1f%%)",
        flagged_path,
        metrics["flagged_rate_union"] * 100,
        metrics["flagged_rate_consensus"] * 100,
    )
    return out, metrics


def load_flagged_dataset(processed_dir: Optional[Path] = None) -> pd.DataFrame:
    processed_dir = Path(processed_dir or DATA_PROCESSED)
    path = processed_dir / FLAGGED_PARQUET
    if not path.exists():
        raise FileNotFoundError(f"Flagged dataset not found at {path}")
    return pd.read_parquet(path)


def load_quality_screen(models_dir: Optional[Path] = None) -> dict[str, Any]:
    models_dir = Path(models_dir or MODELS_DIR)
    path = models_dir / SCREEN_JOBLIB
    if not path.exists():
        raise FileNotFoundError(f"Quality screen artefact not found at {path}")
    return joblib.load(path)
