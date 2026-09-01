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

from retrofittrust.config import DATA_PROCESSED, DATA_RAW, MODELS_DIR, PROJECT_ROOT, REPORTS_FIGURES, SEED
from retrofittrust.quality.ensemble import DataQualityScreen
from retrofittrust.quality.evaluation import evaluate_recall, multi_seed_stability, tune_from_injection
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


# Fallback metadata — never AE features.
_SCREEN_META_COLS = frozenset(
    {"is_quality_screen_fallback", "fallback_source", "synthetic_data_label"}
)


def _strip_screen_meta(df: pd.DataFrame) -> pd.DataFrame:
    drop = [c for c in df.columns if c in _SCREEN_META_COLS]
    return df.drop(columns=drop) if drop else df


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
    exclude |= _SCREEN_META_COLS
    num = df.select_dtypes(include=[np.number]).copy()
    cols = [
        c
        for c in num.columns
        if c not in exclude and not str(c).startswith("quality_") and not str(c).startswith("recon_err")
    ]
    X = num[cols].replace([np.inf, -np.inf], np.nan)
    return X.fillna(X.median(numeric_only=True)).fillna(0.0)


def _ae_matrix_from_preprocessed(X_full: pd.DataFrame, state: Any) -> pd.DataFrame:
    """Numeric feature block only — id / address strings must not reach the AE."""
    names = list(getattr(state, "feature_names", []) or [])
    if names:
        missing = [c for c in names if c not in X_full.columns]
        if missing:
            logger.warning("Preprocess feature list missing %s columns; using numeric select_dtypes", len(missing))
        else:
            return X_full.loc[:, names].astype(float)
    numeric = X_full.select_dtypes(include=[np.number])
    drop = [c for c in numeric.columns if str(c).startswith("quality_")]
    return numeric.drop(columns=drop, errors="ignore")


def _feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, Any]:
    """One-hot + standardised copy for the autoencoder; same index as ``df``."""
    try:
        from retrofittrust.data.preprocess import preprocess

        X_full, state = preprocess(df, fit=True)
        X = _ae_matrix_from_preprocessed(X_full, state)
        if X.shape[1] < 2:
            raise ValueError("preprocess produced too few numeric feature columns")
        return X, state
    except Exception as exc:
        logger.warning("Falling back to numeric columns for the AE (%s)", exc)
        return _numeric_fallback(df), None


def _synthetic_epc_fallback(*, n: int = 800, seed: int = SEED) -> pd.DataFrame:
    """SYNTHETIC DATA — Birmingham-shaped EPC records when merged parquet is absent."""
    rng = np.random.default_rng(seed)
    n = max(200, int(n))
    heating = ["Gas boiler", "Electric storage heaters", "Air source heat pump", "Community scheme"]
    fuels = ["mains gas", "electricity", "oil", "LPG"]
    ages = ["1900-1929", "1930-1949", "1950-1966", "1967-1975", "1976-1982", "1983-1990", "1991-1995", "1996-2002"]
    forms = ["Detached", "Semi-Detached", "Mid-Terrace", "End-Terrace", "Flat", "Maisonette"]
    walls = ["Cavity wall, as built, insulated", "Solid brick, as built, no insulation", "Timber frame, as built"]
    rows: list[dict[str, Any]] = []
    for i in range(n):
        form = str(rng.choice(forms))
        area = float(rng.uniform(35, 180) * (1.4 if form in {"Detached", "Semi-Detached"} else 1.0))
        rows.append(
            {
                "lsoa21cd": f"SYNTH_E010{i % 640:05d}",
                "local_authority_label": "Birmingham",
                "built_form": form,
                "property_type": form,
                "mainheat_description": str(rng.choice(heating)),
                "main_fuel": str(rng.choice(fuels)),
                "total_floor_area": area,
                "construction_age_band": str(rng.choice(ages)),
                "walls_description": str(rng.choice(walls)),
                "current_energy_efficiency": int(rng.integers(20, 85)),
                "potential_energy_efficiency": int(rng.integers(40, 95)),
                "co2_emissions_current": float(rng.uniform(2.0, 12.0)),
                "energy_consumption_current": float(rng.uniform(150, 450)),
                "is_quality_screen_fallback": True,
                "fallback_source": "synthetic_epc",
                "synthetic_data_label": "SYNTHETIC DATA",
            }
        )
    logger.warning(
        "Using %s-row SYNTHETIC Birmingham-shaped EPC frame (merged_lsoa.parquet not ready).",
        n,
    )
    return pd.DataFrame(rows)


def _stratified_lsoa_sample(df: pd.DataFrame, n: int, seed: int = SEED) -> pd.DataFrame:
    """Cap AE training to a PoC-sized frame while covering as many LSOAs as possible."""
    if n <= 0 or len(df) <= n:
        return df
    lsoa_col = next((c for c in ("lsoa21cd", "lsoa_code", "LSOA21CD") if c in df.columns), None)
    if lsoa_col is None:
        return df.sample(n=n, random_state=seed).reset_index(drop=True)

    n_groups = int(df[lsoa_col].nunique(dropna=True))
    per = max(1, n // max(n_groups, 1))
    pieces: list[pd.DataFrame] = []
    for _, group in df.groupby(lsoa_col, dropna=False, sort=False):
        pieces.append(group.sample(n=min(len(group), per), random_state=seed))
    sampled = pd.concat(pieces, ignore_index=False)
    leftover = df.drop(index=sampled.index, errors="ignore")
    if len(sampled) < n and len(leftover) > 0:
        extra = leftover.sample(n=min(n - len(sampled), len(leftover)), random_state=seed)
        sampled = pd.concat([sampled, extra], ignore_index=False)
    if len(sampled) > n:
        sampled = sampled.sample(n=n, random_state=seed)
    logger.info(
        "Quality screen sampled %s / %s rows across %s LSOAs (SEED=%s) — full 476k AE is out of PoC scope.",
        f"{len(sampled):,}",
        f"{len(df):,}",
        f"{sampled[lsoa_col].nunique():,}",
        seed,
    )
    return sampled.reset_index(drop=True)


def load_screening_input(
    merged_path: Path,
    *,
    seed: int = SEED,
    max_fallback_rows: int = 3000,
    max_rows: int | None = 8_000,
) -> tuple[pd.DataFrame, str]:
    """Load merged LSOA data, else Birmingham EPC sample, else synthetic frame.

    When the merged table is large, ``max_rows`` draws a stratified LSOA sample
    so the AutoEncoder stays tractable. Pass ``max_rows=None`` to load everything.
    """
    merged_path = Path(merged_path)
    if merged_path.exists():
        df = (
            pd.read_parquet(merged_path)
            if merged_path.suffix == ".parquet"
            else pd.read_csv(merged_path)
        )
        source = str(merged_path)
        if max_rows is not None and len(df) > max_rows:
            df = _stratified_lsoa_sample(df, max_rows, seed=seed)
            source = f"{merged_path} (stratified sample n={len(df)})"
        return df, source

    epc_candidates = (
        DATA_RAW / "EPC.csv",
        DATA_RAW / "epc_birmingham" / "certificates.csv",
        DATA_RAW / "energy_certficates.csv",
    )
    for epc_path in epc_candidates:
        if not epc_path.exists():
            continue
        logger.warning(
            "Merged dataset not found at %s — loading Birmingham EPC sample from %s "
            "(FALLBACK: not LSOA-merged; run scripts/01_ingest_and_merge.py when ready).",
            merged_path,
            epc_path,
        )
        df = pd.read_csv(epc_path, nrows=max(max_fallback_rows * 3, max_fallback_rows))
        if "local_authority_label" in df.columns:
            mask = df["local_authority_label"].astype(str).str.lower().eq("birmingham")
            df = df.loc[mask]
        if len(df) > max_fallback_rows:
            df = df.sample(n=max_fallback_rows, random_state=seed)
        df = df.copy()
        df["is_quality_screen_fallback"] = True
        df["fallback_source"] = f"raw_epc:{epc_path.name}"
        return df.reset_index(drop=True), f"fallback:raw_epc:{epc_path}"

    frame = _synthetic_epc_fallback(n=min(800, max_fallback_rows), seed=seed)
    return frame, "fallback:synthetic_epc"


def _save_quality_figure(metrics: dict[str, Any], *, path: Path) -> None:
    """Quality-specific checkpoint plot (flag rates + injection recall)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib unavailable — skipping quality figure")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    consensus = metrics.get("flagged_rate_consensus", metrics.get("flagged_rate", 0.0))
    union = metrics.get("flagged_rate_union", consensus)
    lo, hi = LITERATURE_FLAG_RATE_RANGE
    labels = ["consensus", "union"]
    rates = [consensus * 100, union * 100]
    colours = ["#4472C4", "#ED7D31"]
    bars = axes[0].bar(labels, rates, color=colours)
    axes[0].axhspan(lo * 100, hi * 100, alpha=0.15, color="green", label="EPC literature band")
    axes[0].set_ylabel("Flagged rate (%)")
    axes[0].set_title("Operational flag rates")
    axes[0].set_ylim(0, max(100, max(rates) * 1.15))
    for bar, rate in zip(bars, rates):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{rate:.1f}%", ha="center", fontsize=9)
    axes[0].legend(loc="upper right", fontsize=8)

    recall = metrics.get("synthetic_injection_recall", 0.0)
    chance = metrics.get("synthetic_chance_baseline", consensus)
    axes[1].bar(["injection recall", "chance baseline"], [recall * 100, chance * 100], color=["#548235", "#A5A5A5"])
    axes[1].set_ylabel("Rate (%)")
    axes[1].set_title("SYNTHETIC injection evaluation")
    axes[1].set_ylim(0, max(100, recall * 110))
    fig.suptitle("RetrofitTrust quality screen (AE + Isolation Forest)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Quality figure saved to %s", path)
    _save_number_figures(metrics, figures_dir=path.parent)
    write_quality_numbers_markdown(metrics, path=path.parent / "02_quality_numbers.md")


def _save_number_figures(metrics: dict[str, Any], *, figures_dir: Path) -> None:
    """Large-number plates for checkpoint 2 (flag rates and injection recall)."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch
    except ImportError:
        return

    figures_dir.mkdir(parents=True, exist_ok=True)
    consensus = float(metrics.get("flagged_rate_consensus", metrics.get("flagged_rate", 0.0)) or 0.0)
    union = float(metrics.get("flagged_rate_union", consensus) or 0.0)
    lo, hi = LITERATURE_FLAG_RATE_RANGE
    n_rows = int(metrics.get("output_rows") or metrics.get("input_rows") or 0)

    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.2)
    ax.axis("off")
    ax.set_title("Quality-screen flag rates (AE + Isolation Forest)", fontsize=13, pad=8)
    cards = [
        (1.1, "#4472C4", f"{consensus * 100:.1f}%", "Consensus", "high precision"),
        (5.6, "#ED7D31", f"{union * 100:.1f}%", "Union (operational)", "high recall"),
    ]
    for x0, colour, value, title, subtitle in cards:
        ax.add_patch(
            FancyBboxPatch(
                (x0, 2.35),
                3.3,
                3.2,
                boxstyle="round,pad=0.08,rounding_size=0.18",
                facecolor=colour,
                edgecolor="none",
                alpha=0.92,
            )
        )
        ax.text(x0 + 1.65, 4.55, value, ha="center", va="center", fontsize=32, color="white", fontweight="bold")
        ax.text(x0 + 1.65, 3.35, title, ha="center", va="center", fontsize=12, color="white")
        ax.text(x0 + 1.65, 2.85, subtitle, ha="center", va="center", fontsize=9, color="white", alpha=0.9)

    ax.add_patch(
        FancyBboxPatch(
            (1.1, 0.35),
            7.8,
            1.6,
            boxstyle="round,pad=0.06,rounding_size=0.12",
            facecolor="#e8f5e9",
            edgecolor="#2e7d32",
            linewidth=1.2,
        )
    )
    ax.text(
        5.0,
        1.45,
        "EPC literature band  27–60%",
        ha="center",
        va="center",
        fontsize=12,
        color="#1b5e20",
        fontweight="bold",
    )
    in_band = lo <= consensus <= hi and lo <= union <= hi
    status = "both rates sit inside the band" if in_band else "investigate thresholds — rate outside the band"
    n_bit = f"  ·  n = {n_rows:,} rows" if n_rows else ""
    ax.text(
        5.0,
        0.85,
        f"{status}{n_bit}  ·  flagged, never deleted",
        ha="center",
        va="center",
        fontsize=9,
        color="#33691e",
    )
    fig.tight_layout()
    flag_path = figures_dir / "02_flag_rates.png"
    fig.savefig(flag_path, dpi=140, bbox_inches="tight")
    plt.close(fig)

    recall = float(metrics.get("synthetic_injection_recall", 0.0) or 0.0)
    chance = float(metrics.get("synthetic_chance_baseline", union) or 0.0)
    lift_pp = (recall - chance) * 100
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.2)
    ax.axis("off")
    ax.set_title("SYNTHETIC injection recall versus chance", fontsize=13, pad=8)
    ax.add_patch(
        FancyBboxPatch(
            (0.7, 2.2),
            4.2,
            3.4,
            boxstyle="round,pad=0.08,rounding_size=0.18",
            facecolor="#548235",
            edgecolor="none",
        )
    )
    ax.text(2.8, 4.55, f"{recall * 100:.1f}%", ha="center", va="center", fontsize=34, color="white", fontweight="bold")
    ax.text(2.8, 3.35, "Injection recall", ha="center", va="center", fontsize=12, color="white")
    ax.text(2.8, 2.8, "SYNTHETIC DATA", ha="center", va="center", fontsize=9, color="white", alpha=0.9)
    ax.add_patch(
        FancyBboxPatch(
            (5.3, 2.2),
            4.0,
            3.4,
            boxstyle="round,pad=0.08,rounding_size=0.18",
            facecolor="#7f7f7f",
            edgecolor="none",
        )
    )
    ax.text(7.3, 4.55, f"{chance * 100:.1f}%", ha="center", va="center", fontsize=34, color="white", fontweight="bold")
    ax.text(7.3, 3.35, "Chance baseline", ha="center", va="center", fontsize=12, color="white")
    ax.text(7.3, 2.8, "operational flag rate", ha="center", va="center", fontsize=9, color="white", alpha=0.9)
    beats = metrics.get("synthetic_beats_chance")
    if beats is None:
        beats = recall > chance + 0.05
    banner = (
        f"Lift {lift_pp:+.1f} percentage points  ·  "
        f"{'beats chance (+5 pp rule)' if beats else 'does not clearly beat chance'}"
    )
    ax.add_patch(
        FancyBboxPatch(
            (0.7, 0.4),
            8.6,
            1.4,
            boxstyle="round,pad=0.06,rounding_size=0.12",
            facecolor="#fff8e1",
            edgecolor="#f9a825",
            linewidth=1.2,
        )
    )
    ax.text(5.0, 1.1, banner, ha="center", va="center", fontsize=11, color="#5d4e00")
    fig.tight_layout()
    recall_path = figures_dir / "02_injection_recall.png"
    fig.savefig(recall_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    logger.info("Number figures saved to %s and %s", flag_path, recall_path)


def write_quality_numbers_markdown(metrics: dict[str, Any], *, path: Path) -> None:
    """Persist checkpoint-2 rates for the dissertation evidence pack."""
    path.parent.mkdir(parents=True, exist_ok=True)
    consensus = float(metrics.get("flagged_rate_consensus", metrics.get("flagged_rate", 0.0)) or 0.0)
    union = float(metrics.get("flagged_rate_union", consensus) or 0.0)
    recall = float(metrics.get("synthetic_injection_recall", 0.0) or 0.0)
    chance = float(metrics.get("synthetic_chance_baseline", union) or 0.0)
    n_out = metrics.get("output_rows")
    n_in = metrics.get("input_rows")
    n_feat = metrics.get("n_features")
    lo, hi = LITERATURE_FLAG_RATE_RANGE
    tune = metrics.get("threshold_tune") or {}
    by_kind = metrics.get("synthetic_recall_by_kind") or {}
    kind_lines = ""
    if by_kind:
        kind_lines = "\n".join(
            f"- `{kind}`: {float(val) * 100:.1f}%" for kind, val in sorted(by_kind.items())
        )
    beats = metrics.get("synthetic_beats_chance")
    if beats is None:
        beats = recall > chance + 0.05
    in_band_c = lo <= consensus <= hi
    in_band_u = lo <= union <= hi
    n_in_s = f"{n_in:,}" if isinstance(n_in, int) else str(n_in)
    n_out_s = f"{n_out:,}" if isinstance(n_out, int) else str(n_out)
    n_cons = int(round(consensus * n_out)) if isinstance(n_out, int) else None
    n_union = int(round(union * n_out)) if isinstance(n_out, int) else None
    cons_count = f"{consensus * 100:.1f}% ({n_cons:,} rows)" if n_cons is not None else f"{consensus * 100:.1f}%"
    union_count = f"{union * 100:.1f}% ({n_union:,} rows)" if n_union is not None else f"{union * 100:.1f}%"
    body = f"""# Checkpoint 2 — quality-screen numbers

British English. Seed = 42. Flagged records are quarantined, never silently deleted.

## Sample

| Item | Value |
|---|---|
| Input source | {metrics.get("input_source", "unknown")} |
| Input rows | {n_in_s} |
| Output rows (flagged parquet) | {n_out_s} |
| AE features | {n_feat} |
| AE architecture | hidden={metrics.get("ae_hidden", 404)}, bottleneck={metrics.get("ae_bottleneck", 202)} |
| Threshold | k={tune.get("k", "n/a")}, target={tune.get("target_flag_rate", "n/a")}, EVT={tune.get("prefer_evt", "n/a")}, method={metrics.get("ae_threshold_method", "mean_sigma")} |

Row count is unchanged from input to output (quarantine / flag only). LightGBM should pass `sample_weight` (1.0 clean, 0.35 flagged).

## Flag rates

| Mode | Rate | Literature band (27–60%) |
|---|---|---|
| Consensus (`quality_flag`) | **{cons_count}** | {"inside" if in_band_c else "outside"} |
| Union (`quality_flag_union`, operational) | **{union_count}** | {"inside" if in_band_u else "outside"} |

Literature sanity: roughly 27% of EPC records show at least one quality flag; true error rate estimated 36–62% in the wider literature. The operational (union) rate is the high-recall set used for LightGBM sample weights.

## SYNTHETIC injection evaluation

Injected copies only — labelled **SYNTHETIC DATA**, not real assessor errors.

| Metric | Rate |
|---|---|
| Injection recall | **{recall * 100:.1f}%** |
| Chance baseline (operational flag rate) | {chance * 100:.1f}% |
| Lift | {(recall - chance) * 100:+.1f} pp |
| Beats chance (+5 pp) | {"yes" if beats else "no"} |
"""
    if kind_lines:
        body += "\n### Recall by injection kind\n\n" + kind_lines + "\n"
    body += """
## Figures

- `reports/figures/quality_screen_summary.png` — combined checkpoint summary
- `reports/figures/02_flag_rates.png` — consensus vs union vs literature band
- `reports/figures/02_injection_recall.png` — synthetic injection recall vs chance

## Artefacts

- `data/processed/quality_flagged.parquet`
- `models/quality_screen.joblib`
"""
    path.write_text(body, encoding="utf-8")
    logger.info("Quality numbers markdown saved to %s", path)


def run_quality_screen(
    *,
    merged_path: Path,
    interim_dir: Path,
    processed_dir: Path,
    models_dir: Path,
    seed: int = SEED,
    run_stability: bool = False,
    stability_seeds: tuple[int, ...] = (42, 43, 44),
    max_rows: int | None = 8_000,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Train AE + IForest, flag anomalies, write parquet + joblib. Never deletes rows."""
    np.random.seed(seed)
    merged_path = Path(merged_path)
    processed_dir = Path(processed_dir)
    models_dir = Path(models_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    Path(interim_dir).mkdir(parents=True, exist_ok=True)

    df, input_source = load_screening_input(merged_path, seed=seed, max_rows=max_rows)
    if input_source.startswith("fallback:"):
        logger.warning("Quality screen input source: %s", input_source)

    n_in = len(df)
    X, prep_state = _feature_matrix(_strip_screen_meta(df))
    if X.shape[1] < 2:
        raise ValueError("Insufficient features for anomaly screening.")

    ae_batch = min(256, max(16, len(X)))
    screen_kwargs = {
        "random_state": seed,
        "flag_mode": "union",
        "prefer_evt": True,
        "target_flag_rate": 0.35,
        "k": 0.75,
        "ae_kwargs": {
            "epoch_num": 30,
            "batch_size": ae_batch,
            "patience": 6,
            "verbose": 0,
            "preprocessing": False,
        },
    }
    screen = DataQualityScreen(**screen_kwargs)
    screen.fit(X, feature_names=list(X.columns))

    tune_result = tune_from_injection(screen, X, seed=seed)
    logger.info(
        "Threshold tuning (SYNTHETIC injection): k=%.2f target=%.2f recall=%.1f%% flag_rate=%.1f%% evt=%s",
        tune_result.k,
        tune_result.target_flag_rate,
        tune_result.recall * 100,
        tune_result.flag_rate * 100,
        tune_result.prefer_evt,
    )

    flagged = screen.transform(X, feature_names=list(X.columns))
    if len(flagged) != n_in:
        raise RuntimeError("Quality screen changed row count — deletion is forbidden.")

    extra: dict[str, Any] = {}
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
            extra[col] = flagged[col].to_numpy()
    for col in flagged.columns:
        if col.startswith("recon_err__"):
            extra[col] = flagged[col].to_numpy()

    # Aliases for modeling.features / scripts/02 / the FastAPI rank payload.
    extra[CONSENSUS_FLAG_COL] = flagged["flagged_consensus"].astype(int).to_numpy()
    extra[UNION_FLAG_COL] = flagged["flagged_union"].astype(int).to_numpy()
    extra[CONFIDENCE_COL] = flagged[CONFIDENCE_SCORE_COL].to_numpy()
    extra["low_confidence_caveat"] = flagged["inference_caveat"].to_numpy()
    out = pd.concat([df.reset_index(drop=True), pd.DataFrame(extra)], axis=1)

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

    stability_note = (
        "Multi-seed stability not run (AE retraining is costly on large matrices). "
        "Run: python -c \"from retrofittrust.quality.evaluation import multi_seed_stability; "
        "...\" or scripts/02_train_quality_screen.py --stability on a smaller sample."
    )
    stability_metrics: dict[str, Any] = {}
    if run_stability:
        seeds_to_run = stability_seeds
        try:
            stab = multi_seed_stability(X, seeds=seeds_to_run, screen_kwargs=screen_kwargs)
            stability_metrics = {
                "seeds": stab.seeds,
                "mean_jaccard": stab.mean_jaccard,
                "pairwise_jaccard": stab.pairwise_jaccard,
                "flag_rates_by_seed": stab.flag_rates,
            }
            stability_note = stab.notes
            logger.info(stability_note)
            for s1, s2, j in stab.pairwise_jaccard:
                logger.info("  Jaccard(seed %s, %s) = %.3f", s1, s2, j)
        except Exception as exc:
            logger.warning("Multi-seed stability check failed (%s). Document limitation.", exc)

    figure_path = REPORTS_FIGURES / "quality_screen_summary.png"
    metrics: dict[str, Any] = {
        "input_rows": n_in,
        "output_rows": len(out),
        "n_features": int(X.shape[1]),
        "input_source": input_source,
        "flagged_rate": report["operational_flag_rate"],
        "flagged_rate_consensus": report["flag_rate_consensus"],
        "flagged_rate_union": report["flag_rate_union"],
        "synthetic_injection_recall": recall.recall,
        "synthetic_chance_baseline": recall.chance_baseline,
        "synthetic_beats_chance": recall.beats_chance,
        "synthetic_recall_by_kind": recall.recall_by_kind,
        "threshold_tune": {
            "k": tune_result.k,
            "target_flag_rate": tune_result.target_flag_rate,
            "prefer_evt": tune_result.prefer_evt,
            "recall": tune_result.recall,
            "literature_ok": tune_result.literature_ok,
        },
        "literature_band": LITERATURE_FLAG_RATE_RANGE,
        "literature_ok": lit["ok"],
        "ae_threshold_method": (
            screen.ensemble.ae_threshold_.method if screen.ensemble.ae_threshold_ else None
        ),
        "stability": stability_metrics,
        "stability_note": stability_note,
        "output_path": str(flagged_path),
        "model_path": str(model_path),
        "figure_path": str(figure_path),
    }
    _save_quality_figure(metrics, path=figure_path)
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
