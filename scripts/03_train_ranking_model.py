#!/usr/bin/env python
"""
Checkpoint 3 — LightGBM ranking model + SHAP (CURSOR_BUILD_SPEC §8.3).

Trains gradient-boosted tree ranker on quality-screened data when available.
Falls back to merged / EPC+IMD / labelled synthetic data with a logged warning.
Flagged records receive down-weighted sample weights (quarantine — not silent deletion).
Reports cross-validated performance and saves serialised model to models/.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from retrofittrust.config import (  # noqa: E402
    DATA_PROCESSED,
    MODELS_DIR,
    REPORTS_FIGURES,
    SEED,
)
from retrofittrust.modeling import run_ranking_training  # noqa: E402
from retrofittrust.modeling.train import (  # noqa: E402
    DEFAULT_MODEL_PATH,
    RETROFIT_SCORES_CSV,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

FLAGGED_INPUT = DATA_PROCESSED / "quality_flagged.parquet"


def _ensure_dirs() -> None:
    for path in (DATA_PROCESSED, MODELS_DIR, REPORTS_FIGURES):
        path.mkdir(parents=True, exist_ok=True)


def _print_checkpoint(metrics: dict) -> None:
    """Emit checkpoint-3 summary aligned with build spec §8."""
    logger.info("=" * 60)
    logger.info("CHECKPOINT 3 — LightGBM ranking model + SHAP")
    logger.info("=" * 60)

    data_source = metrics.get("data_source")
    if data_source:
        logger.info("  %-28s %s", "data source:", data_source)

    input_rows = metrics.get("input_rows")
    if input_rows is not None:
        logger.info("  %-28s %s rows", "input rows:", f"{input_rows:,}")

    train_rows = metrics.get("train_rows")
    if train_rows is not None:
        logger.info("  %-28s %s rows", "training rows:", f"{train_rows:,}")

    lsoa_rows = metrics.get("lsoa_export_rows")
    if lsoa_rows is not None:
        logger.info("  %-28s %s LSOAs", "consumer export:", f"{lsoa_rows:,}")

    flagged_in_train = metrics.get("flagged_rate_in_train")
    if flagged_in_train is not None:
        logger.info("  %-28s %.1f%%", "flagged in train set:", flagged_in_train * 100)

    cv_rmse = metrics.get("cv_rmse")
    cv_r2 = metrics.get("cv_r2")
    if cv_rmse is not None:
        std = metrics.get("cv_rmse_std")
        if std is not None:
            logger.info("  %-28s %.4f ± %.4f", "CV RMSE:", cv_rmse, std)
        else:
            logger.info("  %-28s %.4f", "CV RMSE:", cv_rmse)
    if cv_r2 is not None:
        logger.info("  %-28s %.4f", "CV R²:", cv_r2)

    rf_rmse = metrics.get("baseline_rf_cv_rmse")
    if rf_rmse is not None:
        logger.info("  %-28s %.4f", "RF baseline CV RMSE:", rf_rmse)

    for label, key in (
        ("SHAP beeswarm:", "shap_beeswarm"),
        ("SHAP bar:", "shap_bar"),
        ("SHAP waterfall:", "shap_waterfall_saved"),
        ("CV metrics figure:", "cv_metrics_figure"),
        ("Weight sensitivity fig:", "weight_sensitivity_figure"),
        ("Ranking numbers md:", "ranking_numbers_md"),
    ):
        value = metrics.get(key)
        if value:
            logger.info("  %-28s %s", label, value)

    scores_path = metrics.get("retrofit_scores_csv", RETROFIT_SCORES_CSV)
    logger.info("  %-28s %s", "Scores table:", scores_path)

    target_note = metrics.get("target_formula")
    if target_note:
        logger.info("  Target: %s", target_note)

    for note in metrics.get("face_validity_notes") or []:
        logger.info("  Face validity: %s", note)

    logger.info("  Model:  %s", DEFAULT_MODEL_PATH)
    logger.info("=" * 60)


def main() -> int:
    np.random.seed(SEED)
    _ensure_dirs()

    logger.info("Starting checkpoint 3 (SEED=%s)", SEED)

    if not FLAGGED_INPUT.exists():
        logger.warning(
            "Quality-flagged dataset not found at %s — "
            "using fallback loaders (merged / EPC+IMD / synthetic).",
            FLAGGED_INPUT,
        )

    metrics = run_ranking_training(
        flagged_path=FLAGGED_INPUT if FLAGGED_INPUT.exists() else None,
        processed_dir=DATA_PROCESSED,
        models_dir=MODELS_DIR,
        reports_dir=REPORTS_FIGURES,
        seed=SEED,
    )

    _print_checkpoint(metrics)

    if not DEFAULT_MODEL_PATH.exists():
        logger.error(
            "Ranking model not found at %s. Training did not complete.",
            DEFAULT_MODEL_PATH,
        )
        return 1

    logger.info("Checkpoint 3 complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
