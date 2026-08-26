#!/usr/bin/env python
"""
Checkpoint 3 — LightGBM ranking model + SHAP (CURSOR_BUILD_SPEC §8.3).

Trains gradient-boosted tree ranker on quality-screened data. Flagged records
receive down-weighted sample weights (quarantine — not silent deletion).
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

FLAGGED_INPUT = DATA_PROCESSED / "quality_flagged.parquet"
RANKING_MODEL = MODELS_DIR / "lgbm_ranker.joblib"


def _ensure_dirs() -> None:
    for path in (DATA_PROCESSED, MODELS_DIR, REPORTS_FIGURES):
        path.mkdir(parents=True, exist_ok=True)


def _print_checkpoint(metrics: dict) -> None:
    """Emit checkpoint-3 summary aligned with build spec §8."""
    logger.info("=" * 60)
    logger.info("CHECKPOINT 3 — LightGBM ranking model + SHAP")
    logger.info("=" * 60)

    input_rows = metrics.get("input_rows")
    if input_rows is not None:
        logger.info("  %-28s %s rows", "input (flagged):", f"{input_rows:,}")

    train_rows = metrics.get("train_rows")
    if train_rows is not None:
        logger.info("  %-28s %s rows", "training rows:", f"{train_rows:,}")

    flagged_in_train = metrics.get("flagged_rate_in_train")
    if flagged_in_train is not None:
        logger.info("  %-28s %.1f%%", "flagged in train set:", flagged_in_train * 100)

    cv_rmse = metrics.get("cv_rmse")
    cv_r2 = metrics.get("cv_r2")
    if cv_rmse is not None:
        logger.info("  %-28s %.4f", "CV RMSE:", cv_rmse)
    if cv_r2 is not None:
        logger.info("  %-28s %.4f", "CV R²:", cv_r2)

    shap_sample = metrics.get("shap_waterfall_saved")
    if shap_sample is not None:
        logger.info("  %-28s %s", "SHAP waterfall saved:", shap_sample)

    target_note = metrics.get("target_formula")
    if target_note:
        logger.info("  Target: %s", target_note)

    logger.info("  Model:  %s", RANKING_MODEL)
    logger.info("=" * 60)


def main() -> int:
    np.random.seed(SEED)
    _ensure_dirs()

    logger.info("Starting checkpoint 3 (SEED=%s)", SEED)

    if not FLAGGED_INPUT.exists():
        logger.error(
            "Quality-flagged dataset not found at %s. "
            "Run scripts/02_train_quality_screen.py first.",
            FLAGGED_INPUT,
        )
        return 1

    result = run_ranking_training(
        flagged_path=FLAGGED_INPUT,
        processed_dir=DATA_PROCESSED,
        models_dir=MODELS_DIR,
        reports_dir=REPORTS_FIGURES,
        seed=SEED,
    )

    if isinstance(result, tuple) and len(result) == 2:
        _model, metrics = result
    elif isinstance(result, dict):
        metrics = result
    else:
        metrics = getattr(result, "metrics", {}) or {}

    _print_checkpoint(metrics)

    if not RANKING_MODEL.exists():
        logger.error(
            "Ranking model not found at %s. Implement retrofittrust.modeling.train.",
            RANKING_MODEL,
        )
        return 1

    logger.info("Checkpoint 3 complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
