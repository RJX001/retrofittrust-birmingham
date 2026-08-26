#!/usr/bin/env python
"""
Checkpoint 2 — Autoencoder + Isolation Forest screening (CURSOR_BUILD_SPEC §8.2).

Trains PyOD ensemble on merged EPC records, attaches anomaly flags and per-feature
reconstruction errors. Quarantines flagged records — never silently deletes.
Reports overall flagged rate against EPC error-rate literature (~27–60%).
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
    DATA_INTERIM,
    DATA_PROCESSED,
    MODELS_DIR,
    SEED,
)
from retrofittrust.quality import run_quality_screen  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

MERGED_INPUT = DATA_PROCESSED / "merged_lsoa.parquet"
FLAGGED_OUTPUT = DATA_PROCESSED / "quality_flagged.parquet"
QUALITY_MODEL = MODELS_DIR / "quality_screen.joblib"

# EPC error-rate literature range (CURSOR_BUILD_SPEC §4)
FLAGGED_RATE_LIT_LOW = 0.27
FLAGGED_RATE_LIT_HIGH = 0.60


def _ensure_dirs() -> None:
    for path in (DATA_INTERIM, DATA_PROCESSED, MODELS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def _print_checkpoint(metrics: dict) -> None:
    """Emit checkpoint-2 summary aligned with build spec §8."""
    logger.info("=" * 60)
    logger.info("CHECKPOINT 2 — Data-quality screening (AE + IForest)")
    logger.info("=" * 60)

    input_rows = metrics.get("input_rows")
    if input_rows is not None:
        logger.info("  %-28s %s rows", "input (merged):", f"{input_rows:,}")

    output_rows = metrics.get("output_rows")
    if output_rows is not None:
        logger.info("  %-28s %s rows", "output (flagged):", f"{output_rows:,}")

    flagged_rate = metrics.get("flagged_rate_consensus")
    if flagged_rate is None:
        flagged_rate = metrics.get("flagged_rate")
    if flagged_rate is not None:
        logger.info("  %-28s %.1f%%", "consensus flagged rate:", flagged_rate * 100)
        if flagged_rate < FLAGGED_RATE_LIT_LOW or flagged_rate > FLAGGED_RATE_LIT_HIGH:
            logger.warning(
                "Flagged rate outside literature range (%.0f–%.0f%%). Investigate "
                "threshold tuning before trusting the screen.",
                FLAGGED_RATE_LIT_LOW * 100,
                FLAGGED_RATE_LIT_HIGH * 100,
            )

    union_rate = metrics.get("flagged_rate_union")
    if union_rate is not None:
        logger.info("  %-28s %.1f%%", "union flagged rate:", union_rate * 100)

    synth_recall = metrics.get("synthetic_injection_recall")
    if synth_recall is not None:
        logger.info("  %-28s %.1f%%", "synthetic injection recall:", synth_recall * 100)
        if synth_recall <= 0.5:
            logger.warning(
                "Synthetic injection recall ≤ 50%% — screen may be no better than chance."
            )

    logger.info("  Output: %s", FLAGGED_OUTPUT)
    logger.info("  Model:  %s", QUALITY_MODEL)
    logger.info("=" * 60)


def main() -> int:
    np.random.seed(SEED)
    _ensure_dirs()

    logger.info("Starting checkpoint 2 (SEED=%s)", SEED)

    if not MERGED_INPUT.exists():
        logger.error(
            "Merged dataset not found at %s. Run scripts/01_ingest_and_merge.py first.",
            MERGED_INPUT,
        )
        return 1

    result = run_quality_screen(
        merged_path=MERGED_INPUT,
        interim_dir=DATA_INTERIM,
        processed_dir=DATA_PROCESSED,
        models_dir=MODELS_DIR,
        seed=SEED,
    )

    if isinstance(result, tuple) and len(result) == 2:
        _df, metrics = result
    elif isinstance(result, dict):
        metrics = result
    else:
        metrics = getattr(result, "metrics", {}) or {}

    _print_checkpoint(metrics)

    if not FLAGGED_OUTPUT.exists() and metrics.get("output_rows", 0) == 0:
        logger.error(
            "Flagged dataset not found at %s. Implement retrofittrust.quality.screen.",
            FLAGGED_OUTPUT,
        )
        return 1

    logger.info("Checkpoint 2 complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
