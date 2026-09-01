#!/usr/bin/env python
"""
Checkpoint 2 — Autoencoder + Isolation Forest screening (CURSOR_BUILD_SPEC §8.2).

Trains PyOD ensemble on merged EPC records, attaches anomaly flags and per-feature
reconstruction errors. Quarantines flagged records — never silently deletes.
Reports overall flagged rate against EPC error-rate literature (~27–60%).

If ``merged_lsoa.parquet`` is not ready, falls back to a Birmingham EPC sample or
a labelled synthetic frame (see ``retrofittrust.quality.screen.load_screening_input``).
"""

from __future__ import annotations

import argparse
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AE + IForest quality screen.")
    parser.add_argument(
        "--stability",
        action="store_true",
        help="Run multi-seed Jaccard overlap (3 seeds; slower on large data).",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=MERGED_INPUT,
        help="Merged LSOA parquet/csv (default: data/processed/merged_lsoa.parquet).",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=8000,
        help="Stratified LSOA sample size for AE training (0 = use full table).",
    )
    return parser.parse_args()


def _ensure_dirs() -> None:
    for path in (DATA_INTERIM, DATA_PROCESSED, MODELS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def _print_checkpoint(metrics: dict) -> None:
    """Emit checkpoint-2 summary aligned with build spec §8."""
    logger.info("=" * 60)
    logger.info("CHECKPOINT 2 — Data-quality screening (AE + IForest)")
    logger.info("=" * 60)

    source = metrics.get("input_source")
    if source:
        logger.info("  %-28s %s", "input source:", source)

    input_rows = metrics.get("input_rows")
    if input_rows is not None:
        logger.info("  %-28s %s rows", "input:", f"{input_rows:,}")

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
                "Consensus rate outside literature range (%.0f–%.0f%%). "
                "Union mode is operational; investigate thresholds if needed.",
                FLAGGED_RATE_LIT_LOW * 100,
                FLAGGED_RATE_LIT_HIGH * 100,
            )

    union_rate = metrics.get("flagged_rate_union")
    if union_rate is not None:
        logger.info("  %-28s %.1f%%", "union flagged rate (ops):", union_rate * 100)
        if union_rate < FLAGGED_RATE_LIT_LOW or union_rate > FLAGGED_RATE_LIT_HIGH:
            logger.warning(
                "Union flagged rate outside literature range (%.0f–%.0f%%). "
                "Investigate threshold tuning before trusting the screen.",
                FLAGGED_RATE_LIT_LOW * 100,
                FLAGGED_RATE_LIT_HIGH * 100,
            )

    synth_recall = metrics.get("synthetic_injection_recall")
    chance = metrics.get("synthetic_chance_baseline")
    if synth_recall is not None:
        logger.info("  %-28s %.1f%%", "synthetic injection recall:", synth_recall * 100)
        if chance is not None:
            logger.info("  %-28s %.1f%%", "chance baseline (flag rate):", chance * 100)
        if synth_recall <= 0.5:
            logger.warning(
                "Synthetic injection recall ≤ 50%% — screen may be no better than chance."
            )
        elif metrics.get("synthetic_beats_chance") is False:
            logger.warning("Injection recall did not clearly beat chance (+5pp).")

    tune = metrics.get("threshold_tune") or {}
    if tune:
        logger.info(
            "  %-28s k=%.2f target=%.2f evt=%s",
            "threshold tuning:",
            tune.get("k", 0),
            tune.get("target_flag_rate", 0),
            tune.get("prefer_evt"),
        )

    stab = metrics.get("stability") or {}
    if stab.get("mean_jaccard") is not None:
        logger.info("  %-28s %.3f", "multi-seed mean Jaccard:", stab["mean_jaccard"])
    elif metrics.get("stability_note"):
        logger.info("  %-28s %s", "stability:", metrics["stability_note"][:50] + "...")

    fig = metrics.get("figure_path")
    if fig:
        logger.info("  Figure: %s", fig)

    logger.info("  Output: %s", FLAGGED_OUTPUT)
    logger.info("  Model:  %s", QUALITY_MODEL)
    logger.info("=" * 60)


def main() -> int:
    args = _parse_args()
    np.random.seed(SEED)
    _ensure_dirs()

    logger.info("Starting checkpoint 2 (SEED=%s)", SEED)
    if not args.input.exists():
        logger.warning(
            "Merged dataset not found at %s — will use EPC sample or synthetic fallback.",
            args.input,
        )

    max_rows = None if args.max_rows <= 0 else args.max_rows
    result = run_quality_screen(
        merged_path=args.input,
        interim_dir=DATA_INTERIM,
        processed_dir=DATA_PROCESSED,
        models_dir=MODELS_DIR,
        seed=SEED,
        run_stability=args.stability,
        max_rows=max_rows,
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
