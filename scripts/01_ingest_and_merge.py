#!/usr/bin/env python
"""
Checkpoint 1 — Data ingestion & Birmingham filtering (CURSOR_BUILD_SPEC §8.1).

Loads EPC (Birmingham LA), IMD 2025, Census TS054 + Central Heating; joins on
2021 LSOA code. Saves intermediates to data/interim and merged output to
data/processed. Reports row counts and join-loss sanity metrics.
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
    DATA_RAW,
    SEED,
)
from retrofittrust.data import run_ingest_and_merge  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

MERGED_OUTPUT = DATA_PROCESSED / "merged_lsoa.parquet"

# Literature sanity bounds for join retention (no silent mass drop)
MIN_JOIN_RETENTION = 0.85


def _ensure_dirs() -> None:
    for path in (DATA_RAW, DATA_INTERIM, DATA_PROCESSED):
        path.mkdir(parents=True, exist_ok=True)


def _print_checkpoint(metrics: dict) -> None:
    """Emit checkpoint-1 summary aligned with build spec §8."""
    logger.info("=" * 60)
    logger.info("CHECKPOINT 1 — Data ingestion & Birmingham merge")
    logger.info("=" * 60)

    for source, count in metrics.get("row_counts", {}).items():
        logger.info("  %-24s %s rows", source + ":", f"{count:,}")

    merged = metrics.get("merged_rows")
    if merged is not None:
        logger.info("  %-24s %s rows", "merged (final):", f"{merged:,}")

    unique_lsoa = metrics.get("unique_lsoa")
    if unique_lsoa is not None:
        logger.info("  %-24s %s", "unique LSOA codes:", f"{unique_lsoa:,}")

    retention = metrics.get("join_retention")
    if retention is not None:
        logger.info("  %-24s %.1f%%", "EPC join retention:", retention * 100)
        if retention < MIN_JOIN_RETENTION:
            logger.warning(
                "Join retention below %.0f%% — investigate silent data loss.",
                MIN_JOIN_RETENTION * 100,
            )

    dropped = metrics.get("rows_dropped_on_join", 0)
    if dropped:
        logger.info("  %-24s %s rows", "dropped on join:", f"{dropped:,}")

    logger.info("  Output: %s", MERGED_OUTPUT)
    logger.info("=" * 60)


def main() -> int:
    np.random.seed(SEED)
    _ensure_dirs()

    logger.info("Starting checkpoint 1 (SEED=%s)", SEED)
    logger.info("Raw data directory: %s", DATA_RAW)

    result = run_ingest_and_merge(
        raw_dir=DATA_RAW,
        interim_dir=DATA_INTERIM,
        processed_dir=DATA_PROCESSED,
        seed=SEED,
    )

    # Sibling agent may return (DataFrame, metrics) or metrics-only dict after saving.
    if isinstance(result, tuple) and len(result) == 2:
        _df, metrics = result
    elif isinstance(result, dict):
        metrics = result
    else:
        metrics = getattr(result, "metrics", {}) or {}

    _print_checkpoint(metrics)

    if not MERGED_OUTPUT.exists() and metrics.get("merged_rows", 0) == 0:
        logger.error(
            "Merged dataset not found at %s. Implement retrofittrust.data.pipeline.",
            MERGED_OUTPUT,
        )
        return 1

    logger.info("Checkpoint 1 complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
