"""Checkpoint-1 adapter — delegates to modular Program 1 loaders.

The full ingest implementation lives in ``load_epc``, ``merge``, ``targets``,
etc. This module only provides the entry points expected by
``scripts/01_ingest_and_merge.py`` and ``api/features.py``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from retrofittrust.config import DATA_PROCESSED, EPC_GAP_WEIGHT, IMD_INCOME_WEIGHT, SEED
from retrofittrust.data.merge import build_merged_dataset
from retrofittrust.data.targets import add_priority_score

logger = logging.getLogger(__name__)


def load_merged_dataset(processed_dir: Path | None = None) -> pd.DataFrame:
    """Load the merged LSOA/property dataset from ``data/processed/``."""
    processed_dir = Path(processed_dir or DATA_PROCESSED)
    for name in (
        "merged_lsoa.parquet",
        "quality_flagged.parquet",
        "merged_with_priority.parquet",
        "merged_lsoa.csv",
    ):
        path = processed_dir / name
        if path.exists():
            return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    interim = processed_dir.parent / "interim" / "merged_with_priority.parquet"
    if interim.exists():
        return pd.read_parquet(interim)
    raise FileNotFoundError(
        f"No merged dataset in {processed_dir}. Run scripts/01_ingest_and_merge.py first."
    )


def run_ingest_and_merge(
    *,
    raw_dir: Path,
    interim_dir: Path,
    processed_dir: Path,
    seed: int = SEED,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load, join on ``lsoa21cd``, attach target; write ``merged_lsoa.parquet``.

    Does **not** run ``preprocess`` — that belongs after program 2 (quality
    screen), per the modular pipeline design.
    """
    np.random.seed(seed)
    raw_dir = Path(raw_dir)
    interim_dir = Path(interim_dir)
    processed_dir = Path(processed_dir)
    interim_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    merged = build_merged_dataset(
        save_interim=True,
        include_geography=True,
        raw_dir=raw_dir,
        external_dir=raw_dir.parent / "external",
    )
    coverage = merged.attrs.get("coverage", {})
    epc_rows = int(coverage.get("epc_rows", len(merged)))

    scored = add_priority_score(merged)
    out_path = processed_dir / "merged_lsoa.parquet"
    scored.to_parquet(out_path, index=False)

    # Mirror to interim path used by run_ingest()
    scored.to_parquet(interim_dir / "merged_with_priority.parquet", index=False)

    imd_matched = int(scored["imd_matched"].sum()) if "imd_matched" in scored.columns else len(scored)
    join_retention = imd_matched / epc_rows if epc_rows else 1.0

    metrics: dict[str, Any] = {
        "row_counts": {
            "epc_birmingham": epc_rows,
            "imd_lsoa": int(coverage.get("imd_lsoas", scored["lsoa21cd"].nunique())),
            "census_lsoa": int(coverage.get("census_lsoas", scored["lsoa21cd"].nunique())),
        },
        "merged_rows": len(scored),
        "unique_lsoa": int(scored["lsoa21cd"].nunique()),
        "join_retention": float(join_retention),
        "rows_dropped_on_join": max(0, epc_rows - len(scored)),
        "target_formula": (
            f"{EPC_GAP_WEIGHT} * normalised_epc_gap + "
            f"{IMD_INCOME_WEIGHT} * normalised_imd_income (higher = priority)"
        ),
        "coverage": coverage,
    }
    logger.info("Saved merged dataset to %s (%s rows)", out_path, f"{len(scored):,}")
    return scored, metrics
