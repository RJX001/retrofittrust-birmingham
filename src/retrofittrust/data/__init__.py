"""Program 1 data pipeline: ingest, join, preprocess, composite target.

Public entry points
-------------------
``load_epc``, ``load_imd``, ``load_census``, ``load_geography``
    Read from immutable ``data/raw`` (geography from ``data/external``).
``merge_datasets`` / ``build_merged_dataset``
    Join on 2021 LSOA code with logged row counts (no silent drops).
``add_priority_score``
    0.6 × normalised EPC gap + 0.4 × normalised IMD income domain.
``preprocess``
    Median impute, missingness flags, standardise numerics, one-hot categoricals.

``run_ingest`` loads, joins and attaches the target. It does **not**
run ``preprocess``: program 2 (anomaly screening) is intended to sit
between the merge and the feature matrix.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ..config import DATA_INTERIM
from ._utils import ensure_logging
from .load_census import load_census
from .load_epc import load_epc
from .load_geography import load_geography
from .load_imd import load_imd
from .merge import build_merged_dataset, merge_datasets
from .pipeline import load_merged_dataset, run_ingest_and_merge
from .preprocess import PreprocessState, preprocess
from .targets import add_priority_score, compute_epc_gap

logger = logging.getLogger(__name__)

__all__ = [
    "PreprocessState",
    "add_priority_score",
    "build_merged_dataset",
    "compute_epc_gap",
    "load_census",
    "load_epc",
    "load_geography",
    "load_imd",
    "merge_datasets",
    "preprocess",
    "run_ingest",
    "run_ingest_and_merge",
    "load_merged_dataset",
]


def run_ingest(
    *,
    save_interim: bool = True,
    include_geography: bool = True,
    raw_dir: Path | None = None,
    external_dir: Path | None = None,
) -> pd.DataFrame:
    """Load sources, join on ``lsoa21cd``, attach ``retrofit_priority_score``.

    Does not one-hot / scale: call :func:`preprocess` after the quality
    screen (program 2), and never drop flagged rows.
    """
    ensure_logging()
    merged = build_merged_dataset(
        save_interim=False,
        include_geography=include_geography,
        raw_dir=raw_dir,
        external_dir=external_dir,
    )
    scored = add_priority_score(merged)
    if save_interim:
        DATA_INTERIM.mkdir(parents=True, exist_ok=True)
        dest = DATA_INTERIM / "merged_with_priority.parquet"
        scored.to_parquet(dest, index=False)
        logger.info("Wrote %s (%s rows)", dest, f"{len(scored):,}")
    return scored
