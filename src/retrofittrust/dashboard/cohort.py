"""Cohort selection helpers for the digital twin dashboard."""

from __future__ import annotations

import pandas as pd

from retrofittrust.config import DEMO_COHORT_LSOA_COUNT, SEED
from retrofittrust.dashboard.data_loader import load_lsoa_dataset


def _find_lsoa_column(df: pd.DataFrame) -> str:
    for col in ("lsoa21cd", "LSOA21CD", "lsoa_code"):
        if col in df.columns:
            return col
    raise ValueError("No LSOA column found in merged dataset")


def _load_merged_lsoa() -> pd.DataFrame:
    frame, _source = load_lsoa_dataset()
    return frame


def select_demo_cohort(n: int | None = None) -> list[str]:
    """Return n LSOA codes for the integration demo (default from config)."""
    n = n or DEMO_COHORT_LSOA_COUNT
    df = _load_merged_lsoa()
    col = _find_lsoa_column(df)
    unique = df[col].drop_duplicates()
    sample_n = min(n, unique.nunique())
    if "priority_score" in df.columns:
        ranked = (
            df.drop_duplicates(subset=[col])
            .sort_values("priority_score", ascending=False)
        )
        return ranked[col].head(sample_n).tolist()
    return unique.sample(n=sample_n, random_state=SEED).tolist()
