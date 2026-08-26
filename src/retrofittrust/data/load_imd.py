"""Load English Indices of Deprivation 2025 at 2021 LSOA level.

Source: GOV.UK statistical release of **30 October 2025** (not 17 November).
Preferred file is File 7 (all ranks, scores, deciles and population
denominators), placed under ``data/raw/imd2025/``.

Ecological fallacy
------------------
IMD scores describe *areas* (LSOAs), not households. Joining an LSOA
income-deprivation score onto an individual EPC dwelling does **not**
mean that household is income-deprived. This is the ecological fallacy
and must be stated wherever the composite priority score is used.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ..config import BIRMINGHAM_LA, DATA_RAW, SEED
from ._utils import (
    BIRMINGHAM_LAD_CODE,
    ensure_logging,
    find_column,
    is_birmingham_lsoa_name,
    list_data_files,
    log_row_count,
    looks_like_birmingham,
    read_table,
    snake_case_columns,
    standardise_lsoa_key,
)

logger = logging.getLogger(__name__)

_ = SEED

# Friendly names used downstream (targets, dashboard, SHAP).
COLUMN_RENAMES = {
    "lsoa_code_2021": "lsoa21cd",
    "lsoa_name_2021": "lsoa21nm",
    "local_authority_district_code_2024": "lad24cd",
    "local_authority_district_name_2024": "lad24nm",
    "local_authority_district_code_2023": "lad23cd",
    "local_authority_district_name_2023": "lad23nm",
    "local_authority_district_code_2021": "lad21cd",
    "local_authority_district_name_2021": "lad21nm",
}


def _discover_imd_files(raw_dir: Path) -> list[Path]:
    folder = raw_dir / "imd2025"
    files = list_data_files(folder)
    if files:
        return files
    # Allow a loosely placed IoD2025 CSV at the raw root.
    extras = [
        p
        for p in list_data_files(raw_dir, recursive=False)
        if "imd" in p.name.lower() or "iod2025" in p.name.lower() or "iod_2025" in p.name.lower()
    ]
    return extras


def _prefer_file7(paths: list[Path]) -> list[Path]:
    """File 7 is the all-in-one ranks/scores/deciles extract."""
    file7 = [
        p
        for p in paths
        if "file_7" in p.name.lower() or "file7" in p.name.lower() or "all_ranks" in p.name.lower()
    ]
    return file7 or paths


def _rename_imd_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = snake_case_columns(df)
    out = out.rename(columns={k: v for k, v in COLUMN_RENAMES.items() if k in out.columns})

    # Domain columns have long official names; map the ones the target uses.
    imd_score = find_column(
        out.columns,
        "index_of_multiple_deprivation_imd_score",
        "imd_score",
        "index_of_multiple_deprivation_score",
    )
    imd_rank = find_column(
        out.columns,
        "index_of_multiple_deprivation_imd_rank_where_1_is_most_deprived",
        "index_of_multiple_deprivation_imd_rank",
        "imd_rank",
    )
    imd_decile = find_column(
        out.columns,
        "index_of_multiple_deprivation_imd_decile_where_1_is_most_deprived_10_of_lsoas",
        "index_of_multiple_deprivation_imd_decile",
        "imd_decile",
    )
    # Income *score* (rate) — not IDACI / IDAOPI supplementary indices.
    income_score = None
    for col in out.columns:
        cs = str(col)
        if "idaci" in cs or "idaopi" in cs or "children" in cs or "older" in cs:
            continue
        if cs in {"income_score_rate", "income_score"} or (
            "income_score" in cs and "rank" not in cs and "decile" not in cs
        ):
            income_score = col
            break
    income_rank = None
    for col in out.columns:
        cs = str(col)
        if "idaci" in cs or "idaopi" in cs or "children" in cs or "older" in cs:
            continue
        if "income_rank" in cs:
            income_rank = col
            break
    income_decile = None
    for col in out.columns:
        cs = str(col)
        if "idaci" in cs or "idaopi" in cs or "children" in cs or "older" in cs:
            continue
        if "income_decile" in cs:
            income_decile = col
            break

    mapping: dict[str, str] = {}
    if imd_score and imd_score != "imd_score":
        mapping[imd_score] = "imd_score"
    if imd_rank and imd_rank != "imd_rank":
        mapping[imd_rank] = "imd_rank"
    if imd_decile and imd_decile != "imd_decile":
        mapping[imd_decile] = "imd_decile"
    if income_score and income_score != "imd_income_score":
        mapping[income_score] = "imd_income_score"
    if income_rank and income_rank != "imd_income_rank":
        mapping[income_rank] = "imd_income_rank"
    if income_decile and income_decile != "imd_income_decile":
        mapping[income_decile] = "imd_income_decile"
    if mapping:
        out = out.rename(columns=mapping)
    return out


def _birmingham_mask(df: pd.DataFrame, local_authority: str) -> pd.Series:
    lad_name = find_column(
        df.columns,
        "lad24nm",
        "lad23nm",
        "lad21nm",
        "local_authority_district_name_2024",
        "local_authority_district_name",
        "ladnm",
        "la_name",
    )
    lad_code = find_column(
        df.columns,
        "lad24cd",
        "lad23cd",
        "lad21cd",
        "local_authority_district_code_2024",
        "ladcd",
    )
    lsoa_name = find_column(df.columns, "lsoa21nm", "lsoa_name_2021", "lsoa_name")

    masks: list[pd.Series] = []
    if lad_name is not None:
        labels = df[lad_name].astype("string")
        masks.append(
            looks_like_birmingham(labels)
            | labels.str.fullmatch(local_authority, case=False, na=False)
        )
    if lad_code is not None:
        masks.append(
            df[lad_code].astype("string").str.strip().str.upper().eq(BIRMINGHAM_LAD_CODE)
        )
    if lsoa_name is not None:
        masks.append(is_birmingham_lsoa_name(df[lsoa_name]))

    if not masks:
        logger.warning(
            "IMD file has no LAD / LSOA name column to filter on; "
            "returning all rows. Downstream merge will still attach by lsoa21cd."
        )
        return pd.Series(True, index=df.index)

    combined = masks[0]
    for extra in masks[1:]:
        combined = combined | extra
    return combined.fillna(False)


def load_imd(
    raw_dir: Path | None = None,
    local_authority: str | None = None,
) -> pd.DataFrame:
    """Load IMD 2025 LSOA statistics and filter to Birmingham.

    Parameters
    ----------
    raw_dir:
        Raw data root (defaults to ``config.DATA_RAW``). Files are expected
        in ``data/raw/imd2025/``.
    local_authority:
        LAD name used to filter (defaults to ``config.BIRMINGHAM_LA``).

    Returns
    -------
    pandas.DataFrame
        One row per Birmingham 2021 LSOA, including at least ``lsoa21cd``
        and, where present, ``imd_income_score`` (rate; higher = more
        income-deprived).

    Notes
    -----
    Rank columns use "1 = most deprived". The composite target in
    ``targets.py`` uses the income *score* (rate), where higher already
    means higher deprivation — do not invert it.
    """
    ensure_logging()
    raw_dir = Path(raw_dir) if raw_dir is not None else DATA_RAW
    local_authority = local_authority or BIRMINGHAM_LA

    paths = _prefer_file7(_discover_imd_files(raw_dir))
    if not paths:
        raise FileNotFoundError(
            f"No IMD 2025 file found in {raw_dir / 'imd2025'}. "
            "Download File 7 (CSV) from the English indices of deprivation "
            "2025 GOV.UK release (published 30 October 2025). "
            "See data/raw/README.md."
        )

    logger.info("IMD files: %s", ", ".join(p.name for p in paths))
    frames = [_rename_imd_columns(read_table(p)) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    df = standardise_lsoa_key(df, source="load_imd")
    df = df.loc[_birmingham_mask(df, local_authority)].copy()

    if df.empty:
        raise ValueError(
            f"IMD 2025 loaded but no rows matched {local_authority!r}. "
            "Check that the file uses 2021 LSOAs and a LAD name/code column."
        )

    before = len(df)
    df = df.drop_duplicates(subset=["lsoa21cd"], keep="first")
    if len(df) < before:
        logger.info("Dropped %s duplicate LSOA rows in IMD", f"{before - len(df):,}")

    if "imd_income_score" not in df.columns:
        if "imd_income_rank" in df.columns:
            logger.warning(
                "IMD income *score* is missing; deriving a higher-is-more-deprived "
                "proxy from income rank (rank 1 = most deprived). Prefer File 7, "
                "which includes Income Score (rate)."
            )
            rank = pd.to_numeric(df["imd_income_rank"], errors="coerce")
            span = rank.max() - rank.min()
            if pd.isna(span) or span == 0:
                df["imd_income_score"] = 0.5
            else:
                # Invert rank so higher value = more deprived (like the official rate).
                df["imd_income_score"] = (rank.max() - rank) / span
        else:
            logger.warning(
                "IMD file has neither income score nor income rank. "
                "The composite target will be incomplete."
            )

    log_row_count(
        "load_imd",
        len(df),
        unique_lsoa=df["lsoa21cd"].nunique(),
        has_income_score="imd_income_score" in df.columns,
    )
    return df.reset_index(drop=True)
