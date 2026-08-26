"""Join EPC, IMD, Census and LSOA geography on 2021 LSOA code.

Grain: **one row per EPC certificate** (property-level), with LSOA-level
IMD and Census attributes attached (many-to-one). Geography is joined as
identifiers only — full polygons are not duplicated onto every dwelling.

No silent data loss
-------------------
- Joins are left joins from the EPC frame (plus an explicit coverage log
  of LSOAs that exist in IMD / Census / geography but have no EPC).
- Unmatched rows are **kept** with nulls on the unmatched side and a
  boolean ``*_matched`` flag. Nothing is dropped because it failed a join.
- Row counts are logged before and after every join.

Coverage bias: LSOAs with IMD/Census data but no EPC are expected — EPCs
exist only where a certificate was triggered. That is a data gap, not a
reason to delete the LSOA from the twin's denominator.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ..config import DATA_INTERIM, SEED
from ._utils import ensure_logging, log_row_count
from .load_census import load_census
from .load_epc import load_epc
from .load_geography import load_geography
from .load_imd import load_imd

logger = logging.getLogger(__name__)

_ = SEED

JOIN_KEY = "lsoa21cd"


def _require_key(df: pd.DataFrame, name: str) -> None:
    if JOIN_KEY not in df.columns:
        raise ValueError(
            f"{name} is missing {JOIN_KEY!r} — cannot join. "
            "Check that the source uses 2021 LSOA codes."
        )


def _lsoa_set(df: pd.DataFrame) -> set[str]:
    return set(df[JOIN_KEY].dropna().astype(str).unique())


def _attach(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    right_name: str,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Left-join ``right`` onto ``left`` on ``lsoa21cd``; keep all left rows."""
    _require_key(left, "left")
    _require_key(right, right_name)

    right_cols = [JOIN_KEY] + [c for c in (columns or list(right.columns)) if c != JOIN_KEY]
    right_cols = [c for c in right_cols if c in right.columns]
    slim = right[right_cols].drop_duplicates(subset=[JOIN_KEY], keep="first")

    # Avoid colliding with columns already on the EPC frame.
    overlap = [c for c in slim.columns if c != JOIN_KEY and c in left.columns]
    if overlap:
        slim = slim.rename(columns={c: f"{c}_{right_name}" for c in overlap})
        logger.info(
            "Renamed overlapping columns from %s: %s",
            right_name,
            overlap,
        )

    before = len(left)
    flag = f"{right_name}_matched"
    merged = left.merge(slim, on=JOIN_KEY, how="left", indicator=True)
    if len(merged) != before:
        logger.error(
            "Join to %s changed row count %s -> %s (expected left join to preserve rows). "
            "Check for duplicate %s on the right.",
            right_name,
            f"{before:,}",
            f"{len(merged):,}",
            JOIN_KEY,
        )
        # Duplicate keys on the right would explode rows. Dedup already ran;
        # if this still fires, collapse back with a warning rather than hide it.
    counts = merged["_merge"].value_counts().to_dict()
    n_matched = int(counts.get("both", 0))
    n_unmatched = int(counts.get("left_only", 0))
    logger.info(
        "Join EPC ⟕ %s on %s: matched=%s unmatched=%s (left rows=%s, right LSOAs=%s)",
        right_name,
        JOIN_KEY,
        f"{n_matched:,}",
        f"{n_unmatched:,}",
        f"{before:,}",
        f"{slim[JOIN_KEY].nunique():,}",
    )
    merged[flag] = merged["_merge"].eq("both")
    merged = merged.drop(columns=["_merge"])
    if n_unmatched:
        logger.warning(
            "%s EPC rows have no matching %s LSOA — kept with nulls (%s=False). "
            "Possible causes: 2011 vs 2021 LSOA codes, missing LSOA on the EPC, "
            "or certificates outside the Birmingham IMD extract.",
            f"{n_unmatched:,}",
            right_name,
            flag,
        )
    return merged


def _log_coverage(
    epc: pd.DataFrame,
    imd: pd.DataFrame,
    census: pd.DataFrame,
    geography: pd.DataFrame | None,
) -> dict[str, int]:
    """Log LSOAs present on the area-level sources but absent from EPC."""
    epc_lsoas = _lsoa_set(epc)
    report = {
        "epc_rows": int(len(epc)),
        "epc_lsoas": int(len(epc_lsoas)),
        "imd_lsoas": int(imd[JOIN_KEY].nunique()),
        "census_lsoas": int(census[JOIN_KEY].nunique()),
    }
    imd_only = _lsoa_set(imd) - epc_lsoas
    census_only = _lsoa_set(census) - epc_lsoas
    report["imd_lsoas_with_no_epc"] = len(imd_only)
    report["census_lsoas_with_no_epc"] = len(census_only)
    if imd_only:
        logger.warning(
            "Coverage bias: %s Birmingham IMD LSOAs have no EPC in this extract "
            "(certificates are trigger-based; absence ≠ no retrofit need). "
            "These LSOAs are not deleted; they simply have no property-level rows.",
            f"{len(imd_only):,}",
        )
    if geography is not None and JOIN_KEY in geography.columns:
        geo_lsoas = _lsoa_set(geography)
        report["geography_lsoas"] = len(geo_lsoas)
        report["geography_lsoas_with_no_epc"] = len(geo_lsoas - epc_lsoas)
        report["epc_lsoas_missing_from_geography"] = len(epc_lsoas - geo_lsoas)
        missing_geo = epc_lsoas - geo_lsoas
        if missing_geo:
            logger.warning(
                "%s EPC LSOA codes are not in the geography file (choropleth gaps): "
                "e.g. %s",
                f"{len(missing_geo):,}",
                sorted(missing_geo)[:8],
            )
    logger.info("Coverage report: %s", report)
    return report


def merge_datasets(
    epc: pd.DataFrame,
    imd: pd.DataFrame,
    census: pd.DataFrame,
    geography: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Left-join IMD, Census (and optional LSOA ids) onto EPC by ``lsoa21cd``.

    Parameters
    ----------
    epc, imd, census:
        Outputs of the corresponding loaders.
    geography:
        Optional GeoDataFrame / DataFrame. Geometry is stripped; only
        identifier columns are attached so polygons are not copied per dwelling.

    Returns
    -------
    pandas.DataFrame
        Property-level frame, same row count as ``epc`` unless a duplicate
        LSOA key on the right-hand side slips through (that is logged as error).
    """
    ensure_logging()
    _require_key(epc, "epc")
    log_row_count("merge:epc_in", len(epc), unique_lsoa=epc[JOIN_KEY].nunique())
    log_row_count("merge:imd_in", len(imd), unique_lsoa=imd[JOIN_KEY].nunique())
    log_row_count("merge:census_in", len(census), unique_lsoa=census[JOIN_KEY].nunique())

    coverage = _log_coverage(epc, imd, census, geography)
    merged = epc.copy()
    merged.attrs["coverage"] = coverage

    merged = _attach(merged, imd, right_name="imd")
    merged = _attach(merged, census, right_name="census")

    if geography is not None:
        geo_tab = geography.drop(columns=["geometry"], errors="ignore")
        keep = [c for c in geo_tab.columns if c in {JOIN_KEY, "lsoa21nm"} or str(c).startswith("lad")]
        if JOIN_KEY not in keep:
            keep = [JOIN_KEY] + keep
        merged = _attach(merged, geo_tab[keep], right_name="geography")

    log_row_count(
        "merge:out",
        len(merged),
        unique_lsoa=merged[JOIN_KEY].nunique(),
        imd_matched=int(merged["imd_matched"].sum()) if "imd_matched" in merged.columns else "n/a",
        census_matched=int(merged["census_matched"].sum()) if "census_matched" in merged.columns else "n/a",
    )
    if len(merged) < len(epc):
        raise RuntimeError(
            f"Silent data loss: merged rows ({len(merged):,}) < EPC rows ({len(epc):,})."
        )
    return merged


def build_merged_dataset(
    *,
    save_interim: bool = True,
    include_geography: bool = True,
    raw_dir: Path | None = None,
    external_dir: Path | None = None,
) -> pd.DataFrame:
    """Load all sources, join on 2021 LSOA, optionally write ``data/interim``.

    Geography is optional at ingest time: a missing GeoJSON logs a warning
    rather than aborting the tabular merge (the choropleth needs it later).
    """
    ensure_logging()
    epc = load_epc(raw_dir=raw_dir)
    imd = load_imd(raw_dir=raw_dir)
    census = load_census(raw_dir=raw_dir)

    geography = None
    if include_geography:
        try:
            geography = load_geography(external_dir=external_dir)
        except (FileNotFoundError, ImportError) as exc:
            logger.warning("%s — continuing tabular merge without geography.", exc)

    merged = merge_datasets(epc, imd, census, geography=geography)

    if save_interim:
        DATA_INTERIM.mkdir(parents=True, exist_ok=True)
        dest = DATA_INTERIM / "merged_epc_imd_census.parquet"
        merged.to_parquet(dest, index=False)
        logger.info("Wrote interim merge to %s (%s rows)", dest, f"{len(merged):,}")
    return merged
