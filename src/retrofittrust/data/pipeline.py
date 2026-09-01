"""Checkpoint-1 adapter — delegates to modular Program 1 loaders.

The full ingest implementation lives in ``load_epc``, ``merge``, ``targets``,
etc. This module only provides the entry points expected by
``scripts/01_ingest_and_merge.py`` and ``api/features.py``.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from retrofittrust.config import DATA_INTERIM, DATA_PROCESSED, EPC_GAP_WEIGHT, IMD_INCOME_WEIGHT, SEED
from retrofittrust.data.load_census import load_census
from retrofittrust.data.load_epc import load_epc
from retrofittrust.data.load_geography import load_geography
from retrofittrust.data.load_imd import load_imd
from retrofittrust.data.merge import merge_datasets
from retrofittrust.data.targets import add_priority_score

logger = logging.getLogger(__name__)

JOIN_AUDIT_PATH = DATA_PROCESSED / "join_audit.json"


def _null_summary(df: pd.DataFrame, columns: list[str] | None = None) -> dict[str, dict[str, float | int]]:
    """Per-column null counts and percentages for dissertation audit."""
    cols = columns or list(df.columns)
    out: dict[str, dict[str, float | int]] = {}
    n = len(df)
    for col in cols:
        if col not in df.columns:
            continue
        nulls = int(df[col].isna().sum())
        if nulls:
            out[col] = {
                "null_count": nulls,
                "null_pct": round(100.0 * nulls / n, 2) if n else 0.0,
            }
    return out


def _build_join_audit(
    *,
    epc: pd.DataFrame,
    imd: pd.DataFrame,
    census: pd.DataFrame,
    merged: pd.DataFrame,
    scored: pd.DataFrame,
    geography_loaded: bool,
    geography_rows: int | None,
    seed: int,
) -> dict[str, Any]:
    coverage = merged.attrs.get("coverage", {})
    join_steps = merged.attrs.get("join_steps", [])

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "join_key": "lsoa21cd",
        "caveats": [
            "Ecological fallacy: IMD and Census are LSOA-level, not household-level.",
            "EPC coverage bias: certificates exist only where triggered (sale, let, new build).",
            "EPC performance gap: modelled ratings diverge from metered use (~16% gas, ~31% electric).",
            "IMD income score derived from rank when File 7 Income Score (rate) is absent.",
            "TS046 central-heating table not present in raw files (ts046 file is a tenure duplicate).",
            "1,153 EPC rows lack lsoa21cd; postcode lookup rows exist but lsoa21cd is null in lookup.",
        ],
        "sources": {
            "epc": {
                "rows": len(epc),
                "unique_lsoa": int(epc["lsoa21cd"].nunique()) if "lsoa21cd" in epc.columns else None,
                "missing_lsoa21cd": int(epc["lsoa21cd"].isna().sum()) if "lsoa21cd" in epc.columns else None,
            },
            "imd": {
                "rows": len(imd),
                "unique_lsoa": int(imd["lsoa21cd"].nunique()),
            },
            "census": {
                "rows": len(census),
                "unique_lsoa": int(census["lsoa21cd"].nunique()),
            },
            "geography": {
                "loaded": geography_loaded,
                "rows": geography_rows,
                "note": (
                    "Missing GeoJSON — choropleth requires "
                    "data/external/lsoa_birmingham.geojson"
                    if not geography_loaded
                    else None
                ),
            },
        },
        "join_steps": join_steps,
        "coverage": coverage,
        "merged": {
            "rows": len(merged),
            "unique_lsoa": int(merged["lsoa21cd"].nunique()) if "lsoa21cd" in merged.columns else None,
            "imd_matched": int(merged["imd_matched"].sum()) if "imd_matched" in merged.columns else None,
            "imd_unmatched": int((~merged["imd_matched"]).sum()) if "imd_matched" in merged.columns else None,
            "census_matched": int(merged["census_matched"].sum()) if "census_matched" in merged.columns else None,
            "census_unmatched": int((~merged["census_matched"]).sum()) if "census_matched" in merged.columns else None,
            "geography_matched": int(merged["geography_matched"].sum())
            if "geography_matched" in merged.columns
            else None,
        },
        "target": {
            "formula": (
                f"{EPC_GAP_WEIGHT} * epc_gap_norm + "
                f"{IMD_INCOME_WEIGHT} * imd_income_norm"
            ),
            "non_null_scores": int(scored["retrofit_priority_score"].notna().sum())
            if "retrofit_priority_score" in scored.columns
            else None,
        },
        "nulls_key_columns": _null_summary(
            scored,
            [
                "lsoa21cd",
                "imd_income_score",
                "imd_matched",
                "census_matched",
                "epc_gap",
                "retrofit_priority_score",
            ],
        ),
    }


def _write_join_audit(audit: dict[str, Any], processed_dir: Path) -> Path:
    processed_dir.mkdir(parents=True, exist_ok=True)
    dest = processed_dir / "join_audit.json"
    dest.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    logger.info("Wrote join audit to %s", dest)
    return dest


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
    external_dir = raw_dir.parent / "external"
    interim_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    epc = load_epc(raw_dir=raw_dir, external_dir=external_dir)
    imd = load_imd(raw_dir=raw_dir)
    census = load_census(raw_dir=raw_dir)

    geography = None
    geography_loaded = False
    geography_rows: int | None = None
    try:
        geography = load_geography(external_dir=external_dir)
        geography_loaded = True
        geography_rows = len(geography)
    except (FileNotFoundError, ImportError, ValueError) as exc:
        logger.warning("%s — continuing tabular merge without geography.", exc)

    merged = merge_datasets(epc, imd, census, geography=geography)

    DATA_INTERIM.mkdir(parents=True, exist_ok=True)
    interim_merge = DATA_INTERIM / "merged_epc_imd_census.parquet"
    interim_out = merged.copy()
    interim_out.attrs.clear()
    interim_out.to_parquet(interim_merge, index=False)
    logger.info("Wrote interim merge to %s (%s rows)", interim_merge, f"{len(merged):,}")

    coverage = merged.attrs.get("coverage", {})
    epc_rows = int(coverage.get("epc_rows", len(merged)))

    scored = add_priority_score(merged)
    out_path = processed_dir / "merged_lsoa.parquet"
    # Drop pandas attrs (join audit metadata) before serialisation.
    scored_clean = scored.copy()
    scored_clean.attrs.clear()
    scored_clean.to_parquet(out_path, index=False)

    # Mirror to interim path used by run_ingest()
    interim_scored = scored_clean.copy()
    interim_scored.to_parquet(interim_dir / "merged_with_priority.parquet", index=False)

    audit = _build_join_audit(
        epc=epc,
        imd=imd,
        census=census,
        merged=merged,
        scored=scored,
        geography_loaded=geography_loaded,
        geography_rows=geography_rows,
        seed=seed,
    )
    _write_join_audit(audit, processed_dir)

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
        "target_formula": audit["target"]["formula"],
        "coverage": coverage,
        "join_audit_path": str(processed_dir / "join_audit.json"),
    }
    logger.info("Saved merged dataset to %s (%s rows)", out_path, f"{len(scored):,}")
    return scored, metrics
