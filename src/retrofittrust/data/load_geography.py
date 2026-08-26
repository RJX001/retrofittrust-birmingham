"""Load 2021 LSOA boundary geometries for Birmingham.

Expected path: ``data/external/lsoa_birmingham.geojson``
(ONS Open Geography Portal, 2021 LSOA **BGC** — generalised clipped —
exported as GeoJSON for web rendering).

A national GeoJSON is accepted; rows are then filtered to Birmingham
using LAD name/code or LSOA name. Full polygons are **not** copied onto
every EPC row in ``merge.py`` (that would explode memory); the dashboard
loads this GeoDataFrame separately for the choropleth.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ..config import BIRMINGHAM_LA, DATA_EXTERNAL, SEED
from ._utils import (
    BIRMINGHAM_LAD_CODE,
    ensure_logging,
    find_column,
    is_birmingham_lsoa_name,
    log_row_count,
    looks_like_birmingham,
    snake_case_columns,
    standardise_lsoa_key,
)

logger = logging.getLogger(__name__)

_ = SEED

GEOJSON_NAME = "lsoa_birmingham.geojson"


def _discover_geo_path(external_dir: Path) -> Path | None:
    preferred = external_dir / GEOJSON_NAME
    if preferred.exists():
        return preferred
    candidates: list[Path] = []
    if external_dir.exists():
        for path in sorted(external_dir.iterdir()):
            if not path.is_file():
                continue
            if path.suffix.lower() in {".geojson", ".json", ".gpkg", ".shp"}:
                candidates.append(path)
    if not candidates:
        return None
    named = [p for p in candidates if "lsoa" in p.name.lower() or "birmingham" in p.name.lower()]
    return (named or candidates)[0]


def _filter_birmingham(gdf, local_authority: str):
    lad_name = find_column(
        gdf.columns,
        "lad24nm",
        "lad23nm",
        "lad22nm",
        "lad21nm",
        "ladnm",
        "la_name",
    )
    lad_code = find_column(
        gdf.columns,
        "lad24cd",
        "lad23cd",
        "lad22cd",
        "lad21cd",
        "ladcd",
    )
    lsoa_name = find_column(gdf.columns, "lsoa21nm", "lsoa11nm", "lsoanm")

    masks: list[pd.Series] = []
    if lad_name is not None:
        labels = gdf[lad_name].astype("string")
        masks.append(
            looks_like_birmingham(labels)
            | labels.str.fullmatch(local_authority, case=False, na=False)
        )
    if lad_code is not None:
        masks.append(
            gdf[lad_code].astype("string").str.strip().str.upper().eq(BIRMINGHAM_LAD_CODE)
        )
    if lsoa_name is not None:
        masks.append(is_birmingham_lsoa_name(gdf[lsoa_name]))

    if not masks:
        logger.warning(
            "Geography file has no LAD/LSOA name to filter on; keeping all %s features. "
            "If this is a national file, clip it to Birmingham before use.",
            f"{len(gdf):,}",
        )
        return gdf

    combined = masks[0]
    for extra in masks[1:]:
        combined = combined | extra
    return gdf.loc[combined.fillna(False)].copy()


def load_geography(
    external_dir: Path | None = None,
    local_authority: str | None = None,
):
    """Load Birmingham 2021 LSOA polygons (GeoJSON / GPKG / shapefile).

    Returns
    -------
    geopandas.GeoDataFrame
        One row per LSOA, with ``lsoa21cd`` and a ``geometry`` column.
    """
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise ImportError(
            "load_geography() requires geopandas (see requirements.txt). "
            "The tabular merge can still run with include_geography=False."
        ) from exc

    ensure_logging()
    external_dir = Path(external_dir) if external_dir is not None else DATA_EXTERNAL
    local_authority = local_authority or BIRMINGHAM_LA

    path = _discover_geo_path(external_dir)
    if path is None:
        raise FileNotFoundError(
            f"No LSOA geography file found in {external_dir}. "
            f"Export ONS 2021 LSOA BGC boundaries as GeoJSON to "
            f"{external_dir / GEOJSON_NAME}. See data/raw/README.md."
        )

    logger.info("Reading geography from %s", path)
    gdf = gpd.read_file(path)
    gdf = gpd.GeoDataFrame(snake_case_columns(gdf), geometry="geometry", crs=gdf.crs)
    gdf = gpd.GeoDataFrame(standardise_lsoa_key(gdf, source="load_geography"), geometry="geometry", crs=gdf.crs)
    gdf = _filter_birmingham(gdf, local_authority)

    if gdf.empty:
        raise ValueError(
            f"Geography loaded from {path.name} but no features matched "
            f"{local_authority!r}."
        )

    before = len(gdf)
    gdf = gdf.drop_duplicates(subset=["lsoa21cd"], keep="first")
    if len(gdf) < before:
        logger.info("Dropped %s duplicate LSOA geometries", f"{before - len(gdf):,}")

    log_row_count(
        "load_geography",
        len(gdf),
        crs=str(gdf.crs),
        unique_lsoa=gdf["lsoa21cd"].nunique(),
    )
    return gdf.reset_index(drop=True)
