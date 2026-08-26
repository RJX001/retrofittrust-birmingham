"""Load Census 2021 tenure (TS054) and central heating at LSOA for Birmingham.

Tables (Nomis / ONS Census 2021):
- **TS054** Tenure of household
- **TS046** Type of central heating in household

Place CSV extracts in ``data/raw/census/``. Nomis wizard downloads with
metadata rows above the header are handled automatically.

These are *household counts at LSOA level*. They are suitable as contextual
features or narrative breakdowns; they are not property-level attributes.
Attaching them to an individual EPC record is an ecological join — see
``targets.py`` for the same caveat on IMD.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ..config import BIRMINGHAM_LA, DATA_RAW, SEED
from ._utils import (
    detect_lsoa_column,
    ensure_logging,
    find_column,
    is_birmingham_lsoa_name,
    list_data_files,
    log_row_count,
    read_table,
    snake_case,
    snake_case_columns,
    standardise_lsoa_key,
)

logger = logging.getLogger(__name__)

_ = SEED

TENURE_TOKENS = ("ts054", "tenure")
HEATING_TOKENS = ("ts046", "heating", "central_heat", "central-heat")


def _discover_census_files(raw_dir: Path) -> list[Path]:
    folder = raw_dir / "census"
    files = list_data_files(folder)
    if files:
        return files
    extras = [
        p
        for p in list_data_files(raw_dir, recursive=False)
        if any(tok in p.name.lower() for tok in (*TENURE_TOKENS, *HEATING_TOKENS, "census"))
    ]
    return extras


def _classify_table(path: Path, df: pd.DataFrame) -> str:
    """Return 'tenure', 'heating', or 'unknown' based on name and headers."""
    name = path.name.lower()
    cols = " ".join(snake_case(str(c)) for c in df.columns)
    if any(tok in name for tok in TENURE_TOKENS) or "tenure" in cols:
        if any(tok in name for tok in HEATING_TOKENS) or (
            "heating" in cols and "tenure" in cols
        ):
            # Combined extract — treat as mixed; caller keeps all columns.
            return "combined"
        return "tenure"
    if any(tok in name for tok in HEATING_TOKENS) or "heating" in cols or "central_heating" in cols:
        return "heating"
    return "unknown"


def _prefix_measure_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Prefix count/rate columns so tenure and heating do not collide."""
    reserved = {"lsoa21cd", "lsoa21nm", "geography", "geography_code", "date"}
    rename = {
        c: f"{prefix}{c}"
        for c in df.columns
        if c not in reserved and not str(c).startswith(prefix)
    }
    return df.rename(columns=rename)


def _add_share_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Convert household counts to shares of the table total, if present.

    Shares are often more comparable across LSOAs of different size than raw
    counts. Original counts are retained.
    """
    total_col = None
    for col in df.columns:
        cs = str(col)
        if not cs.startswith(prefix):
            continue
        if cs.endswith("_share"):
            continue
        if any(tok in cs for tok in ("total", "all_households", "all_categories")):
            total_col = col
            break
    if total_col is None:
        return df
    total = pd.to_numeric(df[total_col], errors="coerce")
    if (total <= 0).all() or total.isna().all():
        return df
    out = df.copy()
    for col in list(out.columns):
        if not str(col).startswith(prefix) or str(col).endswith("_share"):
            continue
        if col == total_col:
            continue
        numeric = pd.to_numeric(out[col], errors="coerce")
        if numeric.notna().sum() == 0:
            continue
        out[f"{col}_share"] = numeric / total.replace(0, pd.NA)
    return out


def _filter_birmingham(df: pd.DataFrame, local_authority: str) -> pd.DataFrame:
    name_col = find_column(
        df.columns,
        "lsoa21nm",
        "geography",
        "lsoa_name",
        "lsoa_name_2021",
        "geography_name",
    )
    if name_col is None:
        logger.warning(
            "Census extract has no geography-name column; cannot filter to "
            "%s by name. Rows with a 2021 LSOA code are kept; merge will "
            "drop non-Birmingham codes via the IMD/EPC join.",
            local_authority,
        )
        return df

    names = df[name_col].astype("string")
    mask = is_birmingham_lsoa_name(names) | names.str.fullmatch(
        local_authority, case=False, na=False
    )
    # Some Nomis extracts use "Birmingham" as the LA and LSOA names like
    # "Birmingham 001A"; also accept names containing the LA as a token.
    if not mask.any():
        mask = names.str.contains(rf"\b{local_authority}\b", case=False, na=False)
    return df.loc[mask].copy()


def _prepare_census_frame(path: Path, df: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    kind = _classify_table(path, df)
    out = snake_case_columns(df)
    name_col = find_column(out.columns, "lsoa21nm", "geography", "lsoa_name", "geography_name")
    if name_col and name_col != "lsoa21nm":
        out = out.rename(columns={name_col: "lsoa21nm"})
    if detect_lsoa_column(out) is None:
        raise ValueError(
            f"{path.name}: no 2021 LSOA code column. Columns: {list(out.columns)}"
        )
    out = standardise_lsoa_key(out, source=f"load_census:{path.name}")
    prefix = {"tenure": "tenure_", "heating": "heat_", "combined": "census_", "unknown": "census_"}.get(
        kind, "census_"
    )
    if kind != "combined":
        out = _prefix_measure_columns(out, prefix)
        out = _add_share_columns(out, prefix)
    else:
        out = _prefix_measure_columns(out, "census_")
    return kind, out


def load_census(
    raw_dir: Path | None = None,
    local_authority: str | None = None,
) -> pd.DataFrame:
    """Load Nomis TS054 tenure and central-heating tables at Birmingham LSOA.

    Multiple files in ``data/raw/census/`` are classified as tenure or
    heating and joined on ``lsoa21cd``. A single combined extract is also
    accepted.

    Returns
    -------
    pandas.DataFrame
        One row per Birmingham 2021 LSOA.
    """
    ensure_logging()
    raw_dir = Path(raw_dir) if raw_dir is not None else DATA_RAW
    local_authority = local_authority or BIRMINGHAM_LA

    paths = _discover_census_files(raw_dir)
    if not paths:
        raise FileNotFoundError(
            f"No Census 2021 files found in {raw_dir / 'census'}. "
            "Download TS054 (Tenure) and TS046 (central heating) at LSOA "
            "for Birmingham from Nomis. See data/raw/README.md."
        )

    logger.info("Census files: %s", ", ".join(p.name for p in paths))
    prepared: list[tuple[str, pd.DataFrame]] = []
    for path in paths:
        raw = read_table(path)
        kind, frame = _prepare_census_frame(path, raw)
        frame = _filter_birmingham(frame, local_authority)
        log_row_count(f"load_census[{path.name}={kind}]", len(frame))
        prepared.append((kind, frame))

    non_empty = [(k, f) for k, f in prepared if not f.empty]
    if not non_empty:
        raise ValueError(
            f"Census files loaded but no rows matched {local_authority!r}. "
            "Filter the Nomis download to Birmingham LSOAs, or check that "
            "geography names look like 'Birmingham 001A'."
        )

    kinds = {k for k, _ in non_empty}
    if kinds <= {"combined", "unknown"} or len(non_empty) == 1:
        df = pd.concat([f for _, f in non_empty], ignore_index=True)
        df = df.drop_duplicates(subset=["lsoa21cd"], keep="first")
    else:
        df = None
        for kind, frame in non_empty:
            frame = frame.drop_duplicates(subset=["lsoa21cd"], keep="first")
            if df is None:
                df = frame
                continue
            before = len(df)
            df = df.merge(
                frame,
                on="lsoa21cd",
                how="outer",
                indicator=True,
                suffixes=("", f"_{kind}"),
            )
            counts = df["_merge"].value_counts().to_dict()
            logger.info(
                "Census %s join on lsoa21cd: left_only=%s right_only=%s both=%s "
                "(starting rows=%s)",
                kind,
                counts.get("left_only", 0),
                counts.get("right_only", 0),
                counts.get("both", 0),
                before,
            )
            df = df.drop(columns=["_merge"])
        assert df is not None

    log_row_count("load_census", len(df), unique_lsoa=df["lsoa21cd"].nunique())
    if "tenure" not in kinds and "combined" not in kinds:
        logger.warning("No TS054 tenure table was detected among census files.")
    if "heating" not in kinds and "combined" not in kinds:
        logger.warning(
            "No central-heating table (TS046) was detected among census files."
        )
    return df.reset_index(drop=True)
