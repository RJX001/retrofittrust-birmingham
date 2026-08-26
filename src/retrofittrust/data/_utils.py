"""Shared helpers for the RetrofitTrust data pipeline.

``data/raw`` is treated as immutable: these functions only read.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterable

import pandas as pd

from ..config import SEED

logger = logging.getLogger(__name__)

# Birmingham Metropolitan District (ONS LAD code). Used when the EPC file
# carries LOCAL_AUTHORITY as a code rather than a name.
BIRMINGHAM_LAD_CODE = "E08000025"

# 2021 LSOA codes are E01 followed by six digits (e.g. E01033557).
LSOA21_PATTERN = re.compile(r"^E01\d{6}$")

# Column-name fragments, preferred first, used to find the 2021 LSOA join key.
LSOA_COLUMN_CANDIDATES: tuple[str, ...] = (
    "lsoa21cd",
    "lsoa_code_2021",
    "lsoa code (2021)",
    "lsoa21",
    "lsoa_21",
    "lsoa-code",
    "lsoa_code",
    "lower_layer_super_output_area_code",
    "geography code",
    "geography_code",
    "geo_code",
)

SKIP_RAW_DIR_PARTS = frozenset({"imd2025", "imd", "census"})


def ensure_logging(level: int = logging.INFO) -> None:
    """Attach a default handler so pipeline logs are visible in scripts."""
    root = logging.getLogger("retrofittrust")
    if root.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    root.addHandler(handler)
    root.setLevel(level)


def snake_case(name: str) -> str:
    """Normalise a header to snake_case (British spelling in comments only)."""
    text = str(name).strip().replace("-", "_").replace("/", "_")
    text = re.sub(r"[^\w]+", "_", text, flags=re.UNICODE)
    text = re.sub(r"_+", "_", text).strip("_").lower()
    return text


def snake_case_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with snake_case column names; collide-safe."""
    out = df.copy()
    counts: dict[str, int] = {}
    new_cols: list[str] = []
    for col in out.columns:
        base = snake_case(str(col)) or "unnamed"
        n = counts.get(base, 0)
        counts[base] = n + 1
        new_cols.append(base if n == 0 else f"{base}_{n + 1}")
    out.columns = new_cols
    return out


def find_column(
    columns: Iterable[str],
    *needles: str,
    contains: bool = False,
) -> str | None:
    """Return the first column whose name matches any needle.

    Matching is case-insensitive. Exact (after snake_case) is tried first,
    then substring if ``contains`` is True or if no exact match exists.
    """
    cols = list(columns)
    lowered = {str(c).strip().lower(): c for c in cols}
    snaked = {snake_case(str(c)): c for c in cols}

    for needle in needles:
        key = needle.strip().lower()
        if key in lowered:
            return lowered[key]
        snake = snake_case(needle)
        if snake in snaked:
            return snaked[snake]

    for needle in needles:
        key = needle.strip().lower()
        snake = snake_case(needle)
        for col in cols:
            cl = str(col).strip().lower()
            cs = snake_case(str(col))
            if key in cl or snake in cs:
                return col
    _ = contains
    return None


def normalise_lsoa_code(series: pd.Series) -> pd.Series:
    """Cast LSOA codes to canonical strings (no trailing ``.0`` from Excel)."""
    s = series.astype("string").str.strip()
    s = s.str.replace(r"\.0$", "", regex=True)
    s = s.replace(
        {
            "nan": pd.NA,
            "None": pd.NA,
            "<NA>": pd.NA,
            "NaT": pd.NA,
            "": pd.NA,
        }
    )
    return s


def detect_lsoa_column(df: pd.DataFrame) -> str | None:
    """Find a 2021 LSOA code column, preferring names that mention 2021."""
    named = find_column(df.columns, *LSOA_COLUMN_CANDIDATES)
    if named is not None:
        return named

    # Fall back to any column whose values look like E01###### codes.
    sample_n = min(len(df), 200)
    if sample_n == 0:
        return None
    sample = df.head(sample_n)
    best: tuple[float, str] | None = None
    for col in df.columns:
        values = normalise_lsoa_code(sample[col].dropna())
        if values.empty:
            continue
        share = values.str.match(LSOA21_PATTERN, na=False).mean()
        if share >= 0.8:
            candidate = (float(share), str(col))
            if best is None or candidate[0] > best[0]:
                best = candidate
    return best[1] if best else None


def standardise_lsoa_key(df: pd.DataFrame, *, source: str) -> pd.DataFrame:
    """Ensure ``lsoa21cd`` exists. Raises if no LSOA column can be found."""
    out = df.copy()
    col = detect_lsoa_column(out)
    if col is None:
        raise ValueError(
            f"{source}: could not find a 2021 LSOA code column. "
            f"Columns were: {list(out.columns)}"
        )
    if col != "lsoa21cd":
        if "lsoa21cd" in out.columns and col != "lsoa21cd":
            logger.warning(
                "%s: renaming %s -> lsoa21cd (existing lsoa21cd will be kept as lsoa21cd_orig)",
                source,
                col,
            )
            out = out.rename(columns={"lsoa21cd": "lsoa21cd_orig"})
        out = out.rename(columns={col: "lsoa21cd"})
    out["lsoa21cd"] = normalise_lsoa_code(out["lsoa21cd"])
    n_missing = int(out["lsoa21cd"].isna().sum())
    if n_missing:
        logger.warning(
            "%s: %s / %s rows have a missing LSOA21 code",
            source,
            f"{n_missing:,}",
            f"{len(out):,}",
        )
    return out


def log_row_count(label: str, n: int, **extra: object) -> None:
    """Log a row count so joins cannot lose rows silently."""
    bits = [f"{label}: {n:,} rows"]
    for key, value in extra.items():
        bits.append(f"{key}={value}")
    logger.info(" | ".join(bits))


def min_max_normalise(series: pd.Series) -> pd.Series:
    """Scale a numeric series to [0, 1] over observed (non-null) values.

    Constant series map to 0.5 (neutral) rather than NaN, so a degenerate
    sample does not silently zero-out the composite target.
    """
    s = pd.to_numeric(series, errors="coerce")
    lo = s.min()
    hi = s.max()
    if pd.isna(lo) or pd.isna(hi):
        logger.warning("min-max normalise: no non-null values")
        return s
    if lo == hi:
        logger.warning(
            "min-max normalise: constant series (value=%s); mapping non-null to 0.5",
            lo,
        )
        return s.mask(s.notna(), 0.5)
    return (s - lo) / (hi - lo)


def looks_like_birmingham(series: pd.Series) -> pd.Series:
    """True where a name/label refers to Birmingham local authority."""
    lowered = series.astype("string").str.strip().str.lower()
    return lowered.eq("birmingham") | lowered.str.startswith("birmingham ")


def is_birmingham_lsoa_name(series: pd.Series) -> pd.Series:
    """True for 2021 LSOA names of the form 'Birmingham 001A'."""
    text = series.astype("string").str.strip()
    return text.str.match(r"(?i)^Birmingham\s+\d", na=False) | text.str.fullmatch(
        r"(?i)Birmingham", na=False
    )


def list_data_files(
    directory: Path,
    suffixes: tuple[str, ...] = (".csv", ".parquet", ".zip", ".xlsx", ".xls"),
    *,
    recursive: bool = True,
) -> list[Path]:
    """List data files under ``directory``, skipping empty placeholders."""
    if not directory.exists():
        return []
    iterator = directory.rglob("*") if recursive else directory.iterdir()
    found: list[Path] = []
    for path in iterator:
        if not path.is_file():
            continue
        if path.name.startswith("."):
            continue
        if path.suffix.lower() not in suffixes:
            continue
        found.append(path)
    return sorted(found)


def read_csv_flexible(path: Path, **kwargs) -> pd.DataFrame:
    """Read a CSV, trying common encodings and Nomis-style metadata headers.

    Nomis wizard downloads often prepend title rows before the real header.
    We try skiprows 0..25 until a plausible header with an LSOA-like column
    appears; if none does, we return the skiprows=0 read.
    """
    encodings = ("utf-8", "utf-8-sig", "cp1252", "latin-1")
    last_error: Exception | None = None
    raw: pd.DataFrame | None = None
    used_encoding = "utf-8"

    for enc in encodings:
        try:
            raw = pd.read_csv(path, encoding=enc, low_memory=False, **kwargs)
            used_encoding = enc
            last_error = None
            break
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
    if raw is None:
        raise last_error or RuntimeError(f"Could not read {path}")

    if detect_lsoa_column(raw) is not None:
        logger.debug("Read %s with encoding=%s (header on row 0)", path.name, used_encoding)
        return raw

    # Nomis / ONS downloads with metadata rows above the header.
    for skip in range(1, 26):
        try:
            trial = pd.read_csv(
                path, encoding=used_encoding, low_memory=False, skiprows=skip, **kwargs
            )
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            continue
        if trial.empty or trial.shape[1] < 2:
            continue
        unnamed = sum(str(c).startswith("Unnamed") for c in trial.columns)
        if unnamed / max(len(trial.columns), 1) > 0.5:
            continue
        if detect_lsoa_column(trial) is not None:
            logger.info(
                "Read %s with encoding=%s, skiprows=%s (Nomis-style header)",
                path.name,
                used_encoding,
                skip,
            )
            return trial

    logger.warning(
        "Read %s with encoding=%s but could not detect an LSOA column; "
        "returning the first-header parse",
        path.name,
        used_encoding,
    )
    return raw


def read_table(path: Path) -> pd.DataFrame:
    """Read CSV / parquet / Excel. Excel requires an engine extra if used."""
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".xlsx", ".xls"}:
        try:
            return pd.read_excel(path)
        except ImportError as exc:
            raise ImportError(
                f"Reading {path.name} requires openpyxl. "
                "Re-export the file as CSV into the same folder, or pip-install openpyxl."
            ) from exc
    return read_csv_flexible(path)


def chunked_csv_filter(
    path: Path,
    predicate,
    *,
    chunksize: int = 100_000,
    encoding: str | None = None,
) -> pd.DataFrame:
    """Read a large CSV in chunks, keeping rows where ``predicate(chunk)`` is True.

    ``predicate`` must return a boolean Series aligned to the chunk index.
    """
    encodings = (encoding,) if encoding else ("utf-8", "utf-8-sig", "cp1252", "latin-1")
    last_error: Exception | None = None
    kept: list[pd.DataFrame] = []
    total_in = 0

    for enc in encodings:
        kept = []
        total_in = 0
        try:
            reader = pd.read_csv(
                path,
                encoding=enc,
                chunksize=chunksize,
                low_memory=False,
            )
            for chunk in reader:
                total_in += len(chunk)
                mask = predicate(chunk)
                subset = chunk.loc[mask]
                if not subset.empty:
                    kept.append(subset)
            last_error = None
            logger.info(
                "Chunked read %s (encoding=%s): scanned %s rows, kept %s",
                path.name,
                enc,
                f"{total_in:,}",
                f"{sum(len(x) for x in kept):,}",
            )
            break
        except UnicodeDecodeError as exc:
            last_error = exc
            continue

    if last_error is not None and not kept:
        raise last_error

    if not kept:
        return pd.DataFrame()
    return pd.concat(kept, ignore_index=True)


# Re-export seed so modules that touch randomness have a single import path.
__all__ = [
    "BIRMINGHAM_LAD_CODE",
    "LSOA21_PATTERN",
    "SEED",
    "SKIP_RAW_DIR_PARTS",
    "chunked_csv_filter",
    "detect_lsoa_column",
    "ensure_logging",
    "find_column",
    "is_birmingham_lsoa_name",
    "list_data_files",
    "log_row_count",
    "looks_like_birmingham",
    "min_max_normalise",
    "normalise_lsoa_code",
    "read_csv_flexible",
    "read_table",
    "snake_case",
    "snake_case_columns",
    "standardise_lsoa_key",
]
