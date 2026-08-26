"""Load domestic EPC records and filter to Birmingham local authority.

EPC bulk files from epc.opendatacommunities.org can exceed 8 GB for England
and Wales. This module therefore:

- prefers an already-filtered folder at ``data/raw/epc_birmingham/``
- reads CSV in chunks when the file is large
- if a zip of the national bulk is present, opens only the Birmingham
  local-authority ``certificates.csv`` member when it can be identified

Caveats (do not treat EPC ratings as metered energy use)
--------------------------------------------------------
- Modelled-vs-metered **performance gap**: around 16% for gas-heated homes
  and around 31% for electrically heated homes (Hardy & Glew 2019; DESNZ).
  Use these data for *relative* retrofit prioritisation, not absolute
  consumption prediction.
- Assessor error: around 6% average change in predicted heating demand
  between assessments of the same dwelling.
- **Coverage bias**: an EPC is only lodged when triggered (sale, let, new
  build, or certain retrofit works). Dwellings without a certificate are
  missing entirely and must not be treated as "no need".
- ``data/raw`` is immutable — this loader never writes back to it.

Join key: 2021 LSOA code (``lsoa21cd``). Older extracts may only carry
2011 LSOA codes; unmatched rates against IMD 2025 may then reflect the
2011–2021 boundary change rather than genuine missingness.
"""

from __future__ import annotations

import logging
import zipfile
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from ..config import BIRMINGHAM_LA, DATA_RAW, SEED
from ._utils import (
    BIRMINGHAM_LAD_CODE,
    SKIP_RAW_DIR_PARTS,
    chunked_csv_filter,
    ensure_logging,
    find_column,
    list_data_files,
    log_row_count,
    looks_like_birmingham,
    snake_case_columns,
    standardise_lsoa_key,
)

logger = logging.getLogger(__name__)

# Large enough that a Birmingham-only extract is usually a single read;
# small enough that the national bulk can be filtered without filling RAM.
DEFAULT_CHUNKSIZE = 100_000
LARGE_FILE_BYTES = 80 * 1024 * 1024  # 80 MiB

_ = SEED  # documented seed; this loader is deterministic given the files


def _la_mask(chunk: pd.DataFrame, local_authority: str) -> pd.Series:
    """Boolean mask of rows belonging to ``local_authority`` (Birmingham)."""
    label_col = find_column(
        chunk.columns,
        "LOCAL_AUTHORITY_LABEL",
        "local-authority-label",
        "local_authority_label",
        "local_authority_name",
        "la_name",
    )
    code_col = find_column(
        chunk.columns,
        "LOCAL_AUTHORITY",
        "local-authority",
        "local_authority_code",
        "lad24cd",
        "lad21cd",
        "lad22cd",
    )
    masks: list[pd.Series] = []
    if label_col is not None:
        labels = chunk[label_col].astype("string").str.strip()
        masks.append(
            looks_like_birmingham(labels)
            | labels.str.fullmatch(local_authority, case=False, na=False)
        )
    if code_col is not None:
        codes = chunk[code_col].astype("string").str.strip().str.upper()
        masks.append(codes.eq(BIRMINGHAM_LAD_CODE))

    if not masks:
        return pd.Series(True, index=chunk.index)

    combined = masks[0]
    for extra in masks[1:]:
        combined = combined | extra
    return combined.fillna(False)


def _is_epc_path(path: Path) -> bool:
    if any(part.lower() in SKIP_RAW_DIR_PARTS for part in path.parts):
        return False
    name = path.name.lower()
    if name.startswith("file_") and "iod" in name:
        return False
    return True


def _discover_epc_files(raw_dir: Path) -> list[Path]:
    """Locate EPC certificate files without picking up IMD/census dumps."""
    preferred_dirs = [
        raw_dir / "epc_birmingham",
        raw_dir / "epc",
        raw_dir / "all-domestic-certificates",
    ]
    found: list[Path] = []
    for folder in preferred_dirs:
        if folder.exists():
            found.extend(p for p in list_data_files(folder) if _is_epc_path(p))

    if found:
        return _prefer_certificates(found)

    extras: list[Path] = []
    for path in list_data_files(raw_dir):
        if not _is_epc_path(path):
            continue
        name = path.name.lower()
        if "certificate" in name or name.startswith("epc") or "domestic" in name:
            extras.append(path)
    if extras:
        return _prefer_certificates(extras)

    # Last resort: any certificates.csv under raw that is not IMD/census.
    certs = [
        p
        for p in raw_dir.rglob("certificates.csv")
        if _is_epc_path(p) and p.is_file()
    ]
    return sorted(certs)


def _prefer_certificates(paths: list[Path]) -> list[Path]:
    certs = [p for p in paths if "certificate" in p.name.lower()]
    if certs:
        return sorted(certs)
    # Skip recommendations.csv unless it is the only file present.
    non_rec = [p for p in paths if "recommendation" not in p.name.lower()]
    return sorted(non_rec or paths)


def _birmingham_zip_member(zf: zipfile.ZipFile) -> str | None:
    """Return the zip member path for Birmingham certificates.csv, if any."""
    members = [
        n
        for n in zf.namelist()
        if n.lower().endswith(".csv") and not n.endswith("/")
    ]
    certs = [n for n in members if "certificate" in Path(n).name.lower()]
    pool = certs or members
    for needle in ("e08000025", "birmingham"):
        hits = [n for n in pool if needle in n.lower().replace("_", "-")]
        if hits:
            return sorted(hits)[0]
    if len(pool) == 1:
        return pool[0]
    return None


def _read_zip_csv(
    path: Path,
    local_authority: str,
    chunksize: int,
) -> pd.DataFrame:
    with zipfile.ZipFile(path) as zf:
        member = _birmingham_zip_member(zf)
        if member is None:
            logger.warning(
                "%s has no identifiable Birmingham certificates.csv; "
                "scanning CSV members in chunks (this may be slow)",
                path.name,
            )
            frames: list[pd.DataFrame] = []
            for name in zf.namelist():
                if not name.lower().endswith("certificates.csv"):
                    continue
                logger.info("Scanning zip member %s", name)
                with zf.open(name) as handle:
                    frames.append(
                        _read_stream_filtered(handle, local_authority, chunksize)
                    )
            if not frames:
                return pd.DataFrame()
            return pd.concat(frames, ignore_index=True)

        logger.info("Reading zip member %s from %s", member, path.name)
        with zf.open(member) as handle:
            return _read_stream_filtered(handle, local_authority, chunksize)


def _read_stream_filtered(
    handle,
    local_authority: str,
    chunksize: int,
) -> pd.DataFrame:
    """Filter a binary CSV stream by local authority, chunked if large."""
    reader = pd.read_csv(handle, chunksize=chunksize, low_memory=False)
    kept: list[pd.DataFrame] = []
    scanned = 0
    for chunk in reader:
        scanned += len(chunk)
        subset = chunk.loc[_la_mask(chunk, local_authority)]
        if not subset.empty:
            kept.append(subset)
    logger.info(
        "Streamed CSV: scanned %s rows, kept %s for %s",
        f"{scanned:,}",
        f"{sum(len(x) for x in kept):,}",
        local_authority,
    )
    if not kept:
        return pd.DataFrame()
    return pd.concat(kept, ignore_index=True)


def _read_one_epc_file(
    path: Path,
    local_authority: str,
    chunksize: int,
) -> pd.DataFrame:
    suffix = path.suffix.lower()
    predicate: Callable[[pd.DataFrame], pd.Series] = (
        lambda chunk: _la_mask(chunk, local_authority)
    )

    if suffix == ".zip":
        return _read_zip_csv(path, local_authority, chunksize)

    if suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError:
            df = pd.read_parquet(path)
            return df.loc[_la_mask(df, local_authority)].copy()

        pf = pq.ParquetFile(path)
        kept: list[pd.DataFrame] = []
        scanned = 0
        for batch in pf.iter_batches(batch_size=chunksize):
            chunk = batch.to_pandas()
            scanned += len(chunk)
            subset = chunk.loc[_la_mask(chunk, local_authority)]
            if not subset.empty:
                kept.append(subset)
        logger.info(
            "Parquet %s: scanned %s rows, kept %s",
            path.name,
            f"{scanned:,}",
            f"{sum(len(x) for x in kept):,}",
        )
        if not kept:
            return pd.DataFrame()
        return pd.concat(kept, ignore_index=True)

    size = path.stat().st_size
    if size >= LARGE_FILE_BYTES:
        logger.info(
            "Large CSV (%s MiB) — chunked read of %s",
            f"{size / (1024 * 1024):.1f}",
            path.name,
        )
        return chunked_csv_filter(path, predicate, chunksize=chunksize)

    df = None
    last_error: Exception | None = None
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            df = pd.read_csv(path, encoding=enc, low_memory=False)
            last_error = None
            break
        except UnicodeDecodeError as exc:
            last_error = exc
    if df is None:
        raise last_error or RuntimeError(f"Could not read {path}")

    if find_column(df.columns, "LOCAL_AUTHORITY_LABEL", "LOCAL_AUTHORITY") is None:
        logger.warning(
            "%s has no local-authority column; keeping all %s rows "
            "(place only Birmingham extracts in data/raw/epc_birmingham/)",
            path.name,
            f"{len(df):,}",
        )
        return df
    return df.loc[_la_mask(df, local_authority)].copy()


def _finalise_epc(df: pd.DataFrame, source: str) -> pd.DataFrame:
    if df.empty:
        return df
    out = snake_case_columns(df)
    try:
        out = standardise_lsoa_key(out, source=source)
    except ValueError:
        # Some EPC extracts omit LSOA; keep POSTCODE so a later NSPL lookup
        # can recover lsoa21cd. Do not drop the rows.
        logger.warning(
            "%s: no LSOA code column found. Join to IMD/Census will be "
            "incomplete until lsoa21cd is added (e.g. via ONS NSPL). "
            "Columns: %s",
            source,
            list(out.columns)[:30],
        )
    log_row_count(source, len(out), unique_lsoa=out["lsoa21cd"].nunique() if "lsoa21cd" in out.columns else "n/a")
    return out


def load_epc(
    raw_dir: Path | None = None,
    local_authority: str | None = None,
    chunksize: int = DEFAULT_CHUNKSIZE,
) -> pd.DataFrame:
    """Load domestic EPCs from ``data/raw`` and filter to Birmingham.

    Parameters
    ----------
    raw_dir:
        Root of the immutable raw tree (defaults to ``config.DATA_RAW``).
    local_authority:
        Local authority *name* as it appears in EPC bulk data
        (defaults to ``config.BIRMINGHAM_LA``).
    chunksize:
        Pandas chunk size for large CSV / parquet / zip members.

    Returns
    -------
    pandas.DataFrame
        Birmingham-filtered certificates. Column names are snake_case;
        the join key is ``lsoa21cd`` when present.

    Raises
    ------
    FileNotFoundError
        If no EPC file can be found under ``raw_dir``.
    ValueError
        If files exist but the Birmingham filter yields zero rows.
    """
    ensure_logging()
    raw_dir = Path(raw_dir) if raw_dir is not None else DATA_RAW
    local_authority = local_authority or BIRMINGHAM_LA

    paths = _discover_epc_files(raw_dir)
    if not paths:
        raise FileNotFoundError(
            "No EPC CSV/parquet/zip found under "
            f"{raw_dir / 'epc_birmingham'} (or {raw_dir}). "
            "See data/raw/README.md for download instructions."
        )

    logger.info("EPC files: %s", ", ".join(p.name for p in paths))
    frames = [_read_one_epc_file(p, local_authority, chunksize) for p in paths]
    frames = [f for f in frames if not f.empty]
    if not frames:
        raise ValueError(
            f"EPC files were found but none contained rows for "
            f"local authority {local_authority!r} "
            f"(LAD code {BIRMINGHAM_LAD_CODE}). Check BIRMINGHAM_LA / filters."
        )

    combined = pd.concat(frames, ignore_index=True)
    # Identical lodgement keys from overlapping extracts: keep first.
    lmk = find_column(combined.columns, "LMK_KEY", "lmk_key")
    if lmk is not None:
        before = len(combined)
        combined = combined.drop_duplicates(subset=[lmk], keep="first")
        dropped = before - len(combined)
        if dropped:
            logger.info("Dropped %s duplicate LMK_KEY rows", f"{dropped:,}")

    return _finalise_epc(combined, source="load_epc")
