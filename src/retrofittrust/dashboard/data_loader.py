"""Cached dataset/geometry loaders for the Streamlit twin."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd

from retrofittrust.api.features import add_composite_score, geojson_path
from retrofittrust.config import DATA_PROCESSED, DEMO_COHORT_LSOA_COUNT, PROJECT_ROOT, SEED
from retrofittrust.ledger.twin_state import fetch_all_lsoa_state

BIRMINGHAM_CENTRE = {"lat": 52.4862, "lon": -1.8904}

# 2021 LSOA codes are E + eight digits (e.g. E01008881).
_VALID_LSOA = re.compile(r"^E\d{8}$")

def _rel(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


_PARQUET_COLUMNS = (
    "lsoa21cd",
    "LSOA21CD",
    "lsoa_code",
    "lsoa21nm",
    "LSOA21NM",
    "current_energy_rating",
    "potential_energy_rating",
    "epc_current",
    "epc_potential",
    "imd_decile",
    "imd_income_score",
    "income_score",
    "epc_gap",
    "retrofit_priority_score",
    "priority_score",
    "flagged_union",
    "quality_flag_union",
    "quality_flag",
    "anomaly_flag",
    "data_quality_flag",
    "n_properties",
)


def _read_parquet_subset(path: Path) -> pd.DataFrame:
    """Read only dashboard columns so a 476k-row property file stays usable."""
    try:
        import pyarrow.parquet as pq

        available = set(pq.read_schema(path).names)
        cols = [c for c in _PARQUET_COLUMNS if c in available]
        if cols:
            return pd.read_parquet(path, columns=cols)
    except Exception:  # noqa: BLE001 — full read if column projection fails
        pass
    return pd.read_parquet(path)


def _canonicalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename_once = {
        "LSOA21CD": "lsoa21cd",
        "lsoa_code": "lsoa21cd",
        "LSOA21NM": "lsoa21nm",
        "current_energy_rating": "epc_current",
        "potential_energy_rating": "epc_potential",
        "imd_income_score": "income_score",
        "retrofit_priority_score": "pipeline_priority",
        "flagged_union": "anomaly_flag",
        "quality_flag_union": "anomaly_flag",
        "quality_flag": "anomaly_flag",
        "data_quality_flag": "anomaly_flag",
    }
    for src, dest in rename_once.items():
        if dest not in out.columns and src in out.columns:
            out[dest] = out[src]
    if "lsoa21cd" not in out.columns:
        raise ValueError(f"No LSOA code column in frame. Columns: {list(out.columns)}")
    return out


def _valid_lsoa_mask(series: pd.Series) -> pd.Series:
    codes = series.astype(str).str.strip()
    return codes.str.match(_VALID_LSOA, na=False)


def _mode_or_first(series: pd.Series) -> Any:
    clean = series.dropna()
    if clean.empty:
        return pd.NA
    modes = clean.mode()
    return modes.iloc[0] if not modes.empty else clean.iloc[0]


def aggregate_to_lsoa(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse property-level EPC rows to one row per 2021 LSOA.

    Drops records with missing/invalid ``lsoa21cd`` (they must not become a
    fake ``<NA>`` polygon on the twin). Counts properties; takes the modal EPC
    band and the first IMD attributes (constant within an LSOA).
    """
    work = _canonicalise_columns(df)
    work = work.loc[_valid_lsoa_mask(work["lsoa21cd"])].copy()
    work["lsoa21cd"] = work["lsoa21cd"].astype(str).str.strip()
    if work.empty:
        raise ValueError("No valid E0* LSOA codes after filtering the processed dataset.")

    if work["lsoa21cd"].nunique() == len(work):
        if "n_properties" not in work.columns:
            work["n_properties"] = 1
        return work.reset_index(drop=True)

    grouped = work.groupby("lsoa21cd", as_index=False)
    named: dict[str, tuple[str, Any]] = {"n_properties": ("lsoa21cd", "size")}
    if "lsoa21nm" in work.columns:
        named["lsoa21nm"] = ("lsoa21nm", "first")
    if "imd_decile" in work.columns:
        named["imd_decile"] = ("imd_decile", "first")
    if "income_score" in work.columns:
        named["income_score"] = ("income_score", "first")
    if "epc_gap" in work.columns:
        named["epc_gap"] = ("epc_gap", "mean")
    if "pipeline_priority" in work.columns:
        named["pipeline_priority"] = ("pipeline_priority", "mean")
    if "priority_score" in work.columns:
        named["priority_score"] = ("priority_score", "mean")
    if "anomaly_flag" in work.columns:
        named["anomaly_flag"] = ("anomaly_flag", "mean")
    if "epc_current" in work.columns:
        named["epc_current"] = ("epc_current", _mode_or_first)
    if "epc_potential" in work.columns:
        named["epc_potential"] = ("epc_potential", _mode_or_first)
    return grouped.agg(**named)


def _overlay_model_scores(frame: pd.DataFrame) -> pd.DataFrame:
    """Prefer LightGBM ``retrofit_scores`` when present; else pipeline composite."""
    out = frame.copy()
    scores_path = DATA_PROCESSED / "retrofit_scores.parquet"
    if not scores_path.exists():
        scores_path = DATA_PROCESSED / "retrofit_scores.csv"
    if scores_path.exists():
        scores = (
            pd.read_parquet(scores_path)
            if scores_path.suffix == ".parquet"
            else pd.read_csv(scores_path)
        )
        if "lsoa_code" in scores.columns and "lsoa21cd" not in scores.columns:
            scores = scores.rename(columns={"lsoa_code": "lsoa21cd"})
        if "lsoa21cd" in scores.columns and "priority_score" in scores.columns:
            slim = scores[["lsoa21cd", "priority_score"]].copy()
            slim["lsoa21cd"] = slim["lsoa21cd"].astype(str).str.strip()
            slim = slim.drop_duplicates(subset=["lsoa21cd"], keep="first")
            out = out.drop(columns=["priority_score"], errors="ignore")
            out = out.merge(slim, on="lsoa21cd", how="left")
    if "priority_score" not in out.columns or out["priority_score"].isna().all():
        if "pipeline_priority" in out.columns:
            out["priority_score"] = pd.to_numeric(out["pipeline_priority"], errors="coerce")
    return out


def _overlay_quality_flags(frame: pd.DataFrame) -> pd.DataFrame:
    """Join LSOA-level anomaly rates from the quality screen if not already present."""
    if "anomaly_flag" in frame.columns and frame["anomaly_flag"].notna().any():
        return frame
    flagged_path = DATA_PROCESSED / "quality_flagged.parquet"
    if not flagged_path.exists():
        return frame
    try:
        flags = _read_parquet_subset(flagged_path)
        flags = _canonicalise_columns(flags)
        flags = flags.loc[_valid_lsoa_mask(flags["lsoa21cd"])]
        if "anomaly_flag" not in flags.columns:
            return frame
        rate = (
            flags.assign(lsoa21cd=flags["lsoa21cd"].astype(str).str.strip())
            .groupby("lsoa21cd", as_index=False)["anomaly_flag"]
            .mean()
        )
        return frame.drop(columns=["anomaly_flag"], errors="ignore").merge(
            rate, on="lsoa21cd", how="left"
        )
    except Exception:  # noqa: BLE001 — flags are optional for the static twin
        return frame


def load_lsoa_dataset() -> tuple[pd.DataFrame, str]:
    """Load LSOA-level twin frame from processed artefacts (not the synthetic demo).

    Property-level ``merged_lsoa.parquet`` (~476k rows) is aggregated here so the
    choropleth stays at 2021 LSOA grain. Model scores overlay when present.
    """
    source_bits: list[str] = []
    raw: pd.DataFrame | None = None

    merged_path = DATA_PROCESSED / "merged_lsoa.parquet"
    flagged_path = DATA_PROCESSED / "quality_flagged.parquet"
    if merged_path.exists():
        raw = _read_parquet_subset(merged_path)
        source_bits.append(_rel(merged_path))
    elif flagged_path.exists():
        raw = _read_parquet_subset(flagged_path)
        source_bits.append(_rel(flagged_path))

    if raw is None:
        scores_path = DATA_PROCESSED / "retrofit_scores.parquet"
        if not scores_path.exists():
            scores_path = DATA_PROCESSED / "retrofit_scores.csv"
        if scores_path.exists():
            raw = (
                pd.read_parquet(scores_path)
                if scores_path.suffix == ".parquet"
                else pd.read_csv(scores_path)
            )
            source_bits.append(_rel(scores_path))

    if raw is None or raw.empty:
        from retrofittrust.api.features import _synthetic_lsoa_frame

        return _synthetic_lsoa_frame(), "synthetic_fallback"

    try:
        frame = aggregate_to_lsoa(raw)
        frame = _overlay_model_scores(frame)
        if (DATA_PROCESSED / "retrofit_scores.parquet").exists() or (
            DATA_PROCESSED / "retrofit_scores.csv"
        ).exists():
            source_bits.append("retrofit_scores")
        frame = _overlay_quality_flags(frame)
        frame = add_composite_score(frame)
        if "pipeline_priority" in frame.columns and "priority_score" in frame.columns:
            # Keep pipeline score visible; choropleth uses priority_score (model preferred).
            pass
        return frame.reset_index(drop=True), "+".join(source_bits)
    except (ValueError, KeyError, TypeError):
        from retrofittrust.api.features import _synthetic_lsoa_frame

        return _synthetic_lsoa_frame(), "synthetic_fallback"


def _synthetic_geojson(codes: list[str]) -> dict[str, Any]:
    """Grid of demo polygons around Birmingham when ONS GeoJSON is absent.

    SYNTHETIC DATA — not 2021 LSOA BGC boundaries. Layout is a compact square
    grid so ~660 LSOAs still sit near the city centre rather than a 5-column strip.
    """
    n = max(len(codes), 1)
    ncols = max(1, int(math.ceil(math.sqrt(n))))
    nrows = max(1, int(math.ceil(n / ncols)))
    width, height = 0.28, 0.22
    cell_w, cell_h = width / ncols, height / nrows
    west = BIRMINGHAM_CENTRE["lon"] - width / 2
    south = BIRMINGHAM_CENTRE["lat"] - height / 2
    features = []
    for i, code in enumerate(codes):
        row, col = divmod(i, ncols)
        x0 = west + col * cell_w
        y0 = south + row * cell_h
        x1, y1 = x0 + cell_w * 0.92, y0 + cell_h * 0.92
        ring = [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]
        features.append(
            {
                "type": "Feature",
                "id": code,
                "properties": {
                    "LSOA21CD": code,
                    "lsoa21cd": code,
                    "LSOA21NM": f"Demo cell {i + 1}",
                    "synthetic": True,
                },
                "geometry": {"type": "Polygon", "coordinates": [ring]},
            }
        )
    return {"type": "FeatureCollection", "features": features}


def detect_featureidkey(geojson: dict[str, Any]) -> str:
    features = geojson.get("features") or []
    if not features:
        return "properties.LSOA21CD"
    props = features[0].get("properties") or {}
    for key in ("LSOA21CD", "lsoa21cd", "LSOA11CD", "code"):
        if key in props:
            return f"properties.{key}"
    return "properties.LSOA21CD"


def load_geometries(lsoa_codes: list[str]) -> tuple[dict[str, Any], str, str]:
    """Return (geojson, featureidkey, source_label). Reprojects to WGS84 for Mapbox."""
    path = geojson_path()
    if path is not None:
        try:
            import geopandas as gpd

            gdf = gpd.read_file(path)
            if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
                gdf = gdf.to_crs(epsg=4326)
            geojson = json.loads(gdf.to_json())
        except Exception:  # noqa: BLE001 — fall back to raw JSON if geopandas/CRS fails
            geojson = json.loads(path.read_text(encoding="utf-8"))
        return geojson, detect_featureidkey(geojson), str(path)
    codes = list(lsoa_codes) or [f"SYNTH_E010{i:05d}" for i in range(DEMO_COHORT_LSOA_COUNT)]
    geojson = _synthetic_geojson(codes)
    return geojson, "properties.lsoa21cd", "synthetic_grid"


def merge_twin_state(df: pd.DataFrame) -> pd.DataFrame:
    """Join live SQLite write-back onto the (possibly cached) base frame."""
    out = df.copy()
    state = fetch_all_lsoa_state()
    codes = out["lsoa21cd"].astype(str)
    out["verified"] = False
    out["epc_uplift_bands"] = 0
    out["verification_status"] = "candidate"
    out["priority_display"] = pd.to_numeric(out.get("priority_score"), errors="coerce")

    if not state:
        return out

    score_map = {
        code: float(entry["priority_score"])
        for code, entry in state.items()
        if entry.get("priority_score") is not None
    }
    if score_map:
        out["priority_display"] = codes.map(score_map).fillna(out["priority_display"])

    verified_codes = {code for code, entry in state.items() if entry.get("verified")}
    if verified_codes:
        is_verified = codes.isin(verified_codes)
        out.loc[is_verified, "verified"] = True
        out.loc[is_verified, "verification_status"] = "verified (SYNTHETIC DATA)"
        uplift = {
            code: int(entry.get("epc_uplift_bands") or 0)
            for code, entry in state.items()
            if entry.get("verified")
        }
        out.loc[is_verified, "epc_uplift_bands"] = codes.map(uplift).fillna(0)

    status = []
    for code in codes:
        entry = state.get(str(code))
        if not entry:
            status.append("candidate")
        elif entry.get("verified"):
            status.append("verified (SYNTHETIC DATA)")
        else:
            meta = entry.get("metadata") or {}
            if meta.get("event") == "works_claimed":
                status.append("works claimed (SYNTHETIC DATA)")
            elif meta.get("cohort") or meta.get("event") == "eligibility":
                status.append("eligibility recorded (SYNTHETIC DATA)")
            else:
                status.append("candidate")
    out["verification_status"] = status
    return out


def apply_epc_uplift(df: pd.DataFrame, bands: int) -> pd.DataFrame:
    """What-if: shift current EPC need toward A without writing the ledger."""
    out = df.copy()
    need = pd.to_numeric(out.get("epc_current_need"), errors="coerce")
    out["epc_need_after"] = (need - bands).clip(lower=1)
    out["epc_gap_after"] = out["epc_need_after"] - pd.to_numeric(
        out.get("epc_potential_need"), errors="coerce"
    ).fillna(1)
    # Assumed fuel-poverty proxy — not the official BEIS statistic.
    imd = pd.to_numeric(out.get("imd_decile"), errors="coerce").fillna(5)
    out["fp_proxy_before"] = ((imd <= 3) & (need >= 5)).astype(int)
    out["fp_proxy_after"] = ((imd <= 3) & (out["epc_need_after"] >= 5)).astype(int)
    return out


def seeded_sample(df: pd.DataFrame, n: int) -> pd.DataFrame:
    n = min(n, len(df))
    if "priority_score" in df.columns:
        return df.sort_values("priority_score", ascending=False).head(n)
    return df.sample(n=n, random_state=SEED)


def build_lsoa_detail(row: pd.Series, state_entry: dict[str, Any] | None = None) -> dict[str, Any]:
    """Structured detail payload for the per-LSOA panel (twin + ledger stub)."""
    state_entry = state_entry or {}
    meta = state_entry.get("metadata") or {}
    outcome = state_entry.get("latest_outcome") or {}

    def _num(key: str) -> float | None:
        val = pd.to_numeric(row.get(key), errors="coerce")
        return None if pd.isna(val) else float(val)

    return {
        "lsoa21cd": str(row.get("lsoa21cd", "")),
        "lsoa21nm": str(row.get("lsoa21nm", "") or ""),
        "epc_current": row.get("epc_current"),
        "epc_potential": row.get("epc_potential"),
        "epc_gap": _num("epc_gap"),
        "epc_current_need": _num("epc_current_need"),
        "imd_decile": _num("imd_decile"),
        "n_properties": _num("n_properties"),
        "priority_score": _num("priority_score"),
        "priority_display": _num("priority_display"),
        "anomaly_flag": row.get("anomaly_flag"),
        "verified": bool(row.get("verified")),
        "verification_status": str(row.get("verification_status", "candidate")),
        "epc_uplift_bands": int(row.get("epc_uplift_bands") or 0),
        "ledger_event": meta.get("event"),
        "ledger_updated": state_entry.get("updated_at"),
        "outcome_epc_before": outcome.get("epc_before"),
        "outcome_epc_after": outcome.get("epc_after"),
        "outcome_recorded_at": outcome.get("recorded_at"),
        "is_synthetic": bool(meta.get("label") == "SYNTHETIC DATA" or outcome),
    }


def whatif_budget_projection(
    df: pd.DataFrame,
    *,
    retrofit_rate_pct: float,
    budget_cap_gbp: float,
    cost_per_lsoa_gbp: float = 8_500.0,
    epc_uplift_bands: int = 2,
) -> dict[str, Any]:
    """Client-side cohort projection — not a second ML model.

    Prioritises by ``priority_display`` (or ``priority_score``), then applies
    budget and retrofit-rate caps. Costs are **SYNTHETIC DATA** for demo only.
    """
    if df.empty:
        return {
            "funded_count": 0,
            "budget_spent_gbp": 0.0,
            "budget_remaining_gbp": budget_cap_gbp,
            "mean_priority_funded": None,
            "fp_proxy_before": 0,
            "fp_proxy_after": 0,
            "funded_codes": [],
        }

    score_col = "priority_display" if "priority_display" in df.columns else "priority_score"
    ranked = df.sort_values(score_col, ascending=False).copy()
    rate_cap = max(0, int(round(len(ranked) * retrofit_rate_pct / 100.0)))
    budget_cap_n = max(0, int(budget_cap_gbp // max(cost_per_lsoa_gbp, 1.0)))
    funded_n = min(len(ranked), rate_cap, budget_cap_n)
    funded = ranked.head(funded_n)

    scenario = apply_epc_uplift(funded, epc_uplift_bands)
    spent = funded_n * cost_per_lsoa_gbp
    mean_pri = (
        float(pd.to_numeric(funded[score_col], errors="coerce").mean()) if funded_n else None
    )

    return {
        "funded_count": funded_n,
        "budget_spent_gbp": spent,
        "budget_remaining_gbp": max(0.0, budget_cap_gbp - spent),
        "mean_priority_funded": mean_pri,
        "fp_proxy_before": int(scenario["fp_proxy_before"].sum()) if funded_n else 0,
        "fp_proxy_after": int(scenario["fp_proxy_after"].sum()) if funded_n else 0,
        "funded_codes": funded["lsoa21cd"].astype(str).tolist(),
        "cost_per_lsoa_gbp": cost_per_lsoa_gbp,
        "retrofit_rate_pct": retrofit_rate_pct,
        "budget_cap_gbp": budget_cap_gbp,
    }
