"""Cached dataset/geometry loaders for the Streamlit twin."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from retrofittrust.api.features import (
    add_composite_score,
    geojson_path,
    load_lsoa_frame,
    normalise_columns,
)
from retrofittrust.config import DEMO_COHORT_LSOA_COUNT, SEED
from retrofittrust.ledger.twin_state import fetch_all_lsoa_state

BIRMINGHAM_CENTRE = {"lat": 52.4862, "lon": -1.8904}


def load_lsoa_dataset() -> tuple[pd.DataFrame, str]:
    """Load LSOA frame; fall back to labelled synthetic demo if merge artefact is unusable."""
    try:
        frame, source = load_lsoa_frame(allow_synthetic_fallback=True)
        return add_composite_score(normalise_columns(frame)), source
    except (ValueError, KeyError, TypeError):
        from retrofittrust.api.features import _synthetic_lsoa_frame

        return _synthetic_lsoa_frame(), "synthetic_fallback"


def _synthetic_geojson(codes: list[str]) -> dict[str, Any]:
    """Grid of demo polygons around Birmingham when ONS GeoJSON is absent.

    SYNTHETIC DATA — not 2021 LSOA BGC boundaries.
    """
    west, south = -1.95, 52.45
    cell_w, cell_h = 0.04, 0.03
    features = []
    for i, code in enumerate(codes):
        row, col = divmod(i, 5)
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
    out["verified"] = False
    out["epc_uplift_bands"] = 0
    out["verification_status"] = "candidate"
    out["priority_display"] = pd.to_numeric(out.get("priority_score"), errors="coerce")

    for i, row in out.iterrows():
        code = str(row["lsoa21cd"])
        entry = state.get(code)
        if not entry:
            continue
        if entry.get("priority_score") is not None:
            out.at[i, "priority_display"] = float(entry["priority_score"])
        if entry.get("verified"):
            out.at[i, "verified"] = True
            out.at[i, "verification_status"] = "verified (SYNTHETIC DATA)"
            out.at[i, "epc_uplift_bands"] = entry.get("epc_uplift_bands") or 0
        else:
            meta = entry.get("metadata") or {}
            if meta.get("event") == "works_claimed":
                out.at[i, "verification_status"] = "works claimed (SYNTHETIC DATA)"
            elif meta.get("cohort") or meta.get("event") == "eligibility":
                out.at[i, "verification_status"] = "eligibility recorded (SYNTHETIC DATA)"
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
