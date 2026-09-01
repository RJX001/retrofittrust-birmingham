# Checkpoint 4 — dashboard numbers

British English. Seed = 42. Static export; FastAPI was not required.

## Dataset

- **Loaded:** yes
- **Source:** `data/processed/merged_lsoa.parquet+retrofit_scores`
- **LSOA count:** 660
- **Properties represented (sum of `n_properties`):** 475,073
- **Invalid LSOA rows:** dropped (join audit records 1,153 EPC rows without `lsoa21cd`)
- **Geometries:** `synthetic_grid`
- **ONS GeoJSON (`data/external/lsoa_birmingham.geojson`):** missing — choropleth uses the Birmingham-centred synthetic grid fallback (not a 3D twin)

## SQLite write-back (`data/processed/twin_state.db`)

Existing `twin_state.db` rows mix older `SYNTH_*` demo codes with a few real E01* LSOAs from earlier integration runs. Only matching codes colour as verified on the live twin.

- **lsoa_state rows:** 20
- **Verified in `lsoa_state`:** 5 (**SYNTHETIC DATA**)
- **`verified_outcomes` rows:** 10 (**SYNTHETIC DATA**)
- **Distinct LSOAs in outcomes:** 5
- **Verified codes matching the live dataset:** 3

## Figures

- Kaleido (Plotly PNG): not installed — choropleth PNG via geopandas/matplotlib; Plotly HTML always written
- `reports/figures/04_priority_bar.png`
- `reports/figures/04_choropleth.html`
- `reports/figures/04_twin_metrics.png`
- `reports/figures/04_choropleth.png`

## Notes

- Dashboard `st.cache_data` still wraps `load_lsoa_dataset` / `load_geometries`.
- API calls in `app.py` fail gracefully when uvicorn is down.
- Grant, works-claimed, and verification fields remain labelled **SYNTHETIC DATA**.
