# RetrofitTrust Birmingham — figure index

Dissertation evidence for the PoC. British English. Seed = 42.
Grant / works / verification payloads shown in ledger and dashboard figures are **SYNTHETIC DATA**.

| File | Caption |
| --- | --- |
| `01_row_counts.png` | Checkpoint 1 row counts: 476,226 Birmingham EPC certificates vs 659 IMD and 659 Census LSOA rows; left join preserves every certificate. |
| `01_join_retention.png` | Join retention on `lsoa21cd`: 475,069 matched (99.76%) and 1,157 unmatched kept (no silent drops). |
| `01_null_summary.png` | Key-column missingness after merge (`lsoa21cd` / IMD income / priority score ≈ 0.24% null; EPC ratings complete). |
| `01_priority_score_hist.png` | Composite retrofit priority (`0.6 × EPC gap + 0.4 × IMD income`) for 475,069 scored certificates (mean 0.421, median 0.443). |
| `01_epc_band_counts.png` | Current EPC band distribution on the Birmingham extract (D largest at 192,798; A–G complete). |
| `01_dataset_numbers.md` | Checkpoint 1 number sheet: row counts, join audit, nulls, target formula, caveats (ecological fallacy, 1,153 missing LSOA codes). |
| `02_flag_rates.png` | Quality-screen flag rates on the 8,000-row stratified sample: consensus 41.8%, union 53.1%, both inside the 27–60% EPC literature band. |
| `02_injection_recall.png` | SYNTHETIC injection evaluation: 70.7% recall vs 53.1% chance baseline (flagged rows quarantined, never deleted). |
| `02_quality_numbers.md` | Checkpoint 2 number sheet: AE architecture, thresholds, flag rates, injection lift. |
| `quality_screen_summary.png` | Combined AE + Isolation Forest summary (operational flag rates and SYNTHETIC injection panel). |
| `03_cv_metrics.png` | LightGBM 5-fold CV vs random-forest baseline (RMSE 0.0079 vs 0.0144; R² 0.996 vs 0.988) on the constructed priority target. |
| `03_weight_sensitivity.png` | Target-weight sensitivity (EPC-gap / IMD-income 0.5/0.5, 0.6/0.4, 0.7/0.3): Spearman ≥ 0.98, top-10 overlap 0.8–1.0. |
| `03_ranking_numbers.md` | Checkpoint 3 number sheet: 8,000 input rows, 660 LSOA consumer export, flagged 53.1% down-weighted not deleted. |
| `shap_beeswarm.png` | Global SHAP beeswarm (TreeExplainer) — correlated EPC/IMD features can share attributed importance. |
| `shap_bar.png` | Global mean \|SHAP\| bar chart for the LightGBM ranker. |
| `shap_waterfall_E01009393.png` | Local SHAP waterfall for LSOA E01009393 (highest saved consumer score). |
| `shap_waterfall_E01009102.png` | Local SHAP waterfall for LSOA E01009102 (earlier training sample). |
| `shap_waterfall_E01009237.png` | Local SHAP waterfall for LSOA E01009237 (earlier training sample). |
| `shap_waterfall_sample.png` | Local SHAP waterfall for a sample property/LSOA row from an earlier training run. |
| `04_choropleth.png` | Streamlit twin choropleth of 660 LSOAs by live priority (ONS GeoJSON missing — Birmingham-centred synthetic grid). |
| `04_choropleth.html` | Interactive Plotly choropleth of the same 660-LSOA priority map. |
| `04_priority_bar.png` | LSOA priority bar chart fallback used when real 2021 BGC geometries are absent. |
| `04_twin_metrics.png` | Twin SQLite snapshot: 20 `lsoa_state` rows, 5 verified (**SYNTHETIC DATA** write-back). |
| `04_dashboard_numbers.md` | Checkpoint 4 number sheet: 660 LSOAs, 475,073 properties represented, GeoJSON missing, write-back counts. |
| `05_ledger_verify.png` | Standalone hash-chain verify badge: `verify_chain()` PASS (24 blocks at capture; genesis + eligibility / works / verification). |
| `05_tamper_demo.png` | In-memory tamper demo: intact chain PASS, after modification FAIL (hash mismatch detected). |
| `05_ledger_numbers.md` | Checkpoint 5 number sheet: SHA-256 simulation, SYNTHETIC DATA labelling, tamper result. |
| `06_integration_loop.png` | Checkpoint 6 five-step loop badge: cohort 10, ranked 10, ledger height 27, chain PASS, SQLite write-back PASS. |
| `06_integration_numbers.md` | Checkpoint 6 number sheet: top LSOA `E01009481` (score 0.5816), 5 verified LSOAs in SQLite, seed 42. |

## How to read these

- **01** = merge / join audit (certificate grain).
- **02** = PyOD AutoEncoder + Isolation Forest screen.
- **03 / shap_*** = LightGBM ranker + TreeExplainer.
- **04** = Streamlit digital twin (static export).
- **05** = hashlib ledger, not a live blockchain.
- **06** = FastAPI integration loop (twin → AI → ledger → SQLite write-back).
