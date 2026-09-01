# Checkpoint 3 — LightGBM ranking numbers

Source: `quality_flagged:quality_flagged.parquet`. Seed = 42. British English.

## Training sample

| Item | Value |
| --- | --- |
| Input rows | 8000 |
| Training rows (non-missing target) | 7984 |
| Features | 2362 |
| LSOA consumer export | 660 |
| Flagged rate in training frame | 53.1% |
| Flagged rows down-weighted (not deleted) | yes |
| Flagged sample weight | 0.35 |
| Target | 0.6 * normalised_epc_gap + 0.4 * normalised_imd_income (config defaults) |

## 5-fold CV (LightGBM)

| Metric | LightGBM | Random Forest baseline |
| --- | --- | --- |
| RMSE | 0.007899 ± 0.000272 | 0.014361 |
| MAE | 0.004445 | — |
| R² | 0.996238 ± 0.000259 | 0.987556 |

High R² is expected: the composite target is a weighted function of the EPC efficiency gap and IMD income need, both of which are present (or recoverable) in the feature matrix. These metrics show that LightGBM reconstructs the *constructed* priority score, not an independently observed retrofit outcome.

## Target-weight sensitivity

Default weights: EPC-gap 0.6, IMD-income 0.4. Sensitivity changes the *formula* only — it does not retrain a separate model family.

- EPC 0.5 / IMD 0.5: Spearman 0.983, top-10 overlap 0.8, mean |score shift| 0.033
- EPC 0.6 / IMD 0.4: Spearman 1.0, top-10 overlap 1.0, mean |score shift| 0.0
- EPC 0.7 / IMD 0.3: Spearman 0.9804, top-10 overlap 1.0, mean |score shift| 0.033

## Limitations (dissertation)

- **SHAP correlated features.** TreeExplainer assumes feature independence. Floor area, habitable-room count and heating cost are correlated in EPC data; IMD score and the income domain are also correlated. Importance can be misattributed within those groups.
- **Ecological fallacy.** IMD and Census attributes are LSOA-level and must not be read as household facts.
- **EPC performance gap.** Modelled SAP points are not metered kWh.
- **No ground-truth priority.** Face-validity checks are indicative only.
