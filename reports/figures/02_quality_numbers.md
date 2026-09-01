# Checkpoint 2 — quality-screen numbers

British English. Seed = 42. Flagged records are quarantined, never silently deleted.

## Sample

| Item | Value |
|---|---|
| Input source | C:\Users\rajan\Desktop\Master Full Resouces\Program Full Build\data\processed\merged_lsoa.parquet (stratified sample n=8000) |
| Input rows | 8,000 |
| Output rows (flagged parquet) | 8,000 |
| AE features | 808 |
| AE architecture | hidden=404, bottleneck=202 |
| Threshold | k=0.25, target=0.35, EVT=False, method=mean_sigma |

Row count is unchanged from input to output (quarantine / flag only). LightGBM should pass `sample_weight` (1.0 clean, 0.35 flagged).

## Flag rates

| Mode | Rate | Literature band (27–60%) |
|---|---|---|
| Consensus (`quality_flag`) | **41.8% (3,343 rows)** | inside |
| Union (`quality_flag_union`, operational) | **53.1% (4,249 rows)** | inside |

Literature sanity: roughly 27% of EPC records show at least one quality flag; true error rate estimated 36–62% in the wider literature. The operational (union) rate is the high-recall set used for LightGBM sample weights.

## SYNTHETIC injection evaluation

Injected copies only — labelled **SYNTHETIC DATA**, not real assessor errors.

| Metric | Rate |
|---|---|
| Injection recall | **70.7%** |
| Chance baseline (operational flag rate) | 53.1% |
| Lift | +17.6 pp |
| Beats chance (+5 pp) | yes |

## Figures

- `reports/figures/quality_screen_summary.png` — combined checkpoint summary
- `reports/figures/02_flag_rates.png` — consensus vs union vs literature band
- `reports/figures/02_injection_recall.png` — synthetic injection recall vs chance

## Artefacts

- `data/processed/quality_flagged.parquet`
- `models/quality_screen.joblib`
