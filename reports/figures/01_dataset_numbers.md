# Checkpoint 1 — merged dataset numbers

Birmingham extract joined on 2021 LSOA (`lsoa21cd`). Generated from `data/processed/join_audit.json` and `merged_lsoa.parquet`. No rows were silently dropped: unmatched certificates are kept with nulls.

## Row counts

| Source | Grain | Rows | Unique LSOA (`lsoa21cd`) |
|---|---|---:|---:|
| EPC (Birmingham) | certificate | 476,226 | 660 |
| IMD 2025 | LSOA | 659 | 659 |
| Census 2021 (Nomis) | LSOA | 659 | 659 |
| Merged (left join from EPC) | certificate | 476,226 | 660 |

## Join retention (`lsoa21cd`)

| Join | Matched | Unmatched (kept) | Retention |
|---|---:|---:|---:|
| EPC left-join IMD | 475,069 | 1,157 | 99.76% |
| EPC left-join Census | 475,069 | 1,157 | 99.76% |

- Missing `lsoa21cd` on EPC: **1,153**
- Extra unmatched with a present LSOA code: **4** (LSOA on the certificate but not in the Birmingham IMD/Census extract).
- Unique EPC LSOAs (660) versus IMD LSOAs (659): one extra LSOA appears on EPC only (`nunique` excludes nulls).

## Key-column nulls

| Column | Null count | Null % |
|---|---:|---:|
| `lsoa21cd` | 1,153 | 0.24 |
| `imd_income_score` | 1,157 | 0.24 |
| `retrofit_priority_score` | 1,157 | 0.24 |
| `current_energy_rating` | 0 | 0.00 |
| `potential_energy_rating` | 0 | 0.00 |
| `current_energy_efficiency` | 0 | 0.00 |
| `epc_gap` | 0 | 0.00 |

## Composite target

- Formula: `0.6 * epc_gap_norm + 0.4 * imd_income_norm`
- Non-null `retrofit_priority_score`: **475,069** / 476,226
- Mean (non-null): **0.4206**; median **0.4429**

## Current energy rating

| Band | Count |
|---|---:|
| A | 1,283 |
| B | 40,742 |
| C | 149,761 |
| D | 192,798 |
| E | 74,176 |
| F | 13,455 |
| G | 4,011 |
| other/NA | 0 |

## Caveats

- **Ecological fallacy:** IMD and Census describe LSOAs, not households.
- **1,153** EPC rows lack `lsoa21cd` (postcode lookup rows exist but `lsoa21cd` is null in the lookup).
- **IMD income rank proxy:** File 7 Income Score (rate) is absent; a higher-is-more-deprived score is derived from income rank.
- **TS046:** the file named central heating in `data/raw` is a tenure duplicate; no genuine heating table was joined.
- **EPC performance gap:** modelled ratings diverge from metered use (~16% gas, ~31% electric).
- **Coverage bias:** EPCs exist only where triggered (sale, let, new build).
- **Geography:** no GeoJSON required for this checkpoint; choropleth is a later agent.

## Figures written

- `C:\Users\rajan\Desktop\Master Full Resouces\Program Full Build\reports\figures\01_row_counts.png`
- `C:\Users\rajan\Desktop\Master Full Resouces\Program Full Build\reports\figures\01_join_retention.png`
- `C:\Users\rajan\Desktop\Master Full Resouces\Program Full Build\reports\figures\01_null_summary.png`
- `C:\Users\rajan\Desktop\Master Full Resouces\Program Full Build\reports\figures\01_priority_score_hist.png`
- `C:\Users\rajan\Desktop\Master Full Resouces\Program Full Build\reports\figures\01_epc_band_counts.png`
