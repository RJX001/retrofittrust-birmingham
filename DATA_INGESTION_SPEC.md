# DATA INGESTION SPEC — Birmingham LSOA Evidence Base

**Project:** Intelligent Urban Decision-Making Framework Using AI, Blockchain and Digital Twins
**Scope of this spec:** Stage 0 only — ingest, clean, aggregate and join three open datasets into a single LSOA-level analysis table for Birmingham.

> **Cursor: read this whole file before writing code.** Build only what is described here. Do not add modelling, dashboards, blockchain code, or visualisation in this stage. Those are separate specs.

---

## 1. Purpose

This stage produces **one file**: an LSOA-level table for Birmingham combining building energy performance, socio-economic deprivation, and housing tenure/heating characteristics.

That table is the empirical substrate for every later stage of the dissertation:

| Later stage | What it consumes |
|---|---|
| Digital twin layer | LSOA-level state variables (the twin's "current state") |
| AI layer | Features for retrofit-priority scoring |
| Blockchain layer | Records provenance and versioning of *this* table |
| Scenario evaluation | Baseline against which interventions are compared |

Design decision that follows from this: **the output must be reproducible and versioned.** Every row must be traceable to a source file and a transformation step, because the blockchain provenance layer later in the project is meaningless if the underlying pipeline is not deterministic.

---

## 2. Datasets

All join on the **2021 LSOA code** (`LSOA21CD`, format `E01xxxxxx`).

### 2.1 EPC Domestic Certificates
- **Source:** `epc.opendatacommunities.org`
- **Download:** Birmingham local authority extract (code `E08000025`), not the full England & Wales bulk file
- **Granularity:** address-level (one row per certificate) — **must be aggregated to LSOA**
- **Key fields:** `LMK_KEY`, `LSOA_CODE`, `CURRENT_ENERGY_RATING`, `POTENTIAL_ENERGY_RATING`, `CURRENT_ENERGY_EFFICIENCY`, `POTENTIAL_ENERGY_EFFICIENCY`, `TOTAL_FLOOR_AREA`, `MAIN_FUEL`, `MAINHEAT_DESCRIPTION`, `WALLS_DESCRIPTION`, `ROOF_DESCRIPTION`, `WINDOWS_DESCRIPTION`, `CONSTRUCTION_AGE_BAND`, `PROPERTY_TYPE`, `BUILT_FORM`, `INSPECTION_DATE`, `TRANSACTION_TYPE`
- **Licence:** OGL v3, with EPC-specific terms of use

### 2.2 English Indices of Deprivation 2025
- **Source:** GOV.UK (MHCLG), published 30 October 2025
- **Granularity:** LSOA (2021 boundaries)
- **Key fields:** IMD score, IMD rank, IMD decile, Income domain, plus other domains if present
- **Licence:** OGL v3

### 2.3 ONS Census 2021 (via Nomis)
- **Source:** `nomisweb.co.uk`
- **Tables:** Tenure (TS054) and Central heating — *confirm the exact table code from your own download; do not assume it*
- **Granularity:** LSOA 2021, filtered to Birmingham
- **Note:** Nomis CSVs frequently carry metadata rows above the header. Detect the header row rather than hardcoding `skiprows`.

### 2.4 LSOA Boundaries (optional at this stage)
- **Source:** ONS Open Geography Portal
- **Version:** 2021 LSOA, Generalised Clipped (BGC), GeoJSON
- Used later for mapping. Ingest only; do not build map code yet.

---

## 3. Repository layout

```
data/
  raw/
    epc/         # unzipped EPC extract (certificates.csv, recommendations.csv)
    imd/         # imd2025_lsoa.csv
    census/      # ts054_tenure_lsoa.csv, central_heating_lsoa.csv
    geo/         # lsoa_2021_bgc.geojson
    README.md    # provenance log — MANDATORY, see §7
  interim/       # per-dataset cleaned + Birmingham-filtered
  processed/     # birmingham_lsoa_master.parquet  <-- the deliverable
src/
  data/
    __init__.py
    paths.py         # centralised path constants
    load_epc.py
    load_imd.py
    load_census.py
    build_master.py  # orchestrator
    validate.py
tests/
  test_join.py
```

`data/raw/**` is already gitignored. Keep it that way.

---

## 4. Module contracts

Each loader is a pure function: raw file path in, tidy DataFrame out. No side effects, no printing to stdout other than via the logger.

### 4.1 `load_epc.py`

```
load_epc_raw(path: Path) -> pd.DataFrame
aggregate_epc_to_lsoa(df: pd.DataFrame) -> pd.DataFrame
```

**Cleaning rules, applied in order:**

1. Normalise the LSOA column to `LSOA21CD` and strip whitespace.
2. Filter to Birmingham LSOAs only (guard against portal over-inclusion).
3. Parse `INSPECTION_DATE` to datetime; drop unparseable rows and log the count.
4. **Deduplicate:** a property can hold multiple certificates over time. Keep the **most recent certificate per address**. Group by `LMK_KEY`'s address components (`ADDRESS1`, `POSTCODE`) or `UPRN` if present; sort by `INSPECTION_DATE` descending; keep first. Log rows dropped.
5. Coerce `CURRENT_ENERGY_EFFICIENCY`, `POTENTIAL_ENERGY_EFFICIENCY`, `TOTAL_FLOOR_AREA` to numeric; set implausible values to NaN (`TOTAL_FLOOR_AREA` outside 10–2000 m²; efficiency scores outside 1–100).
6. Uppercase and strip the energy rating bands; validate they are in `A`–`G`.

**Aggregation to LSOA** — produce one row per LSOA with at minimum:

| Column | Definition |
|---|---|
| `LSOA21CD` | join key |
| `epc_n_certs` | count of deduplicated certificates |
| `epc_mean_current_eff` | mean `CURRENT_ENERGY_EFFICIENCY` |
| `epc_median_current_eff` | median, for skew-robustness |
| `epc_mean_potential_eff` | mean `POTENTIAL_ENERGY_EFFICIENCY` |
| `epc_mean_improvement_gap` | mean of (potential − current) |
| `epc_pct_below_c` | share of certificates rated D–G |
| `epc_pct_f_or_g` | share rated F–G (fuel-poverty proxy) |
| `epc_mean_floor_area` | mean `TOTAL_FLOOR_AREA` |
| `epc_pct_pre_1930` | share in pre-1930 `CONSTRUCTION_AGE_BAND` categories |
| `epc_pct_gas_main_heat` | share with mains gas as `MAIN_FUEL` |

**Coverage caveat that must be recorded, not hidden:** EPC coverage is non-random. Certificates are only produced on sale, new build, or letting, so owner-occupied stock held long-term is systematically under-represented. Emit `epc_n_certs` alongside every mean so downstream code can weight or filter, and state this limitation in the methodology chapter.

### 4.2 `load_imd.py`

```
load_imd(path: Path) -> pd.DataFrame
```

Select and rename to snake_case: `imd_score`, `imd_rank`, `imd_decile`, `income_score`, `income_decile`. Filter to Birmingham. Assert `imd_decile` ∈ 1–10 and that decile 1 is the *most* deprived (check against the source documentation — do not assume the direction).

### 4.3 `load_census.py`

```
load_nomis_csv(path: Path) -> pd.DataFrame   # header-row detection
load_tenure(path: Path) -> pd.DataFrame
load_central_heating(path: Path) -> pd.DataFrame
```

Nomis exports are usually wide, with one column per category and counts as values. Convert counts to **proportions of the LSOA household total**, and retain the denominator as `census_n_households`. Proportions are what later stages need; raw counts invite accidental population-size effects.

Target columns:
- Tenure: `ten_owned_outright`, `ten_owned_mortgage`, `ten_social_rented`, `ten_private_rented`, `ten_rent_free`
- Heating: `heat_none`, `heat_mains_gas_only`, `heat_electric_only`, `heat_other`, `heat_two_or_more`

Category labels vary between Nomis exports. **Map labels explicitly in a dict at the top of the module** rather than relying on column position, and raise a clear error if an unmapped label appears.

### 4.4 `build_master.py`

Orchestrates: load each source → validate individually → left join onto the **canonical LSOA spine**.

The spine is the authoritative list of Birmingham 2021 LSOAs, taken from the census or boundary file — **not** from EPC. Using EPC as the spine would silently drop any LSOA with zero certificates, which is itself a finding.

Join with `how="left"` from the spine, then report per-source match rates. Write to `data/processed/birmingham_lsoa_master.parquet` plus a `.csv` sibling for inspection.

---

## 5. Validation (`validate.py`)

Run automatically at the end of `build_master.py`. Fail loudly.

1. **Spine integrity** — row count equals the number of Birmingham LSOAs; `LSOA21CD` unique and non-null; all match `^E01\d{6}$`.
2. **Join coverage** — log the percentage of spine rows matched by each source. Any source below 95% halts the build with an explanatory error.
3. **Range checks** — all `*_pct_*` and `ten_*`/`heat_*` columns within 0–1; efficiency means within 1–100.
4. **Composition checks** — tenure proportions sum to ≈1.0 (tolerance 0.01); same for heating.
5. **Missingness report** — null count and percentage per column, written to `reports/data_quality.md`.
6. **Sanity correlation** — `imd_score` vs `epc_mean_current_eff` should be weakly negative. This is a smoke test, not a finding; log it, don't assert on it.

---

## 6. Conventions

- Python 3.11+, `pandas`, `pyarrow`, `pathlib`. No absolute paths anywhere — everything through `src/data/paths.py`.
- Type hints on all public functions; short docstrings stating units and granularity.
- `logging` module, not `print`.
- All column names snake_case except `LSOA21CD`.
- Deterministic: no randomness, no network calls at runtime. The pipeline must produce a byte-identical output from identical inputs — this is a precondition for the provenance layer.
- Every magic number (floor-area bounds, coverage threshold, tolerances) declared as a named constant at module top.

---

## 7. `data/raw/README.md` — mandatory

Raw data is gitignored, so this file *is* the audit trail. One entry per dataset:

```markdown
## <Dataset name>
- Source URL:
- Exact query / filter / table code used:
- Date downloaded:
- File name as saved:
- Approx. file size / row count:
- Licence:
- Known limitations:
```

This is not admin overhead. Reproducibility and data provenance are directly assessed in the methodology chapter, and the project's blockchain component is explicitly about provenance — an undocumented pipeline would undercut the dissertation's own argument.

---

## 8. Definition of done

- [ ] `python -m src.data.build_master` runs clean from a fresh clone plus raw files
- [ ] `data/processed/birmingham_lsoa_master.parquet` exists, one row per Birmingham LSOA
- [ ] `reports/data_quality.md` generated
- [ ] All validation checks pass, or failures are explained in the data-quality report
- [ ] `data/raw/README.md` complete for every dataset
- [ ] `tests/test_join.py` passes — covers spine uniqueness, proportion sums, and a known-value spot check for one LSOA

---

## 9. Explicitly out of scope

Do not build in this stage: retrofit scoring or any ML model; digital twin state machine or simulation; smart contracts or hashing logic; maps, dashboards, or Streamlit apps; scenario definitions.

If a task seems to require any of the above, stop and flag it rather than implementing it.
