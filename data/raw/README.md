# Raw data (immutable)

This folder holds **third-party downloads only**. Pipeline code in
`src/retrofittrust/data/` reads from here and never writes back. Place
files in the subfolders below; do not edit them in place.

Join key for all three datasets: **2021 LSOA code**.

---

## 1. Domestic Energy Performance Certificates (EPC)

- **Source:** [epc.opendatacommunities.org](https://epc.opendatacommunities.org/) (register / sign in).
- **What to download:** domestic certificates for **Birmingham** local authority.
  The national England & Wales bulk is ~8.26 GB; prefer the authority-level
  zip (contains `certificates.csv` and `recommendations.csv`).
- **Where to put it:** `data/raw/epc_birmingham/`
  (for example `certificates.csv`, or the zip as downloaded).
- **If you only have the national bulk:** put it under `data/raw/epc/` or
  `data/raw/all-domestic-certificates/`. `load_epc()` will chunk-read and
  filter to `config.BIRMINGHAM_LA` ("Birmingham", LAD code E08000025).
  If the zip still has per-authority folders, the loader opens the
  Birmingham `certificates.csv` member directly.

**Caveats (documented in code, not "fixed" here):**

- Modelled-vs-metered performance gap (~16% gas, ~31% electric).
- Assessor error (~6% change in predicted heating demand).
- Coverage bias: an EPC exists only when triggered (sale, let, new build).

---

## 2. English Indices of Deprivation 2025

- **Source:** [English indices of deprivation 2025](https://www.gov.uk/government/statistics/english-indices-of-deprivation-2025)
  (Ministry of Housing, Communities and Local Government).
- **Published:** **30 October 2025** (uses 2021 LSOA boundaries).
- **What to download:** **File 7** — *All ranks, scores, deciles and population
  denominators* (CSV). This is the all-in-one extract and includes
  **Income Score (rate)**, which the composite target uses.
- **Where to put it:** `data/raw/imd2025/`

`load_imd()` filters to Birmingham on the local-authority name/code
column. Rank columns use "1 = most deprived"; the target uses the
income *score* (higher = more deprived) and does not invert it.

**Ecological fallacy:** IMD describes areas, not households. Do not treat
an LSOA income score as a statement about a specific dwelling.

---

## 3. Census 2021 (Nomis) — tenure and central heating

- **Source:** [Nomis Census 2021](https://www.nomisweb.co.uk/sources/census_2021)
  (Table Finder) or [Census 2021 bulk](https://www.nomisweb.co.uk/sources/census_2021_bulk).
- **Tables:**
  - **TS054** — Tenure of household
  - **TS046** — Type of central heating in household
- **Geography:** 2021 super output areas – lower layer (LSOA), filtered to
  **Birmingham**.
- **Where to put it:** `data/raw/census/`
  (CSV; Nomis wizard downloads with extra header rows are accepted).

Suggested filenames: `census2021-ts054-lsoa.csv`, `census2021-ts046-lsoa.csv`.

---

## Geography (not raw, but required for maps)

ONS Open Geography Portal → 2021 LSOA **BGC** (generalised clipped)
boundaries → export GeoJSON, clipped to Birmingham LAD `E08000025`.

Save as:

```
data/external/lsoa_birmingham.geojson
```

---

## After download

```text
data/raw/epc_birmingham/certificates.csv   (or .zip / .parquet)
data/raw/imd2025/File_7_IoD2025_....csv
data/raw/census/census2021-ts054-lsoa.csv
data/raw/census/census2021-ts046-lsoa.csv
data/external/lsoa_birmingham.geojson
```

Then, from the project root with `src` on `PYTHONPATH`:

```python
from retrofittrust.data import run_ingest
df = run_ingest()
```
