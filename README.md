# RetrofitTrust Birmingham

**Proof-of-concept** for MSc AI dissertation (BCU): *Intelligent Urban Decision-Making Framework Using AI, Blockchain and Digital Twins.*

> This is a research demonstration, not a production system. Synthetic grant/verification data is used and labelled accordingly.

## Structure (Cookiecutter Data Science v2)

```
data/{raw,interim,processed,external}
src/retrofittrust/{data,quality,modeling,ledger,api,dashboard}
models/  notebooks/  reports/figures/  scripts/
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

## Data (manual download — place in `data/raw/`)

1. EPC Domestic bulk — filter to Birmingham LA → `data/raw/epc_birmingham/`
2. English Indices of Deprivation 2025 (LSOA) → `data/raw/imd2025/`
3. Census 2021 Nomis: TS054 Tenure + Central Heating (Birmingham LSOA) → `data/raw/census/`
4. ONS 2021 LSOA BGC boundaries GeoJSON → `data/external/lsoa_birmingham.geojson`

See `CURSOR_BUILD_SPEC.md` for full field lists and join key (2021 LSOA code).

## Run order

1. `python scripts/01_ingest_and_merge.py`
2. `python scripts/02_train_quality_screen.py`
3. `python scripts/03_train_ranking_model.py`
4. `uvicorn retrofittrust.api.main:app --reload --app-dir src`
5. `streamlit run src/retrofittrust/dashboard/app.py`

## Integration demo

End-to-end loop (twin → AI → ledger → write-back). Grant and verification records are **SYNTHETIC DATA**.

**Prerequisites:** pipeline scripts 01–03 complete; Programs 3+4 modules present.

```bash
# Terminal 1 — optional; demo falls back to direct imports if API is down
uvicorn retrofittrust.api.main:app --reload --app-dir src

# Terminal 2 — full five-step loop
python scripts/run_integration_demo.py

# Force direct imports (no HTTP)
python scripts/run_integration_demo.py --force-direct

# Tamper-evidence demo — modify a block, verify_chain() must fail
python scripts/demo_tampering.py
```

Steps executed: cohort selection → `/rank` + `/explain` → ledger eligibility → works claimed → synthetic verification → SQLite write-back → twin state check.

See also:

- [`integration/README.md`](integration/README.md) — viva walkthrough, screenshots, troubleshooting
- [`docs/data-flow-diagram.md`](docs/data-flow-diagram.md) — mermaid data flow (matches code)
- [`docs/architecture-diagram.md`](docs/architecture-diagram.md) — Streamlit ↔ FastAPI ↔ LightGBM/SHAP ↔ ledger ↔ SQLite

Run unit/integration tests:

```bash
python -m unittest discover -s tests -v
```

## Licence

Academic use — Birmingham City University MSc dissertation (RJ, 2025–2026).
