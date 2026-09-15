# RetrofitTrust Birmingham — What Was Built and How to Deploy the Dashboard

**Hand this whole file to Claude, ChatGPT, or another assistant** if you want help deploying. It describes the real code in this repository, not an earlier plan.

| Field | Value |
|---|---|
| Dissertation | Intelligent Urban Decision-Making Framework Using AI, Blockchain and Digital Twins |
| Institution | Birmingham City University, MSc AI (RJ, 2025–2026) |
| Scenario | RetrofitTrust Birmingham — domestic retrofit prioritisation and fuel poverty (Route to Zero) |
| Nature | Research **proof-of-concept**, not a production system |
| Language | British English throughout |
| Seed | `SEED = 42` (`src/retrofittrust/config.py`) |
| Layout | Cookiecutter Data Science v2 |

**Do not** add real blockchain, 3D twins, deep learning as the ranking model, production auth, IoT streaming, or time-series forecasting.

---

## 1. Copy-paste prompt for Claude / ChatGPT

Paste this block plus this file (or the whole repo):

```text
You are helping me deploy RetrofitTrust Birmingham, an MSc AI dissertation PoC.

Goal: get the Streamlit digital twin dashboard running so I can demonstrate it
(viva / screenshare). Prefer a reliable local Windows demo first. Cloud is optional.

Stack that MUST stay as-is:
- Ranking: LightGBM + SHAP TreeExplainer (not deep learning)
- Quality: PyOD AutoEncoder + Isolation Forest (flag/quarantine, never silent delete)
- Dashboard: Streamlit + geopandas + Plotly choropleth + SQLite write-back
- Ledger: Python hashlib SHA-256 hash-chain (NOT Ethereum/Hyperledger)
- Backend: FastAPI with /rank, /explain, /ledger/append, /ledger/verify
- Grant/works/verification payloads are SYNTHETIC DATA and must stay labelled

Read DEPLOYMENT.md in the repo root. Then:
1. Confirm Python version and create a venv from requirements.txt.
2. Start FastAPI AND Streamlit (two processes). Match the dashboard API URL to
   the uvicorn port (see Known issues — dashboard currently defaults to 8001,
   uvicorn defaults to 8000).
3. The dashboard still opens without trained models or merged parquet: it uses a
   labelled synthetic fallback. Full Birmingham maps need processed artefacts
   and ONS GeoJSON (gitignored / not in the clone).
4. Do not invent a Next.js frontend. The UI is Streamlit at
   src/retrofittrust/dashboard/app.py.
5. After it runs, tell me the two URLs and a 2-minute click-through for the viva.

Windows PowerShell. Project root is the retrofittrust-birmingham folder.
```

---

## 2. What to deploy to show the dashboard working

The dashboard is **Program 3**. A convincing demo of the **write-back loop** also needs **Program 4** (FastAPI + ledger + SQLite) running beside it.

### Two processes (this is the demo)

| Process | Command (from project root) | URL |
|---|---|---|
| 1. FastAPI backend | `uvicorn retrofittrust.api.main:app --reload --app-dir src --port 8001` | http://127.0.0.1:8001 |
| 2. Streamlit dashboard | `streamlit run src/retrofittrust/dashboard/app.py` | http://localhost:8501 |

Use **port 8001** for uvicorn unless you change the dashboard default. In `src/retrofittrust/dashboard/app.py`, `DEFAULT_API = "http://127.0.0.1:8001"`. README / sidebar help still mention 8000 — that mismatch is a known bug. Either:

- run uvicorn on **8001** (recommended, no code change), or
- run uvicorn on 8000 and type `http://127.0.0.1:8000` in the Streamlit sidebar **FastAPI base URL**.

### Code that must be present (dashboard demo)

These are the files the running UI actually imports:

```
src/retrofittrust/dashboard/app.py          # Streamlit entry point — THIS is the programme you "deploy"
src/retrofittrust/dashboard/data_loader.py
src/retrofittrust/dashboard/plots.py
src/retrofittrust/dashboard/cohort.py
src/retrofittrust/dashboard/state.py
src/retrofittrust/config.py
src/retrofittrust/api/main.py               # FastAPI — rank, explain, ledger
src/retrofittrust/api/features.py
src/retrofittrust/api/schemas.py
src/retrofittrust/ledger/chain.py
src/retrofittrust/ledger/twin_state.py
src/retrofittrust/ledger/synthetic.py
src/retrofittrust/ledger/tamper.py
src/retrofittrust/modeling/predict.py       # used if models/ranking_model.joblib exists
src/retrofittrust/modeling/explain.py
requirements.txt
```

Runtime artefacts created on first run (gitignored):

| Path | Role |
|---|---|
| `data/processed/twin_state.db` | SQLite twin state — dashboard re-reads this after verification |
| `data/processed/ledger.json` | SHA-256 hash-chain |

### What the dashboard can show *without* training data

If `data/processed/merged_lsoa.parquet` (and scores) are missing — they **are gitignored** and this clone currently has an empty `data/processed/` — the app still starts. It shows a labelled **SYNTHETIC DATA** frame and a Birmingham-centred demo grid instead of a real choropleth. Rank/explain fall back to a composite score if LightGBM is not loaded. Ledger buttons work once FastAPI is up.

That is enough to prove: map/table UI, cohort filters, what-if sliders, SHAP panel, eligibility → works → verification, chain verify, choropleth colour change after write-back.

### What you need for the *full* Birmingham dashboard (660 LSOAs)

These are **not** in git (see `.gitignore`). Copy them from the machine that ran the pipeline, or re-run scripts 01–03 after placing raw downloads.

| Artefact | Why |
|---|---|
| `data/processed/merged_lsoa.parquet` | Merged EPC + IMD + Census at LSOA grain |
| `data/processed/retrofit_scores.csv` (or parquet) | Consumer ranking table |
| `models/ranking_model.joblib` | Trained LightGBM for `/rank` and `/explain` |
| `data/external/lsoa_birmingham.geojson` | Real 2021 LSOA BGC polygons (currently missing — map uses synthetic grid) |
| Optional: quality-flagged parquet | Anomaly flags on the cohort table |

Checkpoint 4 (when those files were present locally): **660 LSOAs**, **475,073** properties represented, geometries = synthetic grid because GeoJSON was still missing.

---

## 3. Local Windows setup (recommended for viva)

Python on the build machine was **3.13.7**. Prefer **3.11 or 3.12** if Streamlit Cloud or Docker is used later (`torch` / `geopandas` wheels are more reliable there).

```powershell
cd "C:\Users\rajan\Desktop\Master Full Resouces\Program Full Build\retrofittrust-birmingham"

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Terminal 1 — API**

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn retrofittrust.api.main:app --reload --app-dir src --port 8001
```

Check: open http://127.0.0.1:8001/health — expect `"status": "ok"`.

**Terminal 2 — dashboard**

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run src/retrofittrust/dashboard/app.py
```

Sidebar should show **API connected**. If it shows **API not reachable**, the port does not match (8000 vs 8001).

### Two-minute viva click-through

1. Confirm sidebar: API connected; note LSOA count and Verified (write-back) metrics.
2. Click **Load demo cohort (10 LSOAs)**.
3. Choropleth (or bar fallback) + cohort table.
4. **Rank selected cohort** — `POST /rank`.
5. Pick an LSOA → **Explain selection** — SHAP waterfall.
6. Same LSOA → **Append eligibility** → **Append works claimed** → **Append verification**.
7. Map / twin status should show verified; priority may drop (PoC decay × 0.5).
8. **Verify chain** — `valid: true`.
9. Optional: **Tampering demo (in-memory copy)** — after-tamper invalid.

Keep the expander **Limitations** visible if an examiner asks.

### Optional integration script (same loop, no clicking)

```powershell
python scripts/run_integration_demo.py
python scripts/demo_tampering.py
```

Then refresh Streamlit. See `integration/README.md`.

---

## 4. Cloud / sharing options (if you do not want “localhost only”)

The original spec mentioned Streamlit Community Cloud. Treat that as **optional**. The honest demo is two local processes.

### Option A — Streamlit Community Cloud (dashboard only)

- Entry file: `src/retrofittrust/dashboard/app.py`
- Needs a **public GitHub repo** and `requirements.txt`
- **Problems:** free tier ~1 GB RAM; this `requirements.txt` includes `torch` (PyOD AutoEncoder) plus geopandas — likely too heavy. FastAPI is a **second process**; Cloud will not start uvicorn unless you hack it into the Streamlit script (not recommended).
- Result: map/filters/what-if may work; **Rank / ledger buttons stay disabled** without an API URL.

Only use this if you also host FastAPI somewhere and paste that HTTPS URL into **FastAPI base URL**.

### Option B — FastAPI on Render / Railway / Fly + Streamlit Cloud

1. Deploy FastAPI (`uvicorn retrofittrust.api.main:app --app-dir src --host 0.0.0.0 --port $PORT`).
2. Deploy Streamlit; set the sidebar URL (or change `DEFAULT_API`) to the public API origin.
3. CORS is already `allow_origins=["*"]` (PoC only).
4. Persist `data/processed/` (ledger.json + twin_state.db) with a volume, or they reset on each deploy.

### Option C — Docker Compose (best remote-style demo)

One container for uvicorn `:8001`, one for Streamlit `:8501`, shared volume for `data/processed` and `models`. A helper AI can write `Dockerfile` + `docker-compose.yml` from this file; they do not exist in the repo yet.

### Option D — Stay local + ngrok / Cloudflare Tunnel

For a supervisor who cannot sit at your PC: tunnel `localhost:8501` (and `:8001` if needed). Simplest after local run works.

**Recommendation:** get Option (local two terminals) working first. Do not start Cloud until `http://localhost:8501` shows **API connected**.

---

## 5. What was built in the whole programme

Four programmes in `src/retrofittrust/`, plus scripts, tests, and dissertation figures. Build order was 1 → 2 → 3 → 4.

```
data/{raw,interim,processed,external}     # raw is immutable; processed is gitignored
src/retrofittrust/{data,quality,modeling,ledger,api,dashboard}
models/   notebooks/   reports/figures/   scripts/   tests/   docs/   integration/
```

### Program 1 — Data pipeline + AI ranking

**Role:** Join Birmingham housing/deprivation data; train an explainable retrofit **priority ranker**.

| Piece | Location |
|---|---|
| Ingest EPC (Birmingham LA) | `src/retrofittrust/data/load_epc.py` |
| IMD 2025 | `src/retrofittrust/data/load_imd.py` |
| Census 2021 TS054 tenure + central heating | `src/retrofittrust/data/load_census.py` |
| Geography / GeoJSON helper | `src/retrofittrust/data/load_geography.py` |
| Join on 2021 LSOA (`lsoa21cd`) | `src/retrofittrust/data/merge.py`, `pipeline.py` |
| Composite target | `0.6 × EPC gap + 0.4 × IMD income` (`config.py`) |
| LightGBM train + RF baseline | `src/retrofittrust/modeling/train.py` |
| Predict / rank | `src/retrofittrust/modeling/predict.py` |
| SHAP TreeExplainer | `src/retrofittrust/modeling/explain.py` |
| Runner | `scripts/01_ingest_and_merge.py`, `scripts/03_train_ranking_model.py` |

**Evidence (when pipeline was run):** 476,226 Birmingham EPCs; 659 IMD + 659 Census LSOAs; 99.76% join retention (unmatched **kept**, not dropped); 660 LSOAs in the ranking export; LightGBM 5-fold RMSE 0.0079 vs RF 0.0144. High R² is expected because the target is a constructed function of features in the matrix — relative ranking, not metered energy.

### Program 2 — Data-quality / anomaly screen

**Role:** Flag implausible EPC-style records **before** ranking. Quarantine / down-weight — **never silent delete**.

| Piece | Location |
|---|---|
| PyOD AutoEncoder (shallow) | `src/retrofittrust/quality/autoencoder.py` |
| Isolation Forest + average ensemble | `src/retrofittrust/quality/ensemble.py` |
| Flags + per-feature reconstruction error | `src/retrofittrust/quality/flags.py`, `screen.py` |
| Synthetic injection evaluation | `src/retrofittrust/quality/evaluation.py` |
| Runner | `scripts/02_train_quality_screen.py` |

**Evidence:** 8,000-row stratified sample; consensus flag 41.8%, union 53.1% (inside ~27–60% EPC literature band); synthetic-injection recall 70.7%; flagged rows get sample weight 0.35 in LightGBM.

### Program 3 — Digital twin dashboard (what you deploy)

**Role:** Interactive Birmingham housing-stock twin. Not 3D. The academic claim is the **closed loop** (UI → AI → ledger → SQLite → map updates), not visual polish.

**Entry point:** `streamlit run src/retrofittrust/dashboard/app.py`

| Feature | What it does |
|---|---|
| Choropleth | Plotly + geopandas; colour by live priority or verified vs candidate |
| Bar fallback | If ONS GeoJSON missing |
| Cohort filters | IMD decile, priority range, EPC band, twin status |
| Demo cohort | Top 10 LSOAs by priority (`cohort.py`, `DEMO_COHORT_LSOA_COUNT = 10`) |
| LSOA detail | EPC current/potential, IMD, anomaly flag, ledger stub, twin status |
| Rank | `POST /rank` — caches scores in SQLite |
| What-if | Client-side EPC uplift, retrofit rate, **SYNTHETIC** budget cap — does **not** write the ledger |
| SHAP panel | `POST /explain` or in-process fallback waterfall |
| Ledger buttons | eligibility → works_claimed → verification (synthetic generator) |
| Verify / tamper | `GET /ledger/verify`, `GET /ledger/tamper-demo` |
| Write-back | Re-reads `twin_state.db` via `db_mtime_token()` so the map updates |
| Caching | `st.cache_data` on dataset and geometries |
| Caveats expander | Ecological fallacy, EPC gap, SHAP correlation, hash-chain not a real chain, SYNTHETIC DATA |

Helper modules: `data_loader.py`, `plots.py`, `export_figures.py` (static PNGs/HTML for the dissertation).

### Program 4 — Ledger + FastAPI integration backend

**Role:** Orchestrate twin → AI → ledger → SQLite. Documented hash-chain simulation, not a live blockchain.

**Entry point:** `uvicorn retrofittrust.api.main:app --app-dir src`

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Model loaded? Ledger valid? |
| POST | `/rank` | Rank LSOA codes; cache to SQLite |
| POST | `/explain` | SHAP (or composite) for one LSOA |
| POST | `/ledger/append` | Append hashed block; update twin state |
| GET | `/ledger/verify` | Walk chain; recent blocks |
| GET | `/ledger/tamper-demo` | In-memory tamper (does not persist) |

Ledger (`ledger/chain.py`): SHA-256 of canonical `json.dumps(..., sort_keys=True)` of the block minus `hash`; each block stores `previous_hash`. Events: `eligibility` | `works_claimed` | `verification`. Payloads from `ledger/synthetic.py` labelled **SYNTHETIC DATA**.

Write-back (`ledger/twin_state.py`): on verification, `verified = 1` and cached `priority_score × 0.5` (`VERIFIED_PRIORITY_DECAY`). Tables: `lsoa_state`, `cohort_selection`, `verified_outcomes`.

Ranking fallbacks inside the API: live LightGBM → saved `retrofit_scores` → composite weights. Dashboard SHAP falls back locally if FastAPI is down; ledger **buttons require** the API.

**Integration demo:** `scripts/run_integration_demo.py` (HTTP, or `--force-direct`). Checkpoint 6: 10 LSOAs ranked, ledger height 27, `verify_chain()` PASS, 5 verified in SQLite, top LSOA `E01009481` (score 0.5816).

**Tests:** `python -m unittest discover -s tests -v`  
`tests/test_ledger_chain.py`, `tests/test_integration_loop.py`

---

## 6. End-to-end loop (dissertation claim)

1. Twin selects a cohort (dashboard or `select_demo_cohort`).
2. `POST /rank` (+ `POST /explain` for one LSOA).
3. Ledger append **eligibility** (and optional **works_claimed**).
4. Ledger append **verification** (synthetic inspection).
5. SQLite updates; Streamlit re-reads; choropleth / status / priority change.

Diagrams that match this code: `docs/data-flow-diagram.md`, `docs/architecture-diagram.md`. Viva notes: `integration/README.md`.

---

## 7. Pinned dependencies (`requirements.txt`)

```
pandas==2.2.3
numpy==2.1.3
scikit-learn==1.5.2
lightgbm==4.5.0
shap==0.46.0
pyod==2.0.2
torch==2.6.0          # pulled in for PyOD AutoEncoder — heavy for Streamlit Cloud
geopandas==1.0.1
plotly==5.24.1
streamlit==1.40.1
fastapi==0.115.5
uvicorn==0.32.1
pydantic==2.10.2
joblib==1.4.2
matplotlib==3.9.2
requests==2.32.3
httpx==0.28.1
pyarrow==18.0.1 / 18.0.0
```

No `packages/` installable package metadata beyond importing `src/` via `--app-dir src` or `sys.path`.

---

## 8. Raw data (only if re-running the pipeline)

`data/raw/` is immutable and gitignored. See `data/raw/README.md`.

1. EPC domestic — Birmingham LA → `data/raw/epc_birmingham/`
2. IMD 2025 File 7 → `data/raw/imd2025/`
3. Census TS054 + TS046 (central heating) at LSOA, Birmingham → `data/raw/census/`
4. ONS 2021 LSOA BGC GeoJSON → `data/external/lsoa_birmingham.geojson`

Then:

```powershell
python scripts/01_ingest_and_merge.py
python scripts/02_train_quality_screen.py
python scripts/03_train_ranking_model.py
```

You do **not** need this to open Streamlit; you **do** need it (or copied processed files) for real Birmingham LSOA names, scores, and a real map.

---

## 9. Known issues the deploying assistant should handle

1. **Port mismatch:** dashboard default `8001`; README/sidebar often say `8000`. Align them.
2. **Gitignored artefacts:** `data/processed/*`, `models/*.joblib`, `*.db` are not in the clone. Synthetic fallback is intentional.
3. **Missing GeoJSON:** choropleth is a demo grid until `data/external/lsoa_birmingham.geojson` exists.
4. **Torch + 1 GB Cloud:** Streamlit Community Cloud may OOM. Local or Docker is safer.
5. **No auth:** CORS `*`. Do not expose the API on the public internet without understanding that.
6. **Windows console Unicode:** logs may garble; exit code 0 still means success.
7. **Corrupt ledger:** delete `data/processed/ledger.json` and re-run the demo.
8. **Parent folder data:** an older pipeline output may live under `Program Full Build\data\processed\` (sibling of this repo), not inside `retrofittrust-birmingham`. Copy parquet/joblib in if you want the 660-LSOA view without re-ingesting.

---

## 10. What not to tell a deploying AI to build

- Next.js / Leaflet rewrite (superseded; UI is Streamlit)
- Solidity / Sepolia / Hyperledger
- User login, secrets manager, Postgres
- 3D / Cesium / Unity twin
- Replacing LightGBM with a neural ranker

---

## 11. Dissertation figures already generated

Tracked under `reports/figures/` (see `INDEX.md`): merge audits (01), quality flags (02), LightGBM/SHAP (03), dashboard choropleth/bar (04), ledger verify + tamper (05), integration loop (06). Grant visuals are **SYNTHETIC DATA**.

---

## 12. Success criteria for “the dashboard is working”

Minimum (synthetic fallback, both processes up):

- [ ] http://localhost:8501 loads **RetrofitTrust Birmingham**
- [ ] Sidebar **API connected** (health JSON ok)
- [ ] Map or bar chart renders
- [ ] Demo cohort of 10 LSOAs loads
- [ ] Rank + Explain produce a table / waterfall
- [ ] Append verification → Verified count or twin status changes after refresh
- [ ] Verify chain returns valid
- [ ] SYNTHETIC DATA captions visible on grant/ledger actions

Full (processed data copied or pipeline re-run):

- [ ] Hundreds of real `E01*` LSOA codes (checkpoint used 660)
- [ ] Real names / EPC / IMD on the detail panel
- [ ] `model_loaded: true` on `/health` if `ranking_model.joblib` is present
- [ ] True Birmingham polygons if GeoJSON is added
