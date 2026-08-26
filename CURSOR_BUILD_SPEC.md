# RetrofitTrust Birmingham — Build Specification for Cursor

**Purpose of this document:** this is the single consolidated technical brief for building the proof-of-concept code. It exists so an AI coding assistant (Cursor) has everything needed in one place, without having to infer decisions already made. Follow this document's decisions exactly; where something is marked "open," ask before assuming.

---

## 0. Project context (read first)

This is the technical proof-of-concept for an MSc Artificial Intelligence dissertation at Birmingham City University. The dissertation title is fixed: **"Intelligent Urban Decision-Making Framework Using AI, Blockchain and Digital Twins."** The illustrative scenario is **RetrofitTrust Birmingham** — domestic retrofit prioritisation and fuel poverty under Birmingham's Route to Zero.

**This is a proof-of-concept, not a production system.** Every decision below has already been made through dedicated research and is intended to stay small, explainable, and well-scoped. Do not:
- upgrade the core model to deep learning
- build a real blockchain network (Hyperledger, Ethereum mainnet/testnet) unless explicitly asked later
- build a 3D or game-engine digital twin
- add features not listed in this spec without checking first

If asked to add something not in this spec, treat it as a candidate for **DEFER** or **REJECT**, and flag the scope question rather than just building it.

**The deliverable that matters most academically is the *research*, not the software.** The code exists to demonstrate the framework, particularly the integration between its four parts. Prioritise that integration working end-to-end over polishing any single component.

---

## 1. The four programs to build

| # | Program | Type | Produces |
|---|---|---|---|
| 1 | Data pipeline + AI ranking model | ML (gradient-boosted trees) + light preprocessing DL | Merged dataset; retrofit priority scores; SHAP explanations |
| 2 | Data-quality screen | Autoencoder + Isolation Forest ensemble | Anomaly flags on EPC records (used in program 1's preprocessing) |
| 3 | Digital twin dashboard | Streamlit app | Interactive maps, cohort filters, what-if scenarios, SHAP viewer |
| 4 | Ledger + integration backend | FastAPI + SQLite + hash-chain | Tamper-evident audit log; orchestrates the twin↔AI↔ledger↔write-back loop |

Build in this order: **1 → 2 (folds into 1's preprocessing) → 3 → 4**. Get the model working on real data before touching the dashboard or ledger.

---

## 2. Data sources and join key

Three datasets, joined on **2021 LSOA code**:

1. **EPC Domestic Energy Performance Certificates** — `epc.opendatacommunities.org`. Full England & Wales bulk download (~8.26GB), **filter to Birmingham local authority only** before doing anything else. Fields: current/potential energy rating, floor area, heating type, wall/roof/window construction, age band, recommended improvements, LSOA code.
2. **English Indices of Deprivation 2025** — GOV.UK, published **30 October 2025** (not 17 November — that date was wrong in earlier project notes and has been corrected). LSOA-level IMD score/rank/decile, plus the income deprivation domain specifically. Uses 2021 LSOA boundaries.
3. **ONS Census 2021 via Nomis** — two tables: **Tenure [TS054]** and **Central Heating**, both at LSOA level, filtered to Birmingham.

**LSOA boundary geometries** (for mapping): ONS Open Geography Portal, 2021 LSOA, use the **Generalised Clipped (BGC)** boundaries for web rendering performance. Export as GeoJSON.

**Known data-quality issues to handle in preprocessing** (this is why program 2 exists):
- EPC modelled-vs-metered performance gap: ~16% for gas-heated homes, ~31% for electrically heated homes
- Assessor error: ~6% average change in predicted heating demand
- Coverage bias: EPCs only exist where triggered by sale/let/new build
- IMD is area-level — never treat an LSOA's IMD score as describing a specific household (ecological fallacy)

---

## 3. Program 1 — Data pipeline + AI ranking model

**Core model: LightGBM (primary) or XGBoost (equally acceptable). NOT deep learning.** This is a settled decision, not open for reconsideration — tree ensembles are the empirically stronger, more interpretable choice for this data type. Random forest as a baseline comparison is fine.

**Target:** a retrofit priority score per property (or aggregated to LSOA level) — frame as a regression on a composite need/benefit measure, or a probability of a "high priority" class combining low EPC efficiency with high deprivation.

**Explainability: SHAP with `TreeExplainer`.**
- Global: `shap.plots.beeswarm` and `shap.plots.bar` for overall feature importance
- Local: `shap.plots.waterfall` per property/LSOA — this is what the dashboard will show to explain individual rankings
- Note in code comments / docstrings: SHAP assumes feature independence and can misattribute importance among correlated features (floor area, room count, heating cost are correlated here) — this is a known limitation to discuss in the dissertation, not something to "fix" in code

**Preprocessing:**
- One-hot encode categoricals (heating type, construction type, tenure) — low cardinality, standard approach
- Standardise numeric features (matters much more for the autoencoder in program 2 than for the tree model itself, but do it consistently across the pipeline)
- Median imputation for missing numerics, explicit "missing" category for categoricals, **plus a missingness indicator flag column** — the pattern of missingness is itself a signal
- Do NOT silently drop flagged/anomalous records — see program 2's downstream handling rule

**Repo structure:** use the **Cookiecutter Data Science (CCDS) v2** layout:
```
data/{raw,interim,processed,external}
notebooks/          (numbered, e.g. 1.0-eda.ipynb — for exploration/narrative only)
src/                (all reusable logic — preprocessing, training, ledger, SHAP utils as importable modules)
models/             (serialised model artifacts via joblib)
reports/figures/
references/
requirements.txt
README.md
```
Treat `data/raw` as immutable — never edit it in place.

---

## 4. Program 2 — Data-quality/anomaly screening (autoencoder)

This runs **before** the model in program 1 trains, as a preprocessing gate.

**Library: PyOD.** Use PyOD's `AutoEncoder`, `IForest` (Isolation Forest), and its `standardizer` / `average` / `maximization` combination utilities — one consistent API instead of hand-rolling PyTorch/Keras plus separate sklearn calls.

**Architecture (autoencoder):**
- Shallow — one encoder hidden layer, small bottleneck, one decoder hidden layer. Do not go deeper; this dataset does not justify it and overfitting is a real risk.
- Bottleneck sized comfortably below half the (one-hot-expanded) input dimensionality — tune empirically by watching validation reconstruction loss.
- Linear output layer (needed to reconstruct standardised numeric values correctly).
- Modest L2 + dropout on hidden layers only, not the bottleneck. Early stopping on validation loss.
- Lightly denoising preferred if easy to implement (small input corruption during training) — not essential for v1.

**Threshold setting — do NOT use an arbitrary "top 5%" cutoff.** Use one of:
- Mean + k·σ of reconstruction error on a clean validation subset, or
- Extreme Value Theory (fit a Generalised Pareto Distribution to the upper tail) — preferred if time allows, more rigorous
- Tune k/threshold against the synthetic anomaly injection set (see evaluation, below)

**Combining with Isolation Forest:**
1. Get raw anomaly scores from both the autoencoder and Isolation Forest
2. Z-score standardise each detector's scores independently (PyOD's `standardizer`)
3. Combine via `average` (for a stable consensus flag) — report both the consensus (high-precision) set and the union (high-recall) set

**Evaluation (no ground truth exists, so):**
- Synthetic anomaly injection: corrupt copies of clean records (swap heating type, scale floor area implausibly, mismatch construction/age) and measure recall
- Manual spot-check of top-N flagged records for face validity
- Sanity-check the overall flagged rate against known EPC error-rate research (documented range: roughly 27% of records show at least one quality flag, true error rate estimated 36–62% in the wider literature) — if your flagged rate is wildly different, investigate before trusting it
- Multi-seed stability check (retrain with different seeds, compare overlap of flagged sets)

**Downstream handling — quarantine and flag, never silently delete.** Attach a data-quality confidence score/flag to each record. Either down-weight flagged records in LightGBM training (sample weights) or exclude from training only while still scoring them at inference with a "low confidence" caveat surfaced in the dashboard. Silent deletion risks systematically excluding unusual-but-real properties (e.g. flats/maisonettes are disproportionately flagged in EPC error research) — an equity problem in a fuel-poverty context.

**Explainability for the anomaly flags too:** compute **per-feature reconstruction error**, not just a single overall anomaly score. This tells you which specific field (e.g. floor area, heating type) looks implausible for a given record — keep this consistent with the SHAP explainability principle used in program 1, rather than making anomaly detection a black box.

---

## 5. Program 3 — Digital twin dashboard

**Framework: Streamlit.** (Not Dash, Panel, Gradio, or a Next.js/FastAPI split — Streamlit is fastest to build well for a solo PoC and integrates cleanly with everything else in this stack. A Next.js rebuild is a legitimate *future* upgrade, not part of this build.)

**Required features:**
- LSOA choropleth map coloured by retrofit priority score — use **geopandas + Plotly** (`px.choropleth_mapbox`) with the Birmingham-filtered 2021 LSOA GeoJSON
- Cohort filter/selection controls (sliders, dropdowns) to select candidate properties/LSOAs
- What-if scenario comparison: select a cohort, show modelled before/after impact (aggregate EPC uplift, estimated fuel-poverty indicator change)
- Embedded SHAP explanation panel — clicking/selecting an LSOA or property shows its waterfall plot
- Use `st.cache_data` for expensive operations (loading the merged dataset, geometries) so the map stays responsive

**The write-back loop — this is the most important feature, don't skip it.** The dashboard must visibly update when a "verified outcome" is recorded via the ledger (program 4). Concretely: persist twin state in SQLite; when program 4 writes a verification record, the dashboard re-reads state on next interaction and the choropleth/cohort table reflects the update. This closed loop — not the visual polish — is what makes this a "digital twin" rather than a static dashboard, and it's the dissertation's central technical claim (see §4.3 of the architecture chapter already written).

**Deployment:** Streamlit Community Cloud, deployed straight from a public GitHub repo. Note the free-tier constraint (1GB RAM per app, sleeps after inactivity) — fine for a PoC demo.

---

## 6. Program 4 — Ledger + integration backend

**Ledger: a Python hash-chain using the standard library `hashlib` (SHA-256). NOT Hyperledger Fabric, NOT a real Ethereum testnet deployment.** This is explicitly permitted and preferred under the project's "documented ledger simulation" scope option — don't second-guess this and reach for a heavier blockchain framework.

**Block structure:**
```python
{
  "index": int,
  "timestamp": iso8601 string,
  "data": {...},          # e.g. {"type": "eligibility"/"works_claimed"/"verification", "lsoa": ..., "details": ...}
  "previous_hash": str,
  "hash": str              # sha256 of the canonical JSON of the block minus "hash" itself
}
```
- **Serialise canonically before hashing** — use `json.dumps(..., sort_keys=True)`. Inconsistent serialisation is the most common cause of spurious chain-integrity failures; get this right from the start.
- Implement a `verify_chain()` method that walks the chain and confirms each block's stored hash matches a recomputed hash, and that each `previous_hash` matches the prior block's actual hash.
- Build a deliberate-tampering demo (modify a historical record, show `verify_chain()` catches it) — this is your evidence that the tamper-evidence property actually works, and it's an easy, concrete thing to screenshot for the dissertation.
- The eligibility → works-claimed → verification sequence is what gets recorded. **The grant/installer transaction data feeding this is synthetic — generate it programmatically and label it clearly in code comments and any UI display ("SYNTHETIC DATA" badge or similar).**

**Backend: FastAPI**, orchestrating:
- Loading the serialised model (`joblib`) and serving `/rank` and `/explain` endpoints
- The ledger module (`/ledger/append`, `/ledger/verify`)
- SQLite persistence (Python's built-in `sqlite3` — no need for Postgres at PoC scale) holding the twin's current state
- Pydantic schemas to validate requests

**The integration sequence to implement end-to-end (this is the demo):**
1. Twin (Streamlit) selects a candidate cohort from the merged dataset
2. Calls the AI service → gets back ranked scores + SHAP explanations
3. A prioritisation decision is appended to the ledger
4. A (synthetic) verification outcome is later appended to the ledger
5. SQLite state updates; twin re-reads and reflects the change

Get this whole loop working manually/scripted before worrying about UI polish on any individual piece.

---

## 7. Explicit non-goals (do not build these without asking)

- Real-time IoT sensor ingestion
- A production authentication/user-management system
- A real deployed blockchain network of any kind
- A 3D/game-engine visualisation
- Deep learning as the core ranking model
- Time-series forecasting on DESNZ consumption data (interesting but out of scope — a static contextual feature is enough if wanted at all)
- Anything that turns this into a deployable commercial product rather than a demonstrable PoC

If you (Cursor) find yourself building something not listed in sections 3–6 above, stop and flag it rather than continuing.

---

## 8. Suggested build sequence with checkpoints

1. **Data ingestion & Birmingham filtering** — get EPC filtered, IMD and Census loaded, joined on LSOA. Checkpoint: a single merged dataframe with sane row counts and no silent data loss.
2. **Autoencoder + Isolation Forest screening** — checkpoint: a flagged-rate that's roughly sane against the ~27–60% EPC error-rate literature range, and synthetic-injection recall that's clearly better than chance.
3. **LightGBM model + SHAP** — checkpoint: a trained model with reasonable cross-validated performance, and a waterfall plot that visibly makes sense for at least one manually-inspected record.
4. **Streamlit dashboard (static first, no write-back yet)** — checkpoint: choropleth renders, cohort filtering works, SHAP panel displays.
5. **Ledger module standalone** — checkpoint: append records, `verify_chain()` passes, tampering demo catches a modification.
6. **FastAPI integration wiring everything together** — checkpoint: the full 5-step loop in §6 runs end-to-end, and the dashboard visibly updates after a write-back.
7. **Polish**: README with setup/run instructions, architecture diagram reference (already exists — see the dissertation Chapter 4 document), pinned `requirements.txt`, fixed random seeds throughout for reproducibility.

---

## 9. Open items (confirm with RJ before assuming)

- Exact composite definition of the "retrofit priority score" target variable (needs a specific formula combining EPC rating gap + IMD income domain, weighted how?)
- Whether to include Census tenure/heating as model features directly or only for stakeholder-analysis narrative
- Exact cohort size for the demo scenario (how many properties/LSOAs to showcase)
- Whether FastAPI is genuinely needed or whether Streamlit alone calling local Python modules is sufficient for the PoC (FastAPI is recommended for portfolio quality, but adds a moving part)

---

*This document supersedes no part of the Master Project Instructions or the Scenario Definition document — it operationalises the technical decisions already researched and agreed. If anything here appears to conflict with those documents, the Master Project Instructions take precedence.*
