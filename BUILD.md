# BUILD.md — RetrofitTrust Birmingham: Technical Proof-of-Concept

**Dissertation:** Intelligent Urban Decision-Making Framework Using AI, Blockchain and Digital Twins
**Scenario:** Domestic retrofit prioritisation and fuel poverty under Birmingham's Route to Zero
**Scope:** Proof-of-concept, not production. Every component exists to demonstrate integration.

This is the build plan for what's left: the AI model, the digital twin dashboard, the blockchain
ledger, and the integration layer connecting them, plus the open project decisions that still
need answers before/while you build.

---

## 0. Repo structure (suggested)

```
retrofittrust-birmingham/
├── ai-model/              # Python — ranking model + explainability
│   ├── data/               # raw + processed (label synthetic clearly)
│   ├── notebooks/
│   ├── src/
│   └── outputs/            # scored properties/LSOAs -> feeds dashboard + ledger
├── ledger/                 # blockchain / permissioned-ledger prototype
│   ├── contracts/           # if Solidity route
│   └── src/
├── dashboard/               # Next.js + TypeScript digital twin
│   ├── app/
│   └── lib/
├── integration/              # scripts/tests proving data actually moves between components
└── docs/
    ├── architecture-diagram.*
    └── data-flow-diagram.*
```

---

## 1. AI component — retrofit priority model

**Goal:** an explainable model that ranks properties/LSOAs by retrofit priority.

### Steps
1. **Acquire data**
   - EPC open data (domestic certificates, register.epcregister.uk / DESNZ) — filter to
     Birmingham postcodes/LSOAs.
   - DESNZ sub-national energy consumption statistics (gas + electricity, LSOA level).
   - Census 2021 (tenure, dwelling age, household composition) via ONS/Nomis.
   - IMD (Index of Multiple Deprivation) — especially the "Living Environment" and "Income"
     domains, relevant to fuel poverty.
2. **Join and clean**
   - Common key: LSOA code (or postcode → LSOA lookup if working at property level).
   - Document every join, every dropped/null row, and every assumption in a data-prep notebook.
3. **Define the target**
   - No single "correct" label exists — this needs to be a constructed priority score, e.g. a
     weighted combination of: EPC band/SAP score, estimated fuel poverty risk (from IMD income
     domain + energy cost proxies), dwelling age/type. Document the weighting rationale — this
     is a methodological decision the dissertation must justify, not something to bury in code.
4. **Model**
   - Start simple: linear/logistic regression or a decision tree as a baseline explainable model.
   - If more structure is needed, gradient boosting (XGBoost/LightGBM) + SHAP for explainability.
   - Do NOT reach for deep learning — no dissertation justification, and it undermines the
     "explainable AI" requirement.
5. **Output**
   - A table: `lsoa_code | priority_score | top_3_contributing_factors | confidence/notes`.
   - Save as `ai-model/outputs/retrofit_scores.csv` (or similar) — this is what the dashboard
     and ledger will consume.
6. **Evaluate**
   - No ground truth exists for "correct" priority, so evaluation is about face validity and
     robustness: sensitivity analysis on weighting choices, sanity-check against known
     high-deprivation Birmingham wards, and a clear explainability example (e.g. SHAP plot for
     one property).

**Still open:** exact scoring formula/weights (needs literature justification — Stage 1 of the
methodology), and whether to work at LSOA or individual-property level (property level is more
realistic for a real system but harder to get clean data for — LSOA is the safer proof-of-concept
choice).

---

## 2. Digital twin component — housing stock dashboard

**Goal:** a data-driven representation of Birmingham's housing stock, not a 3D twin.

### Steps
1. **Data ingestion**
   - Load `retrofit_scores.csv` from the AI component.
   - Load ledger state (grant status, works completed) from the blockchain component once it
     exists — build the dashboard so this can be stubbed with mock data first, then swapped in.
2. **Core views**
   - Map or list view of Birmingham LSOAs coloured/sorted by priority score.
   - Detail view per LSOA: current EPC/energy profile, priority score, contributing factors,
     grant/works status (from ledger).
   - "What-if" scenario controls: e.g. retrofit rate per year, budget cap → recompute a simple
     projection (this can be a lightweight client-side calculation, not another ML model).
3. **Stack**
   - Next.js + TypeScript, a charting library (recharts/visx), and a simple map library
     (Leaflet or Mapbox) if you want geographic visualisation — optional, a sorted table/bar
     chart view is enough for proof-of-concept if time is tight.
4. **What NOT to build**
   - No 3D visualisation, no real-time IoT data feed, no user accounts/auth. This is a
     scenario-model dashboard, per Section 23 of the project brief.

**Still open:** whether to include a map view (nice for the dissertation's visual appeal, but
adds scope — classify under Section 19 if you're unsure whether it's worth the time).

---

## 3. Blockchain component — grant/works verification ledger

**Goal:** demonstrate data provenance and governance — this is deliberately the lightest component.

### Option A — Simple simulated ledger (recommended default)
- An append-only log (e.g. JSON file or lightweight DB table) where each entry is hashed and
  chained to the previous entry's hash (a minimal blockchain data structure), recording:
  `lsoa_code | event_type (eligibility/works_completed/performance_verified) | timestamp |
  data_hash | previous_hash`.
- Enough to demonstrate immutability/tamper-evidence and provenance without needing testnet ETH,
  wallets, or gas fees — much easier to demo and explain in a viva.

### Option B — Testnet smart contract
- Solidity contract on a public testnet (e.g. Sepolia) with functions like
  `recordEligibility()`, `recordWorksCompleted()`, `verifyPerformance()`.
- Use Hardhat for local dev/testing before deploying to testnet.
- More "real" blockchain credibility for the write-up, but more setup overhead (wallet, testnet
  funds, deployment).

**Recommendation:** build Option A first since it's enough to demonstrate the governance/provenance
argument; upgrade to Option B only if time allows and it would meaningfully strengthen the
dissertation's technical contribution.

### Steps (either option)
1. Design the event schema (what gets recorded, what's hashed vs. stored in full).
2. Seed with SYNTHETIC grant/installer/works data — label it synthetic everywhere.
3. Link each entry to an `lsoa_code` from the AI output so the dashboard can join on it.
4. Write a short script/test that proves tamper-evidence (alter an entry, show the hash chain
   breaks).

**Still open:** Option A vs B — worth deciding early since it affects the `ledger/` folder
structure and what the integration layer needs to call.

---

## 4. Integration layer (the most important part)

**Goal:** prove the three components actually connect — this is the dissertation's central claim.

### Minimum viable integration demo
1. Run AI model → produces `retrofit_scores.csv`.
2. Script pushes each LSOA's score (or a hash of it) into the ledger as an "eligibility" event.
3. Dashboard reads both the AI scores and the ledger state, displays them joined on `lsoa_code`.
4. One integration test/script that runs all three steps end-to-end and asserts the data matches
   across all three components.

### Deliverables for the dissertation
- A data-flow diagram (`docs/data-flow-diagram.*`) matching what the code actually does — don't
  let the diagram and the code drift apart.
- A short `integration/README.md` walking through the end-to-end demo, since this is what you'll
  want to screenshot/reference directly in the write-up.

---

## 5. Still-open project decisions (blocking or semi-blocking)

Carried over from Section 26 of the project brief — resolve these alongside building:

1. **Scoring formula/weights** for the AI priority model (needs literature justification).
2. **LSOA vs property-level** granularity.
3. **Ledger option A vs B** (simulated vs testnet smart contract).
4. **Map view or not** in the dashboard.
5. **Data availability check** — confirm which Birmingham/West Midlands datasets are actually
   downloadable in the format you need (EPC bulk download requires a free account; DESNZ/Census/
   IMD are open but check LSOA vs ward boundaries match).
6. **Referencing style** (Harvard assumed) and **ethics approval** status — don't affect the
   code, but don't forget them.

---

## 6. Suggested build order

1. AI model (needs data first — start here, it's the long pole).
2. Simple ledger (Option A) — quick to build, unblocks integration work early.
3. Dashboard — build against mock data first, then wire in real AI + ledger outputs.
4. Integration scripts/tests + data-flow diagram.
5. (If time allows) upgrade ledger to Option B, add map view to dashboard.
