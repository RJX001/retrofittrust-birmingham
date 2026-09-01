# RetrofitTrust Birmingham — Data Flow Diagram

This diagram matches the **actual Python code paths** in `src/retrofittrust/` (Streamlit + FastAPI + hash-chain + SQLite). There is no Next.js frontend in this PoC.

## End-to-end integration loop

```mermaid
flowchart LR
    subgraph Twin["Program 3 — Streamlit dashboard"]
        Cohort["cohort.py\nselect_demo_cohort()"]
        UI["dashboard/app.py\nchoropleth + SHAP panel"]
        StateRead["state.py\nget_lsoa_state()"]
    end

    subgraph API["Program 4 — FastAPI (api/main.py)"]
        Rank["POST /rank"]
        Explain["POST /explain"]
        Append["POST /ledger/append"]
        Verify["GET /ledger/verify"]
    end

    subgraph AI["Program 1 — LightGBM + SHAP"]
        Model["modeling/predict.py\nrank_lsoas()"]
        Shap["modeling/explain.py\nexplain_lsoa()"]
        Fallback["api/features.py\ncomposite fallback"]
    end

    subgraph Ledger["Hash-chain (ledger/chain.py)"]
        Chain["ledger.json\nSHA-256 blocks"]
        Synth["ledger/synthetic.py\nSYNTHETIC DATA events"]
    end

    subgraph DB["SQLite twin state (ledger/twin_state.py)"]
        SQLite["data/processed/twin_state.db"]
    end

    Cohort -->|"lsoa_codes"| Rank
    UI -->|"HTTP or import"| Rank
    UI -->|"HTTP or import"| Explain
    Rank --> Model
    Rank --> Fallback
    Explain --> Shap
    Explain --> Fallback
    Rank -->|"cache_priority_scores"| SQLite
    Append --> Synth
    Append --> Chain
    Append -->|"apply_ledger_event"| SQLite
    StateRead --> SQLite
    UI --> StateRead
    Verify --> Chain
```

## Ledger event sequence (synthetic grant lifecycle)

```mermaid
sequenceDiagram
    participant Demo as run_integration_demo.py
    participant API as FastAPI /ledger/append
    participant Chain as ledger.json
    participant Twin as twin_state.db

    Demo->>API: eligibility (lsoa_code, priority_score)
    API->>Chain: append block (hash chained)
    API->>Twin: cohort + priority cache

    Demo->>API: works_claimed (SYNTHETIC DATA)
    API->>Chain: append block
    API->>Twin: metadata update

    Demo->>API: verification (epc_uplift_bands)
    API->>Chain: append block
    API->>Twin: verified=1, priority decay 50%

    Demo->>API: GET /ledger/verify
    API->>Chain: verify_chain() → valid
```

## Block schema (on disk)

Each block in `data/processed/ledger.json`:

| Field | Type | Description |
|-------|------|-------------|
| `index` | int | Monotonic block number (0 = genesis) |
| `timestamp` | ISO 8601 UTC | Append time |
| `data` | object | Event payload (`type`, `lsoa_code`, …) |
| `previous_hash` | str (64 hex) | SHA-256 of prior block |
| `hash` | str (64 hex) | SHA-256 of canonical JSON (excluding `hash`) |

Event types in `data.type`: `genesis`, `eligibility`, `works_claimed`, `verification`.

Grant/installer/inspection fields are **SYNTHETIC DATA** (see `ledger/synthetic.py`).

## Tamper-evidence demo

```mermaid
flowchart TD
    A["demonstrate_tampering()\ntamper.py"] --> B["Deep-copy chain in memory"]
    B --> C["Alter block[1].data without rehashing"]
    C --> D["compute_block_hash() ≠ stored hash"]
    D --> E["verify_chain() fails"]
    F["demo_tampering.py --persist"] --> G["Optional on-disk tamper\nfor viva screenshot"]
```

## Key files

| Step | Module |
|------|--------|
| Cohort selection | `dashboard/cohort.py`, `scripts/run_integration_demo.py` |
| Ranking | `api/main.py` → `modeling/predict.py` or `api/features.py` |
| SHAP | `api/main.py` → `modeling/explain.py` or `api/features.py` |
| Ledger append | `ledger/chain.py`, `ledger/synthetic.py` |
| Write-back | `ledger/twin_state.py` |
| Dashboard refresh | `dashboard/state.py`, `dashboard/data_loader.py` |
