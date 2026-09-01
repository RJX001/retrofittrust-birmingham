# RetrofitTrust Birmingham — Integration Demo Walkthrough

This folder documents the **end-to-end integration loop** — the dissertation's central technical claim: twin → AI → ledger → SQLite write-back.

Grant, installer, and inspection records are **SYNTHETIC DATA** throughout.

## Prerequisites

```bash
cd "Program Full Build"
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Optional (real Birmingham LSOAs instead of synthetic fallback):

```bash
python scripts/01_ingest_and_merge.py
python scripts/02_train_quality_screen.py
python scripts/03_train_ranking_model.py
```

The integration demo still runs without steps 01–03 using labelled synthetic LSOAs from `api/features.py`.

## Quick run (recommended for viva)

### Terminal 1 — FastAPI (optional but shows HTTP path)

```bash
uvicorn retrofittrust.api.main:app --reload --app-dir src
```

### Terminal 2 — five-step loop

```bash
python scripts/run_integration_demo.py
```

Force direct Python imports (no HTTP):

```bash
python scripts/run_integration_demo.py --force-direct
```

### Terminal 3 — tamper-evidence demo

```bash
python scripts/demo_tampering.py
```

On-disk tamper (shows `verify_chain()` failure on saved file; restore by deleting `data/processed/ledger.json`):

```bash
python scripts/demo_tampering.py --persist
```

## What the script does

| Step | Action | Code path |
|------|--------|-----------|
| 1 | Twin selects cohort (10 LSOAs) | `dashboard/cohort.py` |
| 2 | Rank + SHAP explain top LSOA | `POST /rank`, `POST /explain` or `modeling/*` |
| 3 | Append **eligibility** to ledger | `POST /ledger/append` |
| 3b | Append **works_claimed** (SYNTHETIC DATA) | `POST /ledger/append` |
| 4 | Append **verification** (SYNTHETIC DATA) | `POST /ledger/append` |
| 5 | Confirm SQLite write-back | `ledger/twin_state.py` |

Expected log terminus:

```
CHECKPOINT PASS — dashboard would reflect verification on next Streamlit refresh
Final ledger verify (direct): ok=True detail=ok (N blocks)
Integration loop complete [SYNTHETIC DATA]
```

## Verify via API (curl / browser)

With uvicorn running on port 8000:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/ledger/verify
curl http://127.0.0.1:8000/ledger/tamper-demo
```

Example append (synthetic eligibility):

```bash
curl -X POST http://127.0.0.1:8000/ledger/append ^
  -H "Content-Type: application/json" ^
  -d "{\"event_type\":\"eligibility\",\"lsoa21cd\":\"SYNTH_E01000000\",\"generate_synthetic\":true,\"priority_score\":0.85}"
```

## Dashboard confirmation

```bash
streamlit run src/retrofittrust/dashboard/app.py
```

After the integration script:

- Selected LSOA should show **verified** status where applicable.
- Priority score may reflect write-back decay (50% after verification).
- Ledger panel / state should reflect appended synthetic events.

## Screenshots for dissertation / viva

Capture these while the demo runs:

1. **Integration script output** — all five steps with `CHECKPOINT PASS` and `SYNTHETIC DATA` labels visible.
2. **`GET /ledger/verify` JSON** — `"valid": true` and recent blocks listing eligibility → works_claimed → verification.
3. **`demo_tampering.py` output** — `BEFORE tamper valid=True`, `AFTER tamper valid=False`, hash mismatch error.
4. **Streamlit choropleth** — LSOA colour change or verified badge after write-back refresh.
5. **SHAP panel** — waterfall/bar for the prioritised LSOA (from `/explain` or dashboard click).

Optional: run `demo_tampering.py --persist` then screenshot failed `/ledger/verify` before deleting `data/processed/ledger.json`.

## Automated tests

```bash
python -m unittest discover -s tests -v
```

- `tests/test_ledger_chain.py` — genesis schema, append/verify, tamper detection
- `tests/test_integration_loop.py` — FastAPI full loop + direct import path

## Artefact locations

| File | Purpose |
|------|---------|
| `data/processed/ledger.json` | Hash-chain persistence |
| `data/processed/twin_state.db` | SQLite twin write-back |
| `docs/data-flow-diagram.md` | Mermaid data-flow (matches this code) |
| `docs/architecture-diagram.md` | System architecture mermaid |

## Troubleshooting

| Issue | Fix |
|-------|-----|
| API unreachable | Script auto-falls back to direct imports; or use `--force-direct` |
| Ledger already invalid | Delete `data/processed/ledger.json` and re-run demo |
| Model not found | Demo uses composite weights; train with `03_train_ranking_model.py` for LightGBM |
| Unicode console glitches on Windows | Log content is fine; check exit code `0` |

## Restore clean ledger after `--persist` tamper

```bash
del data\processed\ledger.json
python scripts/run_integration_demo.py --force-direct
```
