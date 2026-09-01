# Checkpoint 6 — integration loop (twin → AI → ledger → SQLite)

Five-step loop from CURSOR_BUILD_SPEC §6. Grant/works/verification payloads
are **SYNTHETIC DATA**. Ledger is a hashlib SHA-256 hash-chain, not a live blockchain.

- Transport: `direct Python imports`
- Cohort size: **10** LSOAs
- Ranked LSOAs: **10**
- Top LSOA: `E01009481` (priority score 0.5816)
- Ledger height: **27** blocks
- Chain `verify_chain()`: **PASS**
- Verified LSOAs in SQLite: **5**
- Write-back checkpoint: **PASS**
- Seed: `42`

## Five steps

| Step | Result |
| --- | --- |
| 1 Twin cohort | 10 LSOAs |
| 2 AI rank | 10 ranked |
| 3 Ledger append | height 27 |
| 4 Verify chain | PASS |
| 5 SQLite write-back | 5 verified |

## Caveats

- Ecological fallacy: LSOA IMD is not household deprivation.
- SHAP TreeExplainer can misattribute among correlated EPC/IMD features.
- EPC modelled-vs-metered gap (~16% gas / ~31% electric).
- `06_integration_loop.png` is a summary badge, not a production monitoring dashboard.
