# Checkpoint 5 — Ledger standalone numbers

**SYNTHETIC DATA.** Grant, installer, and verification records on this chain are generated programmatically. They are not real Birmingham grant awards, installer invoices, or inspections. Amounts are assumed demo figures, not official ECO / Home Upgrade Grant rates.

The ledger is a Python `hashlib` SHA-256 hash-chain simulation, **not** a live blockchain (no Hyperledger, no Ethereum). Blocks are hashed from canonical JSON (`json.dumps(..., sort_keys=True)` excluding the `hash` field). `verify_chain()` recomputes each digest and checks `previous_hash` links.

| Metric | Value |
| --- | --- |
| Ledger path | `data/processed/ledger.json` |
| Block count | 24 |
| `verify_chain()` | **PASS** — ok (24 blocks) |
| `SYNTHETIC DATA` labelled blocks | 23 / 24 (genesis is structural) |
| Tamper demo (in-memory) | intact=PASS; after tamper=FAIL (detected) |
| Tampered block index | 1 |
| Live `ledger.json` modified by demo | no |
| Random seed | 42 |

## Blocks by type

| Type | Count |
| --- | --- |
| `genesis` | 1 |
| `eligibility` | 9 |
| `works_claimed` | 5 |
| `verification` | 9 |

## Figures

- `05_ledger_verify.png` — pass/fail badge and block-type counts
- `05_tamper_demo.png` — before/after `verify_chain()` (intact vs tampered copy)

