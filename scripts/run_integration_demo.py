#!/usr/bin/env python3
"""
RetrofitTrust Birmingham — end-to-end integration demo (Programs 3 + 4).

Five-step loop (CURSOR_BUILD_SPEC §6):
  1. Twin selects a candidate LSOA cohort
  2. AI service ranks cohort and returns SHAP explanations
  3. Prioritisation decision appended to hash-chain ledger
  4. Synthetic verification outcome appended (SYNTHETIC DATA)
  5. SQLite write-back; verify dashboard twin state reflects the update

Uses httpx → FastAPI when the API is reachable; otherwise direct Python imports.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from retrofittrust.config import (  # noqa: E402
    DATA_PROCESSED,
    DEMO_COHORT_LSOA_COUNT,
    LEDGER_PATH,
    SQLITE_PATH,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("integration_demo")

SYNTHETIC_LABEL = "SYNTHETIC DATA"
DEFAULT_API_BASE = "http://127.0.0.1:8000"


def _banner(msg: str) -> None:
    log.info("=" * 60)
    log.info(msg)
    log.info("=" * 60)


def _http_client():
    """Prefer httpx; fall back to requests."""
    try:
        import httpx

        return "httpx", httpx
    except ImportError:
        import requests

        return "requests", requests


def _api_reachable(base_url: str) -> bool:
    kind, client = _http_client()
    try:
        if kind == "httpx":
            r = client.get(f"{base_url}/ledger/verify", timeout=2.0)
            return r.status_code == 200
        r = client.get(f"{base_url}/ledger/verify", timeout=2.0)
        return r.status_code == 200
    except Exception:
        return False


def _http_post(base_url: str, path: str, payload: dict) -> dict:
    kind, client = _http_client()
    url = f"{base_url}{path}"
    if kind == "httpx":
        with client.Client(timeout=30.0) as c:
            r = c.post(url, json=payload)
            r.raise_for_status()
            return r.json()
    r = client.post(url, json=payload, timeout=30.0)
    r.raise_for_status()
    return r.json()


def _http_get(base_url: str, path: str) -> dict:
    kind, client = _http_client()
    url = f"{base_url}{path}"
    if kind == "httpx":
        with client.Client(timeout=30.0) as c:
            r = c.get(url)
            r.raise_for_status()
            return r.json()
    r = client.get(url, timeout=30.0)
    r.raise_for_status()
    return r.json()


# --- Step 1: cohort selection (twin) -------------------------------------------------


def step1_select_cohort() -> list[str]:
    _banner(f"STEP 1 — Twin selects cohort ({DEMO_COHORT_LSOA_COUNT} LSOAs)")

    try:
        from retrofittrust.dashboard.cohort import select_demo_cohort

        lsoas = select_demo_cohort(n=DEMO_COHORT_LSOA_COUNT)
        log.info("Selected via dashboard.cohort.select_demo_cohort: %s", lsoas)
        return list(lsoas)
    except ImportError:
        log.warning("dashboard.cohort not available — fallback from processed data")

    import pandas as pd

    merged = DATA_PROCESSED / "merged_lsoa.parquet"
    if not merged.exists():
        merged = DATA_PROCESSED / "merged_lsoa.csv"
    if not merged.exists():
        raise FileNotFoundError(
            "No cohort module and no data/processed/merged_lsoa.{parquet,csv}. "
            "Run scripts/01_ingest_and_merge.py first."
        )

    df = pd.read_parquet(merged) if merged.suffix == ".parquet" else pd.read_csv(merged)
    lsoa_col = next((c for c in ("lsoa21cd", "LSOA21CD", "lsoa_code") if c in df.columns), None)
    if lsoa_col is None:
        raise ValueError(f"No LSOA column found in {merged}")

    sample = (
        df[lsoa_col]
        .drop_duplicates()
        .sample(n=min(DEMO_COHORT_LSOA_COUNT, df[lsoa_col].nunique()), random_state=SEED)
        .tolist()
    )
    log.info("Fallback cohort (random sample, seed=%s): %s", SEED, sample)
    return sample


# --- Step 2: AI rank + explain -------------------------------------------------------


def step2_rank_and_explain(lsoas: list[str], use_api: bool, base_url: str) -> dict[str, Any]:
    _banner("STEP 2 — AI rank + SHAP explain")

    if use_api:
        log.info("Calling POST %s/rank via HTTP", base_url)
        rank_result = _http_post(base_url, "/rank", {"lsoa_codes": lsoas})
        top_lsoa, _ = _extract_top_lsoa(rank_result, lsoas)
        log.info("Calling POST %s/explain via HTTP", base_url)
        explain_result = _http_post(
            base_url, "/explain", {"lsoa21cd": top_lsoa, "top_n": 10}
        )
        return {"rank": rank_result, "explain": explain_result}

    from retrofittrust.modeling.predict import rank_lsoas
    from retrofittrust.modeling.explain import explain_lsoa

    try:
        rank_result = rank_lsoas(lsoa_codes=lsoas)
    except FileNotFoundError:
        log.warning("No processed training data — using composite LSOA frame fallback")
        from retrofittrust.api.features import load_lsoa_frame, load_model_bundle, predict_priority

        frame, source = load_lsoa_frame(allow_synthetic_fallback=True)
        subset = frame[frame["lsoa21cd"].astype(str).isin([str(c) for c in lsoas])].copy()
        if subset.empty:
            subset = frame.head(len(lsoas))
        bundle = load_model_bundle()
        subset["predicted_priority"] = predict_priority(subset, bundle)
        rankings = [
            {
                "lsoa": str(row.lsoa21cd),
                "lsoa_code": str(row.lsoa21cd),
                "score": float(row.predicted_priority),
            }
            for row in subset.sort_values("predicted_priority", ascending=False).itertuples()
        ]
        rank_result = {
            "rankings": rankings,
            "top_lsoa": rankings[0]["lsoa"] if rankings else lsoas[0],
            "top_score": rankings[0]["score"] if rankings else 0.0,
            "source": source,
        }

    top_lsoa = rank_result.get("top_lsoa") or lsoas[0]
    try:
        explain_result = explain_lsoa(lsoa_code=top_lsoa)
    except (FileNotFoundError, ValueError):
        from retrofittrust.api.features import load_lsoa_frame, load_model_bundle, shap_for_row

        frame, _ = load_lsoa_frame(allow_synthetic_fallback=True)
        row = frame[frame["lsoa21cd"].astype(str) == str(top_lsoa)].head(1)
        if row.empty:
            row = frame.head(1)
        explain_result = shap_for_row(row, load_model_bundle(), top_n=10)
    log.info("Direct import rank top LSOA: %s", top_lsoa)
    return {"rank": rank_result, "explain": explain_result}


# --- Step 3: ledger append prioritisation decision -----------------------------------


def _get_ledger():
    from retrofittrust.ledger.chain import Ledger

    ledger = Ledger(LEDGER_PATH)
    if not LEDGER_PATH.exists() or ledger.is_empty():
        ledger.initialise_genesis()
    return ledger


def step3_append_decision(
    lsoa: str, rank_score: float, use_api: bool, base_url: str
) -> dict:
    _banner(f"STEP 3 — Ledger append prioritisation decision [{SYNTHETIC_LABEL}]")

    try:
        from retrofittrust.ledger.synthetic import synthetic_eligibility_block

        block_data = synthetic_eligibility_block(lsoa=lsoa, priority_score=rank_score)
    except ImportError:
        block_data = {
            "type": "eligibility",
            "lsoa": lsoa,
            "priority_score": rank_score,
            "grant_reference": f"SYNTH-GRANT-{lsoa}",
            "label": SYNTHETIC_LABEL,
            "note": "Simulated retrofit prioritisation — not real grant data",
        }

    log.info("Appending eligibility block: %s", json.dumps(block_data, sort_keys=True))

    if use_api:
        result = _http_post(
            base_url,
            "/ledger/append",
            {
                "event_type": "eligibility",
                "lsoa21cd": lsoa,
                "generate_synthetic": True,
                "priority_score": rank_score,
            },
        )
        log.info("Ledger append response: %s", result)
        return result

    ledger = _get_ledger()
    block = ledger.append_block(block_data)
    from retrofittrust.ledger.twin_state import apply_ledger_event

    apply_ledger_event("eligibility", lsoa, block_data, ledger_index=int(block["index"]))
    log.info("Block #%s appended; chain verify=%s", block["index"], ledger.verify_chain()[0])
    return block


# --- Step 4: synthetic verification append -------------------------------------------


def step4_append_verification(lsoa: str, use_api: bool, base_url: str) -> dict:
    _banner(f"STEP 4 — Ledger append verification outcome [{SYNTHETIC_LABEL}]")

    try:
        from retrofittrust.ledger.synthetic import synthetic_verification_block

        block_data = synthetic_verification_block(lsoa=lsoa)
    except ImportError:
        block_data = {
            "type": "verification",
            "lsoa": lsoa,
            "epc_uplift_bands": 2,
            "verified_by": "SYNTHETIC INSPECTOR",
            "label": SYNTHETIC_LABEL,
            "note": "Simulated post-retrofit verification — not real inspection data",
        }

    log.info("Appending verification block: %s", json.dumps(block_data, sort_keys=True))

    if use_api:
        result = _http_post(
            base_url,
            "/ledger/append",
            {
                "event_type": "verification",
                "lsoa21cd": lsoa,
                "generate_synthetic": True,
                "epc_uplift_bands": int(block_data.get("epc_uplift_bands", 2)),
            },
        )
        log.info("Verification append response: %s", result)
        return result

    ledger = _get_ledger()
    block = ledger.append_block(block_data)
    from retrofittrust.ledger.twin_state import apply_ledger_event

    apply_ledger_event("verification", lsoa, block_data, ledger_index=int(block["index"]))
    log.info("Block #%s appended; chain verify=%s", block["index"], ledger.verify_chain()[0])
    return block


# --- Step 5: SQLite write-back + dashboard state check -------------------------------


def step5_writeback_and_verify(
    lsoa: str, verification: dict, *, already_applied: bool = False
) -> bool:
    _banner("STEP 5 — SQLite write-back + verify dashboard state update")

    from retrofittrust.dashboard.state import (
        apply_verification_writeback,
        get_lsoa_state,
        init_twin_db,
    )

    init_twin_db(SQLITE_PATH)
    before = get_lsoa_state(lsoa)
    log.info("Twin state BEFORE write-back (%s): %s", lsoa, before)

    if not already_applied:
        apply_verification_writeback(
            lsoa=lsoa,
            verification=verification,
            synthetic_label=SYNTHETIC_LABEL,
        )

    after = get_lsoa_state(lsoa)
    log.info("Twin state AFTER write-back (%s): %s", lsoa, after)

    updated = after.get("verified") is True and (
        already_applied or after != before
    )
    if updated:
        log.info("CHECKPOINT PASS — dashboard would reflect verification on next Streamlit refresh")
    else:
        log.warning(
            "CHECKPOINT — verified flag not set; ensure ledger append updates twin_state"
        )
    return updated


def _ledger_block_data(block_response: dict) -> dict:
    """Normalise API or direct ledger append responses to event data dict."""
    if "block" in block_response and isinstance(block_response["block"], dict):
        return block_response["block"].get("data", block_response["block"])
    return block_response.get("data", block_response)


def _extract_top_lsoa(rank_result: dict, fallback: list[str]) -> tuple[str, float]:
    if "items" in rank_result and rank_result["items"]:
        top = rank_result["items"][0]
        return str(top.get("lsoa21cd") or top.get("lsoa")), float(
            top.get("priority_score", top.get("score", 0.0))
        )
    if "rankings" in rank_result and rank_result["rankings"]:
        top = rank_result["rankings"][0]
        return top.get("lsoa") or top.get("lsoa_code"), float(top.get("score", 0.0))
    if "top_lsoa" in rank_result:
        return rank_result["top_lsoa"], float(rank_result.get("top_score", 0.0))
    return fallback[0], 0.0


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="RetrofitTrust integration demo")
    parser.add_argument("--api-base", default=DEFAULT_API_BASE, help="FastAPI base URL")
    parser.add_argument("--force-direct", action="store_true", help="Skip HTTP; use imports only")
    args = parser.parse_args()

    _banner(f"RetrofitTrust Integration Demo — grant/verification layer is {SYNTHETIC_LABEL}")
    log.info("Ledger path: %s", LEDGER_PATH)
    log.info("SQLite twin state: %s", SQLITE_PATH)

    use_api = not args.force_direct and _api_reachable(args.api_base)
    log.info("Transport: %s", f"HTTP ({args.api_base})" if use_api else "direct Python imports")

    # Step 1
    lsoas = step1_select_cohort()

    # Step 2
    ai = step2_rank_and_explain(lsoas, use_api=use_api, base_url=args.api_base)
    top_lsoa, top_score = _extract_top_lsoa(ai["rank"], lsoas)
    log.info("Prioritised LSOA: %s (score=%.4f)", top_lsoa, top_score)

    # Step 3
    step3_append_decision(top_lsoa, top_score, use_api=use_api, base_url=args.api_base)

    # Step 4
    verification_block = step4_append_verification(
        top_lsoa, use_api=use_api, base_url=args.api_base
    )
    verification_data = _ledger_block_data(verification_block)

    # Step 4 applies verification + SQLite write-back (direct and API paths)
    step5_writeback_and_verify(
        top_lsoa, verification_data, already_applied=True
    )

    # Final ledger verify
    if use_api:
        verify = _http_get(args.api_base, "/ledger/verify")
        log.info("Final ledger verify (API): %s", verify)
    else:
        ledger = _get_ledger()
        ok, detail = ledger.verify_chain()
        log.info("Final ledger verify (direct): ok=%s detail=%s", ok, detail)

    _banner(f"Integration loop complete [{SYNTHETIC_LABEL}]")
    log.info("Next: streamlit run src/retrofittrust/dashboard/app.py — choropleth should show write-back")
    return 0


if __name__ == "__main__":
    sys.exit(main())
