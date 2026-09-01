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
    REPORTS_FIGURES,
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
        log.warning("dashboard.cohort not available — using api.features fallback")

    from retrofittrust.api.features import load_lsoa_frame

    frame, source = load_lsoa_frame(allow_synthetic_fallback=True)
    n = min(DEMO_COHORT_LSOA_COUNT, len(frame))
    if "priority_score" in frame.columns:
        ranked = frame.sort_values("priority_score", ascending=False)
        sample = ranked["lsoa21cd"].head(n).astype(str).tolist()
    else:
        sample = (
            frame["lsoa21cd"]
            .drop_duplicates()
            .sample(n=n, random_state=SEED)
            .astype(str)
            .tolist()
        )
    log.info("Fallback cohort from %s (seed=%s): %s", source, SEED, sample)
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
    except Exception as exc:  # noqa: BLE001 — demo must fall back on corrupt/missing artefacts
        log.warning("rank_lsoas failed (%s) — using composite LSOA frame fallback", exc)
        rank_result = None
    if rank_result is None:
        log.warning("Using composite LSOA frame fallback for ranking")
    if rank_result is None:
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
    except Exception:  # noqa: BLE001 — fall back to composite SHAP when model artefact unavailable
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


# --- Step 3b: synthetic works-claimed append -----------------------------------------


def step3b_append_works_claimed(lsoa: str, use_api: bool, base_url: str) -> dict:
    _banner(f"STEP 3b — Ledger append works claimed [{SYNTHETIC_LABEL}]")

    try:
        from retrofittrust.ledger.synthetic import synthetic_works_claimed_block

        block_data = synthetic_works_claimed_block(lsoa=lsoa)
    except ImportError:
        block_data = {
            "type": "works_claimed",
            "lsoa": lsoa,
            "lsoa_code": lsoa,
            "grant_reference": f"SYNTH-GRANT-{lsoa}",
            "label": SYNTHETIC_LABEL,
            "note": "Simulated works-claimed record — not a real installer invoice",
        }

    log.info("Appending works_claimed block: %s", json.dumps(block_data, sort_keys=True))

    if use_api:
        result = _http_post(
            base_url,
            "/ledger/append",
            {
                "event_type": "works_claimed",
                "lsoa21cd": lsoa,
                "generate_synthetic": True,
            },
        )
        log.info("Works-claimed append response: %s", result)
        return result

    ledger = _get_ledger()
    block = ledger.append_block(block_data)
    from retrofittrust.ledger.twin_state import apply_ledger_event

    apply_ledger_event("works_claimed", lsoa, block_data, ledger_index=int(block["index"]))
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


def _verified_count() -> int:
    from retrofittrust.ledger.twin_state import fetch_all_lsoa_state

    state = fetch_all_lsoa_state()
    return sum(1 for entry in state.values() if entry.get("verified"))


def _write_checkpoint6_evidence(
    *,
    cohort: list[str],
    ranked_n: int,
    top_lsoa: str,
    top_score: float,
    ledger_height: int,
    ledger_ok: bool,
    verified_n: int,
    writeback_ok: bool,
    transport: str,
) -> Path:
    """Dissertation evidence for the five-step twin → AI → ledger → write-back loop."""
    REPORTS_FIGURES.mkdir(parents=True, exist_ok=True)
    md_path = REPORTS_FIGURES / "06_integration_numbers.md"
    png_path = REPORTS_FIGURES / "06_integration_loop.png"

    steps = [
        ("1 Twin cohort", f"{len(cohort)} LSOAs"),
        ("2 AI rank", f"{ranked_n} ranked"),
        ("3 Ledger append", f"height {ledger_height}"),
        ("4 Verify chain", "PASS" if ledger_ok else "FAIL"),
        ("5 SQLite write-back", f"{verified_n} verified"),
    ]
    lines = [
        "# Checkpoint 6 — integration loop (twin → AI → ledger → SQLite)",
        "",
        "Five-step loop from CURSOR_BUILD_SPEC §6. Grant/works/verification payloads",
        f"are **{SYNTHETIC_LABEL}**. Ledger is a hashlib SHA-256 hash-chain, not a live blockchain.",
        "",
        f"- Transport: `{transport}`",
        f"- Cohort size: **{len(cohort)}** LSOAs",
        f"- Ranked LSOAs: **{ranked_n}**",
        f"- Top LSOA: `{top_lsoa}` (priority score {top_score:.4f})",
        f"- Ledger height: **{ledger_height}** blocks",
        f"- Chain `verify_chain()`: **{'PASS' if ledger_ok else 'FAIL'}**",
        f"- Verified LSOAs in SQLite: **{verified_n}**",
        f"- Write-back checkpoint: **{'PASS' if writeback_ok else 'FAIL'}**",
        f"- Seed: `{SEED}`",
        "",
        "## Five steps",
        "",
        "| Step | Result |",
        "| --- | --- |",
    ]
    for label, result in steps:
        lines.append(f"| {label} | {result} |")
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- Ecological fallacy: LSOA IMD is not household deprivation.",
            "- SHAP TreeExplainer can misattribute among correlated EPC/IMD features.",
            "- EPC modelled-vs-metered gap (~16% gas / ~31% electric).",
            f"- `{png_path.name}` is a summary badge, not a production monitoring dashboard.",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Wrote %s", md_path)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9.2, 4.6))
        labels = [s[0] for s in steps]
        ok_flags = [True, ranked_n > 0, ledger_height >= 4, ledger_ok, writeback_ok]
        colours = ["#009E73" if flag else "#D55E00" for flag in ok_flags]
        bars = ax.barh(labels[::-1], [1] * 5, color=colours[::-1], height=0.55)
        ax.set_xlim(0, 1.35)
        ax.set_xticks([])
        ax.set_title("Checkpoint 6 — five-step integration loop")
        captions = [s[1] for s in steps]
        for bar, caption, flag in zip(bars, captions[::-1], ok_flags[::-1]):
            ax.text(
                0.04,
                bar.get_y() + bar.get_height() / 2,
                f"{'PASS' if flag else 'FAIL'}  ·  {caption}",
                va="center",
                color="white",
                fontsize=9,
                fontweight="bold",
            )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        fig.text(
            0.01,
            -0.02,
            f"Cohort {len(cohort)} · ranked {ranked_n} · ledger height {ledger_height} · "
            f"verified {verified_n} · {SYNTHETIC_LABEL} · seed={SEED}",
            fontsize=8,
            color="#4D4D4D",
        )
        fig.tight_layout()
        fig.savefig(png_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        log.info("Wrote %s", png_path)
    except Exception as exc:  # noqa: BLE001 — numbers.md is the required artefact
        log.warning("Could not write 06_integration_loop.png (%s)", exc)

    return md_path


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
    rank_payload = ai["rank"] or {}
    ranked_n = len(rank_payload.get("items") or rank_payload.get("rankings") or lsoas)
    log.info("Prioritised LSOA: %s (score=%.4f)", top_lsoa, top_score)

    # Step 3 — eligibility prioritisation
    step3_append_decision(top_lsoa, top_score, use_api=use_api, base_url=args.api_base)

    # Step 3b — synthetic works claimed (eligibility → works → verification)
    step3b_append_works_claimed(top_lsoa, use_api=use_api, base_url=args.api_base)

    # Step 4 — verification outcome
    verification_block = step4_append_verification(
        top_lsoa, use_api=use_api, base_url=args.api_base
    )
    verification_data = _ledger_block_data(verification_block)

    # Step 4 applies verification + SQLite write-back (direct and API paths)
    writeback_ok = step5_writeback_and_verify(
        top_lsoa, verification_data, already_applied=True
    )

    # Final ledger verify
    ledger_height = 0
    ledger_ok = False
    if use_api:
        verify = _http_get(args.api_base, "/ledger/verify")
        log.info("Final ledger verify (API): %s", verify)
        ledger_ok = bool(verify.get("valid"))
        ledger_height = int(verify.get("length") or 0)
    else:
        ledger = _get_ledger()
        ok, detail = ledger.verify_chain()
        log.info("Final ledger verify (direct): ok=%s detail=%s", ok, detail)
        ledger_ok = bool(ok)
        ledger_height = len(ledger.chain)

    verified_n = _verified_count()
    _write_checkpoint6_evidence(
        cohort=lsoas,
        ranked_n=ranked_n,
        top_lsoa=str(top_lsoa),
        top_score=float(top_score),
        ledger_height=ledger_height,
        ledger_ok=ledger_ok,
        verified_n=verified_n,
        writeback_ok=writeback_ok,
        transport=f"HTTP ({args.api_base})" if use_api else "direct Python imports",
    )

    _banner(f"Integration loop complete [{SYNTHETIC_LABEL}]")
    log.info("Next: streamlit run src/retrofittrust/dashboard/app.py — choropleth should show write-back")
    return 0 if writeback_ok and ledger_ok else 1


if __name__ == "__main__":
    sys.exit(main())
