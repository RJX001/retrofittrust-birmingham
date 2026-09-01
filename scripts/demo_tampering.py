#!/usr/bin/env python3
"""
RetrofitTrust Birmingham — deliberate ledger tampering demo.

Demonstrates hash-chain tamper-evidence via an in-memory copy (default) or
optional on-disk tamper (--persist). Grant/verification records are SYNTHETIC DATA.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from retrofittrust.config import LEDGER_PATH  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("demo_tampering")

SYNTHETIC_LABEL = "SYNTHETIC DATA"


def _seed_chain_if_empty() -> None:
    """Ensure the ledger has at least two blocks for a meaningful tamper demo."""
    from retrofittrust.ledger.chain import Ledger
    from retrofittrust.ledger.synthetic import (
        synthetic_eligibility_block,
        synthetic_verification_block,
        synthetic_works_claimed_block,
    )

    ledger = Ledger(LEDGER_PATH)
    if ledger.is_empty():
        ledger.initialise_genesis()
        ledger.append_block(synthetic_eligibility_block(lsoa="E01000001", priority_score=0.75))
        ledger.append_block(synthetic_works_claimed_block(lsoa="E01000001"))
        ledger.append_block(synthetic_verification_block(lsoa="E01000001"))
        log.info("Seeded demo chain with genesis + 3 %s blocks", SYNTHETIC_LABEL)


def main() -> int:
    from retrofittrust.ledger.chain import Ledger
    from retrofittrust.ledger.tamper import demonstrate_tampering, tamper_block_at_index

    parser = argparse.ArgumentParser(description="Ledger tampering demo")
    parser.add_argument(
        "--persist",
        action="store_true",
        help="Tamper the on-disk ledger (verify_chain will fail until ledger.json is deleted)",
    )
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("Ledger tampering demo [%s]", SYNTHETIC_LABEL)
    log.info("=" * 60)
    log.info("Ledger file: %s", LEDGER_PATH)

    _seed_chain_if_empty()

    if args.persist:
        ledger = Ledger(LEDGER_PATH)
        ok_before, detail_before = ledger.verify_chain()
        log.info("BEFORE tamper — verify_chain(): ok=%s (%s)", ok_before, detail_before)
        if not ok_before:
            log.error("Chain already invalid before tamper — fix ledger first")
            return 1
        result = tamper_block_at_index(LEDGER_PATH, block_index=1, persist=True)
        ledger = Ledger(LEDGER_PATH)
        ok_after, detail_after = ledger.verify_chain()
        log.info("AFTER tamper  — verify_chain(): ok=%s (%s)", ok_after, detail_after)
        if ok_after:
            log.error("FAIL — on-disk tamper was not detected")
            return 1
        log.info("PASS — on-disk hash-chain tamper-evidence demonstrated")
        log.info("Restore: delete %s and re-run run_integration_demo.py", LEDGER_PATH)
        return 0

    # Default: in-memory copy only (matches Streamlit /ledger/tamper-demo behaviour)
    demo = demonstrate_tampering(Ledger(LEDGER_PATH))
    log.info("BEFORE tamper — valid=%s (%s)", demo["before_valid"], demo["before_message"])
    log.info("AFTER tamper  — valid=%s errors=%s", demo["after_tamper_valid"], demo["after_errors"])
    if demo["after_tamper_valid"]:
        log.error("FAIL — in-memory tamper was not detected")
        return 1
    log.info("PASS — tamper-evidence property demonstrated (%s)", demo["note"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
