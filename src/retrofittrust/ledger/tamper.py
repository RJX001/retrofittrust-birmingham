"""Deliberate tampering demo for dissertation evidence of tamper-evidence.

Operates on an in-memory copy by default so the live PoC ledger is not destroyed.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from retrofittrust.ledger.chain import Ledger, compute_block_hash


def _verify_blocks(chain: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if not chain:
        return False, ["empty chain"]
    for i, block in enumerate(chain):
        expected = compute_block_hash(block)
        if block.get("hash") != expected:
            errors.append(f"block {i}: hash mismatch (tampered or corrupt)")
        if i == 0:
            continue
        prior = chain[i - 1]
        if block.get("previous_hash") != prior.get("hash"):
            errors.append(f"block {i}: previous_hash does not link to block {i - 1}")
    return (len(errors) == 0), errors


def demonstrate_tampering(ledger: Ledger | None = None) -> dict[str, Any]:
    """Modify a historical record on a copy and show ``verify_chain()`` catching it.

    This is dissertation evidence that the hash-chain is tamper-evident, not an
    attack procedure. The live file on disk is not changed.
    """
    ledger = ledger or Ledger()
    ledger.reload()
    if len(ledger.chain) < 2:
        raise ValueError(
            "Need at least one non-genesis block before running the tampering demo. "
            "Append an eligibility/works/verification event first."
        )

    before_ok, before_msg = ledger.verify_chain()
    tampered = copy.deepcopy(ledger.chain)
    target = tampered[1]
    data = target.setdefault("data", {})
    if isinstance(data, dict):
        data["tampered"] = True
        data["priority_score"] = 999.99
        data["note"] = "DELIBERATELY TAMPERED for dissertation demo"
        data["label"] = "SYNTHETIC DATA"
        data["estimated_award_gbp"] = 999_999

    after_ok, after_errors = _verify_blocks(tampered)
    return {
        "before_valid": before_ok,
        "before_message": before_msg,
        "after_tamper_valid": after_ok,
        "after_errors": after_errors,
        "tampered_block_index": 1,
        "tampered_block_hash_stored": target.get("hash"),
        "tampered_block_hash_recomputed": compute_block_hash(target),
        "persisted": False,
        "note": "Copy only — live ledger.json was not modified",
    }


def tamper_block_at_index(
    ledger_path: Path,
    block_index: int = 1,
    *,
    persist: bool = False,
) -> dict[str, Any]:
    """Optional on-disk tamper for a screenshot of a failed GET /ledger/verify.

    Default ``persist=False`` keeps the live chain intact. Set persist=True only
    when you intentionally want verify_chain() to fail on the saved file.
    """
    path = Path(ledger_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    chain = raw if isinstance(raw, list) else raw.get("chain", raw.get("blocks", []))

    if len(chain) <= block_index:
        raise IndexError(f"Chain has {len(chain)} blocks; need index {block_index}")

    data = chain[block_index].setdefault("data", {})
    if isinstance(data, dict):
        data["tampered"] = True
        data["priority_score"] = 999.99
        data["note"] = "DELIBERATELY TAMPERED for dissertation demo"
        data["label"] = "SYNTHETIC DATA"

    if persist:
        if isinstance(raw, list):
            path.write_text(json.dumps(chain, indent=2), encoding="utf-8")
        else:
            key = "chain" if "chain" in raw else "blocks"
            raw[key] = chain
            path.write_text(json.dumps(raw, indent=2), encoding="utf-8")

    ok, errors = _verify_blocks(chain)
    return {
        "after_tamper_valid": ok,
        "after_errors": errors,
        "persisted": persist,
        "block_index": block_index,
    }
