"""Unit tests for the SHA-256 hash-chain ledger (Program 4)."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from retrofittrust.config import SEED
from retrofittrust.ledger.chain import (
    GENESIS_PREVIOUS,
    Ledger,
    compute_block_hash,
)
from retrofittrust.ledger.synthetic import (
    SYNTHETIC_LABEL,
    generate_event,
    synthetic_eligibility_block,
)
from retrofittrust.ledger.tamper import demonstrate_tampering


class LedgerChainTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.ledger_path = Path(self.tmp.name) / "ledger.json"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_genesis_block_schema(self) -> None:
        ledger = Ledger(self.ledger_path)
        genesis = ledger.initialise_genesis()
        self.assertEqual(genesis["index"], 0)
        self.assertEqual(genesis["previous_hash"], GENESIS_PREVIOUS)
        self.assertIn("timestamp", genesis)
        self.assertIn("data", genesis)
        self.assertIn("hash", genesis)
        self.assertEqual(len(genesis["hash"]), 64)

    def test_canonical_hash_stable(self) -> None:
        block = {
            "index": 1,
            "timestamp": "2026-01-01T00:00:00+00:00",
            "data": {"type": "eligibility", "lsoa_code": "E01000001"},
            "previous_hash": "0" * 64,
        }
        h1 = compute_block_hash(block)
        h2 = compute_block_hash(dict(block))
        self.assertEqual(h1, h2)

    def test_append_and_verify_chain(self) -> None:
        ledger = Ledger(self.ledger_path)
        ledger.initialise_genesis()
        payload = synthetic_eligibility_block(lsoa="E01000001", priority_score=0.82)
        block = ledger.append_block(payload, persist=True)
        self.assertEqual(block["index"], 1)
        self.assertEqual(block["previous_hash"], ledger.chain[0]["hash"])
        ok, msg = ledger.verify_chain()
        self.assertTrue(ok, msg)

    def test_event_types_include_lsoa_code(self) -> None:
        for event_type in ("eligibility", "works_claimed", "verification"):
            payload = generate_event(event_type, "E01000002", priority_score=0.5, seed=SEED)
            self.assertEqual(payload["label"], SYNTHETIC_LABEL)
            self.assertEqual(payload["lsoa_code"], "E01000002")
            self.assertEqual(payload["lsoa"], "E01000002")

    def test_tamper_detected_in_memory(self) -> None:
        ledger = Ledger(self.ledger_path)
        ledger.initialise_genesis()
        ledger.append_block(synthetic_eligibility_block(lsoa="E01000003", priority_score=0.9))
        ledger.save()
        result = demonstrate_tampering(ledger)
        self.assertTrue(result["before_valid"])
        self.assertFalse(result["after_tamper_valid"])
        self.assertTrue(result["after_errors"])

    def test_tamper_without_rehash_fails_verify(self) -> None:
        ledger = Ledger(self.ledger_path)
        ledger.initialise_genesis()
        ledger.append_block({"type": "eligibility", "lsoa_code": "E01000004"})
        tampered = copy.deepcopy(ledger.chain)
        tampered[1]["data"]["priority_score"] = 999.0
        for i, block in enumerate(tampered):
            expected = compute_block_hash(block)
            if block.get("hash") != expected:
                self.assertEqual(i, 1)
                return
        self.fail("Expected hash mismatch on tampered block")


class LedgerPersistenceTests(unittest.TestCase):
    def test_reload_from_disk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ledger.json"
            ledger = Ledger(path)
            ledger.initialise_genesis()
            ledger.append_block({"type": "verification", "lsoa_code": "E01000005"}, persist=True)
            reloaded = Ledger(path)
            self.assertEqual(len(reloaded.chain), 2)
            ok, _ = reloaded.verify_chain()
            self.assertTrue(ok)

    def test_saved_json_has_chain_wrapper(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ledger.json"
            ledger = Ledger(path)
            ledger.initialise_genesis()
            ledger.save()
            raw = json.loads(path.read_text(encoding="utf-8"))
            self.assertIn("chain", raw)
            self.assertEqual(raw["chain"][0]["previous_hash"], GENESIS_PREVIOUS)


if __name__ == "__main__":
    unittest.main()
