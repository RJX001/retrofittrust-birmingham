"""End-to-end integration loop tests (twin → AI → ledger → SQLite write-back)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fastapi.testclient import TestClient

from retrofittrust.api.features import _synthetic_lsoa_frame, add_composite_score
from retrofittrust.api.main import app
from retrofittrust.ledger.chain import Ledger
from retrofittrust.ledger.synthetic import generate_event
from retrofittrust.ledger.tamper import demonstrate_tampering
from retrofittrust.ledger.twin_state import (
    apply_ledger_event,
    get_lsoa_state,
    init_twin_db,
)


def _synthetic_frame():
    return add_composite_score(_synthetic_lsoa_frame(8))


class IntegrationLoopTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.tmp_path = Path(self.tmp.name)
        self.ledger_path = self.tmp_path / "ledger.json"
        self.sqlite_path = self.tmp_path / "twin_state.db"
        self.frame = _synthetic_frame()
        init_twin_db(self.sqlite_path)

        def fake_load_lsoa_frame(*_args, **_kwargs):
            return self.frame.copy(), "test_synthetic"

        patchers = [
            patch("retrofittrust.api.main.LEDGER_PATH", self.ledger_path),
            patch("retrofittrust.api.main.SQLITE_PATH", self.sqlite_path),
            patch("retrofittrust.ledger.chain.LEDGER_PATH", self.ledger_path),
            patch("retrofittrust.ledger.twin_state.SQLITE_PATH", self.sqlite_path),
            patch("retrofittrust.api.main.load_lsoa_frame", side_effect=fake_load_lsoa_frame),
            patch("retrofittrust.api.features.load_lsoa_frame", side_effect=fake_load_lsoa_frame),
            patch("retrofittrust.api.main._rank_via_program1", return_value=None),
        ]
        self._patchers = patchers
        for p in patchers:
            p.start()

        self.client = TestClient(app)

    def tearDown(self) -> None:
        self.client.close()
        for p in self._patchers:
            p.stop()
        self.tmp.cleanup()

    def _sample_lsoa(self) -> str:
        code = str(self.frame["lsoa21cd"].iloc[0])
        frame_resp = self.client.post("/rank", json={"lsoa_codes": [code], "top_k": 1})
        self.assertEqual(frame_resp.status_code, 200, frame_resp.text)
        return frame_resp.json()["items"][0]["lsoa21cd"]

    def test_full_api_loop_eligibility_works_verification(self) -> None:
        lsoa = self._sample_lsoa()

        explain = self.client.post("/explain", json={"lsoa21cd": lsoa, "top_n": 5})
        self.assertEqual(explain.status_code, 200)
        self.assertIn("features", explain.json())

        for event_type, extra in (
            ("eligibility", {"priority_score": 0.88, "generate_synthetic": True}),
            ("works_claimed", {"generate_synthetic": True}),
            ("verification", {"generate_synthetic": True, "epc_uplift_bands": 2}),
        ):
            resp = self.client.post(
                "/ledger/append",
                json={"event_type": event_type, "lsoa21cd": lsoa, **extra},
            )
            self.assertEqual(resp.status_code, 200, resp.text)
            body = resp.json()
            self.assertTrue(body["chain_valid"])
            self.assertTrue(body["twin_state_updated"])
            self.assertEqual(body["block"]["data"]["lsoa_code"], lsoa)

        verify = self.client.get("/ledger/verify")
        self.assertEqual(verify.status_code, 200)
        self.assertTrue(verify.json()["valid"])
        self.assertGreaterEqual(verify.json()["length"], 4)

        state = get_lsoa_state(lsoa, db_path=self.sqlite_path)
        self.assertTrue(state["verified"])
        self.assertIsNotNone(state.get("epc_uplift_bands"))

    def test_direct_import_loop_and_tamper(self) -> None:
        lsoa = str(self.frame["lsoa21cd"].iloc[0])
        ledger = Ledger(self.ledger_path)
        ledger.initialise_genesis()

        for event_type in ("eligibility", "works_claimed", "verification"):
            payload = generate_event(event_type, lsoa, priority_score=0.75, seed=42)
            block = ledger.append_block(payload, persist=True)
            apply_ledger_event(
                event_type,
                lsoa,
                payload,
                ledger_index=int(block["index"]),
                db_path=self.sqlite_path,
            )

        ok, _ = ledger.verify_chain()
        self.assertTrue(ok)
        state = get_lsoa_state(lsoa, db_path=self.sqlite_path)
        self.assertTrue(state["verified"])

        tamper = demonstrate_tampering(ledger)
        self.assertFalse(tamper["after_tamper_valid"])


if __name__ == "__main__":
    unittest.main()
