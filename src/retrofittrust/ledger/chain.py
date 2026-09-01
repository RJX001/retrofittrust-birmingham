"""SHA-256 hash-chain ledger simulation (not a real blockchain).

PoC scope: Python hashlib only. Canonical JSON before hashing is required so
verify_chain() is stable across processes (CURSOR_BUILD_SPEC §6).
"""

from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from retrofittrust.config import LEDGER_PATH

# Genesis previous_hash is a 64-character zero digest (SHA-256 hex length).
GENESIS_PREVIOUS = "0" * 64

_FILE_LOCK = threading.Lock()


def _canonical_json(payload: dict[str, Any]) -> str:
    """Serialise canonically before hashing — sort_keys=True is required."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def compute_block_hash(block: dict[str, Any]) -> str:
    """SHA-256 of the block minus the ``hash`` field itself."""
    body = {k: v for k, v in block.items() if k != "hash"}
    return hashlib.sha256(_canonical_json(body).encode("utf-8")).hexdigest()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def block_lsoa_code(data: dict[str, Any]) -> str | None:
    """Resolve LSOA from ledger event data (``lsoa_code`` or legacy ``lsoa``)."""
    if not isinstance(data, dict):
        return None
    code = data.get("lsoa_code") or data.get("lsoa")
    return str(code) if code is not None else None


class Ledger:
    """Append-only hash-chain persisted as JSON at ``config.LEDGER_PATH``."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path is not None else Path(LEDGER_PATH)
        self.chain: list[dict[str, Any]] = []
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            self.chain = raw
        else:
            self.chain = raw.get("chain", raw.get("blocks", []))

    def reload(self) -> None:
        """Re-read from disk (Streamlit and FastAPI are separate processes)."""
        if self.path.exists():
            self._load()
        else:
            self.chain = []

    def is_empty(self) -> bool:
        return len(self.chain) == 0

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "chain": self.chain,
            "version": 1,
            "note": "RetrofitTrust Birmingham hash-chain simulation — not a cryptocurrency ledger",
        }
        with _FILE_LOCK:
            self.path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    def initialise_genesis(self) -> dict[str, Any]:
        genesis: dict[str, Any] = {
            "index": 0,
            "timestamp": utc_now_iso(),
            "data": {
                "type": "genesis",
                "note": "RetrofitTrust Birmingham PoC ledger (hashlib SHA-256 simulation)",
            },
            "previous_hash": GENESIS_PREVIOUS,
        }
        genesis["hash"] = compute_block_hash(genesis)
        self.chain = [genesis]
        return genesis

    def append_block(self, data: dict[str, Any], *, persist: bool = True) -> dict[str, Any]:
        """Append a block and optionally persist. Reloads from disk first."""
        self.reload()
        if self.is_empty():
            self.initialise_genesis()

        previous = self.chain[-1]
        block: dict[str, Any] = {
            "index": int(previous["index"]) + 1,
            "timestamp": utc_now_iso(),
            "data": data,
            "previous_hash": previous["hash"],
        }
        block["hash"] = compute_block_hash(block)
        self.chain.append(block)
        if persist:
            self.save()
        return block

    def verify_chain(self) -> tuple[bool, str]:
        """Walk the chain: stored hash matches recomputed hash; links are intact."""
        ok, errors = self.verify_chain_detailed()
        if ok:
            return True, f"ok ({len(self.chain)} blocks)"
        return False, "; ".join(errors)

    def verify_chain_detailed(self) -> tuple[bool, list[str]]:
        self.reload()
        errors: list[str] = []
        if self.is_empty():
            return False, ["empty chain"]

        for i, block in enumerate(self.chain):
            expected = compute_block_hash(block)
            if block.get("hash") != expected:
                errors.append(f"block {i}: hash mismatch (tampered or corrupt)")

            if i == 0:
                if block.get("previous_hash") != GENESIS_PREVIOUS:
                    errors.append("genesis previous_hash invalid")
                continue

            prior = self.chain[i - 1]
            if block.get("previous_hash") != prior.get("hash"):
                errors.append(f"block {i}: previous_hash does not link to block {i - 1}")
            if int(block.get("index", -1)) != int(prior.get("index", -1)) + 1:
                errors.append(f"block {i}: index is not prior index + 1")

        return (len(errors) == 0), errors

    def recent_blocks(self, n: int = 8) -> list[dict[str, Any]]:
        self.reload()
        return list(self.chain[-n:])


def load_or_create(path: Path | None = None) -> Ledger:
    ledger = Ledger(path)
    if ledger.is_empty():
        ledger.initialise_genesis()
        ledger.save()
    return ledger


def append_block(data: dict[str, Any], path: Path | None = None) -> dict[str, Any]:
    """Module-level append used by the FastAPI service and the dashboard."""
    return load_or_create(path).append_block(data, persist=True)


def verify_chain(path: Path | None = None) -> tuple[bool, str]:
    """Module-level verify used by GET /ledger/verify."""
    return Ledger(path).verify_chain()
