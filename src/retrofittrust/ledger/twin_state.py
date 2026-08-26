"""SQLite twin-state persistence (write-back loop).

Tables: cohort selections, verified outcomes, priority score cache.
Updated when ledger verification events are appended (CURSOR_BUILD_SPEC §5–6).
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from retrofittrust.config import SQLITE_PATH

# Assumed write-back rule for the PoC: a verified retrofit halves remaining
# modelled need. This is a demonstration decay, not an empirical impact estimate.
VERIFIED_PRIORITY_DECAY = 0.5


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_twin_db(db_path: Path | None = None) -> Path:
    path = Path(db_path) if db_path is not None else Path(SQLITE_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS lsoa_state (
                lsoa_code TEXT PRIMARY KEY,
                priority_score REAL,
                verified INTEGER DEFAULT 0,
                epc_uplift_bands INTEGER,
                metadata_json TEXT,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cohort_selection (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lsoa_codes_json TEXT NOT NULL,
                metadata_json TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS verified_outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lsoa_code TEXT NOT NULL,
                ledger_index INTEGER,
                epc_uplift_bands INTEGER,
                epc_before REAL,
                epc_after REAL,
                fuel_poverty_before REAL,
                fuel_poverty_after REAL,
                details_json TEXT,
                is_synthetic INTEGER DEFAULT 1,
                recorded_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS priority_score_cache (
                lsoa_code TEXT PRIMARY KEY,
                priority_score REAL NOT NULL,
                source TEXT,
                shap_json TEXT,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()
    return path


def _connect(db_path: Path | None = None) -> sqlite3.Connection:
    path = init_twin_db(db_path)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def save_cohort_selection(
    lsoa_codes: list[str],
    metadata: dict[str, Any] | None = None,
    db_path: Path | None = None,
) -> int:
    with _connect(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO cohort_selection (lsoa_codes_json, metadata_json) VALUES (?, ?)",
            (json.dumps(list(lsoa_codes)), json.dumps(metadata or {})),
        )
        for code in lsoa_codes:
            conn.execute(
                """
                INSERT INTO lsoa_state (lsoa_code, verified, metadata_json, updated_at)
                VALUES (?, 0, ?, ?)
                ON CONFLICT(lsoa_code) DO NOTHING
                """,
                (code, json.dumps({"cohort": True}), _utc_now()),
            )
        conn.commit()
        return int(cur.lastrowid)


def cache_priority_scores(
    rows: list[dict[str, Any]],
    *,
    source: str = "model",
    db_path: Path | None = None,
) -> None:
    """Upsert ranked scores so the dashboard can re-read them without the model."""
    now = _utc_now()
    with _connect(db_path) as conn:
        for row in rows:
            code = str(row["lsoa_code"])
            score = float(row["priority_score"])
            shap_json = json.dumps(row.get("shap") or row.get("shap_top_features") or {})
            conn.execute(
                """
                INSERT INTO priority_score_cache (lsoa_code, priority_score, source, shap_json, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(lsoa_code) DO UPDATE SET
                    priority_score = excluded.priority_score,
                    source = excluded.source,
                    shap_json = excluded.shap_json,
                    updated_at = excluded.updated_at
                """,
                (code, score, source, shap_json, now),
            )
            conn.execute(
                """
                INSERT INTO lsoa_state (lsoa_code, priority_score, metadata_json, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(lsoa_code) DO UPDATE SET
                    priority_score = excluded.priority_score,
                    updated_at = excluded.updated_at
                """,
                (code, score, json.dumps({"source": source}), now),
            )
        conn.commit()


def apply_verification_writeback(
    *,
    lsoa: str,
    verification: dict[str, Any],
    synthetic_label: str = "SYNTHETIC DATA",
    ledger_index: int | None = None,
    db_path: Path | None = None,
) -> None:
    """Persist a verified outcome and decay cached priority for that LSOA."""
    uplift = int(verification.get("epc_uplift_bands", 0) or 0)
    meta = {
        "source": "ledger_verification",
        "label": synthetic_label,
        "verification": verification,
        "ledger_index": ledger_index,
    }
    epc_before = verification.get("epc_before")
    epc_after = verification.get("epc_after")
    now = _utc_now()

    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO verified_outcomes (
                lsoa_code, ledger_index, epc_uplift_bands,
                epc_before, epc_after, fuel_poverty_before, fuel_poverty_after,
                details_json, is_synthetic, recorded_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
            """,
            (
                lsoa,
                ledger_index,
                uplift,
                epc_before,
                epc_after,
                verification.get("fuel_poverty_before"),
                verification.get("fuel_poverty_after"),
                json.dumps(meta),
                now,
            ),
        )
        conn.execute(
            """
            INSERT INTO lsoa_state (lsoa_code, verified, epc_uplift_bands, metadata_json, updated_at)
            VALUES (?, 1, ?, ?, ?)
            ON CONFLICT(lsoa_code) DO UPDATE SET
                verified = 1,
                epc_uplift_bands = excluded.epc_uplift_bands,
                metadata_json = excluded.metadata_json,
                updated_at = excluded.updated_at
            """,
            (lsoa, uplift, json.dumps(meta), now),
        )
        cached = conn.execute(
            "SELECT priority_score FROM priority_score_cache WHERE lsoa_code = ?",
            (lsoa,),
        ).fetchone()
        if cached is not None and cached["priority_score"] is not None:
            decayed = float(cached["priority_score"]) * VERIFIED_PRIORITY_DECAY
            conn.execute(
                """
                UPDATE priority_score_cache
                SET priority_score = ?, source = 'writeback', updated_at = ?
                WHERE lsoa_code = ?
                """,
                (decayed, now, lsoa),
            )
            conn.execute(
                """
                UPDATE lsoa_state
                SET priority_score = ?, updated_at = ?
                WHERE lsoa_code = ?
                """,
                (decayed, now, lsoa),
            )
        conn.commit()


def apply_ledger_event(
    event_type: str,
    lsoa: str,
    details: dict[str, Any],
    ledger_index: int | None = None,
    db_path: Path | None = None,
) -> None:
    """Update twin state from any ledger event in the eligibility → verify sequence."""
    if event_type == "eligibility":
        save_cohort_selection([lsoa], metadata={"event": "eligibility", **details}, db_path=db_path)
        score = details.get("priority_score")
        if score is not None:
            cache_priority_scores(
                [{"lsoa_code": lsoa, "priority_score": score}],
                source="ledger_eligibility",
                db_path=db_path,
            )
        return
    if event_type == "works_claimed":
        with _connect(db_path) as conn:
            conn.execute(
                """
                INSERT INTO lsoa_state (lsoa_code, metadata_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(lsoa_code) DO UPDATE SET
                    metadata_json = excluded.metadata_json,
                    updated_at = excluded.updated_at
                """,
                (
                    lsoa,
                    json.dumps({"event": "works_claimed", "label": "SYNTHETIC DATA", **details}),
                    _utc_now(),
                ),
            )
            conn.commit()
        return
    if event_type == "verification":
        apply_verification_writeback(
            lsoa=lsoa,
            verification=details,
            ledger_index=ledger_index,
            db_path=db_path,
        )


def get_lsoa_state(lsoa_code: str, db_path: Path | None = None) -> dict[str, Any]:
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM lsoa_state WHERE lsoa_code = ?",
            (lsoa_code,),
        ).fetchone()
    if row is None:
        return {"lsoa_code": lsoa_code, "verified": False}
    return _row_to_state(row)


def fetch_all_lsoa_state(db_path: Path | None = None) -> dict[str, dict[str, Any]]:
    """Full twin snapshot for dashboard merge (not cached — re-read on interaction)."""
    with _connect(db_path) as conn:
        rows = conn.execute("SELECT * FROM lsoa_state").fetchall()
        cache_rows = conn.execute("SELECT * FROM priority_score_cache").fetchall()
        outcomes = conn.execute(
            "SELECT * FROM verified_outcomes ORDER BY id DESC"
        ).fetchall()

    by_code: dict[str, dict[str, Any]] = {}
    for row in rows:
        by_code[row["lsoa_code"]] = _row_to_state(row)
    for row in cache_rows:
        entry = by_code.setdefault(row["lsoa_code"], {"lsoa_code": row["lsoa_code"], "verified": False})
        if entry.get("priority_score") is None:
            entry["priority_score"] = row["priority_score"]
        entry["cache_source"] = row["source"]
    latest_outcome: dict[str, dict[str, Any]] = {}
    for row in outcomes:
        code = row["lsoa_code"]
        if code not in latest_outcome:
            latest_outcome[code] = dict(row)
    for code, outcome in latest_outcome.items():
        entry = by_code.setdefault(code, {"lsoa_code": code, "verified": True})
        entry["latest_outcome"] = outcome
    return by_code


def latest_cohort(db_path: Path | None = None) -> list[str]:
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT lsoa_codes_json FROM cohort_selection ORDER BY id DESC LIMIT 1"
        ).fetchone()
    if row is None:
        return []
    return list(json.loads(row["lsoa_codes_json"] or "[]"))


def db_mtime_token(db_path: Path | None = None) -> float:
    """Fingerprint so Streamlit can bust caches when write-back occurs."""
    path = Path(db_path) if db_path is not None else Path(SQLITE_PATH)
    if not path.exists():
        return 0.0
    return path.stat().st_mtime


def _row_to_state(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "lsoa_code": row["lsoa_code"],
        "priority_score": row["priority_score"],
        "verified": bool(row["verified"]),
        "epc_uplift_bands": row["epc_uplift_bands"],
        "metadata": json.loads(row["metadata_json"] or "{}"),
        "updated_at": row["updated_at"],
    }
