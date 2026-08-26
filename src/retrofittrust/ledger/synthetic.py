"""Synthetic grant / works / verification payloads.

SYNTHETIC DATA — these records are generated programmatically for the PoC.
They are not real Birmingham grant awards, installer invoices, or inspections.
Amounts are assumed ballpark figures for demonstration only, not official
ECO / Home Upgrade Grant rates.
"""

from __future__ import annotations

import random
from typing import Any, Literal

from retrofittrust.config import SEED

SYNTHETIC_LABEL = "SYNTHETIC DATA"

EventType = Literal["eligibility", "works_claimed", "verification"]

# Assumed demo ranges (GBP) — not sourced from a live scheme tariff.
_AWARD_GBP_RANGE = (3_000, 18_000)
_WORKS_GBP_RANGE = (2_500, 16_000)


def _rng(seed: int | None = None) -> random.Random:
    return random.Random(SEED if seed is None else seed)


def synthetic_eligibility_block(
    *,
    lsoa: str,
    priority_score: float,
    seed: int | None = None,
) -> dict[str, Any]:
    rng = _rng(seed)
    award = rng.randint(*_AWARD_GBP_RANGE)
    return {
        "type": "eligibility",
        "lsoa": lsoa,
        "priority_score": float(priority_score),
        "grant_reference": f"SYNTH-GRANT-{lsoa}",
        "estimated_award_gbp": award,
        "scheme": "SYNTHETIC ECO+/HUG-style grant (demo only)",
        "installer": "SYNTHETIC INSTALLER LTD",
        "label": SYNTHETIC_LABEL,
        "note": "Simulated retrofit prioritisation — not real grant data",
    }


def synthetic_works_claimed_block(
    *,
    lsoa: str,
    grant_reference: str | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    rng = _rng(seed)
    cost = rng.randint(*_WORKS_GBP_RANGE)
    measures = rng.sample(
        ["loft insulation", "cavity wall insulation", "heat pump (air source)", "LED lighting"],
        k=2,
    )
    return {
        "type": "works_claimed",
        "lsoa": lsoa,
        "grant_reference": grant_reference or f"SYNTH-GRANT-{lsoa}",
        "claimed_cost_gbp": cost,
        "measures": measures,
        "installer": "SYNTHETIC INSTALLER LTD",
        "label": SYNTHETIC_LABEL,
        "note": "Simulated works-claimed record — not a real installer invoice",
    }


def synthetic_verification_block(
    *,
    lsoa: str,
    epc_uplift_bands: int = 2,
    seed: int | None = None,
) -> dict[str, Any]:
    rng = _rng(seed)
    return {
        "type": "verification",
        "lsoa": lsoa,
        "epc_uplift_bands": int(epc_uplift_bands),
        "verified_by": "SYNTHETIC INSPECTOR",
        "outcome": "pass",
        "inspection_id": f"SYNTH-INSP-{lsoa}-{rng.randint(1000, 9999)}",
        "label": SYNTHETIC_LABEL,
        "note": "Simulated post-retrofit verification — not real inspection data",
    }


def generate_event(
    event_type: EventType,
    lsoa: str,
    *,
    priority_score: float = 0.0,
    epc_uplift_bands: int = 2,
    extra: dict[str, Any] | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Build a labelled synthetic ledger payload for the eligibility → works → verify sequence."""
    if event_type == "eligibility":
        payload = synthetic_eligibility_block(lsoa=lsoa, priority_score=priority_score, seed=seed)
    elif event_type == "works_claimed":
        payload = synthetic_works_claimed_block(lsoa=lsoa, seed=seed)
    elif event_type == "verification":
        payload = synthetic_verification_block(
            lsoa=lsoa, epc_uplift_bands=epc_uplift_bands, seed=seed
        )
    else:
        raise ValueError(f"Unknown event_type: {event_type}")

    if extra:
        payload = {**payload, **extra, "label": SYNTHETIC_LABEL}
    return payload
