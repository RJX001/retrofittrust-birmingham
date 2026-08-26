"""Hash-chain ledger (Program 4) — SHA-256 simulation, not real blockchain."""

from retrofittrust.ledger.chain import (
    Ledger,
    append_block,
    compute_block_hash,
    load_or_create,
    verify_chain,
)
from retrofittrust.ledger.synthetic import (
    SYNTHETIC_LABEL,
    generate_event,
    synthetic_eligibility_block,
    synthetic_verification_block,
    synthetic_works_claimed_block,
)
from retrofittrust.ledger.tamper import demonstrate_tampering

__all__ = [
    "Ledger",
    "SYNTHETIC_LABEL",
    "append_block",
    "compute_block_hash",
    "demonstrate_tampering",
    "generate_event",
    "load_or_create",
    "synthetic_eligibility_block",
    "synthetic_verification_block",
    "synthetic_works_claimed_block",
    "verify_chain",
]
