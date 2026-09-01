#!/usr/bin/env python3
"""
RetrofitTrust Birmingham — deliberate ledger tampering demo.

Demonstrates hash-chain tamper-evidence via an in-memory copy (default) or
optional on-disk tamper (--persist). Grant/verification records are SYNTHETIC DATA.

Also writes checkpoint 5 evidence to reports/figures/:
  05_ledger_verify.png, 05_tamper_demo.png, 05_ledger_numbers.md
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any

SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from retrofittrust.config import LEDGER_PATH, REPORTS_FIGURES  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("demo_tampering")

SYNTHETIC_LABEL = "SYNTHETIC DATA"

_PASS_GREEN = "#548235"
_FAIL_RED = "#C00000"
_NAVY = "#1F4E79"
_TYPE_COLOURS = {
    "genesis": "#7F7F7F",
    "eligibility": "#4472C4",
    "works_claimed": "#ED7D31",
    "verification": "#70AD47",
}


def _seed_chain_if_needed() -> None:
    """Ensure the ledger has at least one non-genesis block for the tamper demo."""
    from retrofittrust.ledger.chain import Ledger
    from retrofittrust.ledger.synthetic import (
        synthetic_eligibility_block,
        synthetic_verification_block,
        synthetic_works_claimed_block,
    )

    ledger = Ledger(LEDGER_PATH)
    if len(ledger.chain) >= 2:
        return

    if ledger.is_empty():
        ledger.initialise_genesis()
        ledger.save()

    ledger.append_block(synthetic_eligibility_block(lsoa="E01000001", priority_score=0.75))
    ledger.append_block(synthetic_works_claimed_block(lsoa="E01000001"))
    ledger.append_block(synthetic_verification_block(lsoa="E01000001"))
    log.info("Seeded demo chain with genesis + 3 %s blocks", SYNTHETIC_LABEL)


def _short_hash(value: Any, n: int = 16) -> str:
    text = str(value or "")
    if len(text) <= n:
        return text
    return f"{text[:n]}..."


def _block_type_counts(chain: list[dict[str, Any]]) -> Counter:
    return Counter((b.get("data") or {}).get("type", "unknown") for b in chain)


def _synthetic_label_count(chain: list[dict[str, Any]]) -> int:
    return sum(
        1
        for b in chain
        if (b.get("data") or {}).get("label") == SYNTHETIC_LABEL
    )


def save_checkpoint_figures(
    ledger: Any,
    demo: dict[str, Any],
    out_dir: Path | None = None,
) -> dict[str, Path]:
    """Write pass/fail visualisations and the checkpoint numbers markdown."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    out_dir = Path(out_dir) if out_dir is not None else REPORTS_FIGURES
    out_dir.mkdir(parents=True, exist_ok=True)

    chain = list(ledger.chain)
    n_blocks = len(chain)
    ok, msg = True, demo.get("before_message", f"ok ({n_blocks} blocks)")
    if not demo.get("before_valid"):
        ok, msg = False, str(demo.get("before_message", "verify failed"))

    type_counts = _block_type_counts(chain)
    n_synthetic = _synthetic_label_count(chain)
    order = ["genesis", "eligibility", "works_claimed", "verification"]
    types_present = [t for t in order if type_counts.get(t, 0)]
    for extra in sorted(set(type_counts) - set(order)):
        types_present.append(extra)

    verify_path = out_dir / "05_ledger_verify.png"
    tamper_path = out_dir / "05_tamper_demo.png"
    numbers_path = out_dir / "05_ledger_numbers.md"

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))
    fig.suptitle(
        "RetrofitTrust Birmingham — SHA-256 hash-chain verification",
        fontsize=12,
        fontweight="bold",
        color=_NAVY,
    )

    ax_status = axes[0]
    ax_status.set_xlim(0, 10)
    ax_status.set_ylim(0, 10)
    ax_status.axis("off")
    ax_status.set_title("verify_chain()")

    badge_colour = _PASS_GREEN if ok else _FAIL_RED
    badge = FancyBboxPatch(
        (1.1, 4.6),
        7.8,
        4.0,
        boxstyle="round,pad=0.25,rounding_size=0.4",
        facecolor=badge_colour,
        edgecolor="none",
    )
    ax_status.add_patch(badge)
    ax_status.text(
        5.0,
        7.4,
        "PASS" if ok else "FAIL",
        ha="center",
        va="center",
        fontsize=28,
        fontweight="bold",
        color="white",
    )
    ax_status.text(
        5.0,
        5.7,
        f"{n_blocks} blocks",
        ha="center",
        va="center",
        fontsize=13,
        color="white",
    )
    ax_status.text(
        5.0,
        3.4,
        str(msg),
        ha="center",
        va="center",
        fontsize=9,
        color="#333333",
    )
    ax_status.text(
        5.0,
        1.9,
        "hashlib SHA-256 simulation  |  not a real blockchain",
        ha="center",
        va="center",
        fontsize=8,
        color="#555555",
    )
    ax_status.text(
        5.0,
        0.7,
        SYNTHETIC_LABEL,
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
        color="#7F6000",
    )

    ax_bar = axes[1]
    heights = [int(type_counts[t]) for t in types_present]
    colours = [_TYPE_COLOURS.get(t, "#5B9BD5") for t in types_present]
    bars = ax_bar.bar(types_present, heights, color=colours, width=0.65)
    ax_bar.set_ylabel("Block count")
    ax_bar.set_title("Blocks by type")
    ax_bar.set_ylim(0, max(heights + [1]) * 1.25)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    for bar, height in zip(bars, heights):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.15,
            str(height),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax_bar.tick_params(axis="x", labelrotation=12)

    fig.tight_layout()
    fig.savefig(verify_path, dpi=140, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8))
    fig.suptitle(
        "Tamper-evidence demo (in-memory copy — live ledger.json unchanged)",
        fontsize=12,
        fontweight="bold",
        color=_NAVY,
    )

    panels = [
        {
            "ax": axes[0],
            "title": "Before tamper (intact chain)",
            "ok": bool(demo["before_valid"]),
            "subtitle": str(demo.get("before_message", "")),
            "detail": "Stored hashes match recomputed SHA-256;\nprevious_hash links walk the chain.",
        },
        {
            "ax": axes[1],
            "title": "After tamper (historical record edited)",
            "ok": bool(demo["after_tamper_valid"]),
            "subtitle": "; ".join(demo.get("after_errors") or ["no errors"]),
            "detail": (
                f"Block {demo.get('tampered_block_index', 1)} data altered without re-hashing.\n"
                f"stored  {_short_hash(demo.get('tampered_block_hash_stored'))}\n"
                f"recomputed  {_short_hash(demo.get('tampered_block_hash_recomputed'))}"
            ),
        },
    ]

    for panel in panels:
        ax = panel["ax"]
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")
        ax.set_title(panel["title"])
        colour = _PASS_GREEN if panel["ok"] else _FAIL_RED
        card = FancyBboxPatch(
            (0.6, 1.1),
            8.8,
            7.6,
            boxstyle="round,pad=0.2,rounding_size=0.35",
            facecolor="#F7F7F7",
            edgecolor=colour,
            linewidth=2.2,
        )
        ax.add_patch(card)
        badge = FancyBboxPatch(
            (2.2, 6.35),
            5.6,
            1.7,
            boxstyle="round,pad=0.15,rounding_size=0.25",
            facecolor=colour,
            edgecolor="none",
        )
        ax.add_patch(badge)
        ax.text(
            5.0,
            7.2,
            "PASS" if panel["ok"] else "FAIL",
            ha="center",
            va="center",
            fontsize=20,
            fontweight="bold",
            color="white",
        )
        ax.text(
            5.0,
            5.35,
            panel["subtitle"],
            ha="center",
            va="center",
            fontsize=8.5,
            color="#333333",
            wrap=True,
        )
        ax.text(
            5.0,
            3.15,
            panel["detail"],
            ha="center",
            va="center",
            fontsize=8,
            color="#444444",
            family="monospace",
        )

    fig.text(
        0.5,
        0.02,
        f"{SYNTHETIC_LABEL}  |  SHA-256 hashlib hash-chain  |  SEED={SEED}",
        ha="center",
        fontsize=8,
        color="#7F6000",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(tamper_path, dpi=140, bbox_inches="tight")
    plt.close(fig)

    type_rows = "\n".join(
        f"| `{name}` | {type_counts.get(name, 0)} |" for name in types_present
    )
    try:
        ledger_display = LEDGER_PATH.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        ledger_display = str(LEDGER_PATH)
    numbers_path.write_text(
        "\n".join(
            [
                "# Checkpoint 5 — Ledger standalone numbers",
                "",
                f"**{SYNTHETIC_LABEL}.** Grant, installer, and verification records on this "
                "chain are generated programmatically. They are not real Birmingham grant "
                "awards, installer invoices, or inspections. Amounts are assumed demo figures, "
                "not official ECO / Home Upgrade Grant rates.",
                "",
                "The ledger is a Python `hashlib` SHA-256 hash-chain simulation, **not** a live "
                "blockchain (no Hyperledger, no Ethereum). Blocks are hashed from canonical JSON "
                "(`json.dumps(..., sort_keys=True)` excluding the `hash` field). "
                "`verify_chain()` recomputes each digest and checks `previous_hash` links.",
                "",
                "| Metric | Value |",
                "| --- | --- |",
                f"| Ledger path | `{ledger_display}` |",
                f"| Block count | {n_blocks} |",
                f"| `verify_chain()` | **{'PASS' if ok else 'FAIL'}** — {msg} |",
                f"| `{SYNTHETIC_LABEL}` labelled blocks | {n_synthetic} / {n_blocks} (genesis is structural) |",
                f"| Tamper demo (in-memory) | intact={'PASS' if demo['before_valid'] else 'FAIL'}; "
                f"after tamper={'FAIL (detected)' if not demo['after_tamper_valid'] else 'PASS (not detected)'} |",
                f"| Tampered block index | {demo.get('tampered_block_index', 1)} |",
                f"| Live `ledger.json` modified by demo | {'yes' if demo.get('persisted') else 'no'} |",
                f"| Random seed | {SEED} |",
                "",
                "## Blocks by type",
                "",
                "| Type | Count |",
                "| --- | --- |",
                type_rows,
                "",
                "## Figures",
                "",
                f"- `{verify_path.name}` — pass/fail badge and block-type counts",
                f"- `{tamper_path.name}` — before/after `verify_chain()` (intact vs tampered copy)",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    log.info("Wrote %s", verify_path)
    log.info("Wrote %s", tamper_path)
    log.info("Wrote %s", numbers_path)
    return {"verify": verify_path, "tamper": tamper_path, "numbers": numbers_path}


def main() -> int:
    from retrofittrust.ledger.chain import Ledger
    from retrofittrust.ledger.tamper import demonstrate_tampering, tamper_block_at_index

    parser = argparse.ArgumentParser(description="Ledger tampering demo")
    parser.add_argument(
        "--persist",
        action="store_true",
        help="Tamper the on-disk ledger (verify_chain will fail until ledger.json is deleted)",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip writing reports/figures/05_* evidence files",
    )
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("Ledger tampering demo [%s]", SYNTHETIC_LABEL)
    log.info("=" * 60)
    log.info("Ledger file: %s", LEDGER_PATH)

    _seed_chain_if_needed()

    if args.persist:
        ledger = Ledger(LEDGER_PATH)
        ok_before, detail_before = ledger.verify_chain()
        log.info("BEFORE tamper - verify_chain(): ok=%s (%s)", ok_before, detail_before)
        if not ok_before:
            log.error("Chain already invalid before tamper - fix ledger first")
            return 1
        tamper_block_at_index(LEDGER_PATH, block_index=1, persist=True)
        ledger = Ledger(LEDGER_PATH)
        ok_after, detail_after = ledger.verify_chain()
        log.info("AFTER tamper  - verify_chain(): ok=%s (%s)", ok_after, detail_after)
        if ok_after:
            log.error("FAIL - on-disk tamper was not detected")
            return 1
        log.info("PASS - on-disk hash-chain tamper-evidence demonstrated")
        log.info("Restore: delete %s and re-run run_integration_demo.py", LEDGER_PATH)
        return 0

    # Default: in-memory copy only (matches Streamlit /ledger/tamper-demo behaviour)
    ledger = Ledger(LEDGER_PATH)
    demo = demonstrate_tampering(ledger)
    log.info("BEFORE tamper - valid=%s (%s)", demo["before_valid"], demo["before_message"])
    log.info("AFTER tamper  - valid=%s errors=%s", demo["after_tamper_valid"], demo["after_errors"])
    if demo["after_tamper_valid"]:
        log.error("FAIL - in-memory tamper was not detected")
        return 1
    log.info("PASS - tamper-evidence property demonstrated (%s)", demo["note"])

    if not args.no_figures:
        save_checkpoint_figures(ledger, demo)

    return 0


if __name__ == "__main__":
    sys.exit(main())
