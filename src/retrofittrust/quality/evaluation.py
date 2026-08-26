"""Evaluation of the data-quality screen without ground-truth labels.

No real anomaly labels exist on EPC extracts, so we:

1. Inject *synthetic* anomalies into copies of records (heating-type swap,
   implausible floor-area scale, construction/age mismatch). Labelled
   **SYNTHETIC DATA** — not real EPC errors.
2. Measure recall of those injected rows (must beat chance).
3. Spot-check face validity of top-N flags (left to the analyst / notebook).
4. Sanity-check the operational flagged rate against the ~27–60% EPC
   quality-flag literature band.
5. Retrain across seeds and report Jaccard overlap of flagged sets
   (stability).

Tune k / EVT target rate against injection recall; do not pick an arbitrary
top-5% cutoff.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Union

import numpy as np

from retrofittrust.config import SEED
from retrofittrust.quality.flags import (
    LITERATURE_FLAG_RATE_RANGE,
    sanity_check_flag_rate,
)

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None  # type: ignore[assignment]

ArrayLike = Union[np.ndarray, "pd.DataFrame"]  # noqa: F821

# SYNTHETIC DATA — injection kinds used only for evaluation.
INJECTION_HEATING_SWAP = "synthetic_heating_type_swap"
INJECTION_FLOOR_AREA = "synthetic_floor_area_implausible"
INJECTION_CONSTRUCTION_AGE = "synthetic_construction_age_mismatch"
INJECTION_KINDS = (
    INJECTION_HEATING_SWAP,
    INJECTION_FLOOR_AREA,
    INJECTION_CONSTRUCTION_AGE,
)

_HEATING_COLS = (
    "heating_type",
    "MAINHEAT_DESCRIPTION",
    "mainheat_description",
    "MAIN_FUEL",
    "main_fuel",
    "main_heating",
    "MAINHEAT",
)
_FLOOR_COLS = (
    "floor_area",
    "TOTAL_FLOOR_AREA",
    "total_floor_area",
    "total-floor-area",
)
_CONSTRUCTION_COLS = (
    "wall_construction",
    "WALLS_DESCRIPTION",
    "walls_description",
    "construction_type",
    "WALLS_ENERGY_EFF",
    "built_form",
    "BUILT_FORM",
    "PROPERTY_TYPE",
    "property_type",
)
_AGE_COLS = (
    "age_band",
    "CONSTRUCTION_AGE_BAND",
    "construction_age_band",
    "property_age",
)


def _rng(seed: int = SEED) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def _first_col(frame, candidates: Sequence[str]) -> Optional[str]:
    lower = {str(c).lower(): c for c in frame.columns}
    for name in candidates:
        if name in frame.columns:
            return name
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def _cols_matching(names: Sequence[str], needles: Sequence[str]) -> list[str]:
    found: list[str] = []
    lowered = [(n, n.lower()) for n in names]
    for needle in needles:
        needle_l = needle.lower()
        for original, low in lowered:
            if needle_l in low and original not in found:
                found.append(original)
    return found


@dataclass
class InjectionResult:
    """SYNTHETIC DATA — corrupted copies plus a binary injection mask."""

    X_corrupted: Any
    y_injected: np.ndarray
    log: Any
    note: str = (
        "SYNTHETIC DATA — injected for evaluation only; not real EPC assessor errors."
    )
    kinds_used: tuple[str, ...] = INJECTION_KINDS


def _numeric_block(frame, needles: Sequence[str]) -> list[str]:
    """Columns whose names match any needle; prefer numeric (one-hot blocks)."""
    matched = _cols_matching(list(frame.columns), needles)
    numeric = [c for c in matched if pd.api.types.is_numeric_dtype(frame[c])]
    return numeric or matched


def _graft_block(dest, dest_rows, source, source_rows, columns: Sequence[str]) -> None:
    if not columns:
        return
    dest.loc[dest_rows, list(columns)] = source.iloc[list(source_rows)][list(columns)].to_numpy()


def inject_synthetic_anomalies(
    X: ArrayLike,
    *,
    rate: float = 0.15,
    seed: int = SEED,
    kinds: Sequence[str] = INJECTION_KINDS,
) -> InjectionResult:
    """Corrupt copies of rows and *append* them. Original rows are unchanged.

    Domain rules (DataFrame with EPC-like or one-hot columns):
    - swap heating type / one-hot heating block with a donor record
    - scale floor area by 0.05× or 12× (implausible)
    - mismatch construction/age by grafting another row's construction block

    Numeric-matrix fallback: the same three mechanisms on column 0 / 1 / 2.

    Call this on the **same numeric feature matrix** the screen was fitted on
    if you then pass the result to :func:`evaluate_recall`.
    """
    if not 0.0 < rate < 0.5:
        raise ValueError("injection rate must be in (0, 0.5).")
    rng = _rng(seed)
    kinds = tuple(kinds) or INJECTION_KINDS

    if pd is not None and isinstance(X, pd.DataFrame):
        return _inject_dataframe(X, rate=rate, rng=rng, kinds=kinds)
    X_np = np.asarray(X, dtype=float)
    return _inject_numeric(X_np, rate=rate, rng=rng, kinds=kinds)


def _n_inject(n: int, rate: float) -> int:
    return max(1, int(round(n * rate)))


def _inject_dataframe(frame, rate: float, rng: np.random.Generator, kinds: Sequence[str]) -> InjectionResult:
    # Positional concat so y_injected aligns with score_frame row order.
    base = frame.reset_index(drop=True)
    n = len(base)
    n_inj = _n_inject(n, rate)
    source_idx = rng.choice(n, size=n_inj, replace=False)
    copies = base.iloc[list(source_idx)].copy()
    copies.index = pd.RangeIndex(n, n + n_inj)
    assigned = rng.choice(np.asarray(kinds), size=n_inj)
    log_rows: list[dict[str, Any]] = []

    heating_block = _numeric_block(base, _HEATING_COLS) or _numeric_block(
        base, ("heat", "mainheat", "main_fuel", "heating")
    )
    floor_col = _first_col(base, _FLOOR_COLS) or _first_col(
        base, [c for c in base.columns if "floor" in str(c).lower() or "area" in str(c).lower()]
    )
    construction_block = _numeric_block(base, _CONSTRUCTION_COLS) or _numeric_block(
        base, ("wall", "construction", "built_form", "property_type")
    )
    age_block = _numeric_block(base, _AGE_COLS) or _numeric_block(
        base, ("age", "age_band", "construction_age")
    )

    for kind in kinds:
        mask = assigned == kind
        if not np.any(mask):
            continue
        local_rows = np.where(mask)[0]
        dest_index = copies.index[mask]
        donor = rng.integers(0, n, size=len(local_rows))
        if kind == INJECTION_HEATING_SWAP:
            cols = heating_block or [str(base.columns[0])]
            _graft_block(copies, dest_index, base, donor, cols)
            detail = f"swapped heating block {cols}"
        elif kind == INJECTION_FLOOR_AREA:
            numeric_cols = [c for c in base.columns if pd.api.types.is_numeric_dtype(base[c])]
            col = floor_col or (str(numeric_cols[0]) if numeric_cols else str(base.columns[0]))
            factors = rng.choice(np.array([0.05, 12.0]), size=len(local_rows))
            values = pd.to_numeric(copies.loc[dest_index, col], errors="coerce").to_numpy(dtype=float)
            copies.loc[dest_index, col] = values * factors
            detail = f"scaled {col} by 0.05× or 12×"
        else:
            cons_cols = construction_block or [str(base.columns[min(1, len(base.columns) - 1)])]
            age_cols = age_block or [str(base.columns[0])]
            _graft_block(copies, dest_index, base, donor, cons_cols)
            detail = f"grafted construction {cons_cols}; kept age {age_cols}"
        for pos, src in zip(local_rows, source_idx[mask]):
            log_rows.append(
                {
                    "row_position": int(n + pos),
                    "source_index": int(src),
                    "kind": kind,
                    "detail": detail,
                    "synthetic": True,
                    "note": "SYNTHETIC DATA",
                }
            )

    combined = pd.concat([base, copies], axis=0, ignore_index=True)
    y = np.concatenate([np.zeros(n, dtype=int), np.ones(n_inj, dtype=int)])
    log = pd.DataFrame(log_rows)
    return InjectionResult(X_corrupted=combined, y_injected=y, log=log)


def _inject_numeric(
    X: np.ndarray,
    rate: float,
    rng: np.random.Generator,
    kinds: Sequence[str],
) -> InjectionResult:
    n, d = X.shape
    n_inj = _n_inject(n, rate)
    source_idx = rng.choice(n, size=n_inj, replace=False)
    copies = X[source_idx].copy()
    assigned = rng.choice(np.asarray(kinds), size=n_inj)
    heat_idx = 0
    area_idx = min(1, d - 1)
    cons_idx = min(2, d - 1)

    log_rows: list[dict[str, Any]] = []
    for kind in kinds:
        mask = assigned == kind
        if not np.any(mask):
            continue
        rows = np.where(mask)[0]
        if kind == INJECTION_HEATING_SWAP:
            donor = rng.integers(0, n, size=len(rows))
            copies[rows, heat_idx] = X[donor, heat_idx]
            detail = f"swapped feature {heat_idx}"
        elif kind == INJECTION_FLOOR_AREA:
            factors = rng.choice(np.array([0.05, 12.0]), size=len(rows))
            copies[rows, area_idx] = copies[rows, area_idx] * factors
            detail = f"scaled feature {area_idx} by 0.05× or 12×"
        else:
            donor = rng.integers(0, n, size=len(rows))
            copies[rows, cons_idx] = X[donor, cons_idx]
            detail = f"mismatched feature {cons_idx} from donor (kept other slots)"
        for local_i, src in zip(rows, source_idx[mask]):
            log_rows.append(
                {
                    "row_position": int(n + local_i),
                    "source_index": int(src),
                    "kind": kind,
                    "detail": detail,
                    "synthetic": True,
                    "note": "SYNTHETIC DATA",
                }
            )

    combined = np.vstack([X, copies])
    y = np.concatenate([np.zeros(n, dtype=int), np.ones(n_inj, dtype=int)])
    log = pd.DataFrame(log_rows) if pd is not None else log_rows
    return InjectionResult(X_corrupted=combined, y_injected=y, log=log)


@dataclass
class RecallReport:
    recall: float
    recall_by_kind: dict[str, float]
    n_injected: int
    n_caught: int
    flag_mode: str
    chance_baseline: float
    beats_chance: bool
    notes: str = "SYNTHETIC DATA evaluation — injected labels only."


def evaluate_recall(
    screen,
    X: ArrayLike,
    *,
    rate: float = 0.15,
    seed: int = SEED,
    flag_mode: Optional[str] = None,
    kinds: Sequence[str] = INJECTION_KINDS,
) -> RecallReport:
    """Fit is assumed done. Inject synthetic copies, score, measure recall.

    Chance baseline is the operational flagged rate on the original (uninjected)
    matrix — a detector that flags everything would 'recall' 100% but is not
    useful. ``beats_chance`` requires injected recall to exceed that rate.
    """
    injection = inject_synthetic_anomalies(X, rate=rate, seed=seed, kinds=kinds)
    mode = flag_mode or getattr(screen, "flag_mode", "union")
    flag_col = "flagged_union" if mode == "union" else "flagged_consensus"

    original_frame = screen.ensemble.score_frame(X)
    original_rate = float(np.mean(original_frame[flag_col]))

    scored = screen.ensemble.score_frame(injection.X_corrupted)
    y = injection.y_injected.astype(bool)
    pred = np.asarray(scored[flag_col], dtype=bool)
    n_injected = int(y.sum())
    n_caught = int((pred & y).sum())
    recall = n_caught / n_injected if n_injected else 0.0

    recall_by_kind: dict[str, float] = {}
    log = injection.log
    if pd is not None and isinstance(log, pd.DataFrame) and len(log) and "row_position" in log.columns:
        for kind, group in log.groupby("kind"):
            positions = group["row_position"].to_numpy(dtype=int)
            recall_by_kind[str(kind)] = float(pred[positions].mean()) if len(positions) else float("nan")

    return RecallReport(
        recall=float(recall),
        recall_by_kind=recall_by_kind,
        n_injected=n_injected,
        n_caught=n_caught,
        flag_mode=mode,
        chance_baseline=original_rate,
        beats_chance=bool(recall > original_rate + 0.05),
        notes=(
            "SYNTHETIC DATA evaluation — injected labels only. "
            f"Recall {recall:.1%} vs chance (original flag rate) {original_rate:.1%}."
        ),
    )


def _jaccard(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 1.0


@dataclass
class StabilityReport:
    seeds: tuple[int, ...]
    pairwise_jaccard: list[tuple[int, int, float]]
    mean_jaccard: float
    flag_rates: dict[int, float]
    notes: str = ""


def multi_seed_stability(
    X: ArrayLike,
    *,
    seeds: Sequence[int] = (42, 43, 44),
    flag_mode: str = "union",
    screen_kwargs: Optional[dict[str, Any]] = None,
) -> StabilityReport:
    """Retrain the ensemble with different seeds; compare overlap of flagged sets.

    Default three seeds keeps the PoC runtime tractable (the AE is the cost).
    Dissertation write-up can pass five seeds, e.g. ``range(42, 47)``.
    """
    from retrofittrust.quality.ensemble import DataQualityScreen

    kwargs = dict(screen_kwargs or {})
    kwargs.setdefault("flag_mode", flag_mode)
    flag_col = "flagged_union" if flag_mode == "union" else "flagged_consensus"

    flags_by_seed: dict[int, np.ndarray] = {}
    rates: dict[int, float] = {}
    for seed in seeds:
        screen = DataQualityScreen(random_state=int(seed), **kwargs)
        screen.fit(X)
        frame = screen.ensemble.score_frame(X)
        flags = np.asarray(frame[flag_col], dtype=bool)
        flags_by_seed[int(seed)] = flags
        rates[int(seed)] = float(flags.mean())

    pairwise: list[tuple[int, int, float]] = []
    seed_list = [int(s) for s in seeds]
    for i, s1 in enumerate(seed_list):
        for s2 in seed_list[i + 1 :]:
            pairwise.append((s1, s2, _jaccard(flags_by_seed[s1], flags_by_seed[s2])))
    mean_j = float(np.mean([p[2] for p in pairwise])) if pairwise else 1.0
    return StabilityReport(
        seeds=tuple(seed_list),
        pairwise_jaccard=pairwise,
        mean_jaccard=mean_j,
        flag_rates=rates,
        notes=(
            f"Mean pairwise Jaccard={mean_j:.3f} on {flag_mode} flags. "
            "Low overlap means the screen is seed-brittle; report this limitation."
        ),
    )


def literature_sanity(flag_rate: float) -> dict[str, Any]:
    """Wrapper around :func:`sanity_check_flag_rate` with the documented band."""
    result = sanity_check_flag_rate(flag_rate)
    result["expected_band"] = LITERATURE_FLAG_RATE_RANGE
    return result


@dataclass
class TuneResult:
    """Best threshold settings found against SYNTHETIC DATA injection recall."""

    k: float
    target_flag_rate: float
    prefer_evt: bool
    recall: float
    flag_rate: float
    literature_ok: bool
    grid: list[dict[str, Any]]
    notes: str = (
        "Thresholds tuned on SYNTHETIC DATA injection; operational rate should "
        "still sit in the ~27–60% EPC literature band."
    )


def tune_from_injection(
    screen,
    X: ArrayLike,
    *,
    ks: Sequence[float] = (0.25, 0.5, 0.75, 1.0, 1.5),
    target_rates: Sequence[float] = (0.27, 0.30, 0.35, 0.40, 0.45),
    rate: float = 0.15,
    seed: int = SEED,
    flag_mode: Optional[str] = None,
) -> TuneResult:
    """Grid-search k and EVT target rate against synthetic-injection recall.

    Detectors stay fitted; only thresholds are recomputed. Prefers settings
    whose operational flagged rate sits in the ~27–60% literature band, then
    maximises injection recall. Applies the winning settings on ``screen``.
    """
    mode = flag_mode or getattr(screen, "flag_mode", "union")
    flag_col = "flagged_union" if mode == "union" else "flagged_consensus"
    injection = inject_synthetic_anomalies(X, rate=rate, seed=seed)

    grid: list[dict[str, Any]] = []
    lo, hi = LITERATURE_FLAG_RATE_RANGE
    for prefer_evt in (True, False):
        rates = target_rates if prefer_evt else (0.35,)
        k_grid = (0.75,) if prefer_evt else ks
        for target in rates:
            for k in k_grid:
                screen.ensemble.retune_thresholds(
                    prefer_evt=prefer_evt,
                    target_flag_rate=float(target),
                    k=float(k),
                )
                original = screen.ensemble.score_frame(X)
                flag_rate = float(np.mean(original[flag_col]))
                scored = screen.ensemble.score_frame(injection.X_corrupted)
                y = injection.y_injected.astype(bool)
                pred = np.asarray(scored[flag_col], dtype=bool)
                recall = float((pred & y).sum() / y.sum()) if y.sum() else 0.0
                in_band = lo <= flag_rate <= hi
                grid.append(
                    {
                        "prefer_evt": prefer_evt,
                        "k": float(k),
                        "target_flag_rate": float(target),
                        "flag_rate": flag_rate,
                        "recall": recall,
                        "literature_ok": in_band,
                        "threshold_method": (
                            screen.ensemble.ae_threshold_.method
                            if screen.ensemble.ae_threshold_
                            else None
                        ),
                    }
                )

    in_band_rows = [row for row in grid if row["literature_ok"]]
    pool = in_band_rows or grid
    best = max(pool, key=lambda row: (row["recall"], -abs(row["flag_rate"] - 0.35)))
    screen.ensemble.retune_thresholds(
        prefer_evt=best["prefer_evt"],
        target_flag_rate=best["target_flag_rate"],
        k=best["k"],
    )
    return TuneResult(
        k=best["k"],
        target_flag_rate=best["target_flag_rate"],
        prefer_evt=best["prefer_evt"],
        recall=best["recall"],
        flag_rate=best["flag_rate"],
        literature_ok=best["literature_ok"],
        grid=grid,
    )
