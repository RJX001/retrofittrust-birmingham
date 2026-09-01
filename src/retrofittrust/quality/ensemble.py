"""AutoEncoder + Isolation Forest ensemble for EPC anomaly screening.

Combination protocol (CURSOR_BUILD_SPEC §4):
1. Raw scores from :class:`ShallowAutoEncoder` and PyOD :class:`IForest`.
2. Z-score each detector independently via ``pyod.utils.utility.standardizer``.
3. Combine with PyOD ``average`` (consensus, higher precision) and report the
   union of individually thresholded detectors (higher recall).

Thresholds are **not** an arbitrary top-5% cutoff. Preferred: Extreme Value
Theory (Generalised Pareto on the reconstruction-error tail). Fallback:
mean + k·σ on a putative-clean validation subset. k / EVT target rate should
be tuned on synthetic injection (see ``evaluation``) and sanity-checked
against the EPC quality-flag literature band of roughly 27–60%.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence, Union

import numpy as np
from sklearn.utils import check_array

from retrofittrust.config import SEED
from retrofittrust.quality.autoencoder import ShallowAutoEncoder
from retrofittrust.quality.flags import (
    LITERATURE_FLAG_RATE_HIGH,
    LITERATURE_FLAG_RATE_LOW,
    attach_quality_flags,
    sanity_check_flag_rate,
)

ArrayLike = Union[np.ndarray, "pd.DataFrame"]  # noqa: F821 — string forward ref

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None  # type: ignore[assignment]


def _pyod_standardizer(X: np.ndarray, X_t: Optional[np.ndarray] = None, keep_scalar: bool = False):
    from pyod.utils.utility import standardizer

    return standardizer(X, X_t, keep_scalar=keep_scalar)


def _combine_average(score_matrix: np.ndarray) -> np.ndarray:
    """PyOD average combination; numpy mean if ``combo`` is not installed."""
    try:
        from pyod.models.combination import average as pyod_average

        return np.asarray(pyod_average(score_matrix), dtype=float)
    except Exception:
        return np.mean(score_matrix, axis=1)


def _combine_maximization(score_matrix: np.ndarray) -> np.ndarray:
    """PyOD maximization combination; numpy max if ``combo`` is not installed."""
    try:
        from pyod.models.combination import maximization as pyod_maximization

        return np.asarray(pyod_maximization(score_matrix), dtype=float)
    except Exception:
        return np.max(score_matrix, axis=1)


def _to_numpy(X: ArrayLike) -> tuple[np.ndarray, Optional[Sequence[str]], Optional[Any]]:
    if pd is not None and isinstance(X, pd.DataFrame):
        numeric = X.select_dtypes(include=[np.number])
        if numeric.shape[1] == 0:
            raise ValueError("Feature matrix has no numeric columns for anomaly detection.")
        if numeric.shape[1] < X.shape[1]:
            dropped = [c for c in X.columns if c not in numeric.columns]
            if dropped:
                import logging

                logging.getLogger(__name__).debug(
                    "Dropped %s non-numeric columns before scoring: %s",
                    len(dropped),
                    dropped[:5],
                )
        return numeric.to_numpy(dtype=float), list(numeric.columns), X.index
    arr = check_array(X, dtype=np.float64)
    return arr, None, None


@dataclass
class ThresholdResult:
    """Fitted decision threshold plus the method that produced it."""

    threshold: float
    method: str  # "evt" | "mean_sigma"
    k: Optional[float] = None
    target_rate: Optional[float] = None
    tail_quantile: Optional[float] = None
    gpd_shape: Optional[float] = None
    gpd_scale: Optional[float] = None
    notes: str = ""


def _putative_clean(scores: np.ndarray, trim_upper: float = 0.10) -> np.ndarray:
    """Drop the most extreme upper tail before estimating a 'clean' mean/σ."""
    if scores.size == 0:
        return scores
    cap = np.quantile(scores, 1.0 - trim_upper)
    clean = scores[scores <= cap]
    return clean if clean.size >= 10 else scores


def mean_sigma_threshold(
    scores: np.ndarray,
    k: float = 0.75,
    trim_upper: float = 0.10,
) -> ThresholdResult:
    """Mean + k·σ of reconstruction (or detector) scores on a cleaner subset."""
    clean = _putative_clean(np.asarray(scores, dtype=float), trim_upper=trim_upper)
    mu = float(np.mean(clean))
    sigma = float(np.std(clean))
    if sigma < 1e-12:
        sigma = 1e-12
    threshold = mu + float(k) * sigma
    return ThresholdResult(
        threshold=threshold,
        method="mean_sigma",
        k=float(k),
        notes=f"mean={mu:.4f} sigma={sigma:.4f} k={k} (clean subset, top {trim_upper:.0%} trimmed)",
    )


def evt_gpd_threshold(
    scores: np.ndarray,
    target_rate: float = 0.35,
    min_excesses: int = 30,
) -> Optional[ThresholdResult]:
    """Peaks-over-threshold GPD on the upper tail of reconstruction error.

    ``target_rate`` is the operational flag rate we try to land in the
    literature band (~27–60%), not an arbitrary 5% cutoff. The tail threshold
    ``u`` is set *below* that rate so the GPD is fitted to a genuine tail and
    the GPD quantile can still reach ``target_rate``.

    Returns None if the fit is not feasible (too few exceedances, exploding
    shape, or target rate incompatible with the chosen tail).
    """
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    if scores.size < min_excesses * 2:
        return None
    if not (0.05 < target_rate < 0.5):
        return None

    # Tail must be heavier than the target flag rate so z can sit inside it.
    u_quantile = max(0.50, 1.0 - min(0.49, target_rate + 0.15))
    u = float(np.quantile(scores, u_quantile))
    excesses = scores[scores > u] - u
    p_u = float(np.mean(scores > u))
    if excesses.size < min_excesses or p_u <= target_rate:
        return None

    try:
        from scipy.stats import genpareto

        shape, _loc, scale = genpareto.fit(excesses, floc=0)
    except Exception:
        return None

    if not np.isfinite(shape) or not np.isfinite(scale) or scale <= 0:
        return None
    # |ξ| ≳ 1 is an unstable tail for a small PoC sample.
    if abs(float(shape)) >= 1.0:
        return None

    survival = target_rate / p_u  # P(X > z | X > u)
    cdf_at_excess = 1.0 - survival
    if not (0.0 < cdf_at_excess < 1.0):
        return None
    try:
        from scipy.stats import genpareto

        excess_quantile = float(genpareto.ppf(cdf_at_excess, shape, loc=0.0, scale=scale))
    except Exception:
        return None
    if not np.isfinite(excess_quantile) or excess_quantile < 0:
        return None

    threshold = u + excess_quantile
    return ThresholdResult(
        threshold=float(threshold),
        method="evt",
        target_rate=float(target_rate),
        tail_quantile=float(u_quantile),
        gpd_shape=float(shape),
        gpd_scale=float(scale),
        notes=(
            f"GPD POT u=q{u_quantile:.2f}={u:.4f}, ξ={shape:.3f}, σ={scale:.4f}, "
            f"target_rate={target_rate:.2f}"
        ),
    )


def choose_threshold(
    scores: np.ndarray,
    *,
    prefer_evt: bool = True,
    target_rate: float = 0.35,
    k: float = 0.75,
) -> ThresholdResult:
    """Prefer EVT; fall back to mean + k·σ when the GPD fit is not feasible."""
    if prefer_evt:
        evt = evt_gpd_threshold(scores, target_rate=target_rate)
        if evt is not None:
            return evt
        result = mean_sigma_threshold(scores, k=k)
        result.notes += " (EVT not feasible — using mean+k·σ fallback)"
        return result
    return mean_sigma_threshold(scores, k=k)


@dataclass
class AnomalyEnsemble:
    """Fit AE + IForest, standardise scores, combine, threshold.

    Downstream rule (enforced in :mod:`flags`): quarantine / flag only —
    never silently delete. Flagged rows should be down-weighted in LightGBM
    (``sample_weight``) or scored at inference with a low-confidence caveat.
    """

    random_state: int = SEED
    prefer_evt: bool = True
    target_flag_rate: float = 0.35
    k: float = 0.75
    ae_kwargs: dict[str, Any] = field(default_factory=dict)
    iforest_kwargs: dict[str, Any] = field(default_factory=dict)
    ae_verbose: int = 0

    def __post_init__(self) -> None:
        self.ae_: Optional[ShallowAutoEncoder] = None
        self.iforest_ = None
        self.score_scaler_ = None
        self.ae_threshold_: Optional[ThresholdResult] = None
        self.iforest_threshold_: Optional[ThresholdResult] = None
        self.consensus_threshold_: Optional[ThresholdResult] = None
        self.feature_names_: Optional[list[str]] = None
        self.n_features_in_: Optional[int] = None
        self.train_raw_scores_: Optional[np.ndarray] = None

    def fit(self, X: ArrayLike, feature_names: Optional[Sequence[str]] = None) -> "AnomalyEnsemble":
        from pyod.models.iforest import IForest

        X_np, cols, _index = _to_numpy(X)
        self.n_features_in_ = int(X_np.shape[1])
        if feature_names is not None:
            self.feature_names_ = list(feature_names)
        elif cols is not None:
            self.feature_names_ = list(cols)
        else:
            self.feature_names_ = [f"f{i}" for i in range(self.n_features_in_)]

        ae_params = {
            "random_state": self.random_state,
            "verbose": self.ae_verbose,
        }
        ae_params.update(self.ae_kwargs)
        self.ae_ = ShallowAutoEncoder(**ae_params)
        self.ae_.fit(X_np)

        if_params = {
            "n_estimators": 100,
            "max_samples": "auto",
            "contamination": 0.1,
            "random_state": self.random_state,
            "n_jobs": 1,
            "verbose": 0,
        }
        if_params.update(self.iforest_kwargs)
        self.iforest_ = IForest(**if_params)
        self.iforest_.fit(X_np)

        ae_scores = np.asarray(self.ae_.decision_scores_, dtype=float)
        if_scores = np.asarray(self.iforest_.decision_scores_, dtype=float)
        self.train_raw_scores_ = np.column_stack([ae_scores, if_scores])
        _, self.score_scaler_ = _pyod_standardizer(self.train_raw_scores_, keep_scalar=True)

        self.ae_threshold_ = choose_threshold(
            ae_scores,
            prefer_evt=self.prefer_evt,
            target_rate=self.target_flag_rate,
            k=self.k,
        )
        self.iforest_threshold_ = choose_threshold(
            if_scores,
            prefer_evt=False,
            target_rate=self.target_flag_rate,
            k=self.k,
        )
        train_norm = self.score_scaler_.transform(self.train_raw_scores_)
        consensus = _combine_average(train_norm)
        self.consensus_threshold_ = choose_threshold(
            consensus,
            prefer_evt=self.prefer_evt,
            target_rate=self.target_flag_rate,
            k=self.k,
        )
        return self

    def retune_thresholds(
        self,
        *,
        prefer_evt: Optional[bool] = None,
        target_flag_rate: Optional[float] = None,
        k: Optional[float] = None,
    ) -> "AnomalyEnsemble":
        """Recompute EVT / mean+k·σ thresholds from stored training scores.

        Does not refit the AutoEncoder or Isolation Forest — used by
        :func:`tune_from_injection` to grid-search k and target rate.
        """
        self._require_fitted()
        if self.train_raw_scores_ is None:
            raise RuntimeError("No stored training scores to retune from.")
        if prefer_evt is not None:
            self.prefer_evt = bool(prefer_evt)
        if target_flag_rate is not None:
            self.target_flag_rate = float(target_flag_rate)
        if k is not None:
            self.k = float(k)
        ae_scores = self.train_raw_scores_[:, 0]
        if_scores = self.train_raw_scores_[:, 1]
        self.ae_threshold_ = choose_threshold(
            ae_scores,
            prefer_evt=self.prefer_evt,
            target_rate=self.target_flag_rate,
            k=self.k,
        )
        self.iforest_threshold_ = choose_threshold(
            if_scores,
            prefer_evt=False,
            target_rate=self.target_flag_rate,
            k=self.k,
        )
        train_norm = self.score_scaler_.transform(self.train_raw_scores_)
        consensus = _combine_average(train_norm)
        self.consensus_threshold_ = choose_threshold(
            consensus,
            prefer_evt=self.prefer_evt,
            target_rate=self.target_flag_rate,
            k=self.k,
        )
        return self

    def _require_fitted(self) -> None:
        if self.ae_ is None or self.iforest_ is None or self.score_scaler_ is None:
            raise RuntimeError("AnomalyEnsemble is not fitted. Call fit() first.")

    def raw_scores(self, X: ArrayLike) -> np.ndarray:
        """Return ``(n_samples, 2)`` raw scores: columns [autoencoder, iforest]."""
        self._require_fitted()
        X_np, _, _ = _to_numpy(X)
        ae_scores = np.asarray(self.ae_.decision_function(X_np), dtype=float)
        if_scores = np.asarray(self.iforest_.decision_function(X_np), dtype=float)
        return np.column_stack([ae_scores, if_scores])

    def combined_scores(self, X: ArrayLike) -> dict[str, np.ndarray]:
        self._require_fitted()
        raw = self.raw_scores(X)
        normed = self.score_scaler_.transform(raw)
        return {
            "ae_score": raw[:, 0],
            "iforest_score": raw[:, 1],
            "ae_score_z": normed[:, 0],
            "iforest_score_z": normed[:, 1],
            "consensus_score": _combine_average(normed),
            "union_score": _combine_maximization(normed),
        }

    def predict_flags(self, X: ArrayLike) -> dict[str, np.ndarray]:
        scores = self.combined_scores(X)
        ae_flag = scores["ae_score"] > self.ae_threshold_.threshold
        if_flag = scores["iforest_score"] > self.iforest_threshold_.threshold
        consensus_flag = scores["consensus_score"] > self.consensus_threshold_.threshold
        union_flag = ae_flag | if_flag
        return {
            **scores,
            "flagged_ae": ae_flag.astype(bool),
            "flagged_iforest": if_flag.astype(bool),
            "flagged_consensus": consensus_flag.astype(bool),
            "flagged_union": union_flag.astype(bool),
        }

    def per_feature_reconstruction_error(self, X: ArrayLike) -> np.ndarray:
        self._require_fitted()
        X_np, _, _ = _to_numpy(X)
        return self.ae_.per_feature_reconstruction_error(X_np)

    def score_frame(self, X: ArrayLike, feature_names: Optional[Sequence[str]] = None):
        """Build the per-record score / flag / reconstruction-error table."""
        if pd is None:
            raise ImportError("pandas is required for score_frame().")
        X_np, cols, index = _to_numpy(X)
        names = list(feature_names or cols or self.feature_names_ or [f"f{i}" for i in range(X_np.shape[1])])
        flags = self.predict_flags(X_np)
        recon = self.per_feature_reconstruction_error(X_np)
        if len(names) != recon.shape[1]:
            names = [f"f{i}" for i in range(recon.shape[1])]
        data: dict[str, Any] = {key: flags[key] for key in flags}
        for j, name in enumerate(names):
            data[f"recon_err__{name}"] = recon[:, j]
        top_idx = np.argmax(recon, axis=1)
        data["top_implausible_feature"] = np.asarray(names)[top_idx]
        data["top_implausible_error"] = recon[np.arange(len(recon)), top_idx]
        frame = pd.DataFrame(data, index=index)
        return frame


class DataQualityScreen:
    """Pipeline-facing API for Program 1 (ranking) and Program 3 (dashboard).

    Typical modelling-pipeline call::

        from retrofittrust.quality import DataQualityScreen

        screen = DataQualityScreen(random_state=42)
        screen.fit(X_train)                       # preprocessed numeric matrix
        flagged = screen.transform(X)             # same index as X if DataFrame
        weights = flagged["sample_weight"]        # pass to LightGBM
        caveat = flagged["inference_caveat"]      # surface in the twin / API

    ``transform`` never drops rows. Flagged records stay in the dataset with a
    quarantine flag, a confidence score, and a reduced sample weight.
    """

    def __init__(
        self,
        random_state: int = SEED,
        flag_mode: str = "union",
        prefer_evt: bool = True,
        target_flag_rate: float = 0.35,
        k: float = 0.75,
        flagged_sample_weight: float = 0.35,
        ae_kwargs: Optional[dict[str, Any]] = None,
        iforest_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        if flag_mode not in {"union", "consensus"}:
            raise ValueError("flag_mode must be 'union' (high recall) or 'consensus' (high precision).")
        self.flag_mode = flag_mode
        self.flagged_sample_weight = float(flagged_sample_weight)
        self.ensemble = AnomalyEnsemble(
            random_state=random_state,
            prefer_evt=prefer_evt,
            target_flag_rate=target_flag_rate,
            k=k,
            ae_kwargs=ae_kwargs or {},
            iforest_kwargs=iforest_kwargs or {},
        )

    def fit(self, X: ArrayLike, feature_names: Optional[Sequence[str]] = None) -> "DataQualityScreen":
        self.ensemble.fit(X, feature_names=feature_names)
        return self

    def transform(self, X: ArrayLike, feature_names: Optional[Sequence[str]] = None):
        """Attach flags to a copy of ``X`` if it is a DataFrame; else return the score frame.

        Never deletes rows.
        """
        score_frame = self.ensemble.score_frame(X, feature_names=feature_names)
        consensus_thr = (
            self.ensemble.consensus_threshold_.threshold
            if self.ensemble.consensus_threshold_ is not None
            else None
        )
        kwargs = {
            "flag_mode": self.flag_mode,
            "flagged_sample_weight": self.flagged_sample_weight,
            "consensus_threshold": consensus_thr,
        }
        if pd is not None and isinstance(X, pd.DataFrame):
            return attach_quality_flags(X, score_frame, **kwargs)
        return attach_quality_flags(score_frame.copy(), score_frame, **kwargs)

    def retune(self, *, prefer_evt=None, target_flag_rate=None, k=None) -> "DataQualityScreen":
        """Recompute thresholds without refitting detectors (injection tuning)."""
        self.ensemble.retune_thresholds(
            prefer_evt=prefer_evt,
            target_flag_rate=target_flag_rate,
            k=k,
        )
        return self

    def fit_transform(self, X: ArrayLike, feature_names: Optional[Sequence[str]] = None):
        return self.fit(X, feature_names=feature_names).transform(X, feature_names=feature_names)

    def sample_weights(self, X: ArrayLike) -> np.ndarray:
        """LightGBM-ready sample weights (1.0 clean, ``flagged_sample_weight`` if flagged)."""
        frame = self.transform(X)
        return np.asarray(frame["sample_weight"], dtype=float)

    def flag_rate_report(self, X: ArrayLike) -> dict[str, Any]:
        frame = self.ensemble.score_frame(X)
        union_rate = float(np.mean(frame["flagged_union"]))
        consensus_rate = float(np.mean(frame["flagged_consensus"]))
        operational = union_rate if self.flag_mode == "union" else consensus_rate
        return {
            "flag_mode": self.flag_mode,
            "flag_rate_union": union_rate,
            "flag_rate_consensus": consensus_rate,
            "operational_flag_rate": operational,
            "literature_band": (LITERATURE_FLAG_RATE_LOW, LITERATURE_FLAG_RATE_HIGH),
            "literature_ok": sanity_check_flag_rate(operational)["ok"],
            "ae_threshold": self.ensemble.ae_threshold_,
            "iforest_threshold": self.ensemble.iforest_threshold_,
            "consensus_threshold": self.ensemble.consensus_threshold_,
            "ae_architecture": {
                "hidden_dim": self.ensemble.ae_.hidden_dim_ if self.ensemble.ae_ else None,
                "bottleneck_dim": self.ensemble.ae_.bottleneck_dim_ if self.ensemble.ae_ else None,
                "n_features": self.ensemble.n_features_in_,
                "early_stopped": self.ensemble.ae_.stopped_early_ if self.ensemble.ae_ else None,
                "best_epoch": self.ensemble.ae_.best_epoch_ if self.ensemble.ae_ else None,
            },
        }
