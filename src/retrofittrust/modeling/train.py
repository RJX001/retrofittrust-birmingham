"""Train the LightGBM retrofit-priority regressor.

Primary model is LightGBM (not deep learning). An optional Random Forest
baseline is fitted under the same K-fold split for a simple comparison.
Models are serialised with joblib to ``models/ranking_model.joblib``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold

from ..config import MODELS_DIR, SEED as CONFIG_SEED
from .features import (
    TARGET_COLUMN,
    compute_sample_weights,
    resolve_target,
    select_feature_columns,
    to_model_matrix,
)

SEED = 42
assert SEED == CONFIG_SEED

DEFAULT_MODEL_PATH = MODELS_DIR / "ranking_model.joblib"

# Modest defaults for a tabular PoC — not a heavily tuned production model.
LGBM_PARAMS: dict[str, Any] = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 20,
    "reg_lambda": 1.0,
    "random_state": SEED,
    "n_jobs": -1,
    "verbosity": -1,
    "force_col_wise": True,
}

RF_PARAMS: dict[str, Any] = {
    "n_estimators": 200,
    "max_depth": 16,
    "min_samples_leaf": 5,
    "random_state": SEED,
    "n_jobs": -1,
}


def _set_seeds(seed: int = SEED) -> None:
    np.random.seed(seed)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _summarise_folds(fold_metrics: list[dict[str, float]]) -> dict[str, float]:
    keys = [k for k in fold_metrics[0] if k != "fold"]
    summary: dict[str, float] = {}
    for key in keys:
        values = np.array([m[key] for m in fold_metrics], dtype="float64")
        summary[f"{key}_mean"] = float(values.mean())
        summary[f"{key}_std"] = float(values.std(ddof=1) if len(values) > 1 else 0.0)
    return summary


def _make_lgbm(params: dict[str, Any] | None = None) -> lgb.LGBMRegressor:
    merged = {**LGBM_PARAMS, **(params or {})}
    merged.setdefault("random_state", SEED)
    return lgb.LGBMRegressor(**merged)


def _cross_validate(
    estimator_factory,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    sample_weight: np.ndarray | None,
    n_splits: int,
    impute: bool = False,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    n = len(X)
    n_splits = max(2, min(n_splits, n))
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    fold_metrics: list[dict[str, float]] = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X), start=1):
        X_tr = X.iloc[train_idx]
        X_va = X.iloc[val_idx]
        y_tr = y.iloc[train_idx]
        y_va = y.iloc[val_idx]
        w_tr = sample_weight[train_idx] if sample_weight is not None else None

        if impute:
            imputer = SimpleImputer(strategy="median")
            X_tr_fit = imputer.fit_transform(X_tr)
            X_va_fit = imputer.transform(X_va)
        else:
            X_tr_fit, X_va_fit = X_tr, X_va

        model = estimator_factory()
        fit_kwargs: dict[str, Any] = {}
        if w_tr is not None:
            fit_kwargs["sample_weight"] = w_tr
        model.fit(X_tr_fit, y_tr, **fit_kwargs)
        pred = model.predict(X_va_fit)
        scores = _metrics(y_va.to_numpy(), np.asarray(pred))
        scores["fold"] = float(fold)
        fold_metrics.append(scores)

    return fold_metrics, _summarise_folds(fold_metrics)


def train_ranking_model(
    df: pd.DataFrame,
    *,
    target: pd.Series | str | None = None,
    use_sample_weights: bool = True,
    compare_baseline: bool = False,
    n_splits: int = 5,
    lgbm_params: dict[str, Any] | None = None,
    model_path: Path | str | None = None,
    include_census: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Fit LightGBM on a preprocess DataFrame and save the artefact.

    Parameters
    ----------
    df:
        Output of the data/preprocess pipeline (one row per property or LSOA).
    target:
        Column name, aligned Series, or None to use / compute
        ``retrofit_priority_score``.
    use_sample_weights:
        If True, down-weight quality-flagged rows (or use ``sample_weight`` /
        ``quality_confidence``). Flagged rows are still scored at inference.
    compare_baseline:
        If True, run the same K-fold protocol with RandomForestRegressor.
    n_splits:
        K-fold splits (seeded with SEED=42).
    """
    _set_seeds(SEED)

    if df.empty:
        raise ValueError("Cannot train on an empty DataFrame.")

    y = resolve_target(df, target)
    valid = y.notna()
    if not valid.any():
        raise ValueError("Target is entirely missing after resolution.")
    if not valid.all():
        df = df.loc[valid].copy()
        y = y.loc[valid]
        if verbose:
            dropped = int((~valid).sum())
            print(f"Dropped {dropped} row(s) with missing target before training.")

    raw_feature_cols = select_feature_columns(df, include_census=include_census)
    X = to_model_matrix(df, include_census=include_census)
    y = y.loc[X.index]

    weights = compute_sample_weights(df.loc[X.index]) if use_sample_weights else None

    if verbose:
        print(
            f"Training LightGBM on {len(X)} rows x {X.shape[1]} features "
            f"(sample weights={'on' if weights is not None else 'off'})."
        )

    lgbm_fold, lgbm_cv = _cross_validate(
        lambda: _make_lgbm(lgbm_params),
        X,
        y,
        sample_weight=weights,
        n_splits=n_splits,
        impute=False,
    )

    final_model = _make_lgbm(lgbm_params)
    fit_kwargs: dict[str, Any] = {}
    if weights is not None:
        fit_kwargs["sample_weight"] = weights
    final_model.fit(X, y, **fit_kwargs)

    baseline: dict[str, Any] | None = None
    if compare_baseline:
        rf_fold, rf_cv = _cross_validate(
            lambda: RandomForestRegressor(**RF_PARAMS),
            X,
            y,
            sample_weight=weights,
            n_splits=n_splits,
            impute=True,
        )
        baseline = {
            "model_type": "random_forest",
            "params": dict(RF_PARAMS),
            "fold_metrics": rf_fold,
            "cv_metrics": rf_cv,
        }
        if verbose:
            print(
                "Random Forest baseline "
                f"RMSE={rf_cv['rmse_mean']:.4f} +/- {rf_cv['rmse_std']:.4f} | "
                f"R2={rf_cv['r2_mean']:.4f}"
            )

    path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    artefact: dict[str, Any] = {
        "model": final_model,
        "model_type": "lightgbm",
        "feature_names": list(X.columns),
        "raw_feature_columns": raw_feature_cols,
        "target_column": y.name or TARGET_COLUMN,
        "seed": SEED,
        "lgbm_params": {**LGBM_PARAMS, **(lgbm_params or {})},
        "use_sample_weights": bool(use_sample_weights),
        "cv_metrics": lgbm_cv,
        "fold_metrics": lgbm_fold,
        "baseline": baseline,
        "n_samples": int(len(X)),
        "n_features": int(X.shape[1]),
        "include_census": include_census,
    }
    joblib.dump(artefact, path)

    if verbose:
        print(
            f"LightGBM CV RMSE={lgbm_cv['rmse_mean']:.4f} +/- {lgbm_cv['rmse_std']:.4f} | "
            f"MAE={lgbm_cv['mae_mean']:.4f} | R2={lgbm_cv['r2_mean']:.4f}"
        )
        print(f"Saved ranking model to {path}")

    return {
        "model": final_model,
        "artefact": artefact,
        "artefact_path": path,
        "cv_metrics": lgbm_cv,
        "fold_metrics": lgbm_fold,
        "baseline": baseline,
        "n_samples": int(len(X)),
        "n_features": int(X.shape[1]),
        "feature_names": list(X.columns),
    }


def run_ranking_training(
    *,
    flagged_path: Path | str,
    processed_dir: Path | str,
    models_dir: Path | str,
    reports_dir: Path | str,
    seed: int = SEED,
) -> dict[str, Any]:
    """Checkpoint 3 entry point expected by scripts/03_train_ranking_model.py."""
    from retrofittrust.modeling.explain import plot_local_waterfall
    from retrofittrust.modeling.features import quality_flag_mask

    _set_seeds(seed)
    flagged_path = Path(flagged_path)
    models_dir = Path(models_dir)
    reports_dir = Path(reports_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(flagged_path) if flagged_path.suffix == ".parquet" else pd.read_csv(flagged_path)
    model_path = models_dir / "lgbm_ranker.joblib"
    alias_path = models_dir / "ranking_model.joblib"

    result = train_ranking_model(
        df,
        use_sample_weights=True,
        compare_baseline=False,
        model_path=model_path,
        verbose=False,
    )

    # Alias for predict.py / explain.py default path
    joblib.dump(result["artefact"], alias_path)

    flagged_rate = float(quality_flag_mask(df).mean()) if len(df) else 0.0
    cv = result["cv_metrics"]
    shap_path: Path | None = None
    if len(df) > 0:
        shap_path = plot_local_waterfall(
            result["artefact"],
            df.head(1),
            row_index=0,
            out_dir=reports_dir,
            filename="shap_waterfall_sample.png",
        )

    metrics: dict[str, Any] = {
        "input_rows": len(df),
        "train_rows": result["n_samples"],
        "flagged_rate_in_train": flagged_rate,
        "cv_rmse": cv.get("rmse_mean"),
        "cv_r2": cv.get("r2_mean"),
        "shap_waterfall_saved": str(shap_path) if shap_path else None,
        "target_formula": (
            "0.6 * normalised_epc_gap + 0.4 * normalised_imd_income (config defaults)"
        ),
        "model_path": str(model_path),
    }
    return metrics
