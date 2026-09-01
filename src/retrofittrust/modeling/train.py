"""Train the LightGBM retrofit-priority regressor.

Primary model is LightGBM (not deep learning). An optional Random Forest
baseline is fitted under the same K-fold split for a simple comparison.
Models are serialised with joblib to ``models/ranking_model.joblib``.

Quality-flagged rows are down-weighted (never deleted) via sample weights.

SHAP caveat (see also ``explain.py``): TreeExplainer can misattribute
importance among correlated features — floor area, habitable rooms and
heating cost; IMD score versus the income domain. Discuss in the
dissertation; do not treat SHAP values as causal.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold

from ..config import (
    DATA_PROCESSED,
    EPC_GAP_WEIGHT,
    IMD_INCOME_WEIGHT,
    MODELS_DIR,
    SEED as CONFIG_SEED,
)
from .features import (
    TARGET_COLUMN,
    compute_priority_target,
    compute_sample_weights,
    quality_flag_mask,
    resolve_target,
    select_feature_columns,
    to_model_matrix,
)

logger = logging.getLogger(__name__)

SEED = 42
assert SEED == CONFIG_SEED

DEFAULT_MODEL_PATH = MODELS_DIR / "ranking_model.joblib"
RETROFIT_SCORES_CSV = DATA_PROCESSED / "retrofit_scores.csv"
RETROFIT_SCORES_PARQUET = DATA_PROCESSED / "retrofit_scores.parquet"
RANKING_METRICS_JSON = MODELS_DIR / "ranking_metrics.json"
WEIGHT_SENSITIVITY_JSON = MODELS_DIR / "ranking_weight_sensitivity.json"
FACE_VALIDITY_JSON = MODELS_DIR / "ranking_face_validity.json"

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

# Alternative target weights for sensitivity (not separate model families).
SENSITIVITY_WEIGHTS: tuple[tuple[float, float], ...] = (
    (0.5, 0.5),
    (0.6, 0.4),
    (0.7, 0.3),
)

# Birmingham LSOA names / substrings associated with high deprivation (face validity).
HIGH_DEPRIVATION_NAME_HINTS: tuple[str, ...] = (
    "sparkbrook",
    "nechells",
    "aston",
    "ladywood",
    "lozells",
    "handsworth",
    "small heath",
    "saltley",
    "bordesley",
)


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


def _lsoa_column(df: pd.DataFrame) -> str | None:
    for name in ("lsoa21cd", "LSOA21CD", "lsoa_code"):
        if name in df.columns:
            return name
    return None


def _synthetic_training_frame(n: int = 400) -> pd.DataFrame:
    """Labelled in-memory frame when real merged / flagged data are unavailable.

    SYNTHETIC DATA — not real ONS LSOA codes or EPC certificates.
    """
    rng = np.random.default_rng(SEED)
    letters = list("ABCDEFG")
    rows: list[dict[str, Any]] = []
    for i in range(n):
        current_i = int(rng.integers(2, 7))
        potential_i = max(0, current_i - int(rng.integers(1, 4)))
        floor_area = float(rng.uniform(45, 180))
        rooms = max(1, int(round(floor_area / rng.uniform(18, 35))))
        flagged = bool(rng.random() < 0.12)
        rows.append(
            {
                "lsoa21cd": f"SYNTH_E010{i:05d}",
                "lsoa21nm": f"Birmingham demo LSOA {i + 1:02d} (SYNTHETIC DATA)",
                "current_energy_efficiency": float(rng.uniform(20, 75)),
                "potential_energy_efficiency": float(rng.uniform(55, 95)),
                "current_energy_rating": letters[current_i],
                "potential_energy_rating": letters[potential_i],
                "total_floor_area": floor_area,
                "number_habitable_rooms": rooms,
                "heating_cost_current": float(rng.uniform(400, 2200)),
                "imd_decile": int(rng.integers(1, 11)),
                "income_score": float(rng.uniform(0.04, 0.55)),
                "imd_score": float(rng.uniform(5, 45)),
                "quality_flag": flagged,
                "quality_confidence": float(rng.uniform(0.2, 1.0) if flagged else 1.0),
                "data_source_label": "SYNTHETIC DATA",
            }
        )
    frame = pd.DataFrame(rows)
    frame[TARGET_COLUMN] = compute_priority_target(frame)
    return frame


def _minimal_epc_imd_frame() -> pd.DataFrame:
    """EPC + IMD join when full merge (Census) fails — read-only data loaders."""
    from retrofittrust.data.load_epc import load_epc
    from retrofittrust.data.load_imd import load_imd
    from retrofittrust.data.targets import add_priority_score

    epc = load_epc()
    imd = load_imd().drop_duplicates(subset=["lsoa21cd"], keep="first")
    merged = epc.merge(imd, on="lsoa21cd", how="left", suffixes=("", "_imd"))
    return add_priority_score(merged)


def _frame_has_target(df: pd.DataFrame) -> bool:
    if TARGET_COLUMN in df.columns and pd.to_numeric(df[TARGET_COLUMN], errors="coerce").notna().any():
        return True
    sample = df.head(min(200, len(df)))
    try:
        compute_priority_target(sample)
        return True
    except ValueError:
        return False


def _is_usable_training_frame(df: pd.DataFrame) -> bool:
    if df.empty:
        return False
    has_lsoa = _lsoa_column(df) is not None
    imd_hints = any(
        c in df.columns
        for c in (
            "imd_score",
            "income_score",
            "imd_income_score",
            "imd_decile",
            "income_decile",
            TARGET_COLUMN,
        )
    )
    if not has_lsoa and not imd_hints:
        return False
    return _frame_has_target(df)


def _try_enrich_flagged_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Join LSOA (postcode lookup) and IMD when quality output omits them."""
    out = df.copy()
    lsoa_col = _lsoa_column(out)

    if lsoa_col is None and "postcode" in out.columns:
        from retrofittrust.config import DATA_EXTERNAL

        for lookup_name in ("postcode_lsoa_lookup.csv", "pcd_lsoa_lookup.csv"):
            lookup_path = DATA_EXTERNAL / lookup_name
            if not lookup_path.exists():
                continue
            lookup = pd.read_csv(lookup_path, low_memory=False)
            pc_col = next(
                (c for c in lookup.columns if c.lower() in {"postcode", "pcd", "pcds"}),
                None,
            )
            lsoa_lookup_col = next(
                (c for c in lookup.columns if "lsoa" in c.lower()),
                None,
            )
            if pc_col is None or lsoa_lookup_col is None:
                continue
            slim = lookup[[pc_col, lsoa_lookup_col]].drop_duplicates()
            slim.columns = ["postcode_key", "lsoa21cd"]
            out["_postcode_key"] = (
                out["postcode"].astype(str).str.upper().str.replace(" ", "", regex=False)
            )
            slim["postcode_key"] = (
                slim["postcode_key"].astype(str).str.upper().str.replace(" ", "", regex=False)
            )
            out = out.merge(slim, left_on="_postcode_key", right_on="postcode_key", how="left")
            out = out.drop(columns=[c for c in ("_postcode_key", "postcode_key") if c in out.columns])
            lsoa_col = _lsoa_column(out)
            break

    if lsoa_col is not None and lsoa_col != "lsoa21cd":
        out = out.rename(columns={lsoa_col: "lsoa21cd"})

    if TARGET_COLUMN not in out.columns and "lsoa21cd" in out.columns:
        from retrofittrust.data.load_imd import load_imd
        from retrofittrust.data.targets import add_priority_score

        imd = load_imd().drop_duplicates(subset=["lsoa21cd"], keep="first")
        out = out.merge(imd, on="lsoa21cd", how="left", suffixes=("", "_imd"))
        out = add_priority_score(out)

    return out


def load_training_frame(
    *,
    flagged_path: Path | str | None = None,
    processed_dir: Path | str | None = None,
) -> tuple[pd.DataFrame, str]:
    """Resolve training data with a logged fallback chain."""
    processed_dir = Path(processed_dir or DATA_PROCESSED)
    flagged_path = Path(flagged_path) if flagged_path is not None else processed_dir / "quality_flagged.parquet"

    if flagged_path.exists():
        df = (
            pd.read_parquet(flagged_path)
            if flagged_path.suffix == ".parquet"
            else pd.read_csv(flagged_path)
        )
        try:
            enriched = _try_enrich_flagged_frame(df)
        except Exception as exc:
            logger.warning("Could not enrich quality_flagged (%s)", exc)
            enriched = df
        if _is_usable_training_frame(enriched):
            return enriched, f"quality_flagged:{flagged_path.name}"
        logger.warning(
            "quality_flagged at %s lacks LSOA/target after enrichment — skipping.",
            flagged_path,
        )

    for name in ("merged_lsoa.parquet", "merged_with_priority.parquet", "merged_lsoa.csv"):
        path = processed_dir / name
        if path.exists():
            df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
            return df, f"processed:{name}"

    interim = processed_dir.parent / "interim" / "merged_with_priority.parquet"
    if interim.exists():
        return pd.read_parquet(interim), f"interim:{interim.name}"

    try:
        from retrofittrust.data.pipeline import load_merged_dataset

        return load_merged_dataset(processed_dir), "pipeline:load_merged_dataset"
    except (FileNotFoundError, ValueError, ImportError) as exc:
        logger.warning("load_merged_dataset unavailable (%s)", exc)

    try:
        from retrofittrust.data.merge import build_merged_dataset
        from retrofittrust.data.targets import add_priority_score

        merged = build_merged_dataset(save_interim=False)
        return add_priority_score(merged), "merge:build_merged_dataset"
    except Exception as exc:
        logger.warning("Full merge failed (%s); trying EPC+IMD only.", exc)

    try:
        return _minimal_epc_imd_frame(), "fallback:epc_imd_only"
    except Exception as exc:
        logger.warning("EPC+IMD fallback failed (%s); using synthetic frame.", exc)

    logger.warning(
        "Using labelled SYNTHETIC DATA for ranking training — not real Birmingham records."
    )
    return _synthetic_training_frame(), "synthetic_fallback"


def aggregate_to_lsoa(df: pd.DataFrame) -> pd.DataFrame:
    """Mean numeric features and first non-null name per LSOA."""
    lsoa_col = _lsoa_column(df)
    if lsoa_col is None:
        return df.copy()
    if lsoa_col != "lsoa21cd":
        df = df.rename(columns={lsoa_col: "lsoa21cd"})

    if df["lsoa21cd"].nunique() == len(df):
        return df.copy()

    numeric = df.select_dtypes(include=[np.number, "bool", "boolean"]).columns.tolist()
    agg: dict[str, Any] = {c: "mean" for c in numeric if c != "lsoa21cd"}
    for flag_col in ("quality_flag", "quality_flag_union", "anomaly_flag", "low_confidence"):
        if flag_col in df.columns:
            agg[flag_col] = "max"

    grouped = df.groupby("lsoa21cd", as_index=False).agg(agg)
    for name_col in ("lsoa21nm", "lsoa_name"):
        if name_col in df.columns:
            names = df.groupby("lsoa21cd")[name_col].agg(
                lambda s: s.dropna().astype(str).iloc[0] if s.dropna().size else ""
            )
            grouped[name_col] = grouped["lsoa21cd"].map(names)
    return grouped


def _format_top_factors(
    shap_values: np.ndarray,
    feature_names: list[str],
    *,
    top_k: int = 3,
) -> str:
    series = pd.Series(shap_values, index=feature_names)
    top = series.reindex(series.abs().sort_values(ascending=False).index).head(top_k)
    return "; ".join(f"{name} ({val:+.3f})" for name, val in top.items())


def _shap_top_factors_frame(
    artefact: dict[str, Any],
    df: pd.DataFrame,
    *,
    top_k: int = 3,
) -> pd.DataFrame:
    """Per-row top contributing SHAP features (TreeExplainer)."""
    import shap

    from .predict import unwrap_model

    estimator, feature_names, source_columns = unwrap_model(artefact)
    include_census = bool(artefact.get("include_census", True))
    X = to_model_matrix(
        df,
        feature_names,
        source_columns=source_columns,
        include_census=include_census,
    )
    explainer = shap.TreeExplainer(estimator)
    explanation = explainer(X)
    values = np.asarray(explanation.values)
    if values.ndim == 3:
        values = values[:, :, 0]

    factors = [
        _format_top_factors(values[i], list(X.columns), top_k=top_k)
        for i in range(len(X))
    ]
    out = pd.DataFrame(
        {
            "top_3_contributing_factors": factors,
        },
        index=df.index,
    )
    return out


def build_consumer_table(
    artefact: dict[str, Any],
    df: pd.DataFrame,
) -> pd.DataFrame:
    """LSOA-level table for dashboard and ledger consumption."""
    from .predict import score_records

    lsoa_col = _lsoa_column(df)
    if lsoa_col is None:
        raise ValueError("Training frame has no LSOA identifier column.")

    lsoa_df = aggregate_to_lsoa(df) if df[lsoa_col].nunique() < len(df) else df.copy()
    if lsoa_col != "lsoa21cd" and "lsoa21cd" not in lsoa_df.columns:
        lsoa_df = lsoa_df.rename(columns={lsoa_col: "lsoa21cd"})

    scored = score_records(lsoa_df, artefact)
    shap_factors = _shap_top_factors_frame(artefact, lsoa_df, top_k=3)

    consumer = pd.DataFrame(
        {
            "lsoa_code": scored["lsoa21cd"].astype(str),
            "priority_score": scored["predicted_priority"].astype("float64"),
            "top_3_contributing_factors": shap_factors["top_3_contributing_factors"].values,
            "confidence_notes": scored["caveat_note"].fillna("").astype(str),
        }
    )
    if "data_source_label" in lsoa_df.columns:
        synthetic_mask = lsoa_df["data_source_label"].astype(str).str.contains("SYNTHETIC", case=False)
        consumer.loc[synthetic_mask.values, "confidence_notes"] = (
            consumer.loc[synthetic_mask.values, "confidence_notes"]
            .where(
                consumer.loc[synthetic_mask.values, "confidence_notes"].str.len() > 0,
                "SYNTHETIC DATA — demo scores only.",
            )
        )
    consumer = consumer.sort_values("priority_score", ascending=False).reset_index(drop=True)
    return consumer


def weight_sensitivity_analysis(df: pd.DataFrame) -> dict[str, Any]:
    """Rank stability under alternative 0.6/0.4-style target weights (no new model)."""
    lsoa_df = aggregate_to_lsoa(df)
    lsoa_col = _lsoa_column(lsoa_df) or "lsoa21cd"
    if lsoa_col != "lsoa21cd":
        lsoa_df = lsoa_df.rename(columns={lsoa_col: "lsoa21cd"})

    default = compute_priority_target(
        lsoa_df, epc_gap_weight=EPC_GAP_WEIGHT, imd_income_weight=IMD_INCOME_WEIGHT
    )
    default_rank = default.rank(ascending=False, method="average")

    rows: list[dict[str, Any]] = []
    for epc_w, imd_w in SENSITIVITY_WEIGHTS:
        alt = compute_priority_target(lsoa_df, epc_gap_weight=epc_w, imd_income_weight=imd_w)
        alt_rank = alt.rank(ascending=False, method="average")
        spearman = float(default_rank.corr(alt_rank, method="spearman"))
        top_n = min(10, len(lsoa_df))
        default_top = set(default.nlargest(top_n).index)
        alt_top = set(alt.nlargest(top_n).index)
        overlap = len(default_top & alt_top) / top_n if top_n else 0.0
        rows.append(
            {
                "epc_gap_weight": epc_w,
                "imd_income_weight": imd_w,
                "spearman_rank_vs_default": round(spearman, 4),
                f"top_{top_n}_overlap_fraction": round(overlap, 4),
                "mean_score_shift": round(float((alt - default).abs().mean()), 4),
            }
        )

    return {
        "default_weights": {"epc_gap": EPC_GAP_WEIGHT, "imd_income": IMD_INCOME_WEIGHT},
        "note": (
            "Sensitivity on the composite target formula only — not retraining "
            "separate model families. Spearman compares LSOA rank order."
        ),
        "scenarios": rows,
    }


def face_validity_check(
    consumer: pd.DataFrame,
    source_df: pd.DataFrame,
) -> dict[str, Any]:
    """Sanity-check scores against IMD deprivation and known Birmingham area names."""
    lsoa_df = aggregate_to_lsoa(source_df)
    lsoa_col = _lsoa_column(lsoa_df) or "lsoa21cd"
    if lsoa_col != "lsoa21cd":
        lsoa_df = lsoa_df.rename(columns={lsoa_col: "lsoa21cd"})

    merged = consumer.merge(
        lsoa_df[["lsoa21cd"] + [c for c in ("imd_decile", "income_score", "imd_score", "lsoa21nm") if c in lsoa_df.columns]],
        left_on="lsoa_code",
        right_on="lsoa21cd",
        how="left",
    )

    notes: list[str] = []
    if "imd_decile" in merged.columns:
        deprived = merged.nsmallest(10, "imd_decile")
        median_score = float(merged["priority_score"].median())
        deprived_in_top_half = int((deprived["priority_score"] >= median_score).sum())
        notes.append(
            f"Of the 10 most IMD-deprived LSOAs (lowest decile), "
            f"{deprived_in_top_half} sit at or above the cohort median priority score "
            f"({median_score:.3f})."
        )
    elif "income_score" in merged.columns:
        deprived = merged.nlargest(10, "income_score")
        top_quartile = merged["priority_score"] >= merged["priority_score"].quantile(0.75)
        overlap = int(deprived["lsoa_code"].isin(merged.loc[top_quartile, "lsoa_code"]).sum())
        notes.append(
            f"{overlap}/10 highest income-deprivation LSOAs appear in the top quartile "
            "of priority scores (face-validity proxy)."
        )

    if "lsoa21nm" in merged.columns:
        name_lower = merged["lsoa21nm"].fillna("").astype(str).str.lower()
        for hint in HIGH_DEPRIVATION_NAME_HINTS:
            subset = merged[name_lower.str.contains(hint, na=False)]
            if subset.empty:
                continue
            median_score = float(subset["priority_score"].median())
            overall_median = float(merged["priority_score"].median())
            notes.append(
                f"LSOAs matching '{hint}' (n={len(subset)}): median priority "
                f"{median_score:.3f} vs cohort median {overall_median:.3f}."
            )

    return {
        "cohort_lsoa_count": int(len(consumer)),
        "priority_score_median": float(consumer["priority_score"].median()),
        "priority_score_max": float(consumer["priority_score"].max()),
        "low_confidence_count": int(
            consumer["confidence_notes"].astype(str).str.len().gt(0).sum()
        ),
        "sanity_notes": notes,
        "caveat": (
            "No ground-truth retrofit priority exists; checks are indicative only. "
            "IMD is area-level (ecological fallacy)."
        ),
    }


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


def plot_cv_metrics_figure(
    metrics: dict[str, Any],
    out_dir: Path | str,
    *,
    filename: str = "03_cv_metrics.png",
) -> Path:
    """Bar figure of LightGBM CV RMSE / MAE / R² (plus RF baseline if present)."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    dest = out_path / filename

    lgbm_rmse = float(metrics.get("cv_rmse") or 0.0)
    lgbm_mae = float(metrics.get("cv_mae") or 0.0)
    lgbm_r2 = float(metrics.get("cv_r2") or 0.0)
    rf_rmse = metrics.get("baseline_rf_cv_rmse")
    rf_r2 = metrics.get("baseline_rf_cv_r2")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    error_labels = ["RMSE", "MAE"]
    error_lgbm = [lgbm_rmse, lgbm_mae]
    x = np.arange(len(error_labels))
    width = 0.35 if rf_rmse is not None else 0.55
    axes[0].bar(x - (width / 2 if rf_rmse is not None else 0), error_lgbm, width, label="LightGBM", color="#1f4e79")
    if rf_rmse is not None:
        axes[0].bar(x + width / 2, [float(rf_rmse), np.nan], width, label="Random Forest", color="#7a9bb8")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(error_labels)
    axes[0].set_ylabel("Score (priority units, 0–1)")
    axes[0].set_title("Cross-validated error")
    axes[0].legend(frameon=False)
    rmse_std = metrics.get("cv_rmse_std")
    if rmse_std is not None:
        axes[0].errorbar(
            x[0] - (width / 2 if rf_rmse is not None else 0),
            lgbm_rmse,
            yerr=float(rmse_std),
            fmt="none",
            ecolor="black",
            capsize=4,
        )

    r2_labels = ["LightGBM"]
    r2_vals = [lgbm_r2]
    r2_colors = ["#1f4e79"]
    if rf_r2 is not None:
        r2_labels.append("Random Forest")
        r2_vals.append(float(rf_r2))
        r2_colors.append("#7a9bb8")
    axes[1].bar(r2_labels, r2_vals, color=r2_colors, width=0.55)
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_ylabel("R²")
    axes[1].set_title("Cross-validated R²")
    for i, val in enumerate(r2_vals):
        axes[1].text(i, val + 0.02, f"{val:.4f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Checkpoint 3 — LightGBM ranking CV metrics (SEED=42)", fontsize=12)
    fig.tight_layout()
    fig.savefig(dest, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return dest


def plot_weight_sensitivity_figure(
    sensitivity: dict[str, Any],
    out_dir: Path | str,
    *,
    filename: str = "03_weight_sensitivity.png",
) -> Path:
    """Bar figure of rank stability under alternative target weights."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    dest = out_path / filename

    scenarios = sensitivity.get("scenarios") or []
    labels = [
        f"{s.get('epc_gap_weight')}/{s.get('imd_income_weight')}"
        for s in scenarios
    ]
    spearman = [float(s.get("spearman_rank_vs_default") or 0.0) for s in scenarios]
    overlap_key = next(
        (k for s in scenarios for k in s if k.startswith("top_") and k.endswith("_overlap_fraction")),
        "top_10_overlap_fraction",
    )
    overlap = [float(s.get(overlap_key) or 0.0) for s in scenarios]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    ax.bar(x - width / 2, spearman, width, label="Spearman vs default ranks", color="#1f4e79")
    ax.bar(x + width / 2, overlap, width, label="Top-10 LSOA overlap", color="#c47b2b")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("EPC-gap / IMD-income weights")
    ax.set_ylabel("Agreement with default (0.6 / 0.4)")
    ax.set_title("Target-weight sensitivity (formula only — not retrained families)")
    ax.legend(frameon=False, loc="lower right")
    ax.axhline(1.0, color="#888888", linewidth=0.8, linestyle="--")
    fig.tight_layout()
    fig.savefig(dest, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return dest


def write_ranking_numbers_markdown(
    metrics: dict[str, Any],
    sensitivity: dict[str, Any],
    out_dir: Path | str,
    *,
    filename: str = "03_ranking_numbers.md",
) -> Path:
    """Write a British-English CV metrics note for the dissertation appendix."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    dest = out_path / filename

    cv_rmse = metrics.get("cv_rmse")
    cv_rmse_std = metrics.get("cv_rmse_std")
    cv_mae = metrics.get("cv_mae")
    cv_r2 = metrics.get("cv_r2")
    cv_r2_std = metrics.get("cv_r2_std")
    rf_rmse = metrics.get("baseline_rf_cv_rmse")
    rf_r2 = metrics.get("baseline_rf_cv_r2")
    flagged = metrics.get("flagged_rate_in_train")
    downweighted = metrics.get("flagged_downweighted", True)

    scenarios = sensitivity.get("scenarios") or []
    scenario_lines = []
    for s in scenarios:
        overlap_key = next(
            (k for k in s if k.startswith("top_") and k.endswith("_overlap_fraction")),
            "top_10_overlap_fraction",
        )
        scenario_lines.append(
            f"- EPC {s.get('epc_gap_weight')} / IMD {s.get('imd_income_weight')}: "
            f"Spearman {s.get('spearman_rank_vs_default')}, "
            f"top-10 overlap {s.get(overlap_key)}, "
            f"mean |score shift| {s.get('mean_score_shift')}"
        )

    rmse_txt = f"{cv_rmse:.6f}" if isinstance(cv_rmse, (int, float)) else "n/a"
    if isinstance(cv_rmse_std, (int, float)):
        rmse_txt = f"{cv_rmse:.6f} ± {cv_rmse_std:.6f}"
    r2_txt = f"{cv_r2:.6f}" if isinstance(cv_r2, (int, float)) else "n/a"
    if isinstance(cv_r2_std, (int, float)) and isinstance(cv_r2, (int, float)):
        r2_txt = f"{cv_r2:.6f} ± {cv_r2_std:.6f}"
    mae_txt = f"{cv_mae:.6f}" if isinstance(cv_mae, (int, float)) else "n/a"
    flagged_txt = f"{flagged * 100:.1f}%" if isinstance(flagged, (int, float)) else "n/a"

    body = f"""# Checkpoint 3 — LightGBM ranking numbers

Source: `{metrics.get("data_source", "unknown")}`. Seed = 42. British English.

## Training sample

| Item | Value |
| --- | --- |
| Input rows | {metrics.get("input_rows", "n/a")} |
| Training rows (non-missing target) | {metrics.get("train_rows", "n/a")} |
| Features | {metrics.get("n_features", "n/a")} |
| LSOA consumer export | {metrics.get("lsoa_export_rows", "n/a")} |
| Flagged rate in training frame | {flagged_txt} |
| Flagged rows down-weighted (not deleted) | {"yes" if downweighted else "no"} |
| Flagged sample weight | {metrics.get("flagged_sample_weight", 0.35)} |
| Target | {metrics.get("target_formula", "0.6 EPC gap + 0.4 IMD income")} |

## 5-fold CV (LightGBM)

| Metric | LightGBM | Random Forest baseline |
| --- | --- | --- |
| RMSE | {rmse_txt} | {f"{rf_rmse:.6f}" if isinstance(rf_rmse, (int, float)) else "n/a"} |
| MAE | {mae_txt} | — |
| R² | {r2_txt} | {f"{rf_r2:.6f}" if isinstance(rf_r2, (int, float)) else "n/a"} |

High R² is expected: the composite target is a weighted function of the EPC efficiency gap and IMD income need, both of which are present (or recoverable) in the feature matrix. These metrics show that LightGBM reconstructs the *constructed* priority score, not an independently observed retrofit outcome.

## Target-weight sensitivity

Default weights: EPC-gap 0.6, IMD-income 0.4. Sensitivity changes the *formula* only — it does not retrain a separate model family.

{chr(10).join(scenario_lines) if scenario_lines else "- No sensitivity scenarios recorded."}

## Limitations (dissertation)

- **SHAP correlated features.** TreeExplainer assumes feature independence. Floor area, habitable-room count and heating cost are correlated in EPC data; IMD score and the income domain are also correlated. Importance can be misattributed within those groups.
- **Ecological fallacy.** IMD and Census attributes are LSOA-level and must not be read as household facts.
- **EPC performance gap.** Modelled SAP points are not metered kWh.
- **No ground-truth priority.** Face-validity checks are indicative only.
"""
    dest.write_text(body.strip() + "\n", encoding="utf-8")
    return dest


def write_ranking_number_reports(
    metrics: dict[str, Any],
    sensitivity: dict[str, Any],
    reports_dir: Path | str,
) -> dict[str, str]:
    """Save 03_cv_metrics.png, 03_weight_sensitivity.png and 03_ranking_numbers.md."""
    reports_dir = Path(reports_dir)
    cv_fig = plot_cv_metrics_figure(metrics, reports_dir)
    sens_fig = plot_weight_sensitivity_figure(sensitivity, reports_dir)
    md_path = write_ranking_numbers_markdown(metrics, sensitivity, reports_dir)
    return {
        "cv_metrics_figure": str(cv_fig),
        "weight_sensitivity_figure": str(sens_fig),
        "ranking_numbers_md": str(md_path),
    }


def run_ranking_training(
    *,
    flagged_path: Path | str | None = None,
    processed_dir: Path | str | None = None,
    models_dir: Path | str | None = None,
    reports_dir: Path | str | None = None,
    seed: int = SEED,
) -> dict[str, Any]:
    """Checkpoint 3 entry point expected by scripts/03_train_ranking_model.py."""
    from retrofittrust.modeling.explain import (
        plot_global_bar,
        plot_global_beeswarm,
        plot_local_waterfall,
    )

    _set_seeds(seed)
    processed_dir = Path(processed_dir or DATA_PROCESSED)
    models_dir = Path(models_dir or MODELS_DIR)
    reports_dir = Path(reports_dir or MODELS_DIR.parent / "reports" / "figures")
    processed_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    df, data_source = load_training_frame(
        flagged_path=flagged_path,
        processed_dir=processed_dir,
    )
    logger.info("Training data source: %s (%s rows)", data_source, f"{len(df):,}")

    model_path = models_dir / "ranking_model.joblib"
    alias_path = models_dir / "lgbm_ranker.joblib"

    result = train_ranking_model(
        df,
        use_sample_weights=True,
        compare_baseline=True,
        model_path=model_path,
        verbose=False,
    )
    artefact = result["artefact"]
    joblib.dump(artefact, alias_path)

    lsoa_export_df = aggregate_to_lsoa(df)
    consumer = build_consumer_table(artefact, lsoa_export_df)
    consumer.to_csv(RETROFIT_SCORES_CSV, index=False)
    consumer.to_parquet(RETROFIT_SCORES_PARQUET, index=False)
    logger.info(
        "Wrote consumer scores: %s (%s LSOAs)",
        RETROFIT_SCORES_CSV,
        len(consumer),
    )

    shap_beeswarm: Path | None = None
    shap_bar: Path | None = None
    shap_waterfall: Path | None = None
    if len(lsoa_export_df) > 0:
        shap_beeswarm = plot_global_beeswarm(
            artefact,
            lsoa_export_df,
            out_dir=reports_dir,
            filename="shap_beeswarm.png",
        )
        shap_bar = plot_global_bar(
            artefact,
            lsoa_export_df,
            out_dir=reports_dir,
            filename="shap_bar.png",
        )
        lsoa_key = _lsoa_column(lsoa_export_df) or "lsoa21cd"
        lsoa_reset = lsoa_export_df.reset_index(drop=True)
        top_lsoa = str(consumer.iloc[0]["lsoa_code"]) if len(consumer) else None
        row_index = 0
        if top_lsoa is not None:
            hits = lsoa_reset.index[
                lsoa_reset[lsoa_key].astype(str) == top_lsoa
            ]
            if len(hits):
                row_index = int(hits[0])
        waterfall_name = (
            f"shap_waterfall_{top_lsoa}.png"
            if top_lsoa
            else "shap_waterfall_sample.png"
        )
        shap_waterfall = plot_local_waterfall(
            artefact,
            lsoa_reset,
            row_index=row_index,
            out_dir=reports_dir,
            filename=waterfall_name,
        )

    sensitivity = weight_sensitivity_analysis(df)
    face_validity = face_validity_check(consumer, df)
    for path, payload in (
        (WEIGHT_SENSITIVITY_JSON, sensitivity),
        (FACE_VALIDITY_JSON, face_validity),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    flagged_rate = float(quality_flag_mask(df).mean()) if len(df) else 0.0
    cv = result["cv_metrics"]
    baseline_cv = (result.get("baseline") or {}).get("cv_metrics") or {}

    metrics: dict[str, Any] = {
        "data_source": data_source,
        "input_rows": len(df),
        "train_rows": result["n_samples"],
        "lsoa_export_rows": len(consumer),
        "flagged_rate_in_train": flagged_rate,
        "flagged_downweighted": True,
        "flagged_sample_weight": 0.35,
        "cv_rmse": cv.get("rmse_mean"),
        "cv_rmse_std": cv.get("rmse_std"),
        "cv_mae": cv.get("mae_mean"),
        "cv_r2": cv.get("r2_mean"),
        "cv_r2_std": cv.get("r2_std"),
        "baseline_rf_cv_rmse": baseline_cv.get("rmse_mean"),
        "baseline_rf_cv_r2": baseline_cv.get("r2_mean"),
        "n_features": result["n_features"],
        "shap_beeswarm": str(shap_beeswarm) if shap_beeswarm else None,
        "shap_bar": str(shap_bar) if shap_bar else None,
        "shap_waterfall_saved": str(shap_waterfall) if shap_waterfall else None,
        "retrofit_scores_csv": str(RETROFIT_SCORES_CSV),
        "retrofit_scores_parquet": str(RETROFIT_SCORES_PARQUET),
        "target_formula": (
            "0.6 * normalised_epc_gap + 0.4 * normalised_imd_income (config defaults)"
        ),
        "model_path": str(model_path),
        "weight_sensitivity_json": str(WEIGHT_SENSITIVITY_JSON),
        "face_validity_json": str(FACE_VALIDITY_JSON),
        "face_validity_notes": face_validity.get("sanity_notes", []),
    }
    figure_paths = write_ranking_number_reports(metrics, sensitivity, reports_dir)
    metrics.update(figure_paths)
    RANKING_METRICS_JSON.parent.mkdir(parents=True, exist_ok=True)
    RANKING_METRICS_JSON.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return metrics
