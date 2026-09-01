"""Load processed LSOA features and (if present) the ranking model.

Program 1/2 artefacts are produced by sibling modules. This layer is defensive:
missing files fall back to a labelled synthetic demo frame so Programs 3–4 can
still demonstrate the twin → AI → ledger → write-back loop.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from retrofittrust.config import (
    DATA_EXTERNAL,
    DATA_PROCESSED,
    DATA_RAW,
    DEMO_COHORT_LSOA_COUNT,
    EPC_GAP_WEIGHT,
    IMD_INCOME_WEIGHT,
    MODELS_DIR,
    SEED,
)

# SHAP TreeExplainer assumes feature independence; correlated EPC fields
# (floor area, room count, heating cost) can have misattributed importance.
SHAP_CAVEAT = (
    "SHAP TreeExplainer assumes feature independence and can misattribute "
    "importance among correlated features (floor area, room count, heating cost). "
    "LSOA IMD is area-level — ecological fallacy if read as household deprivation. "
    "EPC modelled-vs-metered gap is ~16% (gas) / ~31% (electric); use for ranking, "
    "not absolute consumption prediction."
)

EPC_TO_NEED = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}

_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "lsoa21cd": ("lsoa21cd", "LSOA21CD", "lsoa_code", "lsoa", "LSOA11CD"),
    "lsoa21nm": ("lsoa21nm", "LSOA21NM", "lsoa_name", "LSOA11NM"),
    "imd_decile": ("imd_decile", "IMDDecile", "imd2019_decile", "imd_2025_decile"),
    "income_score": (
        "income_score",
        "IncScore",
        "imd_income_score",
        "income_deprivation",
        "imd_income",
    ),
    "epc_current": (
        "epc_current",
        "current_energy_rating",
        "CURRENT_ENERGY_RATING",
        "mean_epc_current",
        "epc_mean_current",
    ),
    "epc_potential": (
        "epc_potential",
        "potential_energy_rating",
        "POTENTIAL_ENERGY_RATING",
        "mean_epc_potential",
        "epc_mean_potential",
    ),
    "epc_gap": ("epc_gap", "rating_gap", "sap_gap"),
    "priority_score": ("priority_score", "retrofit_priority", "y_hat", "score"),
    "anomaly_flag": ("anomaly_flag", "is_anomaly", "quality_flag", "flagged"),
    "n_properties": ("n_properties", "property_count", "n_epc"),
}


def _first_existing(path_candidates: list[Path]) -> Path | None:
    for path in path_candidates:
        if path.exists():
            return path
    return None


def retrofit_scores_path() -> Path | None:
    """LSOA-level LightGBM scores written by Program 1 (`scripts/03_train_ranking_model.py`)."""
    return _first_existing(
        [
            DATA_PROCESSED / "retrofit_scores.parquet",
            DATA_PROCESSED / "retrofit_scores.csv",
        ]
    )


def processed_frame_path() -> Path | None:
    scores = retrofit_scores_path()
    if scores is not None:
        return scores
    return _first_existing(
        [
            DATA_PROCESSED / "quality_flagged.parquet",
            DATA_PROCESSED / "merged_lsoa.parquet",
            DATA_PROCESSED / "lsoa_priority.parquet",
            DATA_PROCESSED / "birmingham_merged.parquet",
            DATA_PROCESSED / "merged.parquet",
            DATA_PROCESSED / "features.parquet",
            DATA_PROCESSED / "merged_lsoa.csv",
            DATA_PROCESSED / "lsoa_priority.csv",
        ]
    )


def geojson_path() -> Path | None:
    return _first_existing(
        [
            DATA_EXTERNAL / "lsoa_birmingham.geojson",
            DATA_EXTERNAL / "LSOA_2021_BGC_Birmingham.geojson",
            DATA_EXTERNAL / "lsoa_2021_bgc_birmingham.geojson",
            DATA_RAW / "lsoa_birmingham.geojson",
        ]
    )


def model_path() -> Path | None:
    for candidate in (
        MODELS_DIR / "ranking_model.joblib",
        MODELS_DIR / "lgbm_ranker.joblib",
    ):
        if candidate.exists():
            return candidate
    return None


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for canonical, aliases in _COLUMN_ALIASES.items():
        if canonical in out.columns:
            continue
        for alias in aliases:
            if alias in out.columns:
                out[canonical] = out[alias]
                break
    if "lsoa21cd" not in out.columns:
        raise ValueError(f"No LSOA code column in frame. Columns: {list(out.columns)}")
    out["lsoa21cd"] = out["lsoa21cd"].astype(str)
    return out


def epc_to_need(value: Any) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if isinstance(value, (int, float, np.floating, np.integer)):
        return float(value)
    return float(EPC_TO_NEED.get(str(value).strip().upper()[:1], np.nan))


def add_composite_score(df: pd.DataFrame) -> pd.DataFrame:
    """Default composite pending RJ's exact target formula (spec open item).

    priority = 0.6 * normalised EPC gap (room to improve) + 0.4 * IMD income need.
    """
    out = df.copy()
    if "epc_current_need" not in out.columns:
        if "epc_current" in out.columns:
            out["epc_current_need"] = out["epc_current"].map(epc_to_need)
        else:
            out["epc_current_need"] = np.nan
    if "epc_potential_need" not in out.columns:
        if "epc_potential" in out.columns:
            out["epc_potential_need"] = out["epc_potential"].map(epc_to_need)
        else:
            out["epc_potential_need"] = np.nan

    if "epc_gap" not in out.columns:
        out["epc_gap"] = out["epc_current_need"] - out["epc_potential_need"]

    def _minmax(series: pd.Series) -> pd.Series:
        s = pd.to_numeric(series, errors="coerce")
        lo, hi = s.min(), s.max()
        if pd.isna(lo) or pd.isna(hi) or hi == lo:
            return pd.Series(np.zeros(len(s)), index=s.index)
        return (s - lo) / (hi - lo)

    if "imd_need" not in out.columns:
        if "income_score" in out.columns:
            out["imd_need"] = _minmax(out["income_score"])
        elif "imd_decile" in out.columns:
            # Decile 1 = most deprived → invert so higher = more need
            out["imd_need"] = (8 - pd.to_numeric(out["imd_decile"], errors="coerce")) / 7.0
        else:
            out["imd_need"] = 0.5

    gap_norm = _minmax(out["epc_gap"])
    composite = EPC_GAP_WEIGHT * gap_norm.fillna(0) + IMD_INCOME_WEIGHT * out["imd_need"].fillna(0)
    if "priority_score" not in out.columns or out["priority_score"].isna().all():
        out["priority_score"] = composite
    else:
        out["priority_score"] = pd.to_numeric(out["priority_score"], errors="coerce").fillna(composite)
    return out


def _synthetic_lsoa_frame(n: int = DEMO_COHORT_LSOA_COUNT) -> pd.DataFrame:
    """In-memory fallback when Program 1 outputs are not yet on disk.

    SYNTHETIC DATA — demo LSOAs only; not real ONS codes or EPC records.
    """
    rng = np.random.default_rng(SEED)
    n = max(n, DEMO_COHORT_LSOA_COUNT)
    letters = list("ABCDEFG")
    rows = []
    for i in range(n):
        current_i = int(rng.integers(3, 7))  # D–G
        potential_i = max(0, current_i - int(rng.integers(1, 4)))
        rows.append(
            {
                "lsoa21cd": f"SYNTH_E010{i:05d}",
                "lsoa21nm": f"Birmingham demo LSOA {i + 1:02d} (SYNTHETIC DATA)",
                "imd_decile": int(rng.integers(1, 9)),
                "income_score": float(rng.uniform(0.05, 0.45)),
                "epc_current": letters[current_i],
                "epc_potential": letters[potential_i],
                "n_properties": int(rng.integers(80, 250)),
                "anomaly_flag": int(rng.random() < 0.15),
                "is_synthetic_fallback": True,
            }
        )
    return add_composite_score(pd.DataFrame(rows))


def aggregate_to_lsoa(df: pd.DataFrame) -> pd.DataFrame:
    if df["lsoa21cd"].nunique() == len(df):
        return df
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    agg: dict[str, Any] = {c: "mean" for c in numeric if c != "n_properties"}
    if "n_properties" in df.columns:
        agg["n_properties"] = "sum"
    if "anomaly_flag" in df.columns:
        agg["anomaly_flag"] = "mean"
    name_col = "lsoa21nm" if "lsoa21nm" in df.columns else None
    grouped = df.groupby("lsoa21cd", as_index=False).agg(agg)
    if name_col:
        names = df.groupby("lsoa21cd")[name_col].agg(
            lambda s: s.dropna().astype(str).iloc[0] if s.dropna().size else ""
        )
        grouped[name_col] = grouped["lsoa21cd"].map(names)
    if "epc_current" in df.columns and df["epc_current"].dtype == object:
        mode_epc = df.groupby("lsoa21cd")["epc_current"].agg(
            lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0]
        )
        grouped["epc_current"] = grouped["lsoa21cd"].map(mode_epc)
    if "epc_potential" in df.columns and df["epc_potential"].dtype == object:
        mode_epc = df.groupby("lsoa21cd")["epc_potential"].agg(
            lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0]
        )
        grouped["epc_potential"] = grouped["lsoa21cd"].map(mode_epc)
    return grouped


# Property-level merged_lsoa.parquet is ~476k rows — never aggregate it on the
# API hot path when LSOA scores or the quality-screen sample already exist.
_MAX_PROPERTY_ROWS_TO_AGGREGATE = 50_000
_LSOA_CONTEXT_COLUMNS = (
    "lsoa21cd",
    "lsoa21nm",
    "imd_decile",
    "income_score",
    "imd_income_score",
    "epc_current",
    "epc_potential",
    "current_energy_rating",
    "potential_energy_rating",
    "epc_gap",
    "n_properties",
    "anomaly_flag",
    "flagged_union",
    "flagged_consensus",
    "quality_flag",
    "retrofit_priority_score",
)


def load_retrofit_scores() -> tuple[pd.DataFrame, str] | None:
    """Load Program 1 consumer scores (one row per LSOA) when present."""
    path = retrofit_scores_path()
    if path is None:
        return None
    raw = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    if raw is None or len(raw) == 0:
        return None
    frame = normalise_columns(raw)
    if "confidence_notes" in frame.columns and "anomaly_flag" not in frame.columns:
        notes = frame["confidence_notes"].fillna("").astype(str)
        frame["anomaly_flag"] = notes.str.contains(
            "low-confidence|flagged|quarantine", case=False, regex=True
        ).astype(int)
    return add_composite_score(frame), str(path)


def _slim_lsoa_context(df: pd.DataFrame) -> pd.DataFrame:
    """Keep identity / ranking context only — avoid aggregating 1,000+ recon columns."""
    present = [c for c in _LSOA_CONTEXT_COLUMNS if c in df.columns]
    if "lsoa21cd" not in present and "lsoa_code" not in df.columns:
        return df
    slim = df[present].copy() if present else df
    return add_composite_score(aggregate_to_lsoa(normalise_columns(slim)))


def _overlay_saved_scores(context: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """Prefer LightGBM consumer scores where they exist; keep LSOA context columns."""
    score_cols = [c for c in scores.columns if c != "lsoa21cd"]
    ctx = context.drop(columns=[c for c in score_cols if c in context.columns], errors="ignore")
    return ctx.merge(scores, on="lsoa21cd", how="outer")


def load_lsoa_frame(*, allow_synthetic_fallback: bool = True) -> tuple[pd.DataFrame, str]:
    """Return (LSOA-level frame, source label).

    Preference order (fast → slow): retrofit_scores, quality-flagged sample,
    then the full merged extract. Never silently drop flagged records.
    """
    scores_pack = None
    try:
        scores_pack = load_retrofit_scores()
    except (FileNotFoundError, ValueError, ImportError, TypeError, OSError):
        scores_pack = None

    context: pd.DataFrame | None = None
    context_source = ""

    try:
        from retrofittrust.quality.screen import load_flagged_dataset

        flagged = load_flagged_dataset(DATA_PROCESSED)
        if flagged is not None and 0 < len(flagged) <= _MAX_PROPERTY_ROWS_TO_AGGREGATE:
            context = _slim_lsoa_context(pd.DataFrame(flagged))
            context_source = "quality.screen.load_flagged_dataset"
    except (NotImplementedError, FileNotFoundError, ValueError, ImportError, TypeError, OSError):
        pass

    if scores_pack is not None:
        scores_df, scores_src = scores_pack
        if context is not None and "lsoa21cd" in context.columns:
            return _overlay_saved_scores(context, scores_df), f"{scores_src}+{context_source}"
        return scores_df, scores_src

    if context is not None:
        return context, context_source

    try:
        from retrofittrust.data import load_merged_dataset

        merged = load_merged_dataset(DATA_PROCESSED)
        if merged is not None and len(merged):
            work = pd.DataFrame(merged)
            if len(work) > _MAX_PROPERTY_ROWS_TO_AGGREGATE:
                # Full Birmingham extract: do not mean-aggregate 476k × 176 on the API path.
                # Distinct LSOA codes are enough for cohort selection; scores overlay above.
                codes = work
                if "lsoa21cd" not in codes.columns and "lsoa_code" in codes.columns:
                    codes = codes.rename(columns={"lsoa_code": "lsoa21cd"})
                uniq = codes.drop_duplicates(subset=["lsoa21cd"]).copy()
                keep = [c for c in _LSOA_CONTEXT_COLUMNS if c in uniq.columns]
                frame = add_composite_score(normalise_columns(uniq[keep] if keep else uniq))
                return frame, "data.pipeline.load_merged_dataset (deduped, not full aggregate)"
            frame = add_composite_score(aggregate_to_lsoa(normalise_columns(work)))
            return frame, "data.pipeline.load_merged_dataset"
    except (NotImplementedError, FileNotFoundError, ValueError, ImportError, TypeError, OSError):
        pass

    path = processed_frame_path()
    if path is not None:
        raw = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        if len(raw) > _MAX_PROPERTY_ROWS_TO_AGGREGATE:
            raw = raw.head(_MAX_PROPERTY_ROWS_TO_AGGREGATE)
        frame = add_composite_score(aggregate_to_lsoa(normalise_columns(raw)))
        return frame, str(path)

    if not allow_synthetic_fallback:
        raise FileNotFoundError(
            f"No merged LSOA file in {DATA_PROCESSED}. Run scripts/01_ingest_and_merge.py first."
        )
    return _synthetic_lsoa_frame(), "synthetic_fallback"


@lru_cache(maxsize=1)
def load_model_bundle() -> dict[str, Any] | None:
    path = model_path()
    if path is None:
        return None
    import joblib

    loaded = joblib.load(path)
    if isinstance(loaded, dict):
        return loaded
    return {"model": loaded, "feature_names": None, "preprocessor": None}


def model_feature_matrix(df: pd.DataFrame, bundle: dict[str, Any]) -> pd.DataFrame:
    names = bundle.get("feature_names") or bundle.get("feature_cols")
    model = bundle.get("model")
    preprocessor = bundle.get("preprocessor")
    if preprocessor is not None:
        transformed = preprocessor.transform(df)
        if names is None:
            names = [f"f{i}" for i in range(transformed.shape[1])]
        return pd.DataFrame(transformed, columns=list(names), index=df.index)
    if names:
        missing = [c for c in names if c not in df.columns]
        work = df.copy()
        if missing:
            extra = pd.DataFrame(0.0, index=work.index, columns=missing)
            work = pd.concat([work, extra], axis=1)
        return work[list(names)]
    if model is not None and hasattr(model, "feature_name_"):
        names = list(model.feature_name_)
        work = df.copy()
        missing = [c for c in names if c not in work.columns]
        if missing:
            extra = pd.DataFrame(0.0, index=work.index, columns=missing)
            work = pd.concat([work, extra], axis=1)
        return work[names]
    numeric = df.select_dtypes(include=[np.number])
    return numeric


def predict_priority(df: pd.DataFrame, bundle: dict[str, Any] | None) -> np.ndarray:
    if bundle is None or bundle.get("model") is None:
        return pd.to_numeric(df["priority_score"], errors="coerce").fillna(0).to_numpy()
    model = bundle["model"]
    X = model_feature_matrix(df, bundle)
    if hasattr(model, "predict"):
        return np.asarray(model.predict(X), dtype=float)
    raise TypeError("Loaded ranking artefact has no predict()")


def explain_from_saved_scores(lsoa21cd: str, *, top_n: int = 10) -> dict[str, Any] | None:
    """Local explanation from Program 1's stored top contributing factors.

    Used when TreeExplainer cannot run but ``retrofit_scores`` exists.
    """
    pack = load_retrofit_scores()
    if pack is None:
        return None
    frame, source = pack
    row = frame[frame["lsoa21cd"].astype(str) == str(lsoa21cd)].head(1)
    if row.empty:
        return None
    rec = row.iloc[0]
    pred = float(pd.to_numeric(rec.get("priority_score"), errors="coerce") or 0.0)
    features: list[dict[str, Any]] = []
    raw_factors = rec.get("top_3_contributing_factors")
    if isinstance(raw_factors, str) and raw_factors.strip():
        for part in raw_factors.split(";"):
            token = part.strip()
            if not token:
                continue
            name = token
            shap_val = 0.0
            if "(" in token and token.endswith(")"):
                name, _, rest = token.rpartition("(")
                name = name.strip()
                try:
                    shap_val = float(rest.rstrip(")").replace("+", "").strip())
                except ValueError:
                    shap_val = 0.0
            features.append({"feature": name, "value": None, "shap_value": shap_val})
    features = features[:top_n]
    if not features:
        return None
    return {
        "base_value": 0.0,
        "prediction": pred,
        "features": features,
        "method": "saved_retrofit_scores",
        "caveat": SHAP_CAVEAT + f" Factors stored in {source}.",
        "model_loaded": model_path() is not None,
        "lsoa21cd": str(lsoa21cd),
    }


def shap_for_row(
    df_row: pd.DataFrame,
    bundle: dict[str, Any] | None,
    *,
    top_n: int = 10,
) -> dict[str, Any]:
    """Local explanation for one LSOA/property row.

    Limitation: TreeExplainer can misattribute among correlated features.
    """
    if bundle is None or bundle.get("model") is None:
        return _composite_explanation(df_row.iloc[0], top_n=top_n)

    try:
        import shap
    except ImportError:
        return _composite_explanation(df_row.iloc[0], top_n=top_n, extra="shap package not installed")

    model = bundle["model"]
    X = model_feature_matrix(df_row, bundle)
    try:
        explainer = shap.TreeExplainer(model)
        explanation = explainer(X)
        values = np.asarray(explanation.values)
        base = float(np.asarray(explanation.base_values).reshape(-1)[0])
        if values.ndim > 1:
            values = values[0]
        pred = float(base + np.sum(values))
        features = [
            {"feature": str(col), "value": _safe_value(X.iloc[0, i]), "shap_value": float(values[i])}
            for i, col in enumerate(X.columns)
        ]
        features.sort(key=lambda f: abs(f["shap_value"]), reverse=True)
        return {
            "base_value": base,
            "prediction": pred,
            "features": features[:top_n],
            "method": "shap_tree_explainer",
            "caveat": SHAP_CAVEAT,
            "model_loaded": True,
        }
    except Exception as exc:  # noqa: BLE001 — PoC fallback to composite weights
        return _composite_explanation(df_row.iloc[0], top_n=top_n, extra=str(exc))


def _safe_value(val: Any) -> float | str | None:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    if isinstance(val, (int, float, np.floating, np.integer)):
        return float(val)
    return str(val)


def _composite_explanation(row: pd.Series, *, top_n: int, extra: str = "") -> dict[str, Any]:
    gap = float(pd.to_numeric(row.get("epc_gap"), errors="coerce") or 0.0)
    imd = float(pd.to_numeric(row.get("imd_need"), errors="coerce") or 0.0)
    pred = float(pd.to_numeric(row.get("priority_score"), errors="coerce") or 0.0)
    features = [
        {
            "feature": "epc_gap",
            "value": gap,
            "shap_value": EPC_GAP_WEIGHT * gap,
        },
        {
            "feature": "imd_income_need",
            "value": imd,
            "shap_value": IMD_INCOME_WEIGHT * imd,
        },
    ]
    note = (
        "Composite weights (model artefact not loaded). "
        + SHAP_CAVEAT
        + (f" Fallback reason: {extra}" if extra else "")
    )
    return {
        "base_value": 0.0,
        "prediction": pred,
        "features": features[:top_n],
        "method": "composite_weights",
        "caveat": note,
        "model_loaded": False,
    }
