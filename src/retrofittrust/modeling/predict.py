"""Score properties or LSOAs with the trained ranking model.

Flagged / low-confidence records from the quality screen are *not* dropped.
They receive a predicted priority plus an explicit caveat column so the
dashboard can surface uncertainty (CURSOR_BUILD_SPEC §4).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ..config import MODELS_DIR, SEED as CONFIG_SEED
from .features import quality_flag_mask, to_model_matrix

SEED = 42
assert SEED == CONFIG_SEED

DEFAULT_MODEL_PATH = MODELS_DIR / "ranking_model.joblib"
ALT_MODEL_PATH = MODELS_DIR / "lgbm_ranker.joblib"


def _resolve_model_path(model_path: Path | str | None = None) -> Path:
    if model_path is not None:
        return Path(model_path)
    if DEFAULT_MODEL_PATH.exists():
        return DEFAULT_MODEL_PATH
    if ALT_MODEL_PATH.exists():
        return ALT_MODEL_PATH
    return DEFAULT_MODEL_PATH

CAVEAT_NOTE = (
    "Low-confidence score: this record was flagged by the data-quality screen "
    "(anomaly / quarantine). It was not silently excluded; interpret the "
    "ranking with caution."
)


def load_ranking_model(model_path: Path | str | None = None) -> dict[str, Any]:
    """Load the joblib artefact saved by :func:`train_ranking_model`."""
    path = _resolve_model_path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Ranking model not found at {path}. Train it first with "
            "train_ranking_model()."
        )
    artefact = joblib.load(path)
    if not isinstance(artefact, dict) or "model" not in artefact:
        raise ValueError(
            f"Expected a dict artefact with a 'model' key at {path}, "
            f"got {type(artefact)!r}."
        )
    return artefact


def unwrap_model(
    model_or_artefact: Any,
) -> tuple[Any, list[str] | None, list[str] | None]:
    """Return ``(estimator, feature_names, source_columns)`` from a model or artefact."""
    if isinstance(model_or_artefact, dict) and "model" in model_or_artefact:
        names = model_or_artefact.get("feature_names")
        source = model_or_artefact.get("raw_feature_columns")
        if names is not None:
            names = list(names)
        if source is not None:
            source = list(source)
        return model_or_artefact["model"], names, source
    return model_or_artefact, None, None


def score_records(
    df: pd.DataFrame,
    model: Any | None = None,
    *,
    model_path: Path | str | None = None,
    include_census: bool | None = None,
) -> pd.DataFrame:
    """Predict retrofit priority for each row of a preprocess DataFrame.

    Returns a copy of ``df`` with:
    - ``predicted_priority`` — higher = more urgent
    - ``priority_rank`` — 1 = highest priority in this batch
    - ``low_confidence_caveat`` — True for quality-flagged rows
    - ``caveat_note`` — human-readable warning, else empty string
    """
    if df.empty:
        out = df.copy()
        out["predicted_priority"] = pd.Series(dtype="float64")
        out["priority_rank"] = pd.Series(dtype="Int64")
        out["low_confidence_caveat"] = pd.Series(dtype="boolean")
        out["caveat_note"] = pd.Series(dtype="string")
        return out

    if model is None:
        artefact = load_ranking_model(model_path)
    elif isinstance(model, (str, Path)):
        artefact = load_ranking_model(model)
    elif isinstance(model, dict) and "model" in model:
        artefact = model
    else:
        artefact = {"model": model, "feature_names": None}

    estimator, feature_names, source_columns = unwrap_model(artefact)
    if include_census is None:
        include_census = bool(artefact.get("include_census", True))

    X = to_model_matrix(
        df,
        feature_names=feature_names,
        source_columns=source_columns,
        include_census=include_census,
    )
    preds = np.asarray(estimator.predict(X), dtype="float64")

    out = df.copy()
    out["predicted_priority"] = preds
    out["priority_rank"] = (
        out["predicted_priority"].rank(ascending=False, method="min").astype(int)
    )
    caveat = quality_flag_mask(df)
    out["low_confidence_caveat"] = caveat
    if "inference_caveat" in df.columns:
        program2_notes = df["inference_caveat"].fillna("").astype(str)
        out["caveat_note"] = np.where(
            program2_notes.str.len() > 0,
            program2_notes,
            np.where(caveat.to_numpy(), CAVEAT_NOTE, ""),
        )
    elif "low_confidence_caveat" in df.columns and df["low_confidence_caveat"].dtype == object:
        program2_notes = df["low_confidence_caveat"].fillna("").astype(str)
        out["caveat_note"] = np.where(
            program2_notes.str.len() > 0,
            program2_notes,
            np.where(caveat.to_numpy(), CAVEAT_NOTE, ""),
        )
    else:
        out["caveat_note"] = np.where(caveat.to_numpy(), CAVEAT_NOTE, "")
    return out


def _load_scoring_frame(lsoa_codes: list[str]) -> pd.DataFrame:
    """Load processed rows for the requested LSOA codes."""
    from retrofittrust.config import DATA_PROCESSED

    for name in ("quality_flagged.parquet", "merged_lsoa.parquet"):
        path = DATA_PROCESSED / name
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        col = next(
            (c for c in ("lsoa21cd", "LSOA21CD", "lsoa_code") if c in df.columns),
            None,
        )
        if col is None:
            continue
        if col != "lsoa21cd":
            df = df.rename(columns={col: "lsoa21cd"})
        codes = {str(c).upper() for c in lsoa_codes}
        subset = df[df["lsoa21cd"].astype(str).str.upper().isin(codes)]
        if not subset.empty:
            return subset
    raise FileNotFoundError(
        "No processed dataset with matching LSOA codes. "
        "Run scripts/01–03 or place quality_flagged.parquet in data/processed/."
    )


def rank_lsoas(
    *,
    lsoa_codes: list[str],
    model_path: Path | str | None = None,
) -> dict[str, Any]:
    """Rank a cohort of LSOAs — used by the integration demo and FastAPI glue."""
    df = _load_scoring_frame(lsoa_codes)
    scored = score_records(df, model_path=model_path)
    if "lsoa21cd" not in scored.columns:
        raise ValueError("Scoring frame missing lsoa21cd column.")

    agg = (
        scored.groupby("lsoa21cd", as_index=False)
        .agg(
            score=("predicted_priority", "mean"),
            low_confidence=("low_confidence_caveat", "max"),
        )
        .sort_values("score", ascending=False)
    )
    rankings = [
        {
            "lsoa": str(row.lsoa21cd),
            "lsoa_code": str(row.lsoa21cd),
            "score": float(row.score),
            "low_confidence": bool(row.low_confidence),
        }
        for row in agg.itertuples(index=False)
    ]
    top = rankings[0] if rankings else {"lsoa": lsoa_codes[0], "score": 0.0}
    return {
        "rankings": rankings,
        "top_lsoa": top["lsoa"],
        "top_score": float(top["score"]),
    }
