"""SHAP TreeExplainer helpers for the LightGBM ranking model.

Global plots: beeswarm and bar (overall feature importance).
Local plots: waterfall per property / LSOA, written to ``reports/figures/``.

Limitation (do not treat SHAP values as causal attributions):
    TreeExplainer Shapley values assume feature independence when
    attributing credit. Floor area, habitable-room count and heating cost
    are correlated in EPC data, as are IMD score and income-domain score,
    and Census tenure/heating shares that sum toward 100%. Importance can
    therefore be misattributed among correlated groups. This is a known
    limitation to discuss in the dissertation, not something this module
    attempts to "fix".
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from ..config import REPORTS_FIGURES, SEED as CONFIG_SEED
from .features import to_model_matrix
from .predict import load_ranking_model, unwrap_model

SEED = 42
assert SEED == CONFIG_SEED

# Beeswarm on the full Birmingham EPC extract would be slow; subsample.
GLOBAL_PLOT_SAMPLE = 2000
DEFAULT_MAX_DISPLAY = 20


def _ensure_figures_dir(out_dir: Path | str | None = None) -> Path:
    path = Path(out_dir) if out_dir is not None else REPORTS_FIGURES
    path.mkdir(parents=True, exist_ok=True)
    return path


def _resolve_estimator(
    model: Any | None,
    model_path: Path | str | None,
) -> tuple[Any, list[str] | None, list[str] | None, bool]:
    if model is None:
        artefact = load_ranking_model(model_path)
        estimator, names, source = unwrap_model(artefact)
        return estimator, names, source, bool(artefact.get("include_census", True))
    estimator, names, source = unwrap_model(model)
    include_census = True
    if isinstance(model, dict):
        include_census = bool(model.get("include_census", True))
    return estimator, names, source, include_census


def _design_matrix(
    df: pd.DataFrame,
    feature_names: list[str] | None,
    source_columns: list[str] | None,
    include_census: bool,
) -> pd.DataFrame:
    return to_model_matrix(
        df,
        feature_names,
        source_columns=source_columns,
        include_census=include_census,
    )


def build_explainer(model: Any) -> shap.TreeExplainer:
    """Return a TreeExplainer for a fitted LightGBM estimator or artefact."""
    estimator, _, _ = unwrap_model(model)
    # tree_path_dependent is the default for trees and does not require a
    # background sample. Interventional SHAP would need one and still would
    # not solve the correlated-feature attribution issue noted above.
    return shap.TreeExplainer(estimator)


def shap_explanation(
    model: Any | None,
    df: pd.DataFrame,
    *,
    model_path: Path | str | None = None,
    max_rows: int | None = None,
) -> tuple[shap.Explanation, pd.DataFrame]:
    """Compute a SHAP Explanation aligned to the model design matrix."""
    estimator, feature_names, source_columns, include_census = _resolve_estimator(
        model, model_path
    )
    X = _design_matrix(df, feature_names, source_columns, include_census)
    if max_rows is not None and len(X) > max_rows:
        X = X.sample(n=max_rows, random_state=SEED)
    explainer = shap.TreeExplainer(estimator)
    explanation = explainer(X)
    return explanation, X


def shap_values_frame(
    model: Any | None,
    df: pd.DataFrame,
    *,
    model_path: Path | str | None = None,
) -> pd.DataFrame:
    """Per-row SHAP values as a DataFrame (one column per feature)."""
    explanation, X = shap_explanation(model, df, model_path=model_path)
    values = np.asarray(explanation.values)
    if values.ndim == 3:
        # Defensive: some SHAP versions wrap a trailing output dimension.
        values = values[:, :, 0]
    return pd.DataFrame(values, index=X.index, columns=list(X.columns))


def plot_global_beeswarm(
    model: Any | None,
    df: pd.DataFrame,
    *,
    model_path: Path | str | None = None,
    out_dir: Path | str | None = None,
    max_display: int = DEFAULT_MAX_DISPLAY,
    sample_size: int = GLOBAL_PLOT_SAMPLE,
    filename: str = "shap_beeswarm.png",
) -> Path:
    """Save a SHAP beeswarm (global importance + value distribution)."""
    explanation, _ = shap_explanation(
        model, df, model_path=model_path, max_rows=sample_size
    )
    out_path = _ensure_figures_dir(out_dir) / filename
    plt.figure()
    shap.plots.beeswarm(explanation, show=False, max_display=max_display)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
    return out_path


def plot_global_bar(
    model: Any | None,
    df: pd.DataFrame,
    *,
    model_path: Path | str | None = None,
    out_dir: Path | str | None = None,
    max_display: int = DEFAULT_MAX_DISPLAY,
    sample_size: int = GLOBAL_PLOT_SAMPLE,
    filename: str = "shap_bar.png",
) -> Path:
    """Save a SHAP bar chart of mean |SHAP| (global feature importance)."""
    explanation, _ = shap_explanation(
        model, df, model_path=model_path, max_rows=sample_size
    )
    out_path = _ensure_figures_dir(out_dir) / filename
    plt.figure()
    shap.plots.bar(explanation, show=False, max_display=max_display)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
    return out_path


def _row_locator(
    df: pd.DataFrame,
    *,
    row_index: int | None,
    record_id: Any,
    id_column: str | None,
) -> tuple[int, Any]:
    """Return (positional index into df, label used in the filename)."""
    if record_id is not None and id_column:
        if id_column not in df.columns:
            raise KeyError(f"id_column {id_column!r} is not in the DataFrame.")
        matches = df.index[df[id_column].astype(str) == str(record_id)]
        if len(matches) == 0:
            raise ValueError(f"No row with {id_column}={record_id!r}.")
        loc = df.index.get_loc(matches[0])
        if isinstance(loc, slice):
            loc = int(loc.start)
        elif isinstance(loc, np.ndarray):
            loc = int(np.flatnonzero(loc)[0])
        return int(loc), record_id
    if row_index is None:
        row_index = 0
    if row_index < 0 or row_index >= len(df):
        raise IndexError(f"row_index {row_index} is outside [0, {len(df)}).")
    label = record_id if record_id is not None else df.index[row_index]
    return int(row_index), label


def plot_local_waterfall(
    model: Any | None,
    df: pd.DataFrame,
    *,
    row_index: int | None = 0,
    record_id: Any = None,
    id_column: str | None = None,
    model_path: Path | str | None = None,
    out_dir: Path | str | None = None,
    max_display: int = 12,
    filename: str | None = None,
) -> Path:
    """Save a SHAP waterfall plot for one property or LSOA record.

    This is the local explanation the digital-twin dashboard will embed
    when a user selects a row.
    """
    pos, label = _row_locator(
        df, row_index=row_index, record_id=record_id, id_column=id_column
    )
    estimator, feature_names, source_columns, include_census = _resolve_estimator(
        model, model_path
    )
    X = _design_matrix(df, feature_names, source_columns, include_census)
    row = X.iloc[pos : pos + 1]
    explainer = shap.TreeExplainer(estimator)
    explanation = explainer(row)

    safe_label = str(label).replace("/", "_").replace("\\", "_").replace(" ", "_")
    out_name = filename or f"shap_waterfall_{safe_label}.png"
    out_path = _ensure_figures_dir(out_dir) / out_name

    plt.figure()
    shap.plots.waterfall(explanation[0], show=False, max_display=max_display)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
    return out_path


def local_contributions(
    model: Any | None,
    df: pd.DataFrame,
    *,
    row_index: int = 0,
    model_path: Path | str | None = None,
    top_k: int = 10,
) -> dict[str, Any]:
    """Structured local SHAP breakdown (for a future ``/explain`` endpoint)."""
    estimator, feature_names, source_columns, include_census = _resolve_estimator(
        model, model_path
    )
    X = _design_matrix(df, feature_names, source_columns, include_census)
    row = X.iloc[row_index : row_index + 1]
    explainer = shap.TreeExplainer(estimator)
    explanation = explainer(row)
    values = np.asarray(explanation.values).reshape(-1)
    base = float(np.asarray(explanation.base_values).reshape(-1)[0])
    contrib = (
        pd.Series(values, index=list(X.columns))
        .sort_values(key=np.abs, ascending=False)
        .head(top_k)
    )
    return {
        "base_value": base,
        "prediction": base + float(values.sum()),
        "top_contributions": [
            {"feature": str(name), "shap_value": float(val)}
            for name, val in contrib.items()
        ],
    }


def explain_lsoa(
    *,
    lsoa_code: str,
    model_path: Path | str | None = None,
    top_k: int = 10,
) -> dict[str, Any]:
    """Local SHAP explanation for one LSOA — integration demo entry point."""
    from retrofittrust.modeling.predict import _load_scoring_frame

    df = _load_scoring_frame([lsoa_code])
    col = "lsoa21cd"
    match = df[df[col].astype(str).str.upper() == str(lsoa_code).upper()]
    if match.empty:
        raise ValueError(f"LSOA {lsoa_code!r} not found in processed data.")
    result = local_contributions(None, match.reset_index(drop=True), model_path=model_path, top_k=top_k)
    result["lsoa_code"] = str(lsoa_code)
    result["method"] = "shap_tree_explainer"
    result["caveat"] = (
        "SHAP TreeExplainer assumes feature independence; correlated EPC fields "
        "may have misattributed importance. LSOA IMD is area-level (ecological fallacy)."
    )
    return result
