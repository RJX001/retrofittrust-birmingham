"""Program 1 — LightGBM ranking model with SHAP explanations.

Census tenure (TS054) and central-heating shares are included as direct
features. Quality-flagged records are down-weighted in training and scored
at inference with an explicit low-confidence caveat (never silently deleted).
"""

from .explain import (
    build_explainer,
    local_contributions,
    plot_global_bar,
    plot_global_beeswarm,
    plot_local_waterfall,
    shap_values_frame,
    explain_lsoa,
)
from .features import (
    TARGET_COLUMN,
    compute_priority_target,
    compute_sample_weights,
    prepare_feature_frame,
    select_feature_columns,
)
from .predict import load_ranking_model, rank_lsoas, score_records
from .train import (
    DEFAULT_MODEL_PATH,
    build_consumer_table,
    load_training_frame,
    run_ranking_training,
    train_ranking_model,
)

SEED = 42

__all__ = [
    "SEED",
    "DEFAULT_MODEL_PATH",
    "TARGET_COLUMN",
    "train_ranking_model",
    "run_ranking_training",
    "load_training_frame",
    "build_consumer_table",
    "score_records",
    "rank_lsoas",
    "load_ranking_model",
    "select_feature_columns",
    "prepare_feature_frame",
    "compute_priority_target",
    "compute_sample_weights",
    "build_explainer",
    "plot_global_beeswarm",
    "plot_global_bar",
    "plot_local_waterfall",
    "shap_values_frame",
    "local_contributions",
    "explain_lsoa",
]
