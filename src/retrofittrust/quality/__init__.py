"""Program 2 — data-quality / anomaly screening (PyOD AutoEncoder + IForest).

Runs **before** LightGBM training as a preprocessing gate. Flagged records are
quarantined, never silently deleted. The modelling pipeline should:

1. Fit on the preprocessed numeric matrix (one-hot + imputed + standardised).
2. Pass ``sample_weight`` into LightGBM for flagged rows.
3. At inference, surface ``inference_caveat`` / low ``data_quality_confidence``.

Expected operational flagged rate: sanity-check against ~27–60% (EPC quality-flag
literature: ~27% of records show at least one quality flag; true error rate
estimated 36–62%). See :data:`LITERATURE_FLAG_RATE_RANGE`.

``ShallowAutoEncoder`` / ``DataQualityScreen`` require ``pyod`` and PyTorch
(PyOD 2.0.2 is PyTorch-backed). Flag and injection helpers import without them.
"""

from retrofittrust.config import SEED
from retrofittrust.quality.evaluation import (
    INJECTION_KINDS,
    InjectionResult,
    RecallReport,
    StabilityReport,
    TuneResult,
    evaluate_recall,
    inject_synthetic_anomalies,
    literature_sanity,
    multi_seed_stability,
    tune_from_injection,
)
from retrofittrust.quality.flags import (
    CLEAN_SAMPLE_WEIGHT,
    CONFIDENCE_FLAG_COL,
    CONFIDENCE_LABEL_COL,
    CONFIDENCE_SCORE_COL,
    FLAGGED_SAMPLE_WEIGHT,
    LITERATURE_FLAG_RATE_HIGH,
    LITERATURE_FLAG_RATE_LOW,
    LITERATURE_FLAG_RATE_RANGE,
    NEVER_SILENTLY_DELETE,
    attach_quality_flags,
    inference_caveat,
    sample_weights_for_lightgbm,
    sanity_check_flag_rate,
)

__all__ = [
    "SEED",
    "inject_synthetic_anomalies",
    "evaluate_recall",
    "multi_seed_stability",
    "tune_from_injection",
    "literature_sanity",
    "InjectionResult",
    "RecallReport",
    "StabilityReport",
    "TuneResult",
    "INJECTION_KINDS",
    "attach_quality_flags",
    "sample_weights_for_lightgbm",
    "inference_caveat",
    "sanity_check_flag_rate",
    "FLAGGED_SAMPLE_WEIGHT",
    "CLEAN_SAMPLE_WEIGHT",
    "NEVER_SILENTLY_DELETE",
    "LITERATURE_FLAG_RATE_RANGE",
    "LITERATURE_FLAG_RATE_LOW",
    "LITERATURE_FLAG_RATE_HIGH",
    "CONFIDENCE_FLAG_COL",
    "CONFIDENCE_SCORE_COL",
    "CONFIDENCE_LABEL_COL",
]

try:
    from retrofittrust.quality.autoencoder import ShallowAutoEncoder, infer_hidden_dims
    from retrofittrust.quality.ensemble import (
        AnomalyEnsemble,
        DataQualityScreen,
        ThresholdResult,
        choose_threshold,
        evt_gpd_threshold,
        mean_sigma_threshold,
    )
    from retrofittrust.quality.screen import (
        load_flagged_dataset,
        load_quality_screen,
        load_screening_input,
        run_quality_screen,
    )

    __all__ += [
        "ShallowAutoEncoder",
        "infer_hidden_dims",
        "AnomalyEnsemble",
        "DataQualityScreen",
        "ThresholdResult",
        "choose_threshold",
        "evt_gpd_threshold",
        "mean_sigma_threshold",
        "run_quality_screen",
        "load_screening_input",
        "load_flagged_dataset",
        "load_quality_screen",
    ]
except ImportError:
    ShallowAutoEncoder = None  # type: ignore[misc, assignment]
    infer_hidden_dims = None  # type: ignore[misc, assignment]
    AnomalyEnsemble = None  # type: ignore[misc, assignment]
    DataQualityScreen = None  # type: ignore[misc, assignment]
    ThresholdResult = None  # type: ignore[misc, assignment]
    choose_threshold = None  # type: ignore[misc, assignment]
    evt_gpd_threshold = None  # type: ignore[misc, assignment]
    mean_sigma_threshold = None  # type: ignore[misc, assignment]
    run_quality_screen = None  # type: ignore[misc, assignment]
    load_flagged_dataset = None  # type: ignore[misc, assignment]
    load_quality_screen = None  # type: ignore[misc, assignment]
    load_screening_input = None  # type: ignore[misc, assignment]
