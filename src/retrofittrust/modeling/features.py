"""Feature-column contract for the LightGBM ranking model.

Census 2021 tenure (TS054) and central-heating shares are included as
*direct* model features (CURSOR_BUILD_SPEC §9 open item: confirmed for
this PoC), alongside property-level EPC fields and LSOA IMD.

Limitations acknowledged here rather than hidden:
- IMD (and Census) attributes are LSOA-level. Treating them as if they
  describe a specific household is an ecological fallacy.
- Functions accept pandas DataFrames from the data/preprocess pipeline;
  they do not read ``data/raw`` themselves.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import EPC_GAP_WEIGHT, IMD_INCOME_WEIGHT

# Composite target column materialised by preprocess, or computed below.
TARGET_COLUMN = "retrofit_priority_score"

# Optional column: if present, used as LightGBM sample weights as-is.
SAMPLE_WEIGHT_COLUMN = "sample_weight"

# Quality-screen columns (program 2). Used for weights and inference caveats;
# they are NOT ranking features (quarantine-and-flag, never silent delete).
QUALITY_FLAG_COLUMNS = (
    "quality_flag",
    "quality_flag_union",
    "flagged_consensus",
    "flagged_union",
    "anomaly_flag",
    "low_confidence",
    "data_quality_flag",
    "quarantine_flag",
)
QUALITY_CONFIDENCE_COLUMN = "quality_confidence"
QUALITY_CONFIDENCE_ALIASES = (
    "quality_confidence",
    "data_quality_confidence",
)

# Identity / geography keys excluded from the feature matrix.
ID_COLUMNS = (
    "lmk_key",
    "property_id",
    "uprn",
    "address",
    "postcode",
    "postcode_clean",
    "lsoa_code",
    "lsoa21cd",
    "lsoa_name",
    "lsoa21nm",
    "lad_code",
    "lad_name",
    "local_authority",
    "geometry",
    "lat",
    "lon",
    "latitude",
    "longitude",
)

# Columns that must never enter X (target leakage, predictions, caveats).
_EXCLUDE_EXACT = frozenset(
    {
        TARGET_COLUMN,
        SAMPLE_WEIGHT_COLUMN,
        QUALITY_CONFIDENCE_COLUMN,
        "target",
        "y",
        "epc_gap",
        "epc_gap_norm",
        "imd_income_norm",
        "predicted_priority",
        "priority_rank",
        "low_confidence_caveat",
        "caveat_note",
        "shap_base_value",
        "ae_score",
        "iforest_score",
        "consensus_score",
        "union_score",
        "is_quality_screen_fallback",
        "data_quality_label",
        *ID_COLUMNS,
        *QUALITY_FLAG_COLUMNS,
    }
)

_EXCLUDE_PREFIXES = (
    "predicted_",
    "shap_",
    "caveat",
    "recon_err_",
)

# Preferred EPC numeric fields after snake_case preprocess.
# Correlated group (SHAP TreeExplainer caveat): floor area, habitable /
# heated rooms, and heating cost can share credit in local explanations.
EPC_NUMERIC_FEATURES = (
    "current_energy_efficiency",
    "potential_energy_efficiency",
    "total_floor_area",
    "number_habitable_rooms",
    "number_heated_rooms",
    "heating_cost_current",
    "heating_cost_potential",
    "co2_emissions_current",
    "co2_emissions_potential",
    "windows_energy_eff",
    "walls_energy_eff",
    "roof_energy_eff",
    "mainheat_energy_eff",
    "hot_water_energy_eff",
    "lighting_energy_eff",
    "floor_energy_eff",
    "multi_glaze_proportion",
    "extension_count",
    "low_energy_lighting",
    "number_open_fireplaces",
    "floor_height",
    "flat_storey_count",
)

# One-hot (or still-categorical) EPC prefixes.
EPC_CATEGORICAL_PREFIXES = (
    "current_energy_rating_",
    "potential_energy_rating_",
    "property_type_",
    "built_form_",
    "construction_age_band_",
    "main_fuel_",
    "mainheat_",
    "tenure_",  # EPC property tenure; Census uses census_tenure_ / ts054_
    "mains_gas_",
    "glazed_type_",
    "transaction_type_",
    "walls_",
    "roof_",
    "windows_",
)

# IMD 2025 — area-level; ecological-fallacy caveat applies.
# Correlated group (SHAP caveat): imd_score vs income_score / income domain.
IMD_FEATURES = (
    "imd_score",
    "imd_rank",
    "imd_decile",
    "income_score",
    "imd_income_score",
    "income_deprivation_score",
    "income_rank",
    "income_decile",
    "imd_income_rank",
    "imd_income_decile",
)

# Census TS054 tenure — included as direct features (percentages / shares).
CENSUS_TENURE_PREFIXES = (
    "census_tenure_",
    "ts054_",
    "tenure_owned_",
    "tenure_social_",
    "tenure_private_",
    "tenure_rent_",
    "tenure_shared_",
    "pct_owned_",
    "pct_social_rent",
    "pct_private_rent",
)

CENSUS_TENURE_EXACT = (
    "tenure_owned_outright_pct",
    "tenure_owned_mortgage_pct",
    "tenure_shared_ownership_pct",
    "tenure_social_rented_pct",
    "tenure_private_rented_pct",
    "tenure_rent_free_pct",
    "pct_owned_outright",
    "pct_owned_mortgage",
    "pct_social_rented",
    "pct_private_rented",
)

# Census central heating — included as direct features.
CENSUS_HEATING_PREFIXES = (
    "census_heating_",
    "central_heating_",
    "heating_mains_",
    "heating_electric_",
    "heating_oil_",
    "heating_solid_",
    "heating_other_",
    "heating_two_",
    "heating_none_",
    "pct_heating_",
    "pct_gas_heat",
    "pct_electric_heat",
    "pct_no_central_heat",
)

CENSUS_HEATING_EXACT = (
    "heating_mains_gas_pct",
    "heating_electric_pct",
    "heating_oil_pct",
    "heating_solid_fuel_pct",
    "heating_other_pct",
    "heating_two_or_more_pct",
    "heating_none_pct",
)

MISSINGNESS_SUFFIX = "_missing"
MISSINGNESS_PREFIX = "missing_"

# LightGBM rejects JSON-special characters in feature names.
_LGBM_NAME_BAD = str.maketrans({
    "{": "_",
    "}": "_",
    "[": "_",
    "]": "_",
    ":": "_",
    ",": "_",
    '"': "_",
    "'": "_",
    " ": "_",
    "/": "_",
    "\\": "_",
})


def _sanitize_feature_name(name: str) -> str:
    cleaned = str(name).translate(_LGBM_NAME_BAD)
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "feature"


def _sanitize_feature_frame(X: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with LightGBM-safe, unique column names."""
    mapping: dict[str, str] = {}
    used: set[str] = set()
    for col in X.columns:
        base = _sanitize_feature_name(col)
        candidate = base
        suffix = 1
        while candidate in used:
            candidate = f"{base}_{suffix}"
            suffix += 1
        mapping[col] = candidate
        used.add(candidate)
    if mapping and any(a != b for a, b in mapping.items()):
        return X.rename(columns=mapping)
    return X

# Assumed: flagged records keep a non-zero weight so they are not silently
# excluded (CURSOR_BUILD_SPEC §4 downstream handling). Matches quality.flags.
FLAGGED_SAMPLE_WEIGHT = 0.35
MIN_SAMPLE_WEIGHT = 0.1

_RATING_TO_BAND = {
    "A": 1,
    "B": 2,
    "C": 3,
    "D": 4,
    "E": 5,
    "F": 6,
    "G": 7,
}


def _is_excluded(name: str) -> bool:
    lowered = name.lower()
    if lowered in _EXCLUDE_EXACT or name in _EXCLUDE_EXACT:
        return True
    return any(lowered.startswith(p) for p in _EXCLUDE_PREFIXES)


def _has_prefix(name: str, prefixes: tuple[str, ...]) -> bool:
    lowered = name.lower()
    return any(lowered.startswith(p) for p in prefixes)


def select_feature_columns(
    df: pd.DataFrame,
    *,
    include_census: bool = True,
) -> list[str]:
    """Return ranking-feature column names present in ``df``.

    Census tenure and heating columns are always preferred when
    ``include_census`` is True. Remaining numeric preprocess outputs
    (including missingness indicators) are included unless they are
    identifiers, the target, or quality-screen flags.
    """
    columns = list(df.columns)
    selected: list[str] = []
    seen: set[str] = set()

    def _add(name: str) -> None:
        if name in seen or name not in df.columns or _is_excluded(name):
            return
        seen.add(name)
        selected.append(name)

    for name in EPC_NUMERIC_FEATURES:
        _add(name)
    for name in IMD_FEATURES:
        _add(name)

    if include_census:
        for name in CENSUS_TENURE_EXACT + CENSUS_HEATING_EXACT:
            _add(name)
        for name in columns:
            if _has_prefix(name, CENSUS_TENURE_PREFIXES + CENSUS_HEATING_PREFIXES):
                _add(name)

    for name in columns:
        if _has_prefix(name, EPC_CATEGORICAL_PREFIXES):
            _add(name)
        if name.lower().endswith(MISSINGNESS_SUFFIX) or name.lower().startswith(
            MISSINGNESS_PREFIX
        ):
            _add(name)

    # Fallback: remaining numeric columns from the preprocess frame.
    numeric_cols = df.select_dtypes(include=[np.number, "bool", "boolean"]).columns
    for name in numeric_cols:
        _add(name)

    # Leftover low-cardinality categoricals (if preprocess has not one-hot
    # encoded yet). High-cardinality strings (addresses) are already excluded.
    object_cols = df.select_dtypes(include=["object", "category", "string"]).columns
    for name in object_cols:
        if _is_excluded(name):
            continue
        nunique = df[name].nunique(dropna=True)
        if 1 < nunique <= 40:
            _add(name)

    if not selected:
        raise ValueError(
            "No ranking features found in the supplied DataFrame. "
            "The preprocess pipeline should expose EPC, IMD and Census columns."
        )
    return selected


def to_model_matrix(
    df: pd.DataFrame,
    feature_names: list[str] | None = None,
    *,
    source_columns: list[str] | None = None,
    include_census: bool = True,
) -> pd.DataFrame:
    """Build a numeric design matrix aligned to ``feature_names``.

    Leftover object columns are one-hot encoded. At inference, pass the
    training artefact's ``raw_feature_columns`` as ``source_columns`` and
    the persisted dummy-expanded list as ``feature_names``. Missing dummy
    columns are filled with 0. LightGBM handles residual NaNs natively —
    we do not re-impute here.
    """
    if (
        source_columns is None
        and feature_names is not None
        and all(name in df.columns for name in feature_names)
    ):
        X = df.loc[:, feature_names].copy()
        bool_cols = X.select_dtypes(include=["bool", "boolean"]).columns
        for name in bool_cols:
            X[name] = X[name].astype("float64")
        return _sanitize_feature_frame(X.apply(pd.to_numeric, errors="coerce").astype("float64"))

    if source_columns is not None:
        cols = [c for c in source_columns if c in df.columns]
    else:
        cols = select_feature_columns(df, include_census=include_census)

    if not cols and feature_names is None:
        raise ValueError("No feature columns available to build the model matrix.")

    X = df.loc[:, cols].copy() if cols else pd.DataFrame(index=df.index)

    bool_cols = X.select_dtypes(include=["bool", "boolean"]).columns
    for name in bool_cols:
        X[name] = X[name].astype("float64")

    cat_cols = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, dummy_na=False)

    X = X.apply(pd.to_numeric, errors="coerce")

    if feature_names is not None:
        X = X.reindex(columns=feature_names, fill_value=0.0)
    else:
        X = X.dropna(axis=1, how="all")

    X = _sanitize_feature_frame(X.astype("float64"))
    return X


def prepare_feature_frame(
    df: pd.DataFrame,
    feature_names: list[str] | None = None,
    *,
    source_columns: list[str] | None = None,
    include_census: bool = True,
) -> pd.DataFrame:
    """Public alias for :func:`to_model_matrix`."""
    return to_model_matrix(
        df,
        feature_names,
        source_columns=source_columns,
        include_census=include_census,
    )


def _first_present(df: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    lookup = {c.lower(): c for c in df.columns}
    for name in names:
        if name in df.columns:
            return name
        if name.lower() in lookup:
            return lookup[name.lower()]
    return None


def _minmax(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    lo = values.min()
    hi = values.max()
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series(0.5, index=series.index, dtype="float64")
    return (values - lo) / (hi - lo)


def _epc_gap(df: pd.DataFrame) -> pd.Series:
    """Need/benefit proxy: larger gap ⇒ more retrofit headroom.

    Prefers SAP-point gap (potential − current). Letter ratings map A=1…G=7
    so a worse current band yields a larger gap. EPC modelled-vs-metered
    performance gap means this is a *relative* ranking signal, not a
    prediction of metered kWh.
    """
    current_eff = _first_present(
        df, ("current_energy_efficiency", "current_energy_efficiency_sap")
    )
    potential_eff = _first_present(
        df, ("potential_energy_efficiency", "potential_energy_efficiency_sap")
    )
    if current_eff is not None and potential_eff is not None:
        current = pd.to_numeric(df[current_eff], errors="coerce")
        potential = pd.to_numeric(df[potential_eff], errors="coerce")
        return (potential - current).clip(lower=0)

    if current_eff is not None:
        current = pd.to_numeric(df[current_eff], errors="coerce")
        return (100.0 - current).clip(lower=0)

    current_rating = _first_present(
        df, ("current_energy_rating", "current_energy_rating_letter")
    )
    potential_rating = _first_present(
        df, ("potential_energy_rating", "potential_energy_rating_letter")
    )
    if current_rating is not None:
        current_band = (
            df[current_rating].astype(str).str.strip().str.upper().map(_RATING_TO_BAND)
        )
        if potential_rating is not None:
            potential_band = (
                df[potential_rating]
                .astype(str)
                .str.strip()
                .str.upper()
                .map(_RATING_TO_BAND)
            )
            return (current_band - potential_band).clip(lower=0)
        return (current_band - 1).clip(lower=0)

    raise ValueError(
        "Cannot compute EPC gap: need current_energy_efficiency (and ideally "
        "potential_energy_efficiency) or current_energy_rating."
    )


def _imd_income_need(df: pd.DataFrame) -> pd.Series:
    """Higher value = greater income-deprivation need.

    Income *score* rises with deprivation. Rank/decile 1 is most deprived,
    so those are inverted. Area-level only — ecological fallacy applies.
    """
    score_col = _first_present(
        df,
        (
            "income_score",
            "imd_income_score",
            "income_deprivation_score",
            "imd_income_domain_score",
        ),
    )
    if score_col is not None:
        return pd.to_numeric(df[score_col], errors="coerce")

    imd_score = _first_present(df, ("imd_score", "imd_average_score"))
    if imd_score is not None:
        return pd.to_numeric(df[imd_score], errors="coerce")

    decile_col = _first_present(
        df, ("income_decile", "imd_income_decile", "imd_decile")
    )
    if decile_col is not None:
        decile = pd.to_numeric(df[decile_col], errors="coerce")
        return 11.0 - decile  # decile 1 (most deprived) → 10

    rank_col = _first_present(df, ("income_rank", "imd_income_rank", "imd_rank"))
    if rank_col is not None:
        rank = pd.to_numeric(df[rank_col], errors="coerce")
        return -rank  # rank 1 = most deprived → largest need after min-max

    raise ValueError(
        "Cannot compute IMD income need: expected income_score, imd_score, "
        "income_decile or income_rank."
    )


def compute_priority_target(
    df: pd.DataFrame,
    *,
    epc_gap_weight: float = EPC_GAP_WEIGHT,
    imd_income_weight: float = IMD_INCOME_WEIGHT,
) -> pd.Series:
    """Composite retrofit priority score in [0, 1] (higher = more urgent).

    Default weights (``config.EPC_GAP_WEIGHT`` / ``IMD_INCOME_WEIGHT``) are
    0.6 on the normalised EPC efficiency gap and 0.4 on normalised IMD
    income deprivation. Exact weights remain a CURSOR_BUILD_SPEC §9 open
    item; these are the project defaults, not a nationally calibrated index.

    Min-max normalisation is computed on the supplied frame. Prefer
    materialising ``retrofit_priority_score`` in preprocess for a stable
    target across train/serve splits.
    """
    gap = _minmax(_epc_gap(df))
    income_need = _minmax(_imd_income_need(df))
    total = float(epc_gap_weight + imd_income_weight)
    if total <= 0:
        raise ValueError("Target weights must sum to a positive value.")
    score = (epc_gap_weight * gap + imd_income_weight * income_need) / total
    score.name = TARGET_COLUMN
    return score.astype("float64")


def resolve_target(
    df: pd.DataFrame,
    target: pd.Series | str | None = None,
) -> pd.Series:
    """Return the regression target aligned to ``df.index``."""
    if isinstance(target, pd.Series):
        aligned = target.reindex(df.index)
        aligned.name = aligned.name or TARGET_COLUMN
        return aligned.astype("float64")
    if isinstance(target, str):
        if target not in df.columns:
            raise KeyError(f"Target column {target!r} is not in the DataFrame.")
        return pd.to_numeric(df[target], errors="coerce").rename(target)
    if TARGET_COLUMN in df.columns:
        return pd.to_numeric(df[TARGET_COLUMN], errors="coerce").rename(TARGET_COLUMN)
    return compute_priority_target(df)


def _confidence_column(df: pd.DataFrame) -> str | None:
    for name in QUALITY_CONFIDENCE_ALIASES:
        if name in df.columns:
            return name
    return None


def quality_flag_mask(df: pd.DataFrame) -> pd.Series:
    """True where program 2 (or equivalent) has flagged the record."""
    mask = pd.Series(False, index=df.index)
    for name in QUALITY_FLAG_COLUMNS:
        if name not in df.columns:
            continue
        col = df[name]
        if col.dtype == bool or str(col.dtype) == "boolean":
            mask = mask | col.fillna(False).astype(bool)
        else:
            numeric = pd.to_numeric(col, errors="coerce")
            if numeric.notna().any():
                mask = mask | (numeric.fillna(0) > 0)
            else:
                truthy = (
                    col.astype(str)
                    .str.strip()
                    .str.lower()
                    .isin({"1", "true", "yes", "flagged", "quarantine", "low"})
                )
                mask = mask | truthy
    conf_col = _confidence_column(df)
    if conf_col is not None:
        confidence = pd.to_numeric(df[conf_col], errors="coerce")
        mask = mask | (confidence < MIN_SAMPLE_WEIGHT)
    return mask.fillna(False)


def compute_sample_weights(df: pd.DataFrame) -> np.ndarray:
    """Optional LightGBM sample weights from quality flags / confidence.

    Priority:
    1. Explicit ``sample_weight`` column from preprocess.
    2. ``quality_confidence`` in [0, 1], clipped to at least
       ``MIN_SAMPLE_WEIGHT`` so flagged rows are down-weighted, not dropped.
    3. Boolean quality flags → ``FLAGGED_SAMPLE_WEIGHT`` (0.35).
    4. Otherwise uniform 1.0.
    """
    n = len(df)
    if SAMPLE_WEIGHT_COLUMN in df.columns:
        weights = pd.to_numeric(df[SAMPLE_WEIGHT_COLUMN], errors="coerce").fillna(1.0)
        return np.clip(weights.to_numpy(dtype="float64"), MIN_SAMPLE_WEIGHT, None)

    conf_col = _confidence_column(df)
    if conf_col is not None:
        confidence = pd.to_numeric(df[conf_col], errors="coerce")
        weights = confidence.fillna(1.0).to_numpy(dtype="float64")
        flagged = quality_flag_mask(df).to_numpy()
        weights[flagged & (confidence.isna().to_numpy())] = FLAGGED_SAMPLE_WEIGHT
        return np.clip(weights, MIN_SAMPLE_WEIGHT, 1.0)

    weights = np.ones(n, dtype="float64")
    weights[quality_flag_mask(df).to_numpy()] = FLAGGED_SAMPLE_WEIGHT
    return weights


def identity_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Identifier columns useful when scoring / plotting individual records."""
    present = [c for c in ID_COLUMNS if c in df.columns]
    if present:
        return df.loc[:, present].copy()
    return pd.DataFrame(index=df.index)
