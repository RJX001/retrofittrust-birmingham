"""Feature preprocessing for RetrofitTrust ranking / quality-screen inputs.

Steps (applied in this order, no rows dropped)
----------------------------------------------
1. Missingness indicator flags for numeric columns (``<col>_missing``).
   The *pattern* of missingness is itself a signal in EPC data.
2. Median imputation of numeric columns (fitted on the data passed to
   ``fit=True``).
3. Standardisation of imputed numerics (zero mean, unit variance).
   This matters more for the program-2 autoencoder than for LightGBM,
   but is applied consistently across the pipeline.
4. Categorical nulls filled with the explicit label ``missing``, then
   one-hot encoded (low-cardinality heating type, construction, tenure).

Anomaly flags from program 2 must **not** be used here as a reason to
drop rows. Quarantine-and-flag only: unusual-but-real dwellings (flats
and maisonettes are disproportionately flagged in EPC error research)
would otherwise be excluded from a fuel-poverty ranking — an equity harm.

The composite target and its inputs (EPC gap / SAP efficiencies / IMD
income score) are held out of the default feature matrix so LightGBM
cannot trivially recover the label.

EPC performance-gap and coverage-bias caveats still apply to the
resulting matrix: standardised features remain modelled-certificate
quantities, not metered energy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..config import SEED
from ._utils import ensure_logging

logger = logging.getLogger(__name__)

# Identifiers, free text, join flags and the target — never one-hot / scaled.
# Target components are held out of the feature matrix to avoid leaking the
# composite score (gap is a function of current/potential efficiency; the
# 0.4 term is the IMD income score).
_EXCLUDE_EXACT = {
    "lsoa21cd",
    "lsoa21nm",
    "lsoa21nm_imd",
    "lsoa21nm_census",
    "lsoa21nm_geography",
    "lmk_key",
    "uprn",
    "uprn_source",
    "building_reference_number",
    "postcode",
    "address",
    "address1",
    "address2",
    "address3",
    "posttown",
    "geometry",
    "retrofit_priority_score",
    "epc_gap",
    "epc_gap_norm",
    "imd_income_norm",
    "imd_income_score",
    "current_energy_efficiency",
    "potential_energy_efficiency",
    "current_energy_rating",
    "potential_energy_rating",
}

_EXCLUDE_SUFFIXES = ("_matched",)
_EXCLUDE_PREFIXES = ("address",)

# Spec-named categoricals (used if present, even if moderately high cardinality).
PREFERRED_CATEGORICALS = (
    "property_type",
    "built_form",
    "tenure",
    "mainheat_description",
    "main_fuel",
    "construction_age_band",
    "transaction_type",
    "glazed_type",
    "mains_gas_flag",
    "mechanical_ventilation",
    "walls_energy_eff",
    "roof_energy_eff",
    "windows_energy_eff",
    "hot_water_energy_eff",
    "mainheat_energy_eff",
)

HIGH_CARDINALITY_LIMIT = 80
MISSING_LABEL = "missing"


@dataclass
class PreprocessState:
    """Fitted transformers so inference uses the training medians / scaler."""

    numeric_columns: list[str]
    categorical_columns: list[str]
    excluded_columns: list[str]
    numeric_medians: dict[str, float]
    scaler: StandardScaler
    encoder: OneHotEncoder
    feature_names: list[str]
    seed: int = SEED
    missing_flag_columns: list[str] = field(default_factory=list)


def _is_excluded(name: str) -> bool:
    if name in _EXCLUDE_EXACT:
        return True
    if any(name.startswith(p) for p in _EXCLUDE_PREFIXES):
        return True
    if any(name.endswith(s) for s in _EXCLUDE_SUFFIXES):
        return True
    return False


def classify_columns(
    df: pd.DataFrame,
    *,
    categorical_cols: list[str] | None = None,
    numeric_cols: list[str] | None = None,
) -> tuple[list[str], list[str], list[str]]:
    """Split columns into numeric / categorical / excluded.

    High-cardinality text (e.g. free-text wall descriptions) is excluded
    from one-hot encoding and logged — exploding those fields would drown
    the heating-type / tenure dummies the spec actually calls for.
    """
    excluded = [c for c in df.columns if _is_excluded(str(c))]
    usable = [c for c in df.columns if c not in excluded]

    if numeric_cols is not None or categorical_cols is not None:
        numeric = [c for c in (numeric_cols or []) if c in df.columns]
        categorical = [c for c in (categorical_cols or []) if c in df.columns]
        return numeric, categorical, excluded

    numeric: list[str] = []
    categorical: list[str] = []
    skipped_high_card: list[str] = []

    preferred = [c for c in PREFERRED_CATEGORICALS if c in usable]

    for col in usable:
        if col in preferred:
            nunique = df[col].nunique(dropna=True)
            if nunique > HIGH_CARDINALITY_LIMIT:
                skipped_high_card.append(col)
                excluded.append(col)
                continue
            categorical.append(col)
            continue

        series = df[col]
        if str(getattr(series, "dtype", "")) == "geometry":
            excluded.append(col)
            continue
        if pd.api.types.is_bool_dtype(series):
            numeric.append(col)
            continue
        if pd.api.types.is_numeric_dtype(series):
            numeric.append(col)
            continue
        if pd.api.types.is_datetime64_any_dtype(series):
            excluded.append(col)
            continue

        nunique = series.nunique(dropna=True)
        if nunique <= HIGH_CARDINALITY_LIMIT:
            categorical.append(col)
        else:
            skipped_high_card.append(col)
            excluded.append(col)

    if skipped_high_card:
        logger.info(
            "Skipping high-cardinality columns from one-hot (>%s levels): %s",
            HIGH_CARDINALITY_LIMIT,
            skipped_high_card[:20],
        )
    return numeric, categorical, excluded


def _add_missing_flags(df: pd.DataFrame, numeric_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    flags: list[str] = []
    for col in numeric_cols:
        flag = f"{col}_missing"
        out[flag] = out[col].isna().astype(np.int8)
        flags.append(flag)
    return out, flags


def preprocess(
    df: pd.DataFrame,
    *,
    fit: bool = True,
    state: PreprocessState | None = None,
    categorical_cols: list[str] | None = None,
    numeric_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, PreprocessState]:
    """Impute, flag, scale and one-hot-encode. Never drops rows.

    Parameters
    ----------
    df:
        Merged (and usually target-attached) frame.
    fit:
        If True, learn medians / scaler / encoder from ``df``.
        If False, ``state`` from a previous fit is required.
    state:
        Fitted ``PreprocessState`` (required when ``fit=False``).
    categorical_cols, numeric_cols:
        Optional overrides; otherwise dtypes + the preferred list are used.

    Returns
    -------
    X, state
        ``X`` has the same index and row count as ``df``. Identifier
        columns are preserved at the front for later joins / SHAP lookup.
    """
    ensure_logging()
    n_in = len(df)
    if n_in == 0:
        raise ValueError("preprocess() received an empty frame.")

    if not fit:
        if state is None:
            raise ValueError("fit=False requires a previously fitted PreprocessState.")
        numeric_cols = list(state.numeric_columns)
        categorical_cols = list(state.categorical_columns)
        excluded = list(state.excluded_columns)
    else:
        numeric_cols, categorical_cols, excluded = classify_columns(
            df, categorical_cols=categorical_cols, numeric_cols=numeric_cols
        )
        logger.info(
            "preprocess fit: %s numeric, %s categorical, %s excluded",
            len(numeric_cols),
            len(categorical_cols),
            len(excluded),
        )

    work, flag_cols = _add_missing_flags(df, numeric_cols)

    if fit:
        medians: dict[str, float] = {}
        for col in numeric_cols:
            med = pd.to_numeric(work[col], errors="coerce").median()
            if pd.isna(med):
                med = 0.0
                logger.warning("Numeric column %s is all-null; imputing 0.0", col)
            medians[col] = float(med)
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
            dtype=np.float64,
        )
        scaler = StandardScaler()
    else:
        assert state is not None
        medians = dict(state.numeric_medians)
        encoder = state.encoder
        scaler = state.scaler

    imputed = pd.DataFrame(index=work.index)
    for col in numeric_cols:
        imputed[col] = pd.to_numeric(work[col], errors="coerce").fillna(medians[col])

    if fit:
        scaled_arr = scaler.fit_transform(imputed.to_numpy(dtype=float)) if numeric_cols else np.empty((n_in, 0))
    else:
        scaled_arr = scaler.transform(imputed.to_numpy(dtype=float)) if numeric_cols else np.empty((n_in, 0))
    scaled = pd.DataFrame(scaled_arr, index=work.index, columns=numeric_cols)

    cat_raw = pd.DataFrame(index=work.index)
    for col in categorical_cols:
        cat_raw[col] = (
            work[col]
            .astype("string")
            .fillna(MISSING_LABEL)
            .replace({"": MISSING_LABEL, "nan": MISSING_LABEL, "<NA>": MISSING_LABEL})
        )

    if categorical_cols:
        if fit:
            encoded_arr = encoder.fit_transform(cat_raw)
        else:
            encoded_arr = encoder.transform(cat_raw)
        cat_names = list(encoder.get_feature_names_out(categorical_cols))
        encoded = pd.DataFrame(encoded_arr, index=work.index, columns=cat_names)
    else:
        encoded = pd.DataFrame(index=work.index)
        cat_names = []

    flags = work[flag_cols].astype(np.int8) if flag_cols else pd.DataFrame(index=work.index)

    id_cols = [c for c in df.columns if c in _EXCLUDE_EXACT or str(c).endswith("_matched")]
    id_part = df.loc[:, [c for c in id_cols if c in df.columns]].copy()

    X = pd.concat([id_part, scaled, flags, encoded], axis=1)
    # Guard against duplicate column names after concat.
    X = X.loc[:, ~X.columns.duplicated()].copy()

    if len(X) != n_in:
        raise RuntimeError(
            f"Silent data loss in preprocess: {n_in:,} -> {len(X):,} rows."
        )

    feature_names = list(scaled.columns) + list(flags.columns) + cat_names
    fitted = PreprocessState(
        numeric_columns=list(numeric_cols),
        categorical_columns=list(categorical_cols),
        excluded_columns=list(excluded),
        numeric_medians=medians,
        scaler=scaler,
        encoder=encoder,
        feature_names=feature_names,
        seed=SEED,
        missing_flag_columns=list(flag_cols),
    )
    logger.info(
        "preprocess: %s rows, %s feature columns (plus %s id/flag cols kept)",
        f"{len(X):,}",
        f"{len(feature_names):,}",
        f"{len(id_part.columns):,}",
    )
    return X, fitted
