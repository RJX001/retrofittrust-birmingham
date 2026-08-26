"""Composite retrofit priority score (program 1 target).

Definition (weights from ``config.py``)
--------------------------------------
    score = EPC_GAP_WEIGHT * normalised(EPC gap)
          + IMD_INCOME_WEIGHT * normalised(IMD income deprivation)

Defaults: ``EPC_GAP_WEIGHT = 0.6``, ``IMD_INCOME_WEIGHT = 0.4``.
Higher score = higher retrofit priority.

EPC gap
    ``potential − current`` on a scale where a larger gap means more
    modelled improvement headroom.

    Prefer SAP-style efficiencies (``potential_energy_efficiency`` −
    ``current_energy_efficiency``, typically 1–100). If those columns
    are absent, letter ratings are mapped A=7 … G=1 so that
    ``potential − current`` is still non-negative when potential is
    better than current.

    **Performance gap:** EPC modelled ratings diverge from metered use
    (~16% for gas-heated homes, ~31% for electrically heated; Hardy &
    Glew 2019 / DESNZ). The gap used here is therefore a *relative*
    ranking signal, not a prediction of kilowatt-hours saved.

IMD income domain
    ``imd_income_score`` is an LSOA *rate* of income deprivation
    (higher = more deprived). Min–max normalised over the Birmingham
    sample so it shares a [0, 1] scale with the EPC gap.

Ecological fallacy (do not ignore)
    IMD is an **area** statistic. A dwelling in a high-income-deprivation
    LSOA is not therefore an income-deprived household, and vice versa.
    Using the domain as a 0.4 weight on a property-level score is a
    deliberate, documented proxy for neighbourhood need under
    Birmingham's Route to Zero — it is not a household means test.

Coverage bias
    Properties without an EPC never receive a score. That missingness is
    systematic (sale / let / new-build triggers), not random.

Normalisation is min–max over the current frame (Birmingham sample),
not over England. Scores are therefore relative to this PoC extract.
"""

from __future__ import annotations

import logging

import pandas as pd

from ..config import EPC_GAP_WEIGHT, IMD_INCOME_WEIGHT, SEED
from ._utils import ensure_logging, find_column, min_max_normalise

logger = logging.getLogger(__name__)

_ = SEED

# A is best. Mapping so potential − current is positive improvement headroom.
RATING_POINTS = {
    "A": 7.0,
    "B": 6.0,
    "C": 5.0,
    "D": 4.0,
    "E": 3.0,
    "F": 2.0,
    "G": 1.0,
}


def rating_to_points(series: pd.Series) -> pd.Series:
    """Map EPC band letters to A=7 … G=1. Invalid / empty → NA."""
    cleaned = (
        series.astype("string")
        .str.strip()
        .str.upper()
        .str.replace(r"[^A-G]", "", regex=True)
    )
    # Keep only the first letter if the field is e.g. "C ".
    cleaned = cleaned.str[:1]
    return cleaned.map(RATING_POINTS)


def compute_epc_gap(df: pd.DataFrame) -> pd.Series:
    """Potential minus current efficiency; higher = more modelled headroom."""
    sap_pot = find_column(
        df.columns,
        "potential_energy_efficiency",
        "potential-energy-efficiency",
    )
    sap_cur = find_column(
        df.columns,
        "current_energy_efficiency",
        "current-energy-efficiency",
    )
    if sap_pot is not None and sap_cur is not None:
        gap = pd.to_numeric(df[sap_pot], errors="coerce") - pd.to_numeric(
            df[sap_cur], errors="coerce"
        )
        n_neg = int((gap < 0).sum())
        if n_neg:
            logger.warning(
                "%s rows have potential SAP < current SAP (unexpected on a "
                "valid EPC). Negative gaps are kept, not clipped — they will "
                "lower the composite score.",
                f"{n_neg:,}",
            )
        logger.info("EPC gap from SAP efficiencies (%s − %s)", sap_pot, sap_cur)
        return gap

    pot_band = find_column(
        df.columns,
        "potential_energy_rating",
        "potential-energy-rating",
    )
    cur_band = find_column(
        df.columns,
        "current_energy_rating",
        "current-energy-rating",
    )
    if pot_band is None or cur_band is None:
        raise ValueError(
            "Cannot compute EPC gap: need current/potential energy efficiency "
            "or current/potential energy rating columns."
        )
    gap = rating_to_points(df[pot_band]) - rating_to_points(df[cur_band])
    logger.info("EPC gap from letter ratings (%s − %s); A=7 … G=1", pot_band, cur_band)
    return gap


def add_priority_score(
    df: pd.DataFrame,
    *,
    epc_gap_weight: float = EPC_GAP_WEIGHT,
    imd_income_weight: float = IMD_INCOME_WEIGHT,
    income_col: str | None = None,
) -> pd.DataFrame:
    """Attach ``retrofit_priority_score`` and its normalised components.

    Rows with a missing gap or missing IMD income score receive NA for
    the composite (they are not silently filled). That is visible
    missingness, not data loss — callers can still score those dwellings
    on the EPC axis alone if they choose.

    Parameters
    ----------
    df:
        Merged property-level frame (EPC + IMD income domain).
    epc_gap_weight, imd_income_weight:
        Must be the dissertation weights unless RJ changes them.
        Defaults are 0.6 / 0.4.
    income_col:
        Override for the IMD income *score* (rate) column.
    """
    ensure_logging()
    weight_sum = epc_gap_weight + imd_income_weight
    if abs(weight_sum - 1.0) > 1e-9:
        logger.warning(
            "Priority weights sum to %s (expected 1.0). Scores are still a "
            "weighted sum, not a convex combination.",
            weight_sum,
        )

    out = df.copy()
    out["epc_gap"] = compute_epc_gap(out)
    out["epc_gap_norm"] = min_max_normalise(out["epc_gap"])

    col = income_col or find_column(
        out.columns,
        "imd_income_score",
        "income_score_rate",
        "income_score",
    )
    if col is None:
        raise ValueError(
            "Cannot compute priority score: IMD income score column not found. "
            "Load File 7 of IoD 2025 (includes Income Score (rate))."
        )

    # Higher income-score = more deprived = higher priority. Do not invert.
    out["imd_income_norm"] = min_max_normalise(pd.to_numeric(out[col], errors="coerce"))

    n_gap_na = int(out["epc_gap_norm"].isna().sum())
    n_imd_na = int(out["imd_income_norm"].isna().sum())
    if n_gap_na:
        logger.warning(
            "%s / %s rows missing EPC gap after normalisation — composite "
            "will be NA for those rows (not imputed).",
            f"{n_gap_na:,}",
            f"{len(out):,}",
        )
    if n_imd_na:
        logger.warning(
            "%s / %s rows missing IMD income domain — composite will be NA. "
            "Ecological join failed or income score was absent. Rows are kept.",
            f"{n_imd_na:,}",
            f"{len(out):,}",
        )

    out["retrofit_priority_score"] = (
        epc_gap_weight * out["epc_gap_norm"] + imd_income_weight * out["imd_income_norm"]
    )
    logger.info(
        "retrofit_priority_score = %s * epc_gap_norm + %s * imd_income_norm "
        "(non-null scores: %s / %s). "
        "Ecological fallacy: IMD income is LSOA-level, not household-level.",
        epc_gap_weight,
        imd_income_weight,
        f"{out['retrofit_priority_score'].notna().sum():,}",
        f"{len(out):,}",
    )
    return out
