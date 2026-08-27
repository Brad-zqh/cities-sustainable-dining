"""Component-specific diagnostics on an audited reachable-pair table.

The functions in this module deliberately keep component coverage separate
from component values. Missing carbon or nutrition estimates are never
converted to zero, and a raw carbon estimate is never relabelled as a
validated sustainability score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


PRICE_TIER_ORDER = {
    "$50以下": 1.0,
    "$51-100": 2.0,
    "$101-200": 3.0,
    "$201-400": 4.0,
    "$401-800": 5.0,
    "$801以上": 6.0,
}


def normalize_price_order(values: pd.Series) -> pd.Series:
    """Map the platform price labels to an ordinal ceiling order."""

    return values.astype("string").str.strip().map(PRICE_TIER_ORDER).astype(float)


def decay_weight(travel_time: pd.Series, threshold: float, decay: str) -> pd.Series:
    """Return prespecified step or linear walking-time weights."""

    time = pd.to_numeric(travel_time, errors="raise").astype(float)
    within = time.le(float(threshold))
    if decay == "step":
        return within.astype(float)
    if decay == "linear":
        return (1.0 - time.div(float(threshold))).clip(lower=0.0).where(within, 0.0)
    raise ValueError(f"unsupported decay: {decay}")


def component_observation_flags(
    estimate: pd.Series,
    status: pd.Series,
    coverage: pd.Series,
    *,
    positive_only: bool,
) -> tuple[pd.Series, pd.Series]:
    """Return any-observed and complete-observed flags without zero filling."""

    value = pd.to_numeric(estimate, errors="coerce")
    state = status.astype("string").fillna("")
    covered = pd.to_numeric(coverage, errors="coerce")
    valid_value = value.notna() & np.isfinite(value)
    if positive_only:
        valid_value &= value.gt(0)
    observed = valid_value & state.str.startswith("observed_")
    complete = observed & covered.ge(1.0 - 1e-12)
    return observed, complete


def weighted_group_diagnostics(
    frame: pd.DataFrame,
    *,
    group_column: str,
    weight_column: str,
    component: str,
) -> pd.DataFrame:
    """Aggregate coverage and conditional values by one origin identifier."""

    required = {
        group_column,
        weight_column,
        f"{component}_estimate",
        f"{component}_observed",
        f"{component}_complete",
        f"{component}_source_coverage",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"missing columns: {sorted(missing)}")

    work = frame.loc[frame[weight_column].gt(0), list(required)].copy()
    weight = work[weight_column].astype(float)
    observed = work[f"{component}_observed"].astype(bool)
    complete = work[f"{component}_complete"].astype(bool)
    value = pd.to_numeric(work[f"{component}_estimate"], errors="coerce")
    source_coverage = pd.to_numeric(
        work[f"{component}_source_coverage"], errors="coerce"
    )
    work["_total_weight"] = weight
    work["_observed_weight"] = weight.where(observed, 0.0)
    work["_complete_weight"] = weight.where(complete, 0.0)
    work["_observed_value_mass"] = (weight * value).where(observed, 0.0)
    work["_complete_value_mass"] = (weight * value).where(complete, 0.0)
    work["_source_coverage_mass"] = (weight * source_coverage).where(observed, 0.0)
    sums = work.groupby(group_column, observed=True)[
        [
            "_total_weight",
            "_observed_weight",
            "_complete_weight",
            "_observed_value_mass",
            "_complete_value_mass",
            "_source_coverage_mass",
        ]
    ].sum()
    sums[f"{component}_observed_share"] = sums["_observed_weight"].div(
        sums["_total_weight"].replace(0, np.nan)
    )
    sums[f"{component}_complete_share"] = sums["_complete_weight"].div(
        sums["_total_weight"].replace(0, np.nan)
    )
    sums[f"{component}_conditional_mean"] = sums["_observed_value_mass"].div(
        sums["_observed_weight"].replace(0, np.nan)
    )
    sums[f"{component}_complete_conditional_mean"] = sums[
        "_complete_value_mass"
    ].div(sums["_complete_weight"].replace(0, np.nan))
    sums[f"{component}_mean_source_coverage"] = sums[
        "_source_coverage_mass"
    ].div(sums["_observed_weight"].replace(0, np.nan))
    return sums.rename(
        columns={
            "_total_weight": "reachable_opportunity_weight",
            "_observed_weight": f"{component}_observed_weight",
            "_complete_weight": f"{component}_complete_weight",
            "_observed_value_mass": f"{component}_observed_value_mass",
            "_complete_value_mass": f"{component}_complete_value_mass",
            "_source_coverage_mass": f"{component}_source_coverage_mass",
        }
    )
