"""Annual and trailing-window dish-mention component candidates.

The weights are counts of dish mentions in platform reviews.  They are never
interpreted as observed consumption, sales or visits.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


DEFAULT_METRIC_RULES = {
    "carbon_emission_g": "strictly_positive",
    "energy_kcal": "strictly_positive",
    "protein_g": "nonnegative",
    "fat_g": "nonnegative",
    "carbohydrates_g": "nonnegative",
    "salt_g": "nonnegative",
}


def window_years(target_year: int, window: str, available_years: Sequence[int]) -> list[int]:
    years = sorted(int(year) for year in available_years)
    if window == "annual":
        selected = [target_year]
    elif window == "trailing_3_year":
        selected = list(range(target_year - 2, target_year + 1))
    elif window == "legacy_cumulative":
        selected = [year for year in years if year <= target_year]
    else:
        raise ValueError(f"unsupported dish-mention window: {window}")
    missing = sorted(set(selected).difference(years))
    if missing:
        raise ValueError(f"dish count columns missing years required by {window}: {missing}")
    return selected


def valid_metric_mask(values: pd.Series, rule: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    finite = pd.Series(np.isfinite(numeric), index=values.index)
    if rule == "strictly_positive":
        return finite & numeric.gt(0)
    if rule == "nonnegative":
        return finite & numeric.ge(0)
    raise ValueError(f"unsupported metric validity rule: {rule}")


def aggregate_dish_component_chunks(
    chunks: Sequence[pd.DataFrame],
    *,
    years: Sequence[int],
    windows: Sequence[str],
    metric_rules: Mapping[str, str] = DEFAULT_METRIC_RULES,
    id_column: str = "restaurant_id",
) -> pd.DataFrame:
    """Aggregate restricted dish records to restaurant/window sufficient statistics."""

    accumulator: pd.DataFrame | None = None
    available_years: list[int] | None = None
    for chunk in chunks:
        if id_column not in chunk.columns:
            raise ValueError(f"dish chunk missing {id_column}")
        count_columns = [column for column in chunk.columns if column.endswith("_count")]
        year_columns = {int(column.removesuffix("_count")): column for column in count_columns}
        if available_years is None:
            available_years = sorted(year_columns)
        elif available_years != sorted(year_columns):
            raise ValueError("dish-count schema changed between chunks")
        missing_metrics = sorted(set(metric_rules).difference(chunk.columns))
        if missing_metrics:
            raise ValueError(f"dish chunk missing metrics: {missing_metrics}")

        counts = chunk[count_columns].apply(pd.to_numeric, errors="coerce")
        if counts.lt(0).any().any():
            raise ValueError("negative dish-mention count found")
        counts = counts.fillna(0.0)
        working_columns: dict[str, pd.Series] = {id_column: chunk[id_column]}
        for target_year in years:
            for window in windows:
                selected_years = window_years(int(target_year), window, available_years)
                mention_count = counts[[year_columns[year] for year in selected_years]].sum(axis=1)
                suffix = f"{target_year}__{window}"
                working_columns[f"mention_count__{suffix}"] = mention_count
                for metric, rule in metric_rules.items():
                    values = pd.to_numeric(chunk[metric], errors="coerce")
                    valid = valid_metric_mask(values, rule)
                    working_columns[f"covered_mentions__{metric}__{suffix}"] = mention_count.where(valid, 0.0)
                    working_columns[f"weighted_sum__{metric}__{suffix}"] = (mention_count * values).where(valid, 0.0)
        working = pd.DataFrame(working_columns)
        grouped = working.groupby(id_column, sort=False).sum(numeric_only=True)
        accumulator = grouped if accumulator is None else accumulator.add(grouped, fill_value=0.0)

    if accumulator is None:
        raise ValueError("no dish chunks supplied")
    accumulator.index.name = id_column
    return accumulator.reset_index()


def finalize_restaurant_year_candidates(
    sufficient_statistics: pd.DataFrame,
    membership: pd.DataFrame,
    *,
    years: Sequence[int],
    windows: Sequence[str],
    metric_rules: Mapping[str, str] = DEFAULT_METRIC_RULES,
    id_column: str = "restaurant_id",
    early_year_column: str = "operation_early_year",
    latest_year_column: str = "operation_latest_year",
) -> pd.DataFrame:
    """Return one restricted restaurant/year/window row with explicit evidence states."""

    required = {id_column, early_year_column, latest_year_column}
    missing = sorted(required.difference(membership.columns))
    if missing:
        raise ValueError(f"membership table missing columns: {missing}")
    merged = membership[[id_column, early_year_column, latest_year_column]].merge(
        sufficient_statistics,
        on=id_column,
        how="left",
        validate="one_to_one",
    )
    early = pd.to_numeric(merged[early_year_column], errors="coerce")
    latest = pd.to_numeric(merged[latest_year_column], errors="coerce")
    rows: list[pd.DataFrame] = []
    for target_year in years:
        operation = early.le(target_year) & latest.ge(target_year)
        for window in windows:
            suffix = f"{target_year}__{window}"
            mention = pd.to_numeric(merged[f"mention_count__{suffix}"], errors="coerce").fillna(0.0)
            output = pd.DataFrame(
                {
                    id_column: merged[id_column],
                    "year": int(target_year),
                    "dish_mention_window": window,
                    "operation_member": operation,
                    "dish_mention_count": mention,
                    "weight_interpretation": "platform_review_dish_mention_count_not_consumption_or_sales",
                    "model_validation_gate_passed": False,
                    "primary_component_eligible": False,
                }
            )
            for metric in metric_rules:
                covered = pd.to_numeric(
                    merged[f"covered_mentions__{metric}__{suffix}"], errors="coerce"
                ).fillna(0.0)
                weighted_sum = pd.to_numeric(
                    merged[f"weighted_sum__{metric}__{suffix}"], errors="coerce"
                ).fillna(0.0)
                estimate = pd.Series(np.nan, index=merged.index, dtype=float)
                estimate.loc[covered.gt(0)] = weighted_sum.loc[covered.gt(0)] / covered.loc[covered.gt(0)]
                coverage = pd.Series(np.nan, index=merged.index, dtype=float)
                coverage.loc[mention.gt(0)] = covered.loc[mention.gt(0)] / mention.loc[mention.gt(0)]
                status = pd.Series("missing_no_dish_mention_evidence", index=merged.index, dtype=object)
                status.loc[mention.gt(0) & covered.eq(0)] = "missing_no_valid_metric_estimate"
                status.loc[covered.gt(0) & covered.lt(mention)] = "observed_partial_estimate_coverage"
                status.loc[mention.gt(0) & covered.eq(mention)] = "observed_complete_estimate_coverage"
                output[f"{metric}__estimate"] = estimate
                output[f"{metric}__covered_mentions"] = covered
                output[f"{metric}__coverage"] = coverage
                output[f"{metric}__status"] = status
            rows.append(output.loc[operation].copy())
    return pd.concat(rows, ignore_index=True)


def summarize_candidate_coverage(
    candidates: pd.DataFrame,
    *,
    metrics: Sequence[str],
) -> pd.DataFrame:
    """Create disclosure-safe aggregate coverage counts."""

    rows: list[dict[str, int | float | str]] = []
    group_columns = ["year", "dish_mention_window"]
    for (year, window), group in candidates.groupby(group_columns, sort=True):
        base: dict[str, int | float | str] = {
            "year": int(year),
            "dish_mention_window": str(window),
            "operation_restaurant_n": int(len(group)),
            "no_dish_mention_n": int(group["dish_mention_count"].eq(0).sum()),
            "positive_dish_mention_n": int(group["dish_mention_count"].gt(0).sum()),
            "primary_component_eligible": False,
        }
        for metric in metrics:
            status = group[f"{metric}__status"]
            base[f"{metric}__no_valid_estimate_n"] = int(status.eq("missing_no_valid_metric_estimate").sum())
            base[f"{metric}__partial_coverage_n"] = int(status.eq("observed_partial_estimate_coverage").sum())
            base[f"{metric}__complete_coverage_n"] = int(status.eq("observed_complete_estimate_coverage").sum())
            coverage = pd.to_numeric(group[f"{metric}__coverage"], errors="coerce")
            base[f"{metric}__mean_coverage_where_mentions_positive"] = float(coverage.mean()) if coverage.notna().any() else float("nan")
        rows.append(base)
    return pd.DataFrame(rows)
