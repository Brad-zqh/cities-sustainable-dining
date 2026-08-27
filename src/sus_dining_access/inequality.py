"""Population-weighted inequality measures with explicit edge-case handling."""

from __future__ import annotations

import math
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


def _clean(x: Iterable[float], w: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(list(x), dtype=float)
    weights = np.asarray(list(w), dtype=float)
    if values.shape != weights.shape:
        raise ValueError("x and w must have the same shape")
    if np.any(~np.isfinite(values)) or np.any(~np.isfinite(weights)):
        raise ValueError("x and w must be finite; missingness is handled upstream")
    if np.any(values < 0):
        raise ValueError("Inequality outcomes must be nonnegative")
    if np.any(weights <= 0):
        raise ValueError("Population weights must be positive")
    return values, weights


def weighted_mean(x: Iterable[float], w: Iterable[float]) -> float:
    values, weights = _clean(x, w)
    return float(np.average(values, weights=weights))


def weighted_gini(x: Iterable[float], w: Iterable[float]) -> float:
    values, weights = _clean(x, w)
    mean = np.average(values, weights=weights)
    if mean == 0:
        return math.nan
    order = np.argsort(values, kind="mergesort")
    values, weights = values[order], weights[order]
    cumulative_weight = np.cumsum(weights)
    cumulative_value = np.cumsum(values * weights)
    total_weight = cumulative_weight[-1]
    total_value = cumulative_value[-1]
    previous_weight = np.concatenate(([0.0], cumulative_weight[:-1]))
    previous_value = np.concatenate(([0.0], cumulative_value[:-1]))
    area = np.sum((cumulative_value + previous_value) * (cumulative_weight - previous_weight))
    return float(1.0 - area / (total_value * total_weight))


def theil_t(x: Iterable[float], w: Iterable[float]) -> float:
    values, weights = _clean(x, w)
    p = weights / weights.sum()
    mean = np.sum(p * values)
    if mean == 0:
        return math.nan
    ratio = values / mean
    terms = np.zeros_like(ratio)
    positive = values > 0
    terms[positive] = ratio[positive] * np.log(ratio[positive])
    return float(np.sum(p * terms))


def theil_l(x: Iterable[float], w: Iterable[float]) -> float:
    values, weights = _clean(x, w)
    if np.any(values == 0):
        return math.inf
    p = weights / weights.sum()
    mean = np.sum(p * values)
    return float(np.sum(p * np.log(mean / values)))


def atkinson(x: Iterable[float], w: Iterable[float], epsilon: float = 0.5) -> float:
    values, weights = _clean(x, w)
    if epsilon < 0:
        raise ValueError("epsilon must be nonnegative")
    p = weights / weights.sum()
    mean = np.sum(p * values)
    if mean == 0:
        return math.nan
    if epsilon == 1:
        if np.any(values == 0):
            return 1.0
        equally_distributed = math.exp(float(np.sum(p * np.log(values))))
    elif epsilon > 1 and np.any(values == 0):
        return 1.0
    else:
        equally_distributed = float(np.sum(p * values ** (1 - epsilon))) ** (1 / (1 - epsilon))
    return float(1 - equally_distributed / mean)


def fractional_income_ranks(income: Iterable[float], w: Iterable[float]) -> np.ndarray:
    income_values = np.asarray(list(income), dtype=float)
    weights = np.asarray(list(w), dtype=float)
    if income_values.shape != weights.shape or np.any(~np.isfinite(income_values)):
        raise ValueError("income and weights must be complete and aligned")
    if np.any(weights <= 0):
        raise ValueError("weights must be positive")
    ranks = np.empty(len(income_values), dtype=float)
    order = np.argsort(income_values, kind="mergesort")
    sorted_income = income_values[order]
    sorted_weights = weights[order]
    total = sorted_weights.sum()
    cursor = 0.0
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and sorted_income[end] == sorted_income[start]:
            end += 1
        group_weight = sorted_weights[start:end].sum()
        midpoint = (cursor + group_weight / 2) / total
        ranks[order[start:end]] = midpoint
        cursor += group_weight
        start = end
    return ranks


def concentration_index(
    x: Iterable[float], w: Iterable[float], income: Iterable[float]
) -> float:
    values, weights = _clean(x, w)
    p = weights / weights.sum()
    mean = np.sum(p * values)
    if mean == 0:
        return math.nan
    ranks = fractional_income_ranks(income, weights)
    return float(2 * np.sum(p * values * ranks) / mean - 1)


def population_quintile_means(
    x: Iterable[float], w: Iterable[float], income: Iterable[float], groups: int = 5
) -> np.ndarray:
    if groups < 2:
        raise ValueError("groups must be at least 2")
    cuts = np.linspace(0.0, 1.0, groups + 1)
    return population_rank_segment_means(x, w, income, cuts)


def population_rank_segment_means(
    x: Iterable[float],
    w: Iterable[float],
    rank_variable: Iterable[float],
    cutpoints: Sequence[float],
) -> np.ndarray:
    """Population-weighted means across ordered rank segments.

    Geographic-unit weights may cross cutpoints and are split fractionally.
    Exact ties in the ranking variable are pooled before allocation so results
    cannot depend on arbitrary row or identifier order.
    """
    values, weights = _clean(x, w)
    ranks = np.asarray(list(rank_variable), dtype=float)
    cuts = np.asarray(list(cutpoints), dtype=float)
    if len(ranks) != len(values) or np.any(~np.isfinite(ranks)):
        raise ValueError("rank_variable must be complete and aligned")
    if (
        len(cuts) < 2
        or not np.isclose(cuts[0], 0.0)
        or not np.isclose(cuts[-1], 1.0)
        or np.any(np.diff(cuts) <= 0)
    ):
        raise ValueError("cutpoints must increase strictly from 0 to 1")
    order = np.argsort(ranks, kind="mergesort")
    values, weights, ranks = values[order], weights[order], ranks[order]
    total_weight = float(weights.sum())
    segment_edges = cuts * total_weight
    segment_targets = np.diff(segment_edges)
    sums = np.zeros(len(segment_targets))
    assigned = np.zeros(len(segment_targets))
    cursor = 0.0
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and ranks[end] == ranks[start]:
            end += 1
        tied_weight = float(weights[start:end].sum())
        tied_mean = float(np.average(values[start:end], weights=weights[start:end]))
        tied_end = cursor + tied_weight
        # Allocate the tied population interval by exact overlap with every
        # segment. This avoids iterative floating-point residuals when a large
        # tied group spans one or more population cutpoints.
        overlaps = np.maximum(
            0.0,
            np.minimum(tied_end, segment_edges[1:])
            - np.maximum(cursor, segment_edges[:-1]),
        )
        sums += overlaps * tied_mean
        assigned += overlaps
        cursor = tied_end
        start = end
    return np.divide(sums, assigned, out=np.full_like(sums, np.nan), where=assigned > 0)


def zero_access_population_share(x: Iterable[float], w: Iterable[float]) -> float:
    values, weights = _clean(x, w)
    return float(weights[values == 0].sum() / weights.sum())


def palma_style_access_ratio(
    x: Iterable[float], w: Iterable[float], socioeconomic_rank: Iterable[float]
) -> float:
    """Mean access of the top SES decile divided by the bottom 40 percent."""
    bottom_40, _, top_10 = population_rank_segment_means(
        x, w, socioeconomic_rank, [0.0, 0.4, 0.9, 1.0]
    )
    if bottom_40 == 0:
        return math.inf if top_10 > 0 else math.nan
    return float(top_10 / bottom_40)


def group_weighted_means(
    x: Iterable[float], w: Iterable[float], groups: Iterable[object]
) -> pd.DataFrame:
    values, weights = _clean(x, w)
    labels = pd.Series(list(groups), dtype="object")
    if len(labels) != len(values) or labels.isna().any():
        raise ValueError("groups must be complete and aligned")
    frame = pd.DataFrame({"group": labels, "value": values, "weight": weights})
    rows = []
    total = float(weights.sum())
    for group, subset in frame.groupby("group", sort=True, dropna=False):
        group_weight = float(subset["weight"].sum())
        rows.append(
            {
                "group": group,
                "population_weight": group_weight,
                "population_share": group_weight / total,
                "weighted_mean": float(
                    np.average(subset["value"], weights=subset["weight"])
                ),
            }
        )
    return pd.DataFrame(rows)


def group_contrast(
    x: Iterable[float],
    w: Iterable[float],
    groups: Iterable[object],
    *,
    reference_group: object,
    comparison_group: object,
) -> dict[str, float | object]:
    summary = group_weighted_means(x, w, groups).set_index("group")
    if reference_group not in summary.index or comparison_group not in summary.index:
        raise ValueError("reference_group and comparison_group must both be observed")
    reference = float(summary.loc[reference_group, "weighted_mean"])
    comparison = float(summary.loc[comparison_group, "weighted_mean"])
    ratio = comparison / reference if reference > 0 else (math.inf if comparison > 0 else math.nan)
    return {
        "reference_group": reference_group,
        "comparison_group": comparison_group,
        "reference_mean": reference,
        "comparison_mean": comparison,
        "absolute_gap": comparison - reference,
        "relative_ratio": float(ratio),
    }


def theil_decomposition(
    x: Iterable[float], w: Iterable[float], groups: Iterable[object]
) -> pd.DataFrame:
    """Decompose Theil T and L into within- and between-group components."""
    values, weights = _clean(x, w)
    labels = pd.Series(list(groups), dtype="object")
    if len(labels) != len(values) or labels.isna().any():
        raise ValueError("groups must be complete and aligned")
    total_weight = float(weights.sum())
    overall_mean = float(np.average(values, weights=weights))
    totals = {"theil_t": theil_t(values, weights), "theil_l": theil_l(values, weights)}
    if overall_mean == 0:
        return pd.DataFrame(
            [
                {"metric": metric, "total": value, "within": math.nan, "between": math.nan, "status": "undefined_zero_mean"}
                for metric, value in totals.items()
            ]
        )

    frame = pd.DataFrame({"group": labels, "value": values, "weight": weights})
    within_t = 0.0
    between_t = 0.0
    within_l = 0.0
    between_l = 0.0
    l_infinite = False
    for _, subset in frame.groupby("group", sort=True, dropna=False):
        group_values = subset["value"].to_numpy(float)
        group_weights = subset["weight"].to_numpy(float)
        population_share = float(group_weights.sum() / total_weight)
        group_mean = float(np.average(group_values, weights=group_weights))
        mean_ratio = group_mean / overall_mean
        if group_mean > 0:
            within_t += population_share * mean_ratio * theil_t(group_values, group_weights)
            between_t += population_share * mean_ratio * math.log(mean_ratio)
            within_l += population_share * theil_l(group_values, group_weights)
            between_l += population_share * math.log(1.0 / mean_ratio)
        else:
            l_infinite = True
    if l_infinite:
        within_l = math.inf
        between_l = math.inf
    return pd.DataFrame(
        [
            {"metric": "theil_t", "total": totals["theil_t"], "within": float(within_t), "between": float(between_t), "status": "ok"},
            {"metric": "theil_l", "total": totals["theil_l"], "within": float(within_l), "between": float(between_l), "status": "undefined_or_infinite_zero_case" if l_infinite else "ok"},
        ]
    )


def summarize_inequality(
    x: Sequence[float],
    w: Sequence[float],
    income: Sequence[float] | None = None,
    *,
    atkinson_epsilons: Sequence[float] | None = None,
) -> pd.DataFrame:
    """Return the complementary inequality dashboard used by T3/T4.

    Atkinson is excluded unless epsilon values are explicitly supplied because
    it is a normative sensitivity measure whose zero handling and inequality-
    aversion parameter must be frozen in advance.
    """
    values, weights = _clean(x, w)
    metrics = {
        "population_weighted_mean": weighted_mean(values, weights),
        "weighted_gini": weighted_gini(values, weights),
        "theil_t": theil_t(values, weights),
        "theil_l": theil_l(values, weights),
        "zero_access_population_share": zero_access_population_share(values, weights),
        "bottom_access_quintile_mean": float(
            population_rank_segment_means(values, weights, values, [0.0, 0.2, 1.0])[0]
        ),
    }
    for epsilon in atkinson_epsilons or ():
        label = f"{float(epsilon):g}"
        metrics[f"atkinson_{label}"] = atkinson(values, weights, float(epsilon))
    if income is not None:
        quintiles = population_quintile_means(values, weights, income)
        income_segments = population_rank_segment_means(
            values, weights, income, [0.0, 0.4, 0.9, 1.0]
        )
        metrics.update(
            {
                "concentration_index": concentration_index(values, weights, income),
                "income_quintile_low_mean": float(quintiles[0]),
                "income_quintile_high_mean": float(quintiles[-1]),
                "income_quintile_absolute_gap": float(quintiles[-1] - quintiles[0]),
                "income_quintile_relative_ratio": (
                    float(quintiles[-1] / quintiles[0])
                    if quintiles[0] > 0
                    else (math.inf if quintiles[-1] > 0 else math.nan)
                ),
                "income_bottom40_mean": float(income_segments[0]),
                "income_top10_mean": float(income_segments[-1]),
                "palma_top10_bottom40_ratio": palma_style_access_ratio(
                    values, weights, income
                ),
            }
        )
    rows = []
    for metric, value in metrics.items():
        status = "ok"
        if math.isnan(value):
            status = "undefined_zero_mean"
        elif math.isinf(value):
            status = "undefined_or_infinite_zero_case"
        rows.append({"metric": metric, "estimate": value, "status": status})
    return pd.DataFrame(rows)


def bootstrap_percentile_intervals(
    replicates: pd.DataFrame,
    *,
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """Summarize explicit block-bootstrap replicates without hiding failures."""
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must lie strictly between 0 and 1")
    required = {"replicate", "metric", "estimate", "status"}
    missing = sorted(required.difference(replicates.columns))
    if missing:
        raise ValueError(f"bootstrap replicate table missing columns: {missing}")
    alpha = (1.0 - confidence_level) / 2.0
    rows = []
    for metric in pd.unique(replicates["metric"].dropna()):
        subset = replicates.loc[replicates["metric"] == metric]
        valid = pd.to_numeric(
            subset.loc[subset["status"] == "ok", "estimate"], errors="coerce"
        )
        valid = valid[np.isfinite(valid)]
        rows.append(
            {
                "metric": metric,
                "confidence_level": confidence_level,
                "ci_low": float(valid.quantile(alpha)) if len(valid) else math.nan,
                "ci_high": float(valid.quantile(1.0 - alpha)) if len(valid) else math.nan,
                "valid_replicate_n": int(len(valid)),
                "failed_or_nonfinite_n": int(len(subset) - len(valid)),
                "status": "ok" if len(valid) else "no_valid_replicates",
            }
        )
    return pd.DataFrame(rows)


def block_bootstrap(
    data: pd.DataFrame,
    *,
    block_column: str,
    statistic: Callable[[pd.DataFrame], Mapping[str, float]],
    repetitions: int = 1999,
    seed: int = 20210816,
) -> pd.DataFrame:
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    blocks = list(pd.unique(data[block_column].dropna()))
    if not blocks:
        raise ValueError("No nonmissing bootstrap blocks")
    groups = {block: data.loc[data[block_column] == block] for block in blocks}
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for replicate in range(repetitions):
        sampled = rng.choice(blocks, size=len(blocks), replace=True)
        frame = pd.concat([groups[block] for block in sampled], ignore_index=True)
        try:
            estimates = statistic(frame)
            for metric, estimate in estimates.items():
                rows.append({"replicate": replicate, "metric": metric, "estimate": estimate, "status": "ok"})
        except Exception as exc:  # recorded rather than silently discarded
            rows.append({"replicate": replicate, "metric": None, "estimate": np.nan, "status": f"failed:{type(exc).__name__}"})
    return pd.DataFrame(rows)


def block_weight_multiplier_bootstrap(
    data: pd.DataFrame,
    *,
    block_column: str,
    weight_column: str,
    statistic: Callable[[pd.DataFrame], Mapping[str, float]],
    repetitions: int = 1999,
    seed: int = 20210816,
) -> pd.DataFrame:
    """Efficient cluster bootstrap for population-weighted statistics.

    Sampling a block multiple times is represented by multiplying every row's
    population weight by that block's draw count. This is algebraically
    equivalent to concatenating duplicate block rows for statistics that use
    ``weight_column`` as their frequency/population weight.
    """
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    if block_column not in data or weight_column not in data:
        raise KeyError([column for column in (block_column, weight_column) if column not in data])
    if data[block_column].isna().any():
        raise ValueError("block assignments must be complete")
    weights = pd.to_numeric(data[weight_column], errors="coerce").to_numpy(float)
    if np.any(~np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("bootstrap weights must be finite and positive")
    categories = pd.Categorical(data[block_column])
    codes = categories.codes
    block_n = len(categories.categories)
    if block_n == 0:
        raise ValueError("No nonmissing bootstrap blocks")
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for replicate in range(repetitions):
        draws = rng.integers(0, block_n, size=block_n)
        frequencies = np.bincount(draws, minlength=block_n)
        multiplied = weights * frequencies[codes]
        retained = multiplied > 0
        frame = data.loc[retained].copy()
        frame[weight_column] = multiplied[retained]
        try:
            estimates = statistic(frame)
            for metric, estimate in estimates.items():
                rows.append(
                    {"replicate": replicate, "metric": metric,
                     "estimate": estimate, "status": "ok"}
                )
        except Exception as exc:  # recorded rather than silently discarded
            rows.append(
                {"replicate": replicate, "metric": None, "estimate": np.nan,
                 "status": f"failed:{type(exc).__name__}"}
            )
    return pd.DataFrame(rows)
