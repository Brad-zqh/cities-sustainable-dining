"""External-reference benchmark utilities for dish-level model outputs.

This module deliberately separates agreement with an external recipe table from
independent human validation.  It emits aggregate metrics only; dish names and
identifiers remain inside the restricted input boundary.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats


METRIC_NAMES = (
    "bias",
    "mae",
    "rmse",
    "pearson_r",
    "spearman_rho",
    "ccc",
    "r2",
)


DEFAULT_NUTRITION_DENSITY_TARGETS = {
    "protein": 0.0313,
    "carbohydrate": 0.1625,
    "fat": 0.0306,
    "salt": 0.0025,
}

NUTRITION_DENSITY_SCORE_COLUMNS = (
    "DNDS_protein",
    "NNDS_salt",
    "RNDS_carbohydrate",
    "RNDS_fat",
)


def nutrition_density_score_components(
    frame: pd.DataFrame,
    *,
    density_targets: Mapping[str, float] = DEFAULT_NUTRITION_DENSITY_TARGETS,
) -> pd.DataFrame:
    """Return the four legacy-oriented nutrient-density components.

    Invalid physical inputs remain missing. Protein and salt are benefit-oriented
    in the legacy equation; carbohydrate and fat are absolute deviation
    penalties. No sample-dependent filtering or scaling is performed here.
    """

    required_columns = {"energy_kcal", "protein_g", "fat_g", "carbohydrates_g", "salt_g"}
    missing_columns = sorted(required_columns.difference(frame.columns))
    if missing_columns:
        raise ValueError(f"nutrition score is missing columns: {missing_columns}")
    required_targets = {"protein", "carbohydrate", "fat", "salt"}
    if set(density_targets) != required_targets or any(float(value) <= 0 for value in density_targets.values()):
        raise ValueError(f"density_targets must contain positive values for {sorted(required_targets)}")

    ordered = ["energy_kcal", "protein_g", "fat_g", "carbohydrates_g", "salt_g"]
    numeric = frame[ordered].apply(pd.to_numeric, errors="coerce")
    finite = pd.DataFrame(np.isfinite(numeric), index=numeric.index, columns=numeric.columns).all(axis=1)
    physically_valid = finite & numeric["energy_kcal"].gt(0) & numeric.drop(columns="energy_kcal").ge(0).all(axis=1)
    result = pd.DataFrame(np.nan, index=frame.index, columns=NUTRITION_DENSITY_SCORE_COLUMNS, dtype=float)
    valid = numeric.loc[physically_valid]
    if valid.empty:
        return result

    energy = valid["energy_kcal"]
    result.loc[valid.index, "DNDS_protein"] = (
        (valid["protein_g"] / energy) / float(density_targets["protein"])
    )
    result.loc[valid.index, "NNDS_salt"] = (
        1 - (valid["salt_g"] / energy) / float(density_targets["salt"])
    )
    result.loc[valid.index, "RNDS_carbohydrate"] = (
        ((valid["carbohydrates_g"] / energy) - float(density_targets["carbohydrate"])).abs()
        / float(density_targets["carbohydrate"])
    )
    result.loc[valid.index, "RNDS_fat"] = (
        ((valid["fat_g"] / energy) - float(density_targets["fat"])).abs()
        / float(density_targets["fat"])
    )
    return result


def percentile_anchored_nutrition_quality_score(
    components: pd.DataFrame,
    *,
    component_bounds: Mapping[str, tuple[float, float]],
) -> pd.Series:
    """Return a no-deletion percentile-anchored sensitivity score on [0, 1].

    Each finite component is winsorized to precomputed lower and upper anchors
    and scaled to [0, 1]. The benefit-oriented protein and salt components and
    the reverse-coded carbohydrate and fat deviation components are then
    averaged. The anchors are sample-dependent, so this score is suitable for
    sensitivity analysis rather than the primary, fixed-anchor definition.
    """

    missing_columns = sorted(set(NUTRITION_DENSITY_SCORE_COLUMNS).difference(components.columns))
    if missing_columns:
        raise ValueError(f"nutrition components are missing columns: {missing_columns}")
    if set(component_bounds) != set(NUTRITION_DENSITY_SCORE_COLUMNS):
        raise ValueError("component_bounds must cover all four nutrition density components")

    scaled = pd.DataFrame(index=components.index, dtype=float)
    for column in NUTRITION_DENSITY_SCORE_COLUMNS:
        low, high = (float(value) for value in component_bounds[column])
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            raise ValueError(f"invalid percentile anchors for {column}: {(low, high)}")
        values = pd.to_numeric(components[column], errors="coerce")
        scaled[column] = (values.clip(lower=low, upper=high) - low) / (high - low)

    score = (
        scaled["DNDS_protein"]
        + scaled["NNDS_salt"]
        + (1 - scaled["RNDS_carbohydrate"])
        + (1 - scaled["RNDS_fat"])
    ) / 4
    score.name = "nutrition_score_percentile_anchored_0_1"
    return score


def bounded_nutrition_quality_score(
    frame: pd.DataFrame,
    *,
    density_targets: Mapping[str, float] = DEFAULT_NUTRITION_DENSITY_TARGETS,
) -> pd.Series:
    """Return a fixed, sample-independent nutrition score on [0, 1].

    Protein is an adequacy component capped at one. Salt is a moderation
    component, while carbohydrate and fat are balance components centred on
    fixed density targets. Invalid physical inputs remain missing. The score
    deliberately avoids sample-tail deletion and pooled-sample min-max scaling.
    """

    required_columns = {"energy_kcal", "protein_g", "fat_g", "carbohydrates_g", "salt_g"}
    missing_columns = sorted(required_columns.difference(frame.columns))
    if missing_columns:
        raise ValueError(f"nutrition score is missing columns: {missing_columns}")
    required_targets = {"protein", "carbohydrate", "fat", "salt"}
    if set(density_targets) != required_targets or any(float(value) <= 0 for value in density_targets.values()):
        raise ValueError(f"density_targets must contain positive values for {sorted(required_targets)}")

    numeric = frame[list(required_columns)].apply(pd.to_numeric, errors="coerce")
    finite = pd.DataFrame(np.isfinite(numeric), index=numeric.index, columns=numeric.columns).all(axis=1)
    physically_valid = finite & numeric["energy_kcal"].gt(0) & numeric.drop(columns="energy_kcal").ge(0).all(axis=1)
    result = pd.Series(np.nan, index=frame.index, dtype=float, name="nutrition_score_fixed_0_1")
    valid = numeric.loc[physically_valid]
    if valid.empty:
        return result

    energy = valid["energy_kcal"]
    protein = ((valid["protein_g"] / energy) / float(density_targets["protein"])).clip(0, 1)
    salt = (1 - (valid["salt_g"] / energy) / float(density_targets["salt"])).clip(0, 1)
    carbohydrate = (
        1
        - ((valid["carbohydrates_g"] / energy) - float(density_targets["carbohydrate"])).abs()
        / float(density_targets["carbohydrate"])
    ).clip(0, 1)
    fat = (
        1
        - ((valid["fat_g"] / energy) - float(density_targets["fat"])).abs()
        / float(density_targets["fat"])
    ).clip(0, 1)
    result.loc[valid.index] = pd.concat(
        [protein, salt, carbohydrate, fat], axis=1
    ).mean(axis=1)
    return result


def repeated_cross_validated_linear_calibration(
    reference: Sequence[float],
    prediction: Sequence[float],
    *,
    folds: int = 5,
    repeats: int = 10,
    seed: int = 20260818,
) -> np.ndarray:
    """Calibrate non-negative predictions using repeated out-of-fold fits.

    Each observation is predicted only by linear calibration models fitted on
    other observations. Multiple out-of-fold predictions are combined by their
    median. Negative calibrated nutrient values are truncated at zero.
    """

    x = np.asarray(reference, dtype=float)
    y = np.asarray(prediction, dtype=float)
    if x.shape != y.shape:
        raise ValueError("reference and prediction must have the same shape")
    if folds < 2 or repeats < 1:
        raise ValueError("folds must be at least two and repeats must be positive")
    valid = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (y >= 0)
    valid_indices = np.flatnonzero(valid)
    if valid_indices.size < folds:
        raise ValueError("the number of valid pairs must be at least the number of folds")

    rng = np.random.default_rng(seed)
    out_of_fold: list[list[float]] = [[] for _ in range(x.size)]
    for _ in range(repeats):
        shuffled = valid_indices.copy()
        rng.shuffle(shuffled)
        fold_indices = np.array_split(shuffled, folds)
        for held_out in fold_indices:
            training = np.setdiff1d(valid_indices, held_out, assume_unique=False)
            design = np.column_stack([np.ones(training.size), y[training]])
            intercept, slope = np.linalg.lstsq(design, x[training], rcond=None)[0]
            calibrated = np.maximum(intercept + slope * y[held_out], 0.0)
            for index, value in zip(held_out, calibrated, strict=True):
                out_of_fold[int(index)].append(float(value))

    result = np.full(x.shape, np.nan, dtype=float)
    for index in valid_indices:
        values = out_of_fold[int(index)]
        if len(values) != repeats:
            raise RuntimeError("each valid observation must receive one out-of-fold prediction per repeat")
        result[int(index)] = float(np.median(values))
    return result


def _validate_ids(frame: pd.DataFrame, id_column: str, label: str) -> None:
    if id_column not in frame.columns:
        raise ValueError(f"{label} is missing id column {id_column!r}")
    if frame[id_column].isna().any():
        raise ValueError(f"{label} contains missing identifiers")
    if frame[id_column].duplicated().any():
        raise ValueError(f"{label} contains duplicate identifiers")


def concordance_correlation_coefficient(reference: np.ndarray, prediction: np.ndarray) -> float:
    """Return Lin's concordance correlation coefficient using population moments."""

    x = np.asarray(reference, dtype=float)
    y = np.asarray(prediction, dtype=float)
    if x.size < 2 or y.size != x.size:
        return float("nan")
    variance_sum = np.var(x) + np.var(y)
    mean_difference = float(np.mean(x) - np.mean(y))
    denominator = variance_sum + mean_difference**2
    if denominator == 0:
        return 1.0 if np.array_equal(x, y) else float("nan")
    covariance = float(np.mean((x - np.mean(x)) * (y - np.mean(y))))
    return float(2 * covariance / denominator)


def paired_metrics(reference: Sequence[float], prediction: Sequence[float]) -> dict[str, float]:
    """Calculate paired accuracy and agreement metrics on finite observations."""

    x = np.asarray(reference, dtype=float)
    y = np.asarray(prediction, dtype=float)
    if x.shape != y.shape:
        raise ValueError("reference and prediction must have the same shape")
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size < 3:
        raise ValueError("at least three finite paired observations are required")

    residual = y - x
    total_sum_squares = float(np.sum((x - np.mean(x)) ** 2))
    residual_sum_squares = float(np.sum(residual**2))
    pearson = float(stats.pearsonr(x, y).statistic) if np.std(x) > 0 and np.std(y) > 0 else float("nan")
    spearman = float(stats.spearmanr(x, y).statistic) if np.std(x) > 0 and np.std(y) > 0 else float("nan")
    return {
        "bias": float(np.mean(residual)),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "pearson_r": pearson,
        "spearman_rho": spearman,
        "ccc": concordance_correlation_coefficient(x, y),
        "r2": float(1 - residual_sum_squares / total_sum_squares) if total_sum_squares > 0 else float("nan"),
    }


def bootstrap_metric_intervals(
    reference: Sequence[float],
    prediction: Sequence[float],
    *,
    repetitions: int = 1000,
    seed: int = 20260818,
) -> dict[str, tuple[float, float]]:
    """Return deterministic paired percentile-bootstrap 95% intervals."""

    if repetitions < 1:
        raise ValueError("bootstrap repetitions must be positive")
    x = np.asarray(reference, dtype=float)
    y = np.asarray(prediction, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size < 3:
        raise ValueError("at least three finite paired observations are required")

    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {name: [] for name in METRIC_NAMES}
    for _ in range(repetitions):
        indices = rng.integers(0, x.size, size=x.size)
        metrics = paired_metrics(x[indices], y[indices])
        for name, value in metrics.items():
            if np.isfinite(value):
                draws[name].append(value)
    intervals: dict[str, tuple[float, float]] = {}
    for name, values in draws.items():
        if values:
            low, high = np.percentile(values, [2.5, 97.5])
            intervals[name] = (float(low), float(high))
        else:
            intervals[name] = (float("nan"), float("nan"))
    return intervals


def evaluate_external_benchmark(
    reference: pd.DataFrame,
    predictions: Mapping[str, pd.DataFrame],
    *,
    id_column: str,
    variables: Sequence[str],
    ensemble_methods: Sequence[str] = ("median", "mean"),
    bootstrap_repetitions: int = 1000,
    bootstrap_seed: int = 20260818,
) -> pd.DataFrame:
    """Evaluate individual runs and across-run ensembles against one reference.

    The returned table contains aggregate metrics only.  The status columns are
    intentionally fail-closed: an external reference benchmark cannot, by itself,
    pass an independent-human-validation gate.
    """

    if not predictions:
        raise ValueError("at least one prediction frame is required")
    _validate_ids(reference, id_column, "reference")
    missing_reference = [variable for variable in variables if variable not in reference.columns]
    if missing_reference:
        raise ValueError(f"reference is missing variables: {missing_reference}")

    reference_subset = reference[[id_column, *variables]].copy()
    aligned: dict[str, pd.DataFrame] = {}
    for label, frame in predictions.items():
        _validate_ids(frame, id_column, f"prediction {label}")
        missing = [variable for variable in variables if variable not in frame.columns]
        if missing:
            raise ValueError(f"prediction {label} is missing variables: {missing}")
        renamed = frame[[id_column, *variables]].rename(
            columns={variable: f"{variable}__{label}" for variable in variables}
        )
        aligned[label] = reference_subset.merge(renamed, on=id_column, how="inner", validate="one_to_one")

    rows: list[dict[str, object]] = []

    def append_rows(label: str, aggregation: str, frame: pd.DataFrame, prediction_columns: Mapping[str, str], seed_offset: int) -> None:
        for variable_index, variable in enumerate(variables):
            reference_values = pd.to_numeric(frame[variable], errors="coerce").to_numpy(dtype=float)
            prediction_values = pd.to_numeric(frame[prediction_columns[variable]], errors="coerce").to_numpy(dtype=float)
            finite = np.isfinite(reference_values) & np.isfinite(prediction_values)
            metrics = paired_metrics(reference_values[finite], prediction_values[finite])
            intervals = bootstrap_metric_intervals(
                reference_values[finite],
                prediction_values[finite],
                repetitions=bootstrap_repetitions,
                seed=bootstrap_seed + seed_offset + variable_index,
            )
            row: dict[str, object] = {
                "prediction_label": label,
                "aggregation": aggregation,
                "variable": variable,
                "n_reference": int(len(reference_subset)),
                "n_id_aligned": int(len(frame)),
                "n_complete_pairs": int(finite.sum()),
                "reference_mean": float(np.mean(reference_values[finite])),
                "prediction_mean": float(np.mean(prediction_values[finite])),
                "benchmark_status": "external_reference_benchmark_only_not_human_validation",
                "primary_gate_passed": False,
            }
            for metric_name, value in metrics.items():
                row[metric_name] = value
                row[f"{metric_name}_ci_low"] = intervals[metric_name][0]
                row[f"{metric_name}_ci_high"] = intervals[metric_name][1]
            rows.append(row)

    for run_index, (label, frame) in enumerate(aligned.items()):
        append_rows(
            label,
            "single_run",
            frame,
            {variable: f"{variable}__{label}" for variable in variables},
            seed_offset=10_000 * run_index,
        )

    common = reference_subset.copy()
    for label, frame in predictions.items():
        renamed = frame[[id_column, *variables]].rename(
            columns={variable: f"{variable}__{label}" for variable in variables}
        )
        common = common.merge(renamed, on=id_column, how="inner", validate="one_to_one")

    for method_index, method in enumerate(ensemble_methods):
        if method not in {"median", "mean"}:
            raise ValueError(f"unsupported ensemble method: {method}")
        ensemble = common[[id_column, *variables]].copy()
        columns: dict[str, str] = {}
        for variable in variables:
            run_columns = [f"{variable}__{label}" for label in predictions]
            numeric = common[run_columns].apply(pd.to_numeric, errors="coerce")
            output_column = f"{variable}__ensemble_{method}"
            ensemble[output_column] = numeric.median(axis=1, skipna=False) if method == "median" else numeric.mean(axis=1, skipna=False)
            columns[variable] = output_column
        append_rows(
            f"ensemble_{method}",
            method,
            ensemble,
            columns,
            seed_offset=100_000 + 10_000 * method_index,
        )

    return pd.DataFrame(rows)
