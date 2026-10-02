"""Audit the legacy dish nutrition transformation without endorsing it."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping

import numpy as np
import pandas as pd


NUTRIENT_COLUMNS = ("protein_g", "fat_g", "carbohydrates_g", "salt_g")
SCORE_COLUMNS = (
    "DNDS_protein",
    "NNDS_salt",
    "RNDS_carbohydrate",
    "RNDS_fat",
)


@dataclass(frozen=True)
class NutritionLegacyAudit:
    summary: dict[str, int | float | str]
    thresholds: pd.DataFrame
    retained_mask: pd.Series


def audit_legacy_nutrition_transform(
    frame: pd.DataFrame,
    *,
    density_targets: Mapping[str, float],
    lower_quantile: float = 0.0025,
    upper_quantile: float = 0.9975,
) -> NutritionLegacyAudit:
    """Replicate and quantify the legacy sequential row-deletion rule.

    Finite records outside sample quantiles are counted as removed, not called
    erroneous.  The function makes no recommendation that such removal is valid.
    """

    required = {"energy_kcal", *NUTRIENT_COLUMNS}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"missing nutrition columns: {missing}")
    if not 0 <= lower_quantile < upper_quantile <= 1:
        raise ValueError("quantile bounds must satisfy 0 <= lower < upper <= 1")
    target_keys = {"protein", "fat", "carbohydrate", "salt"}
    if set(density_targets) != target_keys or any(float(value) <= 0 for value in density_targets.values()):
        raise ValueError(f"density_targets must contain positive values for {sorted(target_keys)}")

    numeric = frame[["energy_kcal", *NUTRIENT_COLUMNS]].apply(pd.to_numeric, errors="coerce")
    finite = pd.DataFrame(np.isfinite(numeric), index=numeric.index, columns=numeric.columns)
    energy_invalid = ~finite["energy_kcal"] | numeric["energy_kcal"].le(0)
    nutrient_missing_or_nonfinite = ~finite[list(NUTRIENT_COLUMNS)].all(axis=1)
    nutrient_negative = numeric[list(NUTRIENT_COLUMNS)].lt(0).any(axis=1)
    base_valid = ~(energy_invalid | nutrient_missing_or_nonfinite | nutrient_negative)

    valid = numeric.loc[base_valid].copy()
    energy = valid["energy_kcal"]
    scores = pd.DataFrame(index=valid.index)
    scores["DNDS_protein"] = (valid["protein_g"] / energy) / density_targets["protein"]
    scores["NNDS_salt"] = 1 - (valid["salt_g"] / energy) / density_targets["salt"]
    scores["RNDS_carbohydrate"] = (
        (valid["carbohydrates_g"] / energy - density_targets["carbohydrate"]).abs()
        / density_targets["carbohydrate"]
    )
    scores["RNDS_fat"] = (
        (valid["fat_g"] / energy - density_targets["fat"]).abs()
        / density_targets["fat"]
    )
    score_finite = pd.DataFrame(np.isfinite(scores), index=scores.index, columns=scores.columns).all(axis=1)
    retained = pd.Series(False, index=frame.index)
    retained.loc[scores.index[score_finite]] = True

    threshold_rows: list[dict[str, int | float | str]] = []
    independent_tail_flag = pd.Series(False, index=scores.index)
    for score_column in SCORE_COLUMNS:
        base_values = scores.loc[score_finite, score_column]
        base_low = float(base_values.quantile(lower_quantile))
        base_high = float(base_values.quantile(upper_quantile))
        independent_tail_flag.loc[base_values.index] |= base_values.lt(base_low) | base_values.gt(base_high)

        current_index = retained[retained].index
        current_values = scores.loc[current_index, score_column]
        low = float(current_values.quantile(lower_quantile))
        high = float(current_values.quantile(upper_quantile))
        keep = current_values.ge(low) & current_values.le(high)
        before = int(len(current_values))
        retained.loc[current_index[~keep]] = False
        threshold_rows.append(
            {
                "score_column": score_column,
                "lower_quantile": lower_quantile,
                "upper_quantile": upper_quantile,
                "sequential_threshold_low": low,
                "sequential_threshold_high": high,
                "rows_before": before,
                "rows_removed": int((~keep).sum()),
                "rows_after": int(keep.sum()),
            }
        )

    summary: dict[str, int | float | str] = {
        "input_rows": int(len(frame)),
        "invalid_energy_missing_nonfinite_or_nonpositive": int(energy_invalid.sum()),
        "invalid_nutrient_missing_or_nonfinite": int(nutrient_missing_or_nonfinite.sum()),
        "invalid_nutrient_negative": int(nutrient_negative.sum()),
        "base_physically_valid_rows": int(base_valid.sum()),
        "score_nonfinite_rows_after_base_filter": int((~score_finite).sum()),
        "finite_rows_flagged_by_independent_sample_tails": int(independent_tail_flag.sum()),
        "legacy_sequential_quantile_removed_rows": int(score_finite.sum() - retained.sum()),
        "legacy_retained_rows": int(retained.sum()),
        "legacy_retained_share_of_input": float(retained.mean()) if len(retained) else float("nan"),
        "audit_interpretation": "Sample-tail deletion is a modeling choice, not evidence that finite records are erroneous.",
    }
    for column in NUTRIENT_COLUMNS:
        values = numeric[column]
        summary[f"{column}_zero_rows"] = int(values.eq(0).sum())
        summary[f"{column}_negative_rows"] = int(values.lt(0).sum())
        summary[f"{column}_missing_or_nonfinite_rows"] = int((~finite[column]).sum())
    return NutritionLegacyAudit(summary, pd.DataFrame(threshold_rows), retained)
