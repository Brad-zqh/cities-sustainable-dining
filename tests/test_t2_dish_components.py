import unittest

import numpy as np
import pandas as pd

from sus_dining_access.dish_components import (
    aggregate_dish_component_chunks,
    finalize_restaurant_year_candidates,
    summarize_candidate_coverage,
    valid_metric_mask,
    window_years,
)


class DishComponentCandidateTests(unittest.TestCase):
    def test_window_contract(self) -> None:
        years = list(range(2010, 2022))
        self.assertEqual(window_years(2021, "annual", years), [2021])
        self.assertEqual(window_years(2021, "trailing_3_year", years), [2019, 2020, 2021])
        self.assertEqual(window_years(2011, "legacy_cumulative", years), [2010, 2011])

    def test_zero_carbon_is_invalid_but_zero_salt_can_be_observed(self) -> None:
        values = pd.Series([0.0, 1.0, np.nan])
        self.assertEqual(valid_metric_mask(values, "strictly_positive").tolist(), [False, True, False])
        self.assertEqual(valid_metric_mask(values, "nonnegative").tolist(), [True, True, False])

    def test_no_mentions_and_partial_coverage_are_explicit(self) -> None:
        dish = pd.DataFrame(
            {
                "restaurant_id": [1, 1, 2],
                "carbon_emission_g": [100.0, 0.0, 50.0],
                "energy_kcal": [200.0, 100.0, 100.0],
                "protein_g": [10.0, 0.0, 5.0],
                "fat_g": [5.0, 0.0, 2.0],
                "carbohydrates_g": [20.0, 0.0, 10.0],
                "salt_g": [1.0, 0.0, 0.5],
                "2020_count": [0, 0, 0],
                "2021_count": [2, 3, 0],
            }
        )
        sufficient = aggregate_dish_component_chunks(
            [dish], years=[2021], windows=["annual"]
        )
        membership = pd.DataFrame(
            {
                "restaurant_id": [1, 2],
                "operation_early_year": [2020, 2020],
                "operation_latest_year": [2022, 2022],
            }
        )
        result = finalize_restaurant_year_candidates(
            sufficient,
            membership,
            years=[2021],
            windows=["annual"],
        ).set_index("restaurant_id")
        self.assertEqual(result.loc[1, "carbon_emission_g__status"], "observed_partial_estimate_coverage")
        self.assertAlmostEqual(result.loc[1, "carbon_emission_g__coverage"], 2 / 5)
        self.assertAlmostEqual(result.loc[1, "carbon_emission_g__estimate"], 100.0)
        self.assertEqual(result.loc[1, "salt_g__status"], "observed_complete_estimate_coverage")
        self.assertEqual(result.loc[2, "carbon_emission_g__status"], "missing_no_dish_mention_evidence")
        self.assertTrue(np.isnan(result.loc[2, "carbon_emission_g__estimate"]))
        self.assertFalse(bool(result.loc[1, "primary_component_eligible"]))

        summary = summarize_candidate_coverage(
            result.reset_index(),
            metrics=["carbon_emission_g", "salt_g"],
        )
        self.assertEqual(int(summary.iloc[0]["no_dish_mention_n"]), 1)

    def test_negative_count_fails_closed(self) -> None:
        dish = pd.DataFrame(
            {
                "restaurant_id": [1],
                "carbon_emission_g": [1.0],
                "energy_kcal": [1.0],
                "protein_g": [1.0],
                "fat_g": [1.0],
                "carbohydrates_g": [1.0],
                "salt_g": [1.0],
                "2021_count": [-1],
            }
        )
        with self.assertRaises(ValueError):
            aggregate_dish_component_chunks([dish], years=[2021], windows=["annual"])


if __name__ == "__main__":
    unittest.main()
