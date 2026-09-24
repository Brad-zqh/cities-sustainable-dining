import unittest

import numpy as np
import pandas as pd

from sus_dining_access.external_benchmark import (
    bounded_nutrition_quality_score,
    bootstrap_metric_intervals,
    concordance_correlation_coefficient,
    evaluate_external_benchmark,
    paired_metrics,
    nutrition_density_score_components,
    percentile_anchored_nutrition_quality_score,
    repeated_cross_validated_linear_calibration,
)


class ExternalBenchmarkTests(unittest.TestCase):
    def test_fixed_nutrition_score_is_bounded_and_sample_independent(self) -> None:
        target = pd.DataFrame(
            {
                "energy_kcal": [100.0],
                "protein_g": [3.13],
                "fat_g": [3.06],
                "carbohydrates_g": [16.25],
                "salt_g": [0.0],
            }
        )
        first = bounded_nutrition_quality_score(target)
        expanded = pd.concat(
            [
                target,
                pd.DataFrame(
                    {
                        "energy_kcal": [1.0],
                        "protein_g": [1000.0],
                        "fat_g": [1000.0],
                        "carbohydrates_g": [1000.0],
                        "salt_g": [1000.0],
                    }
                ),
            ],
            ignore_index=True,
        )
        second = bounded_nutrition_quality_score(expanded)
        self.assertAlmostEqual(first.iloc[0], 1.0)
        self.assertAlmostEqual(second.iloc[0], first.iloc[0])
        self.assertTrue(second.between(0, 1).all())

    def test_fixed_nutrition_score_leaves_invalid_inputs_missing(self) -> None:
        invalid = pd.DataFrame(
            {
                "energy_kcal": [0.0, 100.0],
                "protein_g": [10.0, -1.0],
                "fat_g": [1.0, 1.0],
                "carbohydrates_g": [10.0, 10.0],
                "salt_g": [1.0, 1.0],
            }
        )
        self.assertTrue(bounded_nutrition_quality_score(invalid).isna().all())

    def test_percentile_anchored_score_winsorizes_without_deleting_finite_rows(self) -> None:
        frame = pd.DataFrame(
            {
                "energy_kcal": [100.0, 100.0, 100.0],
                "protein_g": [0.0, 3.13, 1000.0],
                "fat_g": [0.0, 3.06, 1000.0],
                "carbohydrates_g": [0.0, 16.25, 1000.0],
                "salt_g": [0.0, 0.25, 1000.0],
            }
        )
        components = nutrition_density_score_components(frame)
        bounds = {
            column: (float(components[column].quantile(0.01)), float(components[column].quantile(0.99)))
            for column in components
        }
        score = percentile_anchored_nutrition_quality_score(
            components,
            component_bounds=bounds,
        )
        self.assertEqual(score.notna().sum(), 3)
        self.assertTrue(score.between(0, 1).all())

    def test_percentile_anchored_score_preserves_invalid_inputs_as_missing(self) -> None:
        frame = pd.DataFrame(
            {
                "energy_kcal": [100.0, 0.0],
                "protein_g": [3.13, 3.13],
                "fat_g": [3.06, 3.06],
                "carbohydrates_g": [16.25, 16.25],
                "salt_g": [0.25, 0.25],
            }
        )
        components = nutrition_density_score_components(frame)
        bounds = {
            "DNDS_protein": (0.0, 2.0),
            "NNDS_salt": (-1.0, 1.0),
            "RNDS_carbohydrate": (0.0, 2.0),
            "RNDS_fat": (0.0, 2.0),
        }
        score = percentile_anchored_nutrition_quality_score(
            components,
            component_bounds=bounds,
        )
        self.assertTrue(np.isfinite(score.iloc[0]))
        self.assertTrue(np.isnan(score.iloc[1]))

    def test_cross_validated_calibration_is_deterministic_and_out_of_fold(self) -> None:
        prediction = np.arange(1.0, 21.0)
        reference = 1.5 + 2.0 * prediction
        first = repeated_cross_validated_linear_calibration(
            reference, prediction, folds=5, repeats=4, seed=11
        )
        second = repeated_cross_validated_linear_calibration(
            reference, prediction, folds=5, repeats=4, seed=11
        )
        np.testing.assert_allclose(first, reference, atol=1e-10)
        np.testing.assert_allclose(second, first)

    def test_cross_validated_calibration_preserves_missing_inputs(self) -> None:
        reference = np.arange(1.0, 13.0)
        prediction = reference.copy()
        prediction[3] = np.nan
        calibrated = repeated_cross_validated_linear_calibration(
            reference, prediction, folds=3, repeats=2, seed=5
        )
        self.assertTrue(np.isnan(calibrated[3]))
        np.testing.assert_allclose(calibrated[np.isfinite(calibrated)], reference[np.isfinite(calibrated)])

    def test_identical_values_have_perfect_agreement(self) -> None:
        values = np.array([1.0, 2.0, 4.0, 8.0])
        metrics = paired_metrics(values, values)
        self.assertAlmostEqual(metrics["bias"], 0.0)
        self.assertAlmostEqual(metrics["mae"], 0.0)
        self.assertAlmostEqual(metrics["rmse"], 0.0)
        self.assertAlmostEqual(metrics["pearson_r"], 1.0)
        self.assertAlmostEqual(metrics["spearman_rho"], 1.0)
        self.assertAlmostEqual(metrics["ccc"], 1.0)
        self.assertAlmostEqual(metrics["r2"], 1.0)

    def test_ccc_penalizes_shift_despite_perfect_correlation(self) -> None:
        reference = np.array([1.0, 2.0, 3.0, 4.0])
        prediction = reference + 2.0
        self.assertAlmostEqual(paired_metrics(reference, prediction)["pearson_r"], 1.0)
        self.assertLess(concordance_correlation_coefficient(reference, prediction), 1.0)

    def test_duplicate_ids_fail_closed(self) -> None:
        reference = pd.DataFrame({"dish_id": [1, 1, 2], "energy": [1, 2, 3]})
        prediction = pd.DataFrame({"dish_id": [1, 2, 3], "energy": [1, 2, 3]})
        with self.assertRaises(ValueError):
            evaluate_external_benchmark(
                reference,
                {"run1": prediction},
                id_column="dish_id",
                variables=["energy"],
                bootstrap_repetitions=10,
            )

    def test_ensemble_is_aggregate_only_and_never_passes_human_gate(self) -> None:
        reference = pd.DataFrame({"dish_id": [1, 2, 3, 4], "energy": [10, 20, 30, 40]})
        predictions = {
            "run1": pd.DataFrame({"dish_id": [1, 2, 3, 4], "energy": [9, 19, 29, 39]}),
            "run2": pd.DataFrame({"dish_id": [1, 2, 3, 4], "energy": [11, 21, 31, 41]}),
        }
        result = evaluate_external_benchmark(
            reference,
            predictions,
            id_column="dish_id",
            variables=["energy"],
            bootstrap_repetitions=20,
            bootstrap_seed=7,
        )
        self.assertEqual(set(result["aggregation"]), {"single_run", "median", "mean"})
        ensemble = result[result["aggregation"] == "median"].iloc[0]
        self.assertAlmostEqual(ensemble["mae"], 0.0)
        self.assertEqual(
            ensemble["benchmark_status"],
            "external_reference_benchmark_only_not_human_validation",
        )
        self.assertFalse(bool(ensemble["primary_gate_passed"]))
        self.assertNotIn("dish_id", result.columns)

    def test_bootstrap_is_deterministic(self) -> None:
        reference = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
        prediction = np.array([1.1, 1.9, 3.8, 8.4, 15.0])
        first = bootstrap_metric_intervals(reference, prediction, repetitions=25, seed=3)
        second = bootstrap_metric_intervals(reference, prediction, repetitions=25, seed=3)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
