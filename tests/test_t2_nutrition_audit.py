import unittest

import pandas as pd

from sus_dining_access.nutrition_audit import audit_legacy_nutrition_transform


TARGETS = {
    "protein": 0.0313,
    "carbohydrate": 0.1625,
    "fat": 0.0306,
    "salt": 0.0025,
}


class NutritionLegacyAuditTests(unittest.TestCase):
    def test_invalid_values_are_distinguished_from_sample_tails(self) -> None:
        frame = pd.DataFrame(
            {
                "energy_kcal": [100, 200, 0, 150, 300, 400],
                "protein_g": [5, 10, 2, -1, 30, 8],
                "fat_g": [3, 6, 1, 2, 20, 5],
                "carbohydrates_g": [16, 32, 5, 20, 80, 50],
                "salt_g": [0.2, 0.4, 0.1, 0.2, 3.0, 0.3],
            }
        )
        audit = audit_legacy_nutrition_transform(
            frame,
            density_targets=TARGETS,
            lower_quantile=0.0,
            upper_quantile=1.0,
        )
        self.assertEqual(audit.summary["invalid_energy_missing_nonfinite_or_nonpositive"], 1)
        self.assertEqual(audit.summary["invalid_nutrient_negative"], 1)
        self.assertEqual(audit.summary["base_physically_valid_rows"], 4)
        self.assertEqual(audit.summary["legacy_sequential_quantile_removed_rows"], 0)

    def test_sequential_tail_filter_reports_deletion(self) -> None:
        frame = pd.DataFrame(
            {
                "energy_kcal": [100.0] * 10,
                "protein_g": [3, 3, 3, 3, 3, 3, 3, 3, 3, 100],
                "fat_g": [3.0] * 10,
                "carbohydrates_g": [16.0] * 10,
                "salt_g": [0.2] * 10,
            }
        )
        audit = audit_legacy_nutrition_transform(
            frame,
            density_targets=TARGETS,
            lower_quantile=0.0,
            upper_quantile=0.9,
        )
        self.assertGreater(audit.summary["legacy_sequential_quantile_removed_rows"], 0)
        self.assertLess(audit.summary["legacy_retained_rows"], len(frame))

    def test_bad_target_contract_fails_closed(self) -> None:
        frame = pd.DataFrame(
            {
                "energy_kcal": [100, 200, 300],
                "protein_g": [1, 2, 3],
                "fat_g": [1, 2, 3],
                "carbohydrates_g": [1, 2, 3],
                "salt_g": [1, 2, 3],
            }
        )
        with self.assertRaises(ValueError):
            audit_legacy_nutrition_transform(frame, density_targets={"protein": 1})


if __name__ == "__main__":
    unittest.main()
