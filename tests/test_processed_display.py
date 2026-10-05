"""Disclosure, integrity and frozen-result contract tests for the public bundle."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from reproduce_display import DATA, validate_bundle
from build_display_tables import build_tables


class DisplayBundleTests(unittest.TestCase):
    def test_public_bundle_and_primary_specification(self):
        self.assertEqual(len(validate_bundle()),52)
        primary = pd.read_csv(DATA/'market/quality_market_decomposition.csv')
        comparison = pd.read_csv(DATA/'regression_comparison/income_models_six_and_five.csv')
        six = comparison.loc[comparison.specification.eq('Six-domain primary')].drop(columns='specification')
        keys = ['year','model','predictor']
        pd.testing.assert_frame_equal(primary.sort_values(keys).reset_index(drop=True),six.sort_values(keys).reset_index(drop=True))
        row = primary.query('year == 2024 and model == "Income + market composition" and predictor == "log_income"').iloc[0]
        self.assertEqual(f'{row.standardized_beta:+.3f}','+0.151')
        self.assertEqual(f'{row.ci_low:+.3f}','+0.059')
        self.assertEqual(f'{row.ci_high:+.3f}','+0.243')

    def test_group_weights_and_result_joins(self):
        tables = build_tables()
        groups = tables['census_group_joint_opportunity_2024']
        self.assertEqual(len(groups),23)
        self.assertEqual(groups.is_reference.sum(),6)
        income = groups.loc[groups.domain.eq('Household income')].set_index('group')
        self.assertEqual(income.loc['<HK$10k','group_denominator'],522029)
        contrasts = tables['six_census_domain_contrasts_2024']
        self.assertEqual(len(contrasts),6)
        self.assertTrue(contrasts.q_bh_within_domain_year.between(0,1).all())
        planning = tables['planning_outcomes'].query('budget_sites == 10').set_index('scenario')
        self.assertEqual(f'{100*planning.loc["zero_affordability_gap","zero_affordability_priority_reached_share"]:.2f}','49.70')
        self.assertEqual(f'{100*planning.loc["population_reach","zero_affordability_priority_reached_share"]:.2f}','4.31')

    def fixture(self, root, column='year'):
        path = root/'small.csv'
        pd.DataFrame({column:[2024]}).to_csv(path,index=False)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest = {'files':[dict(path='small.csv',rows=1,columns=[column],sha256=digest,bytes=path.stat().st_size)]}
        (root/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
        pd.DataFrame({'file':['small.csv'],'sha256':[digest]}).to_csv(root/'SHA256SUMS.csv',index=False)

    def test_changed_or_extra_file_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            validate_bundle(root)
            (root/'preview.png').write_bytes(b'not a released asset')
            with self.assertRaisesRegex(ValueError,'Unexpected'):
                validate_bundle(root)
            (root/'preview.png').unlink()
            (root/'small.csv').write_text('year\n2023\n',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'Changed'):
                validate_bundle(root)

    def test_sensitive_field_and_path_traversal_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root,'restaurant_id')
            with self.assertRaisesRegex(ValueError,'Unsafe field'):
                validate_bundle(root)
            manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
            manifest['files'][0]['path'] = '../external.csv'
            (root/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'Invalid bundle path'):
                validate_bundle(root)


if __name__ == '__main__':
    unittest.main()
