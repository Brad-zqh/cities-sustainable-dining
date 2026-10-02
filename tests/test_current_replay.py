"""Synthetic contract checks; fixtures are separate from empirical results."""
from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from verify_current import compare_csv, fresh_output, load_module, TASKS


class CurrentReplayTests(unittest.TestCase):
    def test_current_fig12_uses_fixed_temporal_results_with_external_inputs(self) -> None:
        module = load_module('v140_reader_figure_touchups')
        observed = []
        evidence = SimpleNamespace(
            ROOT=ROOT, DATA_ROOT=Path('/separate_authorized_inputs'),
            export=lambda *args: None,
            sdi=SimpleNamespace(base=SimpleNamespace(YEAR_RAMPS={}), MAP_RAMPS={}),
        )
        evidence.sdi_composite = lambda: observed.append(evidence.TEMPORAL_DATA_DIR)
        with patch.object(module.bridge, 'load_original', return_value=(evidence, ROOT / 'renderer.py')):
            with patch.object(module, 'sha256', return_value='test-only'):
                module.render_12_16('12')
        self.assertEqual(observed, [module.bridge.TEMPORAL])
        self.assertNotEqual(observed[0], evidence.DATA_ROOT / 'source_data/figS_temporal_uncertainty_v4')

    def test_comparison_aligns_keys_and_preserves_nan_bool(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / 'a.csv', Path(directory) / 'b.csv'
            frame = pd.DataFrame({'id': [1, 2], 'x': [np.nan, .5], 'flag': [True, False]})
            frame.to_csv(a, index=False)
            frame.iloc[::-1].to_csv(b, index=False)
            self.assertTrue(compare_csv(a, b, ['id'])['matches_reference'])
            frame.loc[0, 'x'] = 0.
            frame.to_csv(b, index=False)
            self.assertFalse(compare_csv(a, b, ['id'])['matches_reference'])

    def test_comparison_rejects_duplicate_keys_and_changed_identifiers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / 'a.csv', Path(directory) / 'b.csv'
            pd.DataFrame({'id': ['a', 'b'], 'x': [0., 1.]}).to_csv(a, index=False)
            pd.DataFrame({'id': ['a', 'c'], 'x': [0., 1.]}).to_csv(b, index=False)
            self.assertFalse(compare_csv(a, b, ['id'])['matches_reference'])
            pd.DataFrame({'id': ['a', 'a'], 'x': [0., 1.]}).to_csv(b, index=False)
            with self.assertRaises(ValueError):
                compare_csv(a, b, ['id'])

    def test_inputs_and_outputs_are_disjoint_and_previous_runs_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            data = base / 'data'
            data.mkdir()
            for output in (data, data / 'results', base):
                with self.assertRaises(ValueError):
                    fresh_output(output, data)
            output = base / 'results'
            fresh_output(output, data)
            output.mkdir()
            (output / 'previous.txt').write_text('preserve me', encoding='utf-8')
            with self.assertRaises(FileExistsError):
                fresh_output(output, data)
            self.assertEqual((output / 'previous.txt').read_text(encoding='utf-8'), 'preserve me')

    def test_empty_input_does_not_generate_demo_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / 'data').mkdir()
            process = subprocess.run([
                sys.executable, str(ROOT / 'verify_current.py'), '--tasks', 'aggregation',
                '--data-root', str(base / 'data'), '--output-dir', str(base / 'results'), '--authorized'
            ], capture_output=True, text=True, encoding='utf-8', check=False)
            self.assertNotEqual(process.returncode, 0)
            self.assertFalse((base / 'results').exists())

    def test_primary_score_checks_crs_bounds_and_missingness(self) -> None:
        module = load_module('v135_build_fixed_sdi_aggregation_variants')
        weights, _ = module.load_v4_module()
        frame = gpd.GeoDataFrame(
            {**{c: [0., 1., np.nan] for c in weights.COMPONENTS},
             'sdi_equal_arithmetic': [0., 1., np.nan], 'nutrition_coverage': [1., 1., 0.],
             'carbon_coverage': [1., 1., 1.], 't_pop': [1., 2., 3.]},
            geometry=[Point(0, 0), Point(1, 1), Point(2, 2)], crs='EPSG:2326')
        module.verify_primary(frame, weights, (2024, 'lsbg'))
        bad = frame.copy()
        bad.loc[2, 'sdi_equal_arithmetic'] = 0.
        with self.assertRaises(ValueError):
            module.verify_primary(bad, weights, (2024, 'lsbg'))
        bad = frame.copy()
        bad.loc[0, weights.COMPONENTS[0]] = -1.
        with self.assertRaises(ValueError):
            module.verify_primary(bad, weights, (2024, 'lsbg'))
        with self.assertRaises(ValueError):
            module.verify_primary(frame.set_crs(None, allow_override=True), weights, (2024, 'lsbg'))

    def test_current_registry_and_no_private_paths(self) -> None:
        spec = json.loads((ROOT / 'manifests/v180_figure_entry_points.json').read_text(encoding='utf-8'))
        self.assertEqual(set(spec['figures']), {str(i) for i in range(12, 20)})
        self.assertEqual(len(TASKS), 8)
        for record in spec['figures'].values():
            self.assertTrue((ROOT / record['entry_point']).is_file())
        for path in [ROOT / 'verify_current.py', ROOT / 'verify_documents.py',
                     *sorted((ROOT / 'scripts').glob('v4_*.py'))]:
            text = path.read_text(encoding='utf-8')
            for forbidden in ('D:\\OneDrive', 'E:\\UserData', 'C:\\Users', 'ghp_'):
                self.assertNotIn(forbidden, text, path)


if __name__ == '__main__':
    unittest.main()
