"""Synthetic layout and input-validation tests; not empirical evidence."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import v204_cosmetic_figure_refinements as cosmetic
import v203_fig15_six_primary as primary


class FigureCosmeticTests(unittest.TestCase):
    def tearDown(self) -> None:
        plt.close('all')

    def test_numeric_fingerprint_ignores_display_geometry_but_detects_values(self) -> None:
        fig, ax = plt.subplots()
        line, = ax.plot([0, 1], [1, 2], color='#123456')
        before = cosmetic.data_state(fig)
        ax.set_position([.2, .2, .5, .5])
        self.assertEqual(before, cosmetic.data_state(fig))
        line.set_ydata([1, 3])
        self.assertNotEqual(before, cosmetic.data_state(fig))

    def test_six_mm_gap_preserves_artist_values_and_physical_panel_sizes(self) -> None:
        from types import SimpleNamespace
        from unittest.mock import patch
        fig = plt.figure(figsize=(8, 8))
        axes = [fig.add_axes(bounds) for bounds in
                ([.1, .50, .30, .25], [.6, .50, .30, .25],
                 [.1, .10, .30, .25], [.6, .10, .30, .25])]
        for ax, title in zip(axes, ['Income coefficient attenuation',
                                  'Market-composition coefficients',
                                  'Price-sensitive opportunity', 'Lower-price share']):
            ax.set_title(title, loc='left')
        axes[0].plot([.1, .2], [0, 1], label='Synthetic only')
        axes[0].set_xlabel('Synthetic coefficient')
        axes[0].legend(loc='upper center')
        image = axes[1].imshow([[.1, .2], [.3, .4]], aspect='auto')
        fig.colorbar(image, cax=fig.add_axes([.91, .50, .02, .25]))
        fig.canvas.draw()
        sizes = [(ax.get_position().width * fig.get_figwidth(),
                  ax.get_position().height * fig.get_figheight()) for ax in axes]
        before = cosmetic.data_state(fig)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            with patch.object(cosmetic, 'HERE', target), patch.object(cosmetic, 'OUT', target):
                cosmetic.add_gap(fig, 'synthetic', lambda *args: {}, SimpleNamespace())
        self.assertEqual(before, cosmetic.data_state(fig))
        for ax, (width, height) in zip(axes, sizes):
            self.assertAlmostEqual(ax.get_position().width * fig.get_figwidth(), width)
            self.assertAlmostEqual(ax.get_position().height * fig.get_figheight(), height)

    def test_missing_regression_columns_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'synthetic.csv'
            pd.DataFrame({'year': [2011]}).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, 'Missing regression fields'):
                primary.load_regression(path)

    def test_invalid_ci_and_duplicate_keys_fail_closed(self) -> None:
        rows = []
        for year in primary.YEARS:
            for model, predictors in [('Income only', ['log_income']),
                                      ('Income + market composition', ['log_income', 'log_supply', 'low_price_share', 'ageing_share'])]:
                rows.extend({'year': year, 'model': model, 'predictor': predictor,
                             'standardized_beta': .2, 'ci_low': .1, 'ci_high': .3}
                            for predictor in predictors)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'synthetic.csv'
            frame = pd.DataFrame(rows)
            frame.to_csv(path, index=False)
            self.assertEqual(len(primary.load_regression(path)), 20)
            pd.concat([frame, frame.iloc[:1]]).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                primary.load_regression(path)
            frame.loc[0, 'ci_high'] = .15
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, 'interval'):
                primary.load_regression(path)

    def test_cli_help_reads_no_empirical_inputs(self) -> None:
        run = subprocess.run([sys.executable, str(ROOT / 'scripts/v204_cosmetic_figure_refinements.py'), '--help'],
                             capture_output=True, text=True, check=False)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn('--six-domain-regression', run.stdout)


if __name__ == '__main__':
    unittest.main()
