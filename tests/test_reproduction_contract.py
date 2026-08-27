"""Synthetic unit fixtures only; these values never enter manuscript figures."""
from pathlib import Path
import sys
import unittest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]
from presentation_finish import finish
from fig09_social_within_year import bh_adjust, significance_stars
from sus_dining_access.inequality import weighted_gini, zero_access_population_share
from render_base import JOBS
sys.path.insert(0,str(ROOT))
from reproduce import FIGURE_MAP


class ContractTests(unittest.TestCase):
    def test_all_main_data_figures_registered(self):
        self.assertEqual(set(JOBS), set(range(3, 17)))
        self.assertEqual(set(FIGURE_MAP),set(range(3,15)))
        self.assertEqual(FIGURE_MAP[5],'A')
        self.assertEqual(FIGURE_MAP[8],'B')

    def test_star_thresholds(self):
        self.assertEqual(significance_stars(.008), '**')
        self.assertEqual(significance_stars(.03), '*')
        self.assertEqual(significance_stars(.05), '')
        self.assertEqual(significance_stars(.0005), '***')

    def test_bh_known_example(self):
        np.testing.assert_allclose(bh_adjust(np.array([.04, .001, .03])), [.04, .003, .04])

    def test_equal_distribution_gini(self):
        self.assertAlmostEqual(weighted_gini([4, 4, 4], [1, 2, 1]), 0.)

    def test_gini_permutation_invariance(self):
        self.assertAlmostEqual(weighted_gini([0, 2, 5], [2, 3, 1]),
                               weighted_gini([5, 0, 2], [1, 2, 3]))

    def test_zero_access_population_not_row_share(self):
        self.assertAlmostEqual(zero_access_population_share([0, 5, 2], [8, 1, 1]), .8)

    def test_heatmap_rules_preserve_values_and_limits(self):
        fig, ax = plt.subplots()
        values = np.array([[1., np.nan], [3., 4.]])
        im = ax.imshow(values, vmin=0, vmax=5)
        lim = (ax.get_xlim(), ax.get_ylim(), im.get_clim())
        finish(fig, heatmap_grid=True, heatmap_grid_linewidth=.5)
        np.testing.assert_equal(np.asarray(im.get_array()), values)
        self.assertEqual((ax.get_xlim(), ax.get_ylim(), im.get_clim()), lim)
        collections = list(ax.collections)
        finish(fig, heatmap_grid=True, heatmap_grid_linewidth=.5)
        self.assertEqual(list(ax.collections), collections)
        grid = [c for c in collections if c.get_gid() in
                ('heatmap-column-dividers', 'heatmap-row-dividers')]
        self.assertEqual(len(grid), 2)
        for c in grid:
            np.testing.assert_allclose(c.get_linewidths(), .5)
        plt.close(fig)


if __name__ == '__main__':
    unittest.main()
