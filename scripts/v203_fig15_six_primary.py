"""V203 Fig. 15 six-domain primary coefficients and author-approved typography.

Uses the fixed V135 empirical inputs and V148 plot lineage. No estimation.
Maps retain their geographic aspect; statistical panels share a column grid.
"""
from pathlib import Path
import hashlib
import json
import os
import argparse
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox

SCRIPTS = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('CITIES_FIGURE_DIR', str(SCRIPTS.parent / 'figures/v203')))
sys.path.insert(0, str(SCRIPTS))
import v148_panel_precision as prior
import v140_reader_figure_touchups as parent

OUT = ROOT
REGRESSION_FILE = None
YEARS = (2011, 2016, 2021, 2024)
YEAR_COLOURS = {2011: '#BC3E52', 2016: '#C87924', 2021: '#2C875F', 2024: '#2B70AF'}
HEIGHT_FACTOR = 1.40
BASE_EXPORT = prior._base_export


def shade(colour, white_fraction=.42):
    return tuple((1-white_fraction)*np.array(to_rgb(colour)) + white_fraction)


def numeric_axis(ax):
    """Geometry/data fingerprint, independent of grouping artists or colours."""
    lines = sorted(tuple(np.round(row, 12)) for line in ax.lines
                   for row in np.asarray(line.get_xydata(), dtype=float))
    segments = []
    collections = []
    for item in ax.collections:
        if hasattr(item, 'get_segments'):
            segments.extend(tuple(np.round(np.asarray(seg).ravel(), 12))
                            for seg in item.get_segments())
        else:
            collections.append({
                'offsets': np.asarray(item.get_offsets()).tolist(),
                'array': None if item.get_array() is None else np.asarray(item.get_array()).tolist(),
                'paths_sha256': hashlib.sha256(b''.join(np.asarray(p.vertices).tobytes()
                                                       for p in item.get_paths())).hexdigest(),
            })
    return {'line_points': lines, 'ci_segments': sorted(segments),
            'collections': collections,
            'images': [np.asarray(im.get_array()).tolist() for im in ax.images],
            'limits': [np.asarray(ax.get_xlim()).tolist(), np.asarray(ax.get_ylim()).tolist()],
            'ticks': [np.asarray(ax.get_xticks()).tolist(), np.asarray(ax.get_yticks()).tolist()]}


def fingerprint(state):
    return hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()


def load_regression(path: Path) -> pd.DataFrame:
    """Validate an explicit coefficient table without estimating a model."""
    frame = pd.read_csv(path)
    required = {'year', 'model', 'predictor', 'standardized_beta', 'ci_low', 'ci_high'}
    if not required.issubset(frame.columns):
        raise ValueError(f'Missing regression fields: {sorted(required - set(frame.columns))}')
    if frame.duplicated(['year', 'model', 'predictor']).any():
        raise ValueError('Duplicate regression keys')
    expected_keys = {(year, 'Income only', 'log_income') for year in YEARS}
    expected_keys.update((year, 'Income + market composition', predictor)
                         for year in YEARS for predictor in
                         ('log_income', 'log_supply', 'low_price_share', 'ageing_share'))
    keys = set(frame[['year', 'model', 'predictor']].itertuples(index=False, name=None))
    if not expected_keys.issubset(keys):
        raise ValueError('Required benchmark years or model predictors are missing')
    values = frame[['ci_low', 'standardized_beta', 'ci_high']].to_numpy(float)
    if not np.isfinite(values).all() or (values[:, 0] > values[:, 1]).any() or (values[:, 1] > values[:, 2]).any():
        raise ValueError('Non-finite estimates or invalid interval endpoints')
    return frame


def recolour_coefficients(ax, regression):
    """Reuse the exact source estimates and interval endpoints, one hue/year."""
    old_limits = ax.get_xlim(), ax.get_ylim()
    title = ax.get_title(loc='left')
    ax.clear()
    subset = regression.loc[regression.predictor.eq('log_income')]
    rows = []
    models = [('Income only', -.10, 'dark'),
              ('Income + market composition', .10, 'light')]
    for model, offset, lightness in models:
        values = subset.loc[subset.model.eq(model)].set_index('year').loc[list(YEARS)]
        for index, year in enumerate(YEARS):
            row = values.loc[year]
            x = float(row.standardized_beta)
            low, high = float(row.ci_low), float(row.ci_high)
            y = index + offset
            colour = YEAR_COLOURS[year] if lightness == 'dark' else shade(YEAR_COLOURS[year])
            error = np.array([[x-low], [high-x]])
            ax.errorbar([x], [y], xerr=error, fmt='none', ecolor=colour,
                        elinewidth=5.0, alpha=.12, capsize=0, zorder=1)
            ax.errorbar([x], [y], xerr=error, fmt='o', color=colour, ecolor=colour,
                        elinewidth=1.25, capsize=2.1, markersize=4.5,
                        markeredgecolor='white', markeredgewidth=.5, zorder=3)
            rows.append({'year': year, 'model': model, 'coefficient': x,
                         'ci_low': low, 'ci_high': high, 'y': y,
                         'colour': colour, 'shade': lightness})
    ax.axvline(0, color='#7B858C', linewidth=.7, linestyle='--')
    ax.set_yticks(np.arange(4), YEARS)
    ax.set_xlim(old_limits[0]); ax.set_ylim(old_limits[1])
    ax.set_xlabel('Standardized income coefficient (95% CI)')
    ax.set_title(title, loc='left', pad=4, fontweight='normal')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(False); ax.set_axisbelow(True)
    ax.text(-.16, 1.02, 'b', transform=ax.transAxes, fontsize=9.2,
            fontweight='bold', va='bottom', ha='right')
    # Neutral samples encode shade, not a second conflicting year-colour key.
    ax.legend(handles=[
        Line2D([], [], marker='o', color='#444444', lw=1.25, markersize=4.5,
               markeredgecolor='white', markeredgewidth=.5, label='Income only (dark)'),
        Line2D([], [], marker='o', color='#AAAAAA', lw=1.25, markersize=4.5,
               markeredgecolor='white', markeredgewidth=.5, label='Adjusted model (light)'),
    ], loc='upper center', bbox_to_anchor=(.50, -.28/HEIGHT_FACTOR), ncol=2,
       frameon=False, fontsize=6.7, handlelength=1.1, handletextpad=.4, columnspacing=.8)
    return rows


def refined_export(fig, stem):
    # Keep automatic tick positions fixed when larger type changes locator density.
    from matplotlib.ticker import FixedLocator
    for ax in fig.axes:
        ax.xaxis.set_major_locator(FixedLocator(ax.get_xticks()))
        ax.yaxis.set_major_locator(FixedLocator(ax.get_yticks()))
    before = [numeric_axis(ax) for ax in fig.axes]
    coefficient = next(ax for ax in fig.axes if ax.get_title(loc='left') == 'Income coefficient attenuation')
    market = next(ax for ax in fig.axes if ax.get_title(loc='left') == 'Market-composition coefficients')
    key = market.images[0].colorbar.ax
    if REGRESSION_FILE is None:
        raise ValueError('Provide the separately authorized six-domain regression CSV explicitly')
    regression = load_regression(Path(REGRESSION_FILE))
    rows = recolour_coefficients(coefficient, regression)
    expected = regression.loc[regression.model.eq('Income + market composition')].pivot(
        index='predictor', columns='year', values='standardized_beta').loc[
            ['log_income','log_supply','low_price_share','ageing_share'], list(YEARS)].to_numpy(float)
    market.images[0].set_data(expected)
    for item in market.texts:
        if item.get_text() == 'c':
            continue
        col, row = map(lambda v: int(round(v)), item.get_position())
        if 0 <= row < 4 and 0 <= col < 4:
            item.set_text(f'{expected[row, col]:+.2f}')
    coefficient.xaxis.set_major_locator(FixedLocator(before[fig.axes.index(coefficient)]['ticks'][0]))
    # V189 had a 12 mm canvas extension. Add just the taller b/c row on top.
    old_height = fig.get_figheight()
    row_added = coefficient.get_position().height*old_height*(HEIGHT_FACTOR-1)
    extra = 12/25.4 + row_added
    fig.set_size_inches(fig.get_figwidth(), old_height+extra, forward=True)
    positions = []
    for ax in fig.axes:
        box = ax.get_position(); physical_y = box.y0*old_height
        if box.y0 >= .73: shift = 12/25.4 + row_added
        elif box.y0 >= .49: shift = 9/25.4 + row_added
        elif box.y0 >= .30: shift = 4/25.4
        else: shift = 0
        height = box.height*old_height
        if ax in (coefficient, market, key): height *= HEIGHT_FACTOR
        new = [box.x0, (physical_y+shift)/(old_height+extra), box.width,
               height/(old_height+extra)]
        ax.set_position(new)
        positions.append({'title': ax.get_title(loc='left') or ax.get_title(),
                          'old': list(box.bounds), 'new': new})
    for item in fig.texts:
        x, y = item.get_position()
        shift = (12/25.4 + row_added if y >= .73 else
                 9/25.4 + row_added if y >= .49 else 4/25.4 if y >= .30 else 0)
        item.set_y((y*old_height+shift)/(old_height+extra))
    parent.typography(fig)
    # Centre every lower panel beneath its map column. Equal-sized plotting
    # rectangles are used, with a separate gutter for the heatmap colour key.
    maps = [next(ax for ax in fig.axes if ax.get_title() == str(year)) for year in YEARS]
    centres = [maps[i].get_position().x0 + maps[i].get_position().width/2 for i in (0, 1)]
    panel_width = .350
    count = next(ax for ax in fig.axes if ax.get_title(loc='left') == 'Price-sensitive opportunity')
    share = next(ax for ax in fig.axes if ax.get_title(loc='left') == 'Lower-price share')
    panel_height = coefficient.get_position().height
    for ax, centre in ((coefficient, centres[0]), (market, centres[1]),
                       (count, centres[0]), (share, centres[1])):
        box = ax.get_position()
        ax.set_position([centre-panel_width/2, box.y0, panel_width, panel_height])
    box = market.get_position()
    key.set_position([box.x1+.012, box.y0, .012, box.height])
    # Panel letters use the same column guide, independently of tick-label width.
    for ax, letter, x in ((maps[0], 'a', .035), (coefficient, 'b', .035),
                          (market, 'c', .495), (count, 'd', .035)):
        labels = [item for item in ax.texts if item.get_text() == letter]
        assert len(labels) == 1, f'Panel label {letter} not found uniquely'
        labels[0].set_transform(fig.transFigure)
        labels[0].set_position((x, ax.get_position().y1+.010))
        labels[0].set_ha('left'); labels[0].set_va('bottom')
    # V194: larger type, equal tick-label padding and a measured legend lane.
    from matplotlib.text import Text
    for item in fig.findobj(Text):
        if item.get_text().strip():
            item.set_fontsize(max(8.0, item.get_fontsize()*1.18))
    for item in market.texts:
        if item.get_text().strip() != 'c':
            item.set_fontsize(9.2)
    market.set_yticks(market.get_yticks(),
                     [t.get_text().replace('Restaurant supply', 'Restaurant\nsupply')
                      for t in market.get_yticklabels()])
    for ax in (coefficient, market, count, share):
        ax.tick_params(axis='x', which='major', pad=3, length=2, width=.5)
        ax.tick_params(axis='y', which='major', pad=3)
        ax.tick_params(axis='both', which='major', labelsize=8.2)
        ax.xaxis.labelpad = 2
        ax._left_title.set_fontsize(9.4)
    # Position the b legend just below its x-axis label, not near panel d.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    label_bottom = coefficient.xaxis.label.get_window_extent(renderer).y0
    legend_top = (label_bottom - fig.dpi*(.8/25.4))/(fig.dpi*fig.get_figheight())
    legend = coefficient.get_legend()
    legend.borderaxespad = 0
    legend.set_bbox_to_anchor((centres[0], legend_top), transform=fig.transFigure)
    fig.canvas.draw()
    label_box = coefficient.xaxis.label.get_window_extent(fig.canvas.get_renderer())
    legend_box = legend.get_window_extent(fig.canvas.get_renderer())
    assert legend_box.y1 < label_box.y0, 'b legend overlaps x-axis label'
    final_boxes = {name:list(ax.get_position().bounds) for name, ax in
                   [('b',coefficient),('c',market),('d_left',count),('d_right',share)]}
    np.testing.assert_allclose([coefficient.get_position().width, coefficient.get_position().height],
                               [market.get_position().width, market.get_position().height], atol=1e-12)
    np.testing.assert_allclose([count.get_position().width, count.get_position().height],
                               [share.get_position().width, share.get_position().height], atol=1e-12)
    after = [numeric_axis(ax) for ax in fig.axes]
    if before != after:
        (ROOT/'fig15_geometry_debug.json').write_text(json.dumps({'before':before,'after':after}, indent=2), encoding='utf-8')
    changed_axes = {fig.axes.index(coefficient), fig.axes.index(market)}
    assert all(a == b for i, (a, b) in enumerate(zip(before, after)) if i not in changed_axes), 'An unrelated panel changed'
    heatmap = market.images[0].get_array()
    expected = regression.loc[regression.model.eq('Income + market composition')].pivot(
        index='predictor', columns='year', values='standardized_beta').loc[
            ['log_income','log_supply','low_price_share','ageing_share'], list(YEARS)].to_numpy(float)
    np.testing.assert_array_equal(heatmap, expected)
    parent.OUT = OUT
    outputs = BASE_EXPORT(fig, stem)
    # Export the requested b/c row separately for immediate visual inspection.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [ax.get_tightbbox(renderer) for ax in (coefficient, market, key)]
    row_bbox = Bbox.union(boxes).transformed(fig.dpi_scale_trans.inverted()).expanded(1.025, 1.065)
    fig.savefig(OUT/'Fig15_bc_Year_Colours_Taller_Row.png', dpi=450,
                bbox_inches=row_bbox, facecolor='white')
    audit = {'numeric_sha256_before': fingerprint(before), 'numeric_sha256_after': fingerprint(after),
             'numeric_arrays_ci_limits_ticks_identical': before == after,
             'heatmap_exactly_equals_source_csv': True, 'year_colours': YEAR_COLOURS,
             'shade_mapping': {'Income only':'dark', 'Adjusted model':'light'},
             'b_c_height_factor': HEIGHT_FACTOR, 'row_added_mm': row_added*25.4,
             'canvas_mm': [fig.get_figwidth()*25.4, fig.get_figheight()*25.4],
             'v189_map_and_bc_gaps_preserved': True, 'other_panel_colours_unchanged': True,
             'shared_column_centres': centres, 'statistical_panel_rectangles': final_boxes,
             'all_statistical_panels_same_width_height': True,
             'coefficient_rows': rows, 'positions': positions, 'outputs': outputs,
             'font_multiplier': 1.18, 'heatmap_number_font_pt': 9.2,
             'b_legend_gap_below_xlabel_mm': .8, 'major_x_tick_padding_pt': 3,
             'empirical_recomputation': False,
             'coefficient_source': 'Explicit six-domain primary CSV; estimates are not fitted by this renderer',
             'unrelated_panel_arrays_identical': True}
    (ROOT/'fig15_style_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    return outputs


def render() -> dict:
    """Render from explicit six-domain estimates without fitting a model."""
    if REGRESSION_FILE is None or not Path(REGRESSION_FILE).is_file():
        raise FileNotFoundError('An authorized six-domain regression CSV is required')
    ROOT.mkdir(parents=True, exist_ok=True)
    prior._base_export = refined_export
    parent.export = prior.polished_export
    parent.key_left = prior.prior.right_key
    parent.typography = prior.polished_typography
    before_inputs = parent.bridge.assert_inputs()
    result = parent.render_15()
    assert before_inputs == parent.bridge.assert_inputs(), 'Fixed source data changed'
    (ROOT/'fig15_source_audit.json').write_text(json.dumps({
        'fixed_inputs': before_inputs, 'render': result,
        'six_domain_regression_sha256': hashlib.sha256(Path(REGRESSION_FILE).read_bytes()).hexdigest(),
        'empirical_recomputation': False}, indent=2), encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--six-domain-regression', required=True, type=Path)
    args = parser.parse_args()
    REGRESSION_FILE = args.six_domain_regression
    render()
    print(OUT)
