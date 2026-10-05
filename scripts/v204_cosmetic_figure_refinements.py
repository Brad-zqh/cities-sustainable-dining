"""Author-requested Fig. 3/15/17 cosmetic changes using authorized fixed data.

No manuscript edits and no estimation. Exports 600-dpi PNG plus vector masters.
"""
from pathlib import Path
import hashlib
import json
import sys
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')

SCRIPTS = Path(__file__).resolve().parent
HERE = Path(os.environ.get('CITIES_FIGURE_DIR', str(SCRIPTS.parent / 'figures/v204')))
sys.path.insert(0, str(SCRIPTS))
OUT = HERE
SIX_DOMAIN_REGRESSION = None
VISUAL_DATA_ROOT = None


def data_state(fig) -> str:
    """Native artist values/colours; display positions intentionally excluded."""
    result = []
    for ax in fig.axes:
        result.append({
            'limits': [list(ax.get_xlim()), list(ax.get_ylim())],
            'lines': [(np.asarray(l.get_xydata()).tolist(), str(l.get_color())) for l in ax.lines],
            'patches': [(p.get_path().vertices.tolist(), str(p.get_facecolor()),
                         list(p.get_bbox().bounds) if hasattr(p, 'get_bbox') else None)
                        for p in ax.patches],
            'collections': [(np.asarray(c.get_offsets()).tolist(),
                             np.asarray(c.get_facecolors()).tolist(),
                             np.asarray(c.get_edgecolors()).tolist(),
                             [np.asarray(s).tolist() for s in c.get_segments()] if hasattr(c, 'get_segments') else None,
                             None if c.get_array() is None else np.asarray(c.get_array()).tolist())
                            for c in ax.collections],
            'images': [np.asarray(im.get_array()).tolist() for im in ax.images],
        })
    return hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()


def add_gap(fig, stem: str, base_export, parent) -> dict:
    """Grow canvas 6 mm, translate upper rows, leave d row physical size fixed."""
    from matplotlib.text import Text
    from matplotlib.transforms import Bbox
    fig.canvas.draw()
    before = data_state(fig)
    axes_by_title = {ax.get_title(loc='left'): ax for ax in fig.axes}
    count = axes_by_title['Price-sensitive opportunity']
    share = axes_by_title['Lower-price share']
    coefficient = axes_by_title['Income coefficient attenuation']
    market = axes_by_title['Market-composition coefficients']
    old_h = fig.get_figheight()
    gap = 6 / 25.4
    new_h = old_h + gap
    positions = {ax: ax.get_position().frozen() for ax in fig.axes}
    labels = [(item, item.get_position(), item.axes in (count, share))
              for item in fig.findobj(Text) if item.get_transform() is fig.transFigure]
    legend = coefficient.get_legend()
    fig.canvas.draw()
    old_anchor = legend.get_bbox_to_anchor().frozen()
    old_gap = (legend.get_window_extent().y0 - count._left_title.get_window_extent().y1) / fig.dpi * 25.4
    fig.set_size_inches(fig.get_figwidth(), new_h, forward=True)
    for ax, box in positions.items():
        shift = 0 if ax in (count, share) else gap
        ax.set_position([box.x0, (box.y0 * old_h + shift) / new_h,
                         box.width, box.height * old_h / new_h])
    for item, (x, y), bottom_row in labels:
        item.set_position((x, (y * old_h + (0 if bottom_row else gap)) / new_h))
    fig.canvas.draw()
    legend.set_bbox_to_anchor(((old_anchor.x0 + old_anchor.width / 2) / fig.bbox.width,
                              (old_anchor.y1 / fig.dpi + gap) / new_h), transform=fig.transFigure)
    fig.canvas.draw()
    after = data_state(fig)
    assert before == after, 'Data or colours changed during spacing adjustment'
    new_gap = (legend.get_window_extent().y0 - count._left_title.get_window_extent().y1) / fig.dpi * 25.4
    assert abs(new_gap - old_gap - 6) < .05
    parent.OUT = OUT
    exported = base_export(fig, 'Fig15_RowGap_Expanded_600dpi')
    # Native plot export of the panels shown in the user's screenshot.
    boxes = [ax.get_tightbbox(fig.canvas.get_renderer()) for ax in (coefficient, market, count, share, market.images[0].colorbar.ax)]
    boxes.append(legend.get_window_extent())
    crop = Bbox.union(boxes).transformed(fig.dpi_scale_trans.inverted()).expanded(1.035, 1.04)
    fig.savefig(OUT / 'Fig15_Lower_Panels_RowGap_600dpi.png', dpi=600, bbox_inches=crop, facecolor='white')
    (HERE / 'fig15_gap_audit.json').write_text(json.dumps({
        'data_colour_fingerprint_before': before, 'data_colour_fingerprint_after': after,
        'data_and_colours_identical': True, 'added_gap_mm': 6,
        'old_legend_to_d_title_gap_mm': old_gap, 'new_legend_to_d_title_gap_mm': new_gap,
        'outputs': exported, 'manuscript_changed': False}, indent=2), encoding='utf-8')
    return exported


def render15() -> None:
    """Reuse V203 six-domain display; add only the requested 6-mm row gap."""
    import v203_fig15_six_primary as source
    source.ROOT = HERE
    source.OUT = OUT
    source.REGRESSION_FILE = SIX_DOMAIN_REGRESSION
    native_export = source.BASE_EXPORT
    source.BASE_EXPORT = lambda fig, stem: add_gap(fig, stem, native_export, source.parent)
    source.render()


def render17() -> None:
    import pandas as pd
    import matplotlib as mpl
    import v138_reproduce_visual_polish as bridge
    groups, source = bridge.load_original('fig09_social_within_year')
    groups.SOURCE = bridge.DOWNSTREAM
    groups.OUT = OUT
    groups.STEM = 'Fig17_Larger_Subgroup_Legends_600dpi'
    contrasts = pd.read_csv(bridge.CONTRASTS / 'within_year_contrasts.csv')
    groups.within_year_contrasts = lambda replicates, points: contrasts.copy()
    original_save = mpl.figure.Figure.savefig
    adjusted = False

    def save(fig, target, *args, **kwargs):
        nonlocal adjusted
        if not adjusted:
            before = data_state(fig)
            audit = []
            for ax in fig.axes[:6]:
                legend = ax.get_legend()
                old_size = legend.get_texts()[0].get_fontsize()
                handles = legend.legend_handles
                labels = [t.get_text() for t in legend.get_texts()]
                legend.remove()
                new_legend = ax.legend(handles, labels, title='Subgroup · $N$', ncol=1,
                          loc='upper right', bbox_to_anchor=(.995, .995),
                          fontsize=6.25, title_fontsize=6.6, handlelength=.90,
                          handletextpad=.36, labelspacing=.18, borderaxespad=.12)
                audit.append({'panel': ax.get_title(loc='left'), 'old_font_pt': old_size,
                              'new_font_pt': 6.25, 'title_font_pt': 6.6})
            fig.canvas.draw()
            # Find the nearest empty lane for the larger Ethnicity legend.
            # Keep all bar heights, limits, intervals and significance marks fixed.
            from matplotlib.transforms import Bbox
            ax = fig.axes[0]
            legend = ax.get_legend()
            renderer = fig.canvas.get_renderer()
            obstacles = [item.get_window_extent(renderer).expanded(1.01, 1.03)
                         for item in [*ax.lines, *ax.patches, *ax.texts, ax._left_title]
                         if item.get_visible()]
            chosen = None
            for y in np.arange(.995, 1.241, .01):
                for x in (.995, .90, .80, .70):
                    legend.set_bbox_to_anchor((x, float(y)), transform=ax.transAxes)
                    box = legend.get_window_extent(renderer)
                    if not any(box.overlaps(other) for other in obstacles):
                        chosen = [x, float(y)]
                        break
                if chosen is not None:
                    break
            assert chosen is not None, 'No clear lane for Ethnicity legend'
            audit[0]['overlap_free_anchor'] = chosen
            fig.canvas.draw()
            after = data_state(fig)
            assert before == after, 'Bar values, intervals, stars or colours changed'
            (HERE / 'fig17_legend_audit.json').write_text(json.dumps({
                'data_colour_fingerprint_before': before, 'data_colour_fingerprint_after': after,
                'data_and_colours_identical': True, 'legends': audit,
                'source_script': str(source), 'source_data': str(bridge.DOWNSTREAM),
                'contrast_source': str(bridge.CONTRASTS / 'within_year_contrasts.csv'),
                'manuscript_changed': False}, indent=2), encoding='utf-8')
            adjusted = True
        if str(target).endswith('.tiff'):
            return None
        return original_save(fig, target, *args, **kwargs)

    mpl.figure.Figure.savefig = save
    try:
        groups.main()
    finally:
        mpl.figure.Figure.savefig = original_save


def render03() -> None:
    import v140_reader_figure_touchups as parent
    from matplotlib.text import Text
    parent.OUT = OUT
    original_loader = parent.bridge.load_original

    def loader(name):
        module, path = original_loader(name)
        if name != 'render_study_benchmark_composite':
            return module, path
        # Fig. 3's historical bundle is separately supplied; it is not part of
        # the fixed V180 result tree and must never fall back into the clone.
        visual_root = Path(VISUAL_DATA_ROOT)
        module.study.DATA_ROOT = visual_root
        module.study.SOURCE = visual_root / 'source_data/fig03_study_area_annual_distribution_v1'
        module.study.GEOMETRY = visual_root / 'source_data/fig_v4_four_year/2024_lsbg_components.gpkg'
        module.benchmark.SOURCE_BUNDLE = visual_root / 'source_data/figS_openrice_fehd_v4'
        module.benchmark.RAW_LSBG = module.benchmark.SOURCE_BUNDLE / 'lsbg_2021_simplified.gpkg'
        module.shared.DATA_ROOT = visual_root
        original_export = module.shared.export

        def export(fig, stem, content):
            before = data_state(fig)
            titles = ['LSBG agreement', 'Bias and agreement limits', 'Availability diagnostics']
            panels = [next(ax for ax in fig.axes if ax.get_title(loc='left') == title) for title in titles]
            old_h, width = fig.get_figheight(), fig.get_figwidth()
            # Shared square size and aligned row edges for all three panels.
            common_width = panels[0].get_position().width
            for ax in panels:
                box = ax.get_position()
                ax.set_position([box.x0, box.y0, common_width, box.height])
            old_positions = {ax: ax.get_position().frozen() for ax in fig.axes}
            # Largest reduction preserves the original clearance above this row.
            reduction = max(old_positions[ax].height * old_h - old_positions[ax].width * width for ax in panels)
            clearance = 5 / 25.4
            new_h = old_h - reduction + clearance
            labels = [(t, t.get_position(), t.axes in panels)
                      for t in fig.findobj(Text) if t.get_transform() is fig.transFigure]
            fig.set_size_inches(width, new_h, forward=True)
            for ax, box in old_positions.items():
                if ax in panels:
                    physical_height = box.width * width
                    physical_y = box.y0 * old_h
                else:
                    physical_height = box.height * old_h
                    physical_y = box.y0 * old_h - reduction + clearance
                ax.set_position([box.x0, physical_y / new_h, box.width, physical_height / new_h])
            for text, (x, y), in_row in labels:
                text.set_position((x, (y * old_h - (0 if in_row else reduction - clearance)) / new_h))
            fig.canvas.draw()
            for ax in panels:
                box = ax.get_position()
                assert abs(box.height * new_h - box.width * width) < 1e-10
            after = data_state(fig)
            assert before == after, 'Benchmark data or colours changed'
            from matplotlib.transforms import Bbox
            boxes = [ax.get_tightbbox(fig.canvas.get_renderer()) for ax in panels]
            crop = Bbox.union(boxes).transformed(fig.dpi_scale_trans.inverted()).expanded(1.035, 1.08)
            fig.savefig(OUT / 'Fig03_Bottom_Panels_Square_600dpi.png', dpi=600, bbox_inches=crop, facecolor='white')
            (HERE / 'fig03_square_audit.json').write_text(json.dumps({
                'data_colour_fingerprint_before': before, 'data_colour_fingerprint_after': after,
                'data_and_colours_identical': True, 'canvas_reduction_mm': reduction * 25.4,
                'panel_physical_mm': [{'title': ax.get_title(loc='left'),
                                      'width_mm': ax.get_position().width * width * 25.4,
                                      'height_mm': ax.get_position().height * new_h * 25.4}
                                     for ax in panels], 'manuscript_changed': False}, indent=2), encoding='utf-8')
            result = original_export(fig, stem, content)
            from shutil import copy2
            for suffix in ('.png', '.svg', '.pdf'):
                copy2(OUT / (stem + suffix), OUT / ('Fig03_Benchmark_Square_Bottom_Panels_600dpi' + suffix))
            return result

        module.shared.export = export
        return module, path

    parent.bridge.load_original = loader
    try:
        parent.render_03()
    finally:
        parent.bridge.load_original = original_loader


def main() -> None:
    """Require explicit external inputs and preserve all previous outputs."""
    global HERE, OUT, SIX_DOMAIN_REGRESSION, VISUAL_DATA_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--figure', required=True, choices=('03', '15', '17'))
    parser.add_argument('--data-root', required=True, type=Path,
                        help='Separately authorized input tree; not supplied by this clone')
    parser.add_argument('--output', required=True, type=Path, help='Fresh output directory')
    parser.add_argument('--six-domain-regression', type=Path,
                        help='Explicit six-domain primary coefficient CSV, required for Fig. 15')
    parser.add_argument('--visual-data-root', type=Path,
                        help='External historical source_data tree for Fig. 3 (defaults to --data-root)')
    args = parser.parse_args()
    if not args.data_root.is_dir():
        parser.error('--data-root must identify an existing authorized input tree')
    if args.output.exists():
        parser.error('--output must be a fresh directory; previous exports are preserved')
    if args.figure == '15' and (args.six_domain_regression is None or not args.six_domain_regression.is_file()):
        parser.error('Fig. 15 requires --six-domain-regression with an existing CSV')
    HERE = OUT = args.output.resolve()
    SIX_DOMAIN_REGRESSION = args.six_domain_regression
    VISUAL_DATA_ROOT = (args.visual_data_root or args.data_root).resolve()
    os.environ['CITIES_DATA_ROOT'] = str(args.data_root.resolve())
    os.environ['CITIES_FIGURE_DIR'] = str(OUT)
    OUT.mkdir(parents=True)
    {'15': render15, '17': render17, '03': render03}[args.figure]()
    print(OUT)


if __name__ == '__main__':
    main()
