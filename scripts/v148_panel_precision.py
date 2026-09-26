"""Restricted, presentation-only V148 refinements of four V147 result figures.

The fixed empirical source tables and estimators are unchanged.  This wrapper
keeps the established plotting lineage and changes only panel geometry and
display encoding where the observed planning-map share is nearly discrete.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap, to_rgba
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle

import v140_reader_figure_touchups as parent
import v147_reader_figure_refinements as prior


OUT = parent.ROOT / "outputs/restricted/v148_panel_precision"
parent.OUT = OUT
prior.OUT = OUT
_load_original = parent.bridge.load_original
_base_typography = prior.typography_with_line_key
_base_export = prior._original_export

YEAR_COLORS = {
    2011: "#BF4B5E", 2016: "#2D76B7", 2021: "#328858", 2024: "#D58B29"
}


def polished_typography(fig: plt.Figure, floor: float = 6.7) -> None:
    """Apply the V147 text rules, then refine three identified panels only."""
    _base_typography(fig, floor)
    for ax in fig.axes:
        title = ax.get_title(loc="left")
        if title == "2024 subgroup contrasts":
            # Shift the complete j matrix and key, including its row-label
            # footprint, into the same visual left column as a/d/g.
            delta = -.060
            box = ax.get_position()
            ax.set_position([box.x0 + delta, box.y0, box.width, box.height])
            key = ax.images[0].colorbar.ax
            box = key.get_position()
            key.set_position([box.x0 + delta, box.y0, box.width, box.height])
        elif title == "Component weights":
            # The first two collections are the Equal-six halo and hollow
            # circles. Change both; leave PC1 squares and entropy triangles.
            if len(ax.collections) < 6:
                raise AssertionError("Fig18b weight markers unexpectedly changed")
            red = YEAR_COLORS[2011]
            ax.collections[0].set_facecolor([to_rgba(red, .12)])
            ax.collections[1].set_edgecolor(red)
            legend = ax.get_legend()
            if legend is None or not legend.legend_handles:
                raise AssertionError("Fig18b weight legend unavailable")
            legend.legend_handles[0].set_edgecolor(red)
        elif ax.get_xlabel() == "Jaccard overlap" and ax.images:
            # Increase width and height together so the 4x4 matrix keeps the
            # same physical cell aspect and the right-side key remains clear.
            ax.set_position([.595, .035, .310, .175])
            key = ax.images[0].colorbar.ax
            key.set_position([.917, .035, .012, .175])
            for item in fig.texts:
                if item.get_text() == "h":
                    item.set_x(.567)
                elif item.get_text() == "Selection overlap · K=10":
                    item.set_x(.595)


def year_coded_price_panel(fig: plt.Figure) -> dict:
    """Reuse the V147 paired-dot geometry; assign one map-matched hue per year."""
    values = prior._redraw_price_panel(fig)
    count = next(ax for ax in fig.axes if ax.get_title(loc="left") == "Price-sensitive opportunity")
    share = next(ax for ax in fig.axes if ax.get_title(loc="left") == "Lower-price share")
    count.set_position([.110, .050, .350, .200])
    share.set_position([.590, .050, .350, .200])
    years = (2011, 2016, 2021, 2024)
    colors = [YEAR_COLORS[year] for year in years]
    if len(count.lines) != 4 or len(count.collections) != 2 or len(share.collections) != 3:
        raise AssertionError("Fig15d artist structure unexpectedly changed")
    for line, color in zip(count.lines, colors):
        line.set_color(to_rgba(color, .28))
        line.set_linewidth(2.0)
    # Category uses shape; colour consistently identifies the map year.
    count.collections[0].set_edgecolors(colors)
    count.collections[1].set_edgecolors(colors)
    square = MarkerStyle("s").get_path().transformed(MarkerStyle("s").get_transform())
    count.collections[1].set_paths([square])
    legend = count.get_legend()
    if legend is not None:
        legend.remove()
    count.legend(handles=[
        Line2D([], [], marker="o", color="none", markerfacecolor="white",
               markeredgecolor="#394B55", markeredgewidth=1.25, markersize=5.0,
               label="All restaurants"),
        Line2D([], [], marker="s", color="none", markerfacecolor="white",
               markeredgecolor="#394B55", markeredgewidth=1.25, markersize=4.8,
               label="≤HK$100 restaurants"),
    ], loc="upper center", bbox_to_anchor=(.50, .99), ncol=2,
       frameon=False, fontsize=6.7, handletextpad=.35, columnspacing=.8)
    share.collections[0].set_colors([to_rgba(color, .10) for color in colors])
    share.collections[1].set_colors([to_rgba(color, .32) for color in colors])
    share.collections[2].set_edgecolors(colors)
    for item, color in zip(share.texts, colors):
        if item.get_text().endswith("%"):
            item.set_color(color)
    return values


def polished_export(fig: plt.Figure, stem: str):
    if stem == "Fig15_Price_Composition_Socioeconomic_Associations":
        coefficient = next(ax for ax in fig.axes
                           if ax.get_title(loc="left") == "Income coefficient attenuation")
        market = next(ax for ax in fig.axes
                      if ax.get_title(loc="left") == "Market-composition coefficients")
        # Same top edge as V147; b/c are 20% taller while the d title, b axis
        # label and b legend each retain their own vertical reading space.
        coefficient.set_position([.100, .330, .345, .150])
        old = market.get_position()
        market.set_position([old.x0, .330, old.width, .150])
        key = market.images[0].colorbar.ax
        old_key = key.get_position()
        key.set_position([old_key.x0, .330, old_key.width, .150])
        coefficient.legend(frameon=False, loc="upper center",
                           bbox_to_anchor=(.50, -.28), ncol=2, fontsize=6.7,
                           handletextpad=.4, columnspacing=.8)
        year_coded_price_panel(fig)
        polished_typography(fig)
    return _base_export(fig, stem)


def patched_loader(name: str):
    module, source = _load_original(name)
    if name != "render_planning_composite":
        return module, source
    original_draw = module.draw_map

    def categorical_map(fig, ax, cax, frame, sites, selections, strategy, index):
        result = original_draw(fig, ax, cax, frame, sites, selections, strategy, index)
        # Fixed source check: among 1,744 LSBGs there are 1,645 zeros, 98
        # complete (100%) shares, and one interior share of 34.3438%.
        denominator = frame.t_pop.to_numpy(float)
        share = frame.zero_affordability_priority_population.to_numpy(float) / np.where(
            denominator > 0, denominator, np.nan)
        finite = share[np.isfinite(share)]
        interior = finite[(finite > 0) & (finite < 1)]
        if (len(finite) != 1744 or np.count_nonzero(finite == 0) != 1645
                or np.count_nonzero(np.isclose(finite, 1)) != 98
                or len(interior) != 1
                or not np.isclose(interior[0], .343437595829522)):
            raise AssertionError("Planning-map distribution changed; review classification")
        mapped = [artist for artist in ax.collections
                  if artist.get_array() is not None and len(artist.get_array())]
        if len(mapped) != 1:
            raise AssertionError("Planning map colour artist unexpectedly changed")
        artist = mapped[0]
        recorded = np.asarray(artist.get_array(), dtype=float)
        levels = np.where(np.isclose(recorded, 0), 0,
                          np.where(np.isclose(recorded, 1), 2, 1))
        ramp = module.MAP_RAMPS[strategy]
        cmap = ListedColormap([ramp[0], ramp[1], ramp[-1]])
        norm = BoundaryNorm([-.5, .5, 1.5, 2.5], cmap.N)
        artist.set_array(levels)
        artist.set_cmap(cmap)
        artist.set_norm(norm)
        position = cax.get_position()
        cax.clear()
        colorbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                                boundaries=[-.5, .5, 1.5, 2.5], spacing="uniform")
        colorbar.set_ticks([0, 1, 2], labels=["0", "34.3", "100"])
        colorbar.ax.set_title("%", fontsize=6.4, pad=3)
        colorbar.ax.tick_params(labelsize=6.2, length=2, width=.5, pad=1.5)
        colorbar.outline.set_edgecolor("#253039")
        colorbar.outline.set_linewidth(.5)
        cax.set_position(position)
        for sample in result["map_samples"]:
            value = sample["share"]
            level = 0 if np.isclose(value, 0) else 2 if np.isclose(value, 1) else 1
            sample["rgba"] = list(cmap(norm(level)))
        result["display_classes_percent"] = [0, 34.3, 100]
        return result

    module.draw_map = categorical_map
    return module, source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=("14", "15", "18b", "19"), required=True)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    before = parent.bridge.assert_inputs()
    parent.bridge.load_original = patched_loader
    parent.key_left = prior.right_key
    parent.typography = polished_typography
    parent.export = polished_export
    renderers = {"14": parent.render_14, "15": parent.render_15,
                 "18b": parent.render_18b, "19": parent.render_19}
    result = renderers[args.only]()
    if before != parent.bridge.assert_inputs():
        raise AssertionError("Fixed V135 empirical inputs changed during V148 rendering")
    result.update(status="RESTRICTED_PRESENTATION_ONLY", figure=args.only,
                  fixed_input_sha256=before, public_release_authorized=False,
                  no_empirical_recomputation=True,
                  figure14_j_shift_figure_fraction=-.060 if args.only == "14" else None,
                  figure15_year_coding=YEAR_COLORS if args.only == "15" else None,
                  figure18b_equal_six_outline=YEAR_COLORS[2011] if args.only == "18b" else None,
                  figure19_map_display="observed 0/34.3/100% classes" if args.only == "19" else None)
    (OUT / f"V148_FIG{args.only}_PRESENTATION_AUDIT.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
