"""Restricted V147 presentation changes on top of the fixed V140 figures.

Only heatmap colour-key placement, the Fig. 14 line legend, and the Fig. 15d
display geometry are changed. The V135 empirical inputs are hash-checked before
and after rendering. Existing V140 output files remain untouched.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import v140_reader_figure_touchups as parent


OUT = parent.ROOT / "outputs/restricted/v147_reader_figure_refinements"
parent.OUT = OUT
_original_typography = parent.typography
_original_export = parent.export


def right_key(image, bounds, *, title="", label=None):
    """Relocate the existing key to the right without changing its scale."""
    fig = image.axes.figure
    previous = image.colorbar
    ticks = list(previous.get_ticks()) if previous is not None else None
    if previous is not None:
        if not title:
            title = previous.ax.get_title()
        if label is None:
            label = previous.ax.xaxis.label.get_text() or previous.ax.yaxis.label.get_text()
        previous.remove()

    ax = image.axes
    position = ax.get_position()
    _, bottom, width, height = bounds
    # A vertical label needs more outer margin than a short key title.
    key_right_edge = .935 if label else .970
    key_left_edge = min(position.x1 + .012, key_right_edge - width)
    if position.x1 + .012 > key_left_edge:
        shift = position.x1 + .012 - key_left_edge
        ax.set_position([position.x0 - shift, position.y0, position.width, position.height])
    cax = fig.add_axes([key_left_edge, bottom, width, height])
    cb = fig.colorbar(image, cax=cax, orientation="vertical", ticks=ticks)
    cb.ax.yaxis.set_ticks_position("right")
    cb.ax.yaxis.set_label_position("right")
    cb.ax.tick_params(labelsize=6.7, length=2, width=.5, pad=1.5)
    cb.outline.set_visible(True)
    cb.outline.set_color("#253039")
    cb.outline.set_linewidth(.55)
    if title:
        cb.ax.set_title(title, fontsize=6.7, pad=3)
    if label:
        cb.set_label(label, fontsize=6.7, labelpad=4)
    return cb


def typography_with_line_key(fig: plt.Figure, floor: float = 6.7):
    """Use line samples for the four concentration curves in Fig. 14b."""
    _original_typography(fig, floor)
    matches = [ax for ax in fig.axes if ax.get_title(loc="left") == "Concentration gap"]
    for ax in matches:
        old = ax.get_legend()
        if old is None:
            raise AssertionError("Fig. 14b year legend is missing")
        labels = [item.get_text() for item in old.get_texts()]
        colours = [handle.get_color() for handle in old.legend_handles]
        if labels != ["2011", "2016", "2021", "2024"]:
            raise AssertionError(f"Unexpected concentration legend: {labels}")
        old.remove()
        handles = [Line2D([], [], color=colour, lw=1.8, ls="-", marker=None,
                          solid_capstyle="round", label=label)
                   for colour, label in zip(colours, labels)]
        ax.legend(handles=handles, ncol=2, loc="upper left",
                  bbox_to_anchor=(.08, 1.02), frameon=False, fontsize=7.0,
                  handlelength=1.8, handletextpad=.45, columnspacing=.8)


def _redraw_price_panel(fig: plt.Figure):
    """Four-year paired dots and direct-label share plot; exact values retained."""
    count = next(ax for ax in fig.axes if ax.get_title(loc="left") == "Price-sensitive opportunity")
    share = next(ax for ax in fig.axes if ax.get_title(loc="left") == "Lower-price share")
    summary = pd.read_csv(parent.bridge.PRICE_MARKET / "price_market_summary.csv")
    summary = summary.set_index("year").loc[[2011, 2016, 2021, 2024]]
    years = summary.index.to_numpy(int)
    total = summary["population_weighted_total_supply_per1000"].to_numpy(float)
    cheaper = summary["population_weighted_low_price_per1000"].to_numpy(float)
    fraction = 100 * summary["population_weighted_low_price_share"].to_numpy(float)
    if not np.all((cheaper <= total) & (fraction >= 0) & (fraction <= 100)):
        raise AssertionError("Invalid fixed price-market summary")

    count.clear()
    share.clear()
    count.set_position([.110, .050, .350, .240])
    share.set_position([.590, .050, .350, .240])
    y = np.arange(4)[::-1]
    blue, red, navy = "#367FAD", "#B74652", "#285A7A"
    for yy, a, b in zip(y, total, cheaper):
        count.plot([b, a], [yy, yy], color="#C8D3D9", lw=2.2,
                   solid_capstyle="round", zorder=1)
    count.scatter(total, y, s=43, facecolors="white", edgecolors=blue,
                  linewidths=1.8, zorder=3, label="All restaurants")
    count.scatter(cheaper, y, s=43, facecolors="white", edgecolors=red,
                  linewidths=1.8, zorder=3, label="≤HK$100 restaurants")
    count.set(yticks=y, yticklabels=years, ylim=(-.55, 3.95), xlim=(0, 3.9),
              xlabel="Restaurants per 1,000 residents")
    count.tick_params(axis="y", length=0)
    count.set_title("Price-sensitive opportunity", loc="left", pad=8,
                    fontsize=8.1, fontweight="normal")
    count.text(-.16, 1.03, "d", transform=count.transAxes, fontsize=9.2,
               fontweight="bold", ha="right", va="bottom")
    count.legend(loc="upper center", bbox_to_anchor=(.50, .99), ncol=2,
                 frameon=False, fontsize=6.7, handletextpad=.35, columnspacing=.8)
    count.spines[["top", "right", "left"]].set_visible(False)

    # The dot strip uses a labelled 0–100% reference scale, so the four
    # closely spaced estimates are compared without a visually false baseline.
    share.hlines(y, 0, 100, color="#E7EDEF", lw=5.0, zorder=1)
    share.hlines(y, 0, fraction, color="#B7CCD8", lw=5.0, zorder=2)
    share.scatter(fraction, y, s=53, marker="D", facecolors="white",
                  edgecolors=navy, linewidths=1.7, zorder=3)
    for yy, value in zip(y, fraction):
        share.text(value + 3.5, yy, f"{value:.1f}%", va="center", ha="left",
                   fontsize=7.0, color=navy)
    share.set(yticks=y, yticklabels=years, ylim=(-.55, 3.95), xlim=(0, 116),
              xticks=[0, 25, 50, 75, 100], xlabel="Population-weighted area share (%)")
    share.tick_params(axis="y", length=0)
    share.set_title("Lower-price share", loc="left", pad=8,
                    fontsize=8.1, fontweight="normal")
    share.spines[["top", "right", "left"]].set_visible(False)
    return {str(y): {"all_per_1000": float(a), "low_price_per_1000": float(b),
                     "low_price_share_percent": float(c)}
            for y, a, b, c in zip(years, total, cheaper, fraction)}


def export_with_price_panel(fig: plt.Figure, stem: str):
    if stem == "Fig15_Price_Composition_Socioeconomic_Associations":
        coefficient = next(ax for ax in fig.axes
                           if ax.get_title(loc="left") == "Income coefficient attenuation")
        market = next(ax for ax in fig.axes
                      if ax.get_title(loc="left") == "Market-composition coefficients")
        coefficient.set_position([.100, .355, .345, .125])
        old = market.get_position()
        market.set_position([old.x0, .355, old.width, .125])
        colorbar = market.images[0].colorbar.ax
        old_key = colorbar.get_position()
        colorbar.set_position([old_key.x0, .355, old_key.width, .125])
        _redraw_price_panel(fig)
        typography_with_line_key(fig)
    return _original_export(fig, stem)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=("12", "14", "15", "16", "18b", "19"), required=True)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    before = parent.bridge.assert_inputs()
    parent.key_left = right_key
    parent.typography = typography_with_line_key
    parent.export = export_with_price_panel
    renderers = {
        "12": lambda: parent.render_12_16("12"),
        "14": parent.render_14,
        "15": parent.render_15,
        "16": lambda: parent.render_12_16("16"),
        "18b": parent.render_18b,
        "19": parent.render_19,
    }
    result = renderers[args.only]()
    if before != parent.bridge.assert_inputs():
        raise AssertionError("Fixed empirical inputs changed during V147 rendering")
    if args.only == "19":
        path = OUT / "planning_audit.json"
        audit = json.loads(path.read_text(encoding="utf-8"))
        audit["heatmap_colorbar_side"] = "right"
        path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    result.update(status="RESTRICTED_PRESENTATION_ONLY", figure=args.only,
                  fixed_input_sha256=before, public_release_authorized=False,
                  no_empirical_recomputation=True,
                  heatmap_colorbar_side="right",
                  concentration_legend="solid line samples" if args.only == "14" else None,
                  price_panel="paired dots and share strips" if args.only == "15" else None)
    (OUT / f"V147_FIG{args.only}_PRESENTATION_AUDIT.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
