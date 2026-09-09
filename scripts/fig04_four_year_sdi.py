# Academic Figure Skill Asset Confirmation (verified against assets/figures/)
# (a-d) maps -> existing Cities cartographic production system -> param inherit
# (e,g) line trends -> assets/figures/LineTrend -> param inherit
# (f) forest/effect intervals -> cross-type inherit from GroupedBarChart -> param inherit
# RULE: "native run" = load pre-rendered PNG via Image.open().ax.imshow().
#       "param inherit" = drawing function below that copies Class A/B/C values.
#       If a panel says "native run" and you write a drawing function, you broke the contract.

"""Four-year SDI maps with a red 2011 ramp, 0.9 map scale and matched lines."""

from __future__ import annotations

import os
from pathlib import Path
import sys

import matplotlib as mpl

from v5_cities_visual_system import FONT_FAMILY

# Academic Figure Skill Typography Baseline — COPY VERBATIM, place at TOP of script
mpl.rcParams.update({
    "font.family": FONT_FAMILY,
    "font.sans-serif": [FONT_FAMILY, "Helvetica", "Nimbus Sans", "DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 8,
    "figure.titlesize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.frameon": False,
})

# Academic Figure Skill Nature/Cell/Science Color Palette -- COPY VERBATIM
CATEGORICAL = ["#2166AC", "#B2182B", "#1B7837", "#F1A340", "#762A83", "#666666"]
CATEGORICAL_EXTENDED = [
    "#2166AC", "#B2182B", "#1B7837", "#F1A340", "#762A83", "#666666",
    "#4393C3", "#D6604D", "#5AAE61", "#B35806", "#9970AB", "#999999",
]
DIVERGING   = ["#2166AC", "#F7F7F7", "#B2182B"]
SEQUENTIAL  = ["#F7FBFF", "#6BAED6", "#08306B"]
ACCENT_RED  = "#B2182B"
GREY        = "#999999"
BLACK       = "#222222"

# Academic Figure Skill Export Baseline — COPY VERBATIM
mpl.rcParams.update({
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
})

def save_cns_figure(fig, filename):
    """Standard Academic Figure Skill export: vector PDF + 300dpi PNG preview."""
    fig.savefig(f"{filename}.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(f"{filename}.png", bbox_inches="tight", dpi=300)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"
PROJECT_SCRIPTS = ROOT / "scripts"
for entry in (SCRIPT_DIR, PROJECT_SCRIPTS, FIGURES):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import v10_plot_four_year_sdi_nature_pdf as base  # noqa: E402
from v5_cities_visual_system import COLORS, COMPONENT_COLORS, MM, clean_axis, panel_label  # noqa: E402


OUT = Path(os.environ.get("CITIES_FIGURE_DIR", str(ROOT / "figures")))
STEM = "Fig3_FourYear_Strict_SDI_v25_RIGHT_CBAR_MATCHED_LINES"
YEARS = base.YEARS
FIELD = base.FIELD

# Hue identifies the map year, while lightness continues to encode SDI.
# The same anchor hues are reused below so the quantitative panels belong to
# the same visual system without implying that hue encodes a new variable.
MAP_RAMPS = {
    2011: ["#F8D7D4", "#ECA7A2", "#DD7771", "#C64B47", "#9F2F32"],
    2016: ["#D8EAF5", "#AACFE6", "#72ADD2", "#3D83B7", "#1D527D"],
    2021: ["#DCEED7", "#AED7A7", "#78BB78", "#439956", "#1F6B3B"],
    2024: ["#FBE2C8", "#F7BE82", "#ED9550", "#D86B29", "#9F3F17"],
}
YEAR_ANCHOR = {year: ramp[-1] for year, ramp in MAP_RAMPS.items()}
YEAR_MID = {year: ramp[2] for year, ramp in MAP_RAMPS.items()}
COMPONENT_MATCHED = {
    "Environmental sustainability": YEAR_ANCHOR[2021],
    "Cuisine diversity": YEAR_ANCHOR[2016],
    "Practice": YEAR_ANCHOR[2024],
    "Hygiene": YEAR_MID[2016],
    "Nutrition": YEAR_MID[2011],
    "Carbon": YEAR_MID[2021],
}

# The three diagnostic panels use independent sequential families so that
# hue identifies the panel while lightness carries ordering within the panel.
RED_DIAGNOSTIC = ["#F2B8B4", "#E48680", "#C95450", "#8F2630"]
BLUE_DIAGNOSTIC = ["#D7E9F5", "#B4D5E9", "#88BAD9", "#599BC4", "#3279AC", "#174F78"]
GREEN_DIAGNOSTIC = ["#DCEED7", "#AED7A7", "#78BB78", "#1F6B3B"]


def trajectory_panel(ax: plt.Axes, annual: pd.DataFrame, letter: str = "e") -> None:
    work = annual.loc[annual["metric"].eq(FIELD)].set_index("year").loc[YEARS]
    x = np.asarray(YEARS, dtype=float)
    mean = work["population_weighted_mean"].to_numpy(float)
    lo = work["ci_low"].to_numpy(float)
    hi = work["ci_high"].to_numpy(float)
    ax.set_facecolor("white")
    for xx, yy, lower, upper, colour in zip(x, mean, lo, hi, RED_DIAGNOSTIC):
        ax.errorbar([xx], [yy], yerr=[[yy - lower], [upper - yy]], fmt="none",
                    ecolor=colour, elinewidth=2.8, capsize=2.1, capthick=2.4,
                    alpha=0.13, zorder=1)
        ax.errorbar([xx], [yy], yerr=[[yy - lower], [upper - yy]], fmt="none",
                    ecolor=colour, elinewidth=0.62, capsize=1.7, capthick=0.62,
                    alpha=0.75, zorder=2)
    segments = np.stack([
        np.column_stack([x[:-1], mean[:-1]]),
        np.column_stack([x[1:], mean[1:]]),
    ], axis=1)
    segment_colours = RED_DIAGNOSTIC[1:]
    ax.add_collection(LineCollection(segments, colors=segment_colours,
                                     linewidths=5.2, alpha=0.13,
                                     capstyle="round", joinstyle="round", zorder=1))
    ax.add_collection(LineCollection(segments, colors=segment_colours,
                                     linewidths=1.55,
                                     capstyle="round", joinstyle="round", zorder=3))
    for xx, yy, colour in zip(x, mean, RED_DIAGNOSTIC):
        ax.scatter(xx, yy, s=20, color=colour, edgecolor="white",
                   linewidth=0.55, zorder=4)
    for index, (xx, yy, upper) in enumerate(zip(x, mean, hi)):
        text_x = xx - 0.10 if index == len(x) - 1 else xx
        align = "right" if index == len(x) - 1 else "center"
        ax.text(text_x, upper + 0.0030, f"{yy:.3f}", ha=align, va="bottom",
                fontsize=5.8, color=RED_DIAGNOSTIC[index])
    ax.axhline(mean[0], color="#AEB9BE", lw=0.55, ls=(0, (3, 2)), zorder=0)
    ax.set_xticks(YEARS)
    ax.set_xlim(2010.2, 2024.8)
    ax.set_ylim(0.365, 0.462)
    ax.set_ylabel("Population-weighted SDI")
    ax.set_xlabel("Year")
    clean_axis(ax)
    panel_label(ax, letter, x=-0.11, y=1.07)
    ax.text(0, 1.07, "Temporal change", transform=ax.transAxes, ha="left",
            va="top", fontsize=7.8, fontweight="bold")
    ax.text(0.99, 0.02, "95% DCCA-block CI", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=5.4, color=COLORS["ink"])


def component_panel(ax: plt.Axes, changes: pd.DataFrame, letter: str = "f") -> None:
    order = ["sustainability_score", "diversity_score", "practice_score",
             "hygiene_score", "nutrition_score", "carbon_score"]
    work = changes.loc[(changes["year"].eq(2024)) & changes["metric"].isin(order)].set_index("metric").loc[order]
    y = np.arange(len(order))[::-1]
    ax.set_facecolor("white")
    for yy, metric, colour in zip(y, order, BLUE_DIAGNOSTIC[::-1]):
        row = work.loc[metric]
        value, lo, hi = float(row["difference"]), float(row["ci_low"]), float(row["ci_high"])
        label = base.COMPONENT_LABELS[metric]
        ax.plot([lo, hi], [yy, yy], color=colour, lw=4.2, alpha=0.16,
                solid_capstyle="round", zorder=1)
        ax.plot([lo, hi], [yy, yy], color=colour, lw=0.82,
                solid_capstyle="round", zorder=2)
        ax.scatter(value, yy, s=28, color=colour, edgecolor="white", linewidth=0.55, zorder=3)
        ax.text(hi + 0.007, yy, f"{value:+.3f}", ha="left", va="center",
                fontsize=5.7, color=colour)
    ax.axvline(0, color="#7E8B91", lw=0.65, zorder=0)
    ax.set_yticks(y, [base.COMPONENT_LABELS[m] for m in order])
    ax.set_xlim(-0.07, 0.205)
    ax.set_xticks([-0.05, 0.00, 0.05, 0.10, 0.15])
    ax.set_xlabel("Change in score, 2024 minus 2011")
    ax.tick_params(axis="y", length=0, pad=2)
    clean_axis(ax)
    panel_label(ax, letter, x=-0.16, y=1.07)
    ax.text(0, 1.07, "Component-specific change", transform=ax.transAxes,
            ha="left", va="top", fontsize=7.8, fontweight="bold")
    ax.text(0.99, 0.02, "95% DCCA-block CI", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=5.4, color=COLORS["ink"])


def coverage_panel(ax: plt.Axes, coverage: pd.DataFrame, letter: str = "g") -> None:
    strict = coverage.loc[coverage["metric"].eq(FIELD)].set_index("year").loc[YEARS]
    high = coverage.loc[coverage["metric"].eq("equal_six_highcoverage")].set_index("year").loc[YEARS]
    x = np.asarray(YEARS)
    series = [
        (100 * strict["population_coverage_share"].to_numpy(float), "o", "-", "Six-component coverage"),
        (100 * high["population_coverage_share"].to_numpy(float), "s", (0, (3, 1.8)), "High-coverage subset"),
    ]
    ax.set_facecolor("white")
    for y, marker, linestyle, label in series:
        for index in range(len(x) - 1):
            colour = GREEN_DIAGNOSTIC[index + 1]
            ax.plot(x[index:index + 2], y[index:index + 2], color=colour,
                    lw=5.2, alpha=0.13, linestyle=linestyle,
                    solid_capstyle="round", solid_joinstyle="round", zorder=1)
            ax.plot(x[index:index + 2], y[index:index + 2], color=colour,
                    lw=1.45, linestyle=linestyle,
                    solid_capstyle="round", solid_joinstyle="round", zorder=3)
        for xx, yy, colour in zip(x, y, GREEN_DIAGNOSTIC):
            ax.scatter(xx, yy, s=18, marker=marker, color=colour,
                       edgecolor="white", linewidth=0.45, zorder=4)
    ax.axhline(50, color="#B7C0C4", lw=0.55, ls=(0, (3, 2)))
    ax.text(2024.25, series[0][0][-1], f"{series[0][0][-1]:.1f}%", ha="left", va="center", fontsize=5.8, color=GREEN_DIAGNOSTIC[-1])
    ax.text(2024.25, series[1][0][-1], f"{series[1][0][-1]:.1f}%", ha="left", va="center", fontsize=5.8, color=GREEN_DIAGNOSTIC[-1])
    ax.set_xticks(YEARS)
    ax.set_xlim(2010.2, 2026.4)
    ax.set_ylim(0, 78)
    ax.set_yticks([0, 25, 50, 75])
    ax.set_ylabel("Population covered (%)")
    ax.set_xlabel("Year")
    clean_axis(ax)
    panel_label(ax, letter, x=-0.13, y=1.07)
    ax.text(0, 1.07, "Coverage audit", transform=ax.transAxes, ha="left",
            va="top", fontsize=7.8, fontweight="bold")
    handles = [
        Line2D([0], [0], color="#4C5960", lw=1.45, marker="o", ms=3.8,
               label="Six-component coverage"),
        Line2D([0], [0], color="#4C5960", lw=1.45, linestyle=(0, (3, 1.8)),
               marker="s", ms=3.5, label="High-coverage subset"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 0.52), ncol=1,
              handlelength=1.5, fontsize=5.6, borderaxespad=0)


def main() -> int:
    base.YEAR_RAMPS.update(MAP_RAMPS)
    frames = base.load_maps()
    breaks = base.pooled_quintiles(frames)
    data_dir = ROOT / "source_data" / "figS_temporal_uncertainty_v4"
    annual = pd.read_csv(data_dir / "annual_block_bootstrap_intervals.csv")
    changes = pd.read_csv(data_dir / "change_from_2011_intervals.csv")
    coverage = pd.read_csv(data_dir / "metric_coverage_audit.csv")
    all_bounds = np.array([frames[y].total_bounds for y in YEARS])
    common_bounds = np.array([all_bounds[:, 0].min(), all_bounds[:, 1].min(),
                              all_bounds[:, 2].max(), all_bounds[:, 3].max()])
    # Expanding the shared extent by 1/0.9 makes every geography occupy exactly
    # 90% of its former linear size while preserving a common centre and scale.
    centre = np.array([(common_bounds[0] + common_bounds[2]) / 2,
                       (common_bounds[1] + common_bounds[3]) / 2])
    half_span = np.array([(common_bounds[2] - common_bounds[0]) / 2,
                          (common_bounds[3] - common_bounds[1]) / 2]) / 0.9
    scaled_bounds = np.array([centre[0] - half_span[0], centre[1] - half_span[1],
                              centre[0] + half_span[0], centre[1] + half_span[1]])

    fig = plt.figure(figsize=(183 * MM, 190 * MM), facecolor="white")
    outer = fig.add_gridspec(3, 1, height_ratios=[1.72, 0.10, 0.72],
                             left=0.055, right=0.958, top=0.982, bottom=0.075, hspace=0.045)
    maps = outer[0].subgridspec(2, 2, wspace=0.10, hspace=0.095)
    for idx, (year, letter) in enumerate(zip(YEARS, "abcd")):
        row, col = divmod(idx, 2)
        cell = maps[row, col].subgridspec(1, 2, width_ratios=[1.0, 0.046], wspace=0.018)
        map_ax = fig.add_subplot(cell[0, 0])
        cax = fig.add_subplot(cell[0, 1])
        base.draw_map(map_ax, cax, frames[year], year, letter, breaks, scaled_bounds)
        position = cax.get_position()
        cax.set_position([position.x0, position.y0 + .12 * position.height,
                          position.width, .76 * position.height])
        cax.yaxis.set_ticks_position("right")
        cax.yaxis.set_label_position("right")
        cax.tick_params(labelleft=False, labelright=True, pad=1.0, labelsize=5.0)

    key_ax = fig.add_subplot(outer[1])
    key_ax.set_axis_off()
    key_ax.legend(handles=[
        Patch(facecolor="#F3F5F6", edgecolor="#C2CACE", lw=0.45, label="Incomplete six-component data"),
        Patch(facecolor="#FFFFFF", edgecolor="#D2D8DB", lw=0.45, label="No platform-listed outlet"),
    ], loc="center", ncol=2, frameon=False, fontsize=5.5, handlelength=1.2, columnspacing=1.0)

    bottom = outer[2].subgridspec(1, 3, width_ratios=[0.95, 1.25, 0.95], wspace=0.52)
    trajectory_panel(fig.add_subplot(bottom[0, 0]), annual)
    component_panel(fig.add_subplot(bottom[0, 1]), changes)
    coverage_panel(fig.add_subplot(bottom[0, 2]), coverage)

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"pooled_break": breaks}).to_csv(OUT / "Fig3_v23_map_breaks.csv", index=False)
    base_path = OUT / STEM
    for label in fig.findobj(mpl.text.Text):
        label.set_color("#151515")
    save_cns_figure(fig, str(base_path))
    fig.savefig(base_path.with_suffix(".png"), bbox_inches="tight", dpi=600)
    plt.close(fig)
    print(base_path.with_suffix(".png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
