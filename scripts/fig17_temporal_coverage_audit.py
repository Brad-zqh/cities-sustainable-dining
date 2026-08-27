"""Nature-style audit of temporal coverage and specification sensitivity."""

from __future__ import annotations

from pathlib import Path
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "scripts"
if str(FIGURES) not in sys.path:
    sys.path.insert(0, str(FIGURES))
from v5_cities_visual_system import COLORS, FONT_FAMILY, YEAR_COLORS, configure as base_configure, panel_label, save_bundle

SOURCE = ROOT / "source_data" / "figS_temporal_uncertainty_v4"
OUT = Path(os.environ.get("CITIES_FIGURE_DIR", str(ROOT / "figures")))
OUT.mkdir(parents=True, exist_ok=True)
STEM_NAME = "Fig17_Temporal_Coverage_Audit_NATURE"

YEARS = [2011, 2016, 2021, 2024]
INK = COLORS["ink"]
MUTED = COLORS["muted"]
GRID = COLORS["grid"]
BLUE = COLORS["blue_dark"]
PURPLE = COLORS["violet"]
GOLD = COLORS["gold"]
TEAL = COLORS["teal"]


def configure() -> None:
    base_configure()
    mpl.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "font.sans-serif": [FONT_FAMILY, "Helvetica", "Nimbus Sans", "DejaVu Sans"],
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 6.8,
            "axes.linewidth": 0.75,
            "axes.edgecolor": INK,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def panel(ax: plt.Axes, letter: str) -> None:
    panel_label(ax, letter, x=-0.10, y=1.07, fontsize=10.0)


def tidy(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.55, zorder=0)
    ax.set_axisbelow(True)


def line_ci(ax: plt.Axes, data: pd.DataFrame, metric: str, color: str,
            label: str, marker: str) -> None:
    d = data.loc[data["metric"].eq(metric)].set_index("year").loc[YEARS]
    x = np.asarray(YEARS)
    y = d["population_weighted_mean"].to_numpy(float)
    lo = d["ci_low"].to_numpy(float)
    hi = d["ci_high"].to_numpy(float)
    ax.fill_between(x, lo, hi, color=color, alpha=0.14, linewidth=0)
    ax.plot(x, y, color=color, lw=1.65, marker=marker, ms=4.4,
            markeredgecolor="white", markeredgewidth=0.55, label=label, zorder=3)


def save(fig: plt.Figure) -> None:
    save_bundle(fig, OUT, STEM_NAME, preview_dpi=360)


def main() -> None:
    configure()
    annual = pd.read_csv(SOURCE / "annual_block_bootstrap_intervals.csv")
    changes = pd.read_csv(SOURCE / "change_from_2011_intervals.csv")
    coverage = pd.read_csv(SOURCE / "metric_coverage_audit.csv")

    fig = plt.figure(figsize=(183 / 25.4, 142 / 25.4), facecolor="white")
    gs = fig.add_gridspec(2, 2, left=0.155, right=0.955, bottom=0.105,
                          top=0.955, wspace=0.52, hspace=0.48)

    ax = fig.add_subplot(gs[0, 0])
    line_ci(ax, annual, "equal_six_strict", BLUE, "Six components", "o")
    line_ci(ax, annual, "equal_five_no_practice", COLORS["green"], "Without practice", "s")
    line_ci(ax, annual, "equal_six_highcoverage", GOLD, "High-coverage subset", "D")
    ax.set_xticks(YEARS)
    ax.set_ylim(0.34, 0.52)
    ax.set_ylabel("Population-weighted SDI")
    ax.set_title("Temporal SDI across specifications", loc="left", fontweight="bold", pad=5)
    ax.legend(frameon=False, loc="upper left", handlelength=1.8)
    tidy(ax)
    panel(ax, "a")

    ax = fig.add_subplot(gs[0, 1])
    for metric, color, marker, label in [
        ("equal_six_strict", BLUE, "o", "Six-component complete case"),
        ("equal_six_highcoverage", GOLD, "D", "High-coverage nutrition/carbon"),
    ]:
        d = coverage.loc[coverage["metric"].eq(metric)].set_index("year").loc[YEARS]
        ax.plot(YEARS, 100 * d["population_coverage_share"], color=color,
                lw=5.2, alpha=.12, solid_capstyle="round", zorder=1)
        ax.plot(YEARS, 100 * d["population_coverage_share"], color=color, marker=marker,
                lw=1.65, ms=4.4, markeredgecolor="white", markeredgewidth=0.55,
                label=label, zorder=2)
    ax.set_xticks(YEARS)
    ax.set_ylim(0, 82)
    ax.set_ylabel("Population represented (%)")
    ax.set_title("Coverage after the early platform period", loc="left",
                 fontweight="bold", pad=5)
    ax.legend(frameon=False, loc="lower right", handlelength=1.8)
    tidy(ax)
    panel(ax, "b")

    ax = fig.add_subplot(gs[1, 0])
    order = [
        "nutrition_score", "carbon_score", "sustainability_score", "hygiene_score",
        "practice_score", "diversity_score", "equal_six_strict",
        "equal_five_no_practice", "equal_six_highcoverage",
    ]
    labels = [
        "Nutrition", "Carbon", "Environmental\nsustainability", "Hygiene", "Practice",
        "Cuisine diversity", "Six components", "Without practice", "High-coverage subset",
    ]
    d = changes.loc[changes["year"].eq(2024)].set_index("metric").loc[order]
    y = np.arange(len(order))[::-1]
    colors = [COLORS["coral"], COLORS["blue"], COLORS["teal"], COLORS["green"],
              COLORS["gold"], COLORS["violet"], BLUE, COLORS["green"], GOLD]
    for yy, (_, row), color in zip(y, d.iterrows(), colors):
        ax.plot([row["ci_low"], row["ci_high"]], [yy, yy], color=color,
                lw=5.8, alpha=.13, solid_capstyle="round", zorder=1)
        ax.plot([row["ci_low"], row["ci_high"]], [yy, yy], color=color, lw=1.45)
        ax.scatter(row["difference"], yy, s=27, color=color, edgecolor="white",
                   linewidth=0.55, zorder=3)
    ax.axvline(0, color=INK, lw=0.75, ls="--")
    ax.axhline(2.5, color=GRID, lw=0.75)
    ax.set_yticks(y, labels, color="black")
    ax.set_xlim(-0.065, 0.19)
    ax.set_xlabel("Change in score, 2024 minus 2011")
    ax.set_title("Component-specific change, 2011–2024", loc="left", fontweight="bold", pad=5)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(False)
    ax.set_axisbelow(True)
    panel(ax, "c")

    ax = fig.add_subplot(gs[1, 1])
    component_order = [
        "nutrition_score", "carbon_score", "sustainability_score",
        "hygiene_score", "practice_score", "diversity_score",
    ]
    component_labels = [
        "Nutrition", "Carbon", "Environmental\nsustainability", "Hygiene",
        "Practice", "Cuisine diversity",
    ]
    matrix = np.array([
        coverage.loc[coverage["metric"].eq(metric)].set_index("year")
        .loc[YEARS, "population_coverage_share"].to_numpy(float)
        for metric in component_order
    ]) * 100
    cmap = LinearSegmentedColormap.from_list(
        "coverage_blue_red",
        ["#2166AC", "#67A9CF", "#D1E5F0", "#F7F7F7", "#FDDBC7", "#EF8A62", "#B2182B"],
    )
    image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=50, vmax=85)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            ax.text(j, i, f"{value:.0f}", ha="center", va="center",
                    fontsize=6.2, color="white" if value >= 72 else "black")
    ax.set_xticks(range(4), YEARS, color="black")
    ax.set_yticks(range(6), component_labels, color="black")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Population coverage by component", loc="left", fontweight="bold", pad=5)
    cbar = fig.colorbar(image, ax=ax, orientation="vertical", fraction=0.046, pad=0.035)
    # The panel title supplies the quantity; retain just its unit on the bar.
    cbar.ax.set_title("%", fontsize=7.2, pad=4)
    cbar.outline.set_linewidth(0.55)
    panel(ax, "d")

    save(fig)
    plt.close(fig)
    print(OUT / f"{STEM_NAME}.png")


if __name__ == "__main__":
    main()
