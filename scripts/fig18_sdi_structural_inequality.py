"""Nature-style four-year structural inequality analysis for area SDI."""

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

SOURCE = ROOT / "source_data" / "figS_sdi_structural_inequality_v4"
OUT = Path(os.environ.get("CITIES_FIGURE_DIR", str(ROOT / "figures")))
OUT.mkdir(parents=True, exist_ok=True)
STEM_NAME = "Fig18_FourYear_SDI_Structural_Inequality_NATURE"

YEARS = [2011, 2016, 2021, 2024]
INK = COLORS["ink"]
MUTED = COLORS["muted"]
GRID = COLORS["grid"]


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


def interval_axis(ax: plt.Axes, table: pd.DataFrame, metric: str,
                  title: str, xlabel: str, show_years: bool = True) -> None:
    subset = table.loc[table["metric"].eq(metric)].set_index("year").loc[YEARS]
    y = np.arange(len(YEARS))[::-1]
    for yi, year in zip(y, YEARS):
        row = subset.loc[year]
        ax.errorbar(
            row["estimate"], yi,
            xerr=[[row["estimate"] - row["lower_95"]], [row["upper_95"] - row["estimate"]]],
            fmt="none", ecolor=YEAR_COLORS[year], elinewidth=5.2,
            alpha=.13, capsize=0, zorder=1,
        )
        ax.errorbar(
            row["estimate"], yi,
            xerr=[[row["estimate"] - row["lower_95"]], [row["upper_95"] - row["estimate"]]],
            fmt="o", ms=4.6, color=YEAR_COLORS[year], ecolor=YEAR_COLORS[year],
            elinewidth=1.35, capsize=2.5, markeredgecolor="white", markeredgewidth=0.55,
            zorder=3,
        )
    ax.set_yticks(y, YEARS if show_years else [""] * len(YEARS), color="black")
    ax.set_title(title, loc="left", fontweight="bold", pad=5)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", color=GRID, linewidth=0.55)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)


def save(fig: plt.Figure) -> None:
    save_bundle(fig, OUT, STEM_NAME, preview_dpi=360)


def main() -> None:
    configure()
    intervals = pd.read_csv(SOURCE / "sdi_inequality_block_intervals.csv")
    quintiles = pd.read_csv(SOURCE / "sdi_income_quintile_means.csv")
    curves = pd.read_csv(SOURCE / "sdi_income_concentration_curves.csv")
    components = pd.read_csv(SOURCE / "component_income_concentration.csv")

    fig = plt.figure(figsize=(183 / 25.4, 146 / 25.4), facecolor="white")
    # Reserve a true outer margin for the heat-map colour bar and its label.
    # The former 0.965 right bound was safe on-screen but could clip during
    # Word/PDF placement and raster trimming.
    gs = fig.add_gridspec(2, 2, left=0.09, right=0.925, bottom=0.115,
                          top=0.955, wspace=0.34, hspace=0.46)

    sub = gs[0, 0].subgridspec(1, 2, wspace=0.50)
    ax_a1 = fig.add_subplot(sub[0, 0])
    ax_a2 = fig.add_subplot(sub[0, 1])
    interval_axis(ax_a1, intervals, "weighted_gini", "Gini coefficient", "Gini")
    interval_axis(ax_a2, intervals, "theil_t", "Theil T", "Theil T", show_years=False)
    panel(ax_a1, "a")

    ax = fig.add_subplot(gs[0, 1])
    ax.axhline(0, color="#8D989F", lw=0.9, ls="--", zorder=2)
    for year in YEARS:
        d = curves.loc[curves["year"].eq(year)]
        gap = 100 * (d["cumulative_sdi_fraction"] - d["population_fraction"])
        # A translucent halo follows the observed curve itself. It does not
        # represent an interval or shade the space between the curve and zero.
        ax.plot(d["population_fraction"], gap, color=YEAR_COLORS[year],
                lw=5.4, alpha=.12, solid_capstyle="round", zorder=1)
        ax.plot(
            d["population_fraction"], gap,
            color=YEAR_COLORS[year], lw=1.7, label=str(year), zorder=3,
        )
    ax.set_title("Income-ranked concentration gap", loc="left", fontweight="bold", pad=5)
    ax.set_xlabel("Cumulative population, low to high income")
    ax.set_ylabel("SDI share minus population share (pp)")
    ax.set_xlim(0, 1)
    ax.grid(False)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(.53, .995), handlelength=1.7,
              columnspacing=0.9)
    panel(ax, "b")

    ax = fig.add_subplot(gs[1, 0])
    for year in YEARS:
        d = quintiles.loc[quintiles["year"].eq(year)]
        ax.plot(d["income_quintile"], d["mean_sdi"], color=YEAR_COLORS[year],
                lw=5.0, alpha=.12, solid_capstyle="round", zorder=1)
        ax.plot(d["income_quintile"], d["mean_sdi"], color=YEAR_COLORS[year],
                lw=1.65, marker="o", ms=4.2, markeredgecolor="white",
                markeredgewidth=0.55, label=str(year), zorder=2)
    ax.set_title("Income gradient in area SDI", loc="left", fontweight="bold", pad=5)
    ax.set_xlabel("Income population quintile")
    ax.set_ylabel("Population-weighted mean SDI")
    ax.set_xticks(range(1, 6), ["Q1\nlowest", "Q2", "Q3", "Q4", "Q5\nhighest"])
    ax.grid(False)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=2, loc="upper left", handlelength=1.7,
              columnspacing=0.9)
    panel(ax, "c")

    ax = fig.add_subplot(gs[1, 1])
    order = ["Nutrition", "Carbon", "Cuisine diversity", "Sustainability text", "Hygiene", "Practice tag*"]
    display_order = ["Nutrition", "Carbon", "Cuisine diversity", "Environmental\nsustainability", "Hygiene", "Practice"]
    matrix = components.pivot(index="component", columns="year", values="concentration_index").loc[order, YEARS]
    limit = max(0.10, float(np.nanmax(np.abs(matrix.to_numpy()))))
    muted_diverging = LinearSegmentedColormap.from_list(
        "muted_income_concentration",
        ["#4575B4", "#B9D5E8", "#F7F7F7", "#F4C2B5", "#C84A4A"],
    )
    image = ax.imshow(matrix, cmap=muted_diverging, vmin=-limit, vmax=limit, aspect="auto")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix.iloc[row, column]
            ax.text(column, row, f"{value:+.02f}", ha="center", va="center",
                    fontsize=6.2, color="white" if abs(value) > 0.62 * limit else "black")
    ax.set_xticks(range(len(YEARS)), YEARS, color="black")
    ax.set_yticks(range(len(order)), display_order, color="black")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Income concentration by SDI component", loc="left", fontweight="bold", pad=5)
    cbar = fig.colorbar(image, ax=ax, orientation="vertical", fraction=0.040, pad=0.028)
    cbar.set_label("Income concentration index", fontsize=7.0, labelpad=5)
    cbar.ax.tick_params(labelsize=6.8)
    cbar.outline.set_linewidth(0.55)
    panel(ax, "d")

    save(fig)
    plt.close(fig)
    print(OUT / f"{STEM_NAME}.png")


if __name__ == "__main__":
    main()
