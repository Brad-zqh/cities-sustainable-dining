from __future__ import annotations

from pathlib import Path
import sys

# Allow this script to run directly from a clean repository clone without
# requiring users to set PYTHONPATH manually. Shared Nature-style helpers live
# in the sibling figures directory.
FIGURES_DIR = Path(__file__).resolve().parents[1] / "figures"
if str(FIGURES_DIR) not in sys.path:
    sys.path.insert(0, str(FIGURES_DIR))

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D

from v4_plot_four_year_results import (
    MM,
    add_panel_label,
    configure_matplotlib,
    plot_choropleth,
    pooled_quantile_breaks,
    save_figure,
)


YEARS = [2011, 2016, 2021, 2024]
ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DIR = ROOT / "source_data" / "figS_price_market_v4"
DATA_DIR = (
    PUBLIC_DIR
    if (PUBLIC_DIR / "analysis_contract.json").exists()
    else ROOT / "outputs" / "restricted" / "v4_price_market_decomposition"
)

COLORS = {
    "income": "#2B6CB0",
    "adjusted": "#C94157",
    "supply": "#347FC0",
    "low_count": "#C94157",
    "low_share": "#184A78",
    "ink": "#202A32",
    "muted": "#6C7880",
    "grid": "#DDE3E6",
}

YEAR_RAMPS = {
    2011: ["#FBEDEC", "#F3B6B2", "#D96B66", "#9F2F32"],
    2016: ["#EDF5FA", "#B7D4EA", "#6BA5CC", "#1F5A8A"],
    2021: ["#EEF7EF", "#B9DDBB", "#72B77B", "#246B3B"],
    2024: ["#FFF3E0", "#F9D39A", "#EDA64A", "#B5651D"],
}


def load_frames() -> dict[int, gpd.GeoDataFrame]:
    frames = {}
    for year in YEARS:
        frame = gpd.read_file(DATA_DIR / f"{year}_lsbg_price_market.gpkg").to_crs(2326)
        frame["geometry"] = frame.geometry.simplify(18.0, preserve_topology=True)
        # The shared cartographic function uses restaurant_n to distinguish empty units.
        frame["restaurant_n"] = frame["active_restaurant_n"]
        frames[year] = frame
    return frames


def style_stat_axis(ax: plt.Axes, grid_axis: str = "x") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(False)
    ax.set_axisbelow(True)


def plot_income_attenuation(ax: plt.Axes, regression: pd.DataFrame) -> None:
    subset = regression.loc[regression["predictor"].eq("log_income")].copy()
    y = np.arange(len(YEARS))
    for model, color, offset, label in [
        ("Income only", COLORS["income"], -0.10, "Income only"),
        ("Income + market composition", COLORS["adjusted"], 0.10, "Adjusted model"),
    ]:
        data = subset.loc[subset["model"].eq(model)].set_index("year").reindex(YEARS)
        x = data["standardized_beta"].to_numpy(dtype=float)
        low = data["ci_low"].to_numpy(dtype=float)
        high = data["ci_high"].to_numpy(dtype=float)
        # A pale stroke follows each audited interval; it is estimate emphasis,
        # not panel-background shading.
        ax.errorbar(
            x,
            y + offset,
            xerr=np.vstack([x - low, high - x]),
            fmt="none",
            ecolor=color,
            elinewidth=5.0,
            alpha=0.12,
            capsize=0,
            zorder=1,
        )
        ax.errorbar(
            x,
            y + offset,
            xerr=np.vstack([x - low, high - x]),
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=1.25,
            capsize=2.1,
            markersize=4.5,
            markeredgecolor="white",
            markeredgewidth=0.5,
            label=label,
            zorder=3,
        )
    ax.axvline(0, color="#7B858C", linewidth=0.7, linestyle="--")
    ax.set_yticks(y, YEARS)
    ax.invert_yaxis()
    ax.set_xlabel("Standardized income coefficient (95% CI)")
    ax.set_title("Income coefficient attenuation", loc="left", pad=4, fontweight="bold")
    style_stat_axis(ax, "x")
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.50, -0.16),
        handletextpad=0.5,
        borderaxespad=0,
    )
    add_panel_label(ax, "b", x=-0.16, y=1.02)


def plot_full_model_heatmap(ax: plt.Axes, regression: pd.DataFrame) -> None:
    labels = {
        "log_income": "Income",
        "log_supply": "Restaurant supply",
        "low_price_share": "Low-price share",
        "ageing_share": "Ageing",
    }
    subset = regression.loc[regression["model"].eq("Income + market composition")].copy()
    matrix = (
        subset.pivot(index="predictor", columns="year", values="standardized_beta")
        .reindex(labels)
        .reindex(columns=YEARS)
    )
    matrix.index = [labels[index] for index in matrix.index]
    values = matrix.to_numpy(dtype=float)
    norm = TwoSlopeNorm(vmin=-0.40, vcenter=0.0, vmax=0.40)
    readable_diverging = LinearSegmentedColormap.from_list(
        "readable_market_coefficients", ["#A9CBE5", "#FFFFFF", "#E8A5A3"], N=256
    )
    image = ax.imshow(values, cmap=readable_diverging, norm=norm, aspect="auto")
    ax.set_xticks(np.arange(len(YEARS)), YEARS)
    ax.set_yticks(np.arange(len(matrix.index)), matrix.index)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            ax.text(
                column,
                row,
                f"{value:+.2f}",
                ha="center",
                va="center",
                fontsize=6.4,
                color="#151515",
            )
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Market-composition coefficients", loc="left", pad=4, fontweight="bold")
    cbar = ax.figure.colorbar(image, ax=ax, orientation="horizontal", fraction=0.055, pad=0.16, aspect=28)
    cbar.set_label("Association with area quality score", labelpad=2)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=2, width=0.5)
    add_panel_label(ax, "c", x=-0.16, y=1.02)


def plot_price_supply_trajectory(
    ax_count: plt.Axes,
    ax_share: plt.Axes,
    summary: pd.DataFrame,
) -> None:
    summary = summary.set_index("year").reindex(YEARS)
    # Pale strokes follow the estimates themselves; they are not panel-background shading.
    ax_count.plot(
        YEARS,
        summary["population_weighted_total_supply_per1000"],
        color=COLORS["supply"],
        linewidth=5.2,
        alpha=0.12,
        solid_capstyle="round",
        zorder=1,
    )
    ax_count.plot(
        YEARS,
        summary["population_weighted_total_supply_per1000"],
        color=COLORS["supply"],
        marker="o",
        linewidth=1.8,
        markersize=4.4,
        markeredgecolor="white",
        markeredgewidth=0.5,
        label="All restaurants",
        zorder=2,
    )
    ax_count.plot(
        YEARS,
        summary["population_weighted_low_price_per1000"],
        color=COLORS["low_count"],
        linewidth=5.2,
        alpha=0.12,
        solid_capstyle="round",
        zorder=1,
    )
    ax_count.plot(
        YEARS,
        summary["population_weighted_low_price_per1000"],
        color=COLORS["low_count"],
        marker="o",
        linewidth=1.8,
        markersize=4.4,
        markeredgecolor="white",
        markeredgewidth=0.5,
        label="≤HK$100 restaurants",
        zorder=2,
    )
    ax_count.set_xticks(YEARS)
    ax_count.tick_params(axis="x", labelbottom=False)
    ax_count.set_ylabel("Restaurants per 1,000")
    style_stat_axis(ax_count, "y")
    ax_share.plot(
        YEARS,
        100 * summary["population_weighted_low_price_share"],
        color=COLORS["low_share"],
        linewidth=4.6,
        linestyle="--",
        alpha=0.10,
        zorder=1,
    )
    ax_share.plot(
        YEARS,
        100 * summary["population_weighted_low_price_share"],
        color=COLORS["low_share"],
        marker="D",
        linewidth=1.35,
        linestyle="--",
        markersize=3.8,
        markeredgecolor="white",
        markeredgewidth=0.45,
        label="≤HK$100 share",
        zorder=2,
    )
    ax_share.set_xticks(YEARS)
    ax_share.set_xlabel("Year")
    ax_share.set_ylabel("Low-price share (%)")
    style_stat_axis(ax_share, "y")
    ax_count.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.00, 1.01),
        borderaxespad=0,
        handlelength=1.7,
        fontsize=5.3,
    )
    ax_share.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.00, 1.00),
        borderaxespad=0,
        handlelength=1.7,
        fontsize=5.3,
    )
    ax_count.set_title("Price-sensitive opportunity", loc="left", pad=4, fontweight="bold")
    add_panel_label(ax_count, "d", x=-0.16, y=1.03)


def main() -> None:
    configure_matplotlib()
    frames = load_frames()
    regression = pd.read_csv(DATA_DIR / "quality_market_decomposition.csv")
    summary = pd.read_csv(DATA_DIR / "price_market_summary.csv")

    breaks = pooled_quantile_breaks(frames, "low_price_share", quantiles=5)
    fig = plt.figure(figsize=(183 * MM, 210 * MM), constrained_layout=False)
    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=[1.64, 1.00],
        left=0.050,
        right=0.958,
        top=0.965,
        bottom=0.105,
        hspace=0.16,
    )
    map_grid = outer[0].subgridspec(2, 2, wspace=0.075, hspace=0.055)
    map_axes = []
    for index, year in enumerate(YEARS):
        ax = fig.add_subplot(map_grid[index // 2, index % 2])
        cmap = LinearSegmentedColormap.from_list(f"price_share_{year}", YEAR_RAMPS[year], N=256)
        norm = BoundaryNorm(breaks, cmap.N)
        plot_choropleth(
            ax,
            frames[year],
            "low_price_share",
            cmap,
            norm,
            title=str(year),
            # Every spatial panel carries its own scale bar and north arrow.
            show_cartography=True,
        )
        cax = ax.inset_axes([1.006, .10, .030, .81], transform=ax.transAxes)
        local_bar = fig.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=cax,
            orientation="vertical",
            boundaries=breaks,
            ticks=breaks,
        )
        local_bar.ax.yaxis.set_ticks_position("right")
        local_bar.ax.set_yticklabels([
            f"{max(0, 100 * value):.0f}" for value in breaks
        ])
        local_bar.ax.set_title("%", fontsize=5.5, pad=1.5)
        local_bar.ax.tick_params(labelsize=5.2, length=1.5, width=.45, pad=1.0)
        local_bar.outline.set_linewidth(.45)
        map_axes.append(ax)
    add_panel_label(map_axes[0], "a", x=-0.09, y=1.02)

    bottom = outer[1].subgridspec(1, 3, width_ratios=[1.04, 0.88, 1.05], wspace=0.48)
    plot_income_attenuation(fig.add_subplot(bottom[0, 0]), regression)
    plot_full_model_heatmap(fig.add_subplot(bottom[0, 1]), regression)
    trajectory = bottom[0, 2].subgridspec(2, 1, height_ratios=[0.62, 0.38], hspace=0.18)
    plot_price_supply_trajectory(
        fig.add_subplot(trajectory[0, 0]),
        fig.add_subplot(trajectory[1, 0]),
        summary,
    )

    save_figure(fig, "FigS_FourYear_Price_Market_Decomposition_Cities_RR_v15_RIGHT_CBAR")
    plt.close(fig)


if __name__ == "__main__":
    main()
