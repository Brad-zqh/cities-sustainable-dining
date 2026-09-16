"""Nature-style figure for quality-context-qualified affordable access.

All displayed values are read from the released aggregate source bundle.  The
maps show counts of low-price restaurants reachable in 15 minutes that are
located in destination LSBGs with area SDI >= 0.45.  They do not display a
restaurant-level SDI.
"""

from __future__ import annotations

from pathlib import Path
import os
import sys

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
if str(FIGURES) not in sys.path:
    sys.path.insert(0, str(FIGURES))

from v5_cities_visual_system_tnr import (  # noqa: E402
    COLORS,
    clean_axis,
    configure,
    north_arrow,
    outer_boundary,
    panel_label,
    save_bundle,
    segmented_scale_bar,
)


YEARS = (2016, 2021, 2024)
DATA = ROOT / "source_data" / "fig13_joint_quality_affordable_access_v1"
NETWORK_DATA = ROOT / "source_data" / "fig04_network_price_v4"
OUT = Path(os.environ.get("CITIES_FIGURE_DIR", str(ROOT / "figures")))
STEM = "Fig7_Joint_Quality_Affordable_Access_v2_TNR"

YEAR_COLORS = {2016: "#C94A4A", 2021: "#397FAF", 2024: "#3C9261"}
YEAR_CMAPS = {
    # The lightest positive class is deliberately darker than the neutral
    # zero fill so that small, low-count urban LSBGs remain visible at print size.
    2016: ListedColormap(["#F5B4B3", "#ED8584", "#DB5B61", "#C33849", "#852337"]),
    2021: ListedColormap(["#B3D5EF", "#83BCE1", "#529DCD", "#3277AA", "#174C75"]),
    2024: ListedColormap(["#B9DFB8", "#89CA93", "#58AE72", "#348C58", "#175E40"]),
}
OUTCOME_COLORS = {
    "all": "#7D8790",
    # The lower-row comparison uses one stable two-group semantic throughout:
    # blue = price-only comparator; red = jointly qualified opportunity.
    "low": "#397FAF",
    "joint": "#C94A4A",
}
BREAKS = np.array([1, 25, 75, 225, 450, 900], dtype=float)


def load_data():
    points = pd.read_csv(DATA / "joint_access_point_estimates.csv")
    intervals = pd.read_csv(DATA / "joint_access_dcca_block_intervals.csv")
    network = pd.read_csv(NETWORK_DATA / "network_price_inequality_point_estimates.csv")
    network = network.loc[network["threshold_min"].eq(15.0)].copy()
    maps = {}
    for year in YEARS:
        frame = gpd.read_file(DATA / f"{year}_joint_access_geometry.gpkg").to_crs(2326)
        frame["geometry"] = frame.geometry.simplify(18, preserve_topology=True)
        maps[year] = frame
    return points, intervals, network, maps


def map_panel(ax, cax, frame: gpd.GeoDataFrame, year: int) -> None:
    norm = BoundaryNorm(BREAKS, len(BREAKS) - 1)
    available = frame["network_available"].fillna(False).astype(bool)
    values = pd.to_numeric(frame["joint_access"], errors="coerce")
    frame.plot(ax=ax, color="#FFFFFF", edgecolor="#D6DCE0", linewidth=0.11, alpha=.34)
    missing = frame.loc[~available | values.isna()]
    zero_joint = available & values.eq(0)
    low_price_values = pd.to_numeric(frame["low_price_access"], errors="coerce")
    price_only = frame.loc[zero_joint & low_price_values.gt(0)]
    no_low_price = frame.loc[zero_joint & low_price_values.fillna(0).eq(0)]
    positive = frame.loc[available & values.gt(0)]
    if not missing.empty:
        missing.plot(ax=ax, color="#FFFFFF", edgecolor="#AEB8BE", linewidth=0.18, hatch="////", alpha=.42)
    if not no_low_price.empty:
        no_low_price.plot(ax=ax, color="#F0F2F3", edgecolor="#D5DADD", linewidth=0.105, alpha=.38)
    if not price_only.empty:
        price_only.plot(ax=ax, color="#E6EDF1", edgecolor="#D2D9DD", linewidth=0.09, alpha=.42)
    if not positive.empty:
        positive.plot(
            ax=ax,
            column="joint_access",
            cmap=YEAR_CMAPS[year],
            norm=norm,
            edgecolor="#FFFFFF",
            linewidth=0.055,
            alpha=.91,
        )
    xmin, ymin, xmax, ymax = frame.total_bounds
    dx, dy = xmax - xmin, ymax - ymin
    ax.set_xlim(xmin - 0.01 * dx, xmax + 0.04 * dx)
    ax.set_ylim(ymin - 0.15 * dy, ymax + 0.025 * dy)
    ax.set_axis_off()
    ax.set_aspect("equal")
    outer_boundary(ax, frame, linewidth=0.28)
    ax.set_title(str(year), loc="center", x=0.50, y=0.97, pad=0,
                 fontsize=8.6, fontweight="normal")
    # Compact upper-left placement, deliberately below the panel letter/year.
    # Keep the compass in the upper-left map corner while leaving a clear
    # gutter between it and the panel letter at the extreme left.
    north_arrow(ax, x=0.105, y=0.940, height=0.048)
    segmented_scale_bar(ax, length_km=10, x=0.06, y=0.030)
    residents = pd.to_numeric(frame.get("t_pop"), errors="coerce").fillna(0)
    denominator = float(residents.loc[available & values.notna()].sum())
    covered = float(residents.loc[available & values.gt(0)].sum())
    if denominator <= 0:
        raise ValueError(f"No audited population denominator for {year}")
    # Population coverage and the joint eligibility definition are reported
    # in the manuscript caption, not as explanatory sentences on the map.

    cbar = plt.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=YEAR_CMAPS[year]),
        cax=cax,
        orientation="vertical",
        spacing="uniform",
    )
    cbar.set_ticks(BREAKS)
    cbar.set_ticklabels([str(int(value)) for value in BREAKS])
    cbar.ax.yaxis.set_ticks_position("right")
    cbar.ax.yaxis.set_label_position("right")
    cbar.ax.tick_params(labelsize=5.5, length=1.5, pad=1.2)
    cbar.ax.set_title("n", fontsize=5.7, fontweight="normal", pad=2.0)
    cbar.outline.set_linewidth(0.45)


def select_point(points: pd.DataFrame, year: int, outcome: str, metric: str) -> float:
    row = points.loc[
        points["year"].eq(year)
        & points["outcome"].eq(outcome)
        & points["threshold_min"].eq(15.0)
        & points["price_ceiling_hkd"].eq(100)
        & points["destination_area_sdi_threshold"].eq(0.45)
    ].iloc[0]
    return float(row[metric])


def interval(intervals: pd.DataFrame, year: int, outcome: str, metric: str):
    row = intervals.loc[
        intervals["year"].eq(year)
        & intervals["outcome"].eq(outcome)
        & intervals["metric"].eq(metric)
    ].iloc[0]
    return float(row["point_estimate"]), float(row["ci_low"]), float(row["ci_high"])


def opportunity_funnel(ax, points: pd.DataFrame, network: pd.DataFrame) -> None:
    y_positions = np.arange(len(YEARS))[::-1]
    for y_pos, year in zip(y_positions, YEARS):
        all_mean = float(
            network.loc[
                network["year"].eq(year)
                & network["outcome"].eq("opportunity_n_all_valid_tiers"),
                "population_weighted_mean",
            ].iloc[0]
        )
        low_mean = select_point(points, year, "low_price_access", "population_weighted_mean")
        joint_mean = select_point(points, year, "joint_access", "population_weighted_mean")
        ax.plot([joint_mean, all_mean], [y_pos, y_pos], color="#D5DADD", lw=2.0, zorder=0)
        for value, key, marker in (
            (all_mean, "all", "o"),
            (low_mean, "low", "s"),
            (joint_mean, "joint", "D"),
        ):
            ax.scatter(
                value,
                y_pos,
                s=31,
                marker=marker,
                color=OUTCOME_COLORS[key],
                edgecolor="white",
                linewidth=0.55,
                zorder=3,
            )
        retained = 100 * joint_mean / low_mean
        ax.text(
            joint_mean,
            y_pos - 0.23,
            f"{retained:.0f}% retained",
            color=OUTCOME_COLORS["joint"],
            fontsize=5.8,
            fontweight="bold",
            ha="left",
        )
    ax.set_yticks(y_positions, YEARS)
    ax.set_xlabel("Population-weighted mean\nopportunity count", labelpad=3)
    ax.set_title("Opportunity screen", loc="left", pad=4, fontsize=7.2, fontweight="normal")
    ax.set_xlim(left=0)
    # Reserve a real legend lane above the three observed years; a legend
    # inside the 2021 row previously obscured both the estimate and its label.
    ax.set_ylim(-0.48, 3.25)
    ax.tick_params(axis="y", length=0)
    clean_axis(ax)
    handles = [
        Line2D([], [], marker="o", ls="none", color=OUTCOME_COLORS["all"], label="All valid-price"),
        Line2D([], [], marker="s", ls="none", color=OUTCOME_COLORS["low"], label="≤HK$100"),
        Line2D([], [], marker="D", ls="none", color=OUTCOME_COLORS["joint"], label="≤HK$100 + area SDI ≥0.45"),
    ]
    ax.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(1.01, .985),
        fontsize=4.65,
        handletextpad=0.32,
        labelspacing=0.24,
        markerscale=0.72,
    )


def coefficient_panel(ax, intervals: pd.DataFrame, metric: str, title: str, scale=1.0, zero=False) -> None:
    y_base = np.arange(len(YEARS))[::-1]
    for outcome, color, offset in (
        ("low_price_access", OUTCOME_COLORS["low"], 0.10),
        ("joint_access", OUTCOME_COLORS["joint"], -0.10),
    ):
        for y, y_pos in zip(YEARS, y_base + offset):
            value, lo, hi = interval(intervals, y, outcome, metric)
            value, lo, hi = value * scale, lo * scale, hi * scale
            ax.fill_betweenx(
                [y_pos - 0.065, y_pos + 0.065],
                lo,
                hi,
                color=color,
                alpha=0.18,
                linewidth=0,
                zorder=1,
            )
            ax.plot([lo, hi], [y_pos, y_pos], color=color, lw=1.05, solid_capstyle="round", zorder=2)
            ax.scatter(value, y_pos, s=24, color=color, edgecolor="white", linewidth=0.45, zorder=3)
    if zero:
        ax.axvline(0, color="#9AA3A9", lw=0.65, ls=(0, (3, 2)), zorder=0)
    ax.set_yticks(y_base, YEARS)
    ax.set_ylim(-0.45, len(YEARS) - 0.05)
    ax.set_title(title, loc="left", pad=4, fontsize=7.1, fontweight="normal")
    ax.tick_params(axis="y", length=0, pad=2)
    clean_axis(ax)
    ax.legend(
        handles=[
            Line2D([], [], marker="o", ls="-", color=OUTCOME_COLORS["low"], label="≤HK$100"),
            Line2D([], [], marker="o", ls="-", color=OUTCOME_COLORS["joint"], label="Joint"),
        ],
        loc="upper right",
        bbox_to_anchor=(1.01, .99),
        fontsize=4.55,
        handlelength=1.05,
        handletextpad=0.25,
        labelspacing=0.22,
        markerscale=0.55,
    )


def sensitivity_tile(ax, points: pd.DataFrame) -> None:
    subset = points.loc[
        points["outcome"].eq("joint_access_sensitivity")
        & points["threshold_min"].eq(15.0)
        & points["price_ceiling_hkd"].eq(100)
    ].copy()
    matrix = subset.pivot(
        index="year", columns="destination_area_sdi_threshold", values="zero_access_population_share"
    ).reindex(index=YEARS, columns=[0.40, 0.45, 0.50])
    means = subset.pivot(
        index="year", columns="destination_area_sdi_threshold", values="population_weighted_mean"
    ).reindex(index=YEARS, columns=[0.40, 0.45, 0.50])
    # Low-to-high multihue sequential ramp: blue is low and red is high.
    # No white midpoint or centred normalization is used, so the scale does
    # not imply a signed/diverging statistic.
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "zero_share_blue_red",
        ["#2166AC", "#92C5DE", "#F7F7F7", "#F4A582", "#B2182B"],
    )
    image = ax.imshow(matrix.to_numpy(float) * 100, cmap=cmap, vmin=10, vmax=70, aspect="auto")
    for i, year in enumerate(YEARS):
        for j, threshold in enumerate((0.40, 0.45, 0.50)):
            zero_value = 100 * float(matrix.loc[year, threshold])
            mean_value = float(means.loc[year, threshold])
            color = "#151515"
            ax.text(j, i - 0.09, f"{zero_value:.0f}%", ha="center", va="center", fontsize=6.0, color=color, fontweight="bold")
            ax.text(j, i + 0.20, f"n̄={mean_value:.0f}", ha="center", va="center", fontsize=4.9, color=color)
    ax.set_xticks(range(3), ["0.40", "0.45", "0.50"])
    ax.set_yticks(range(3), YEARS)
    ax.set_xlabel("Area SDI threshold")
    ax.set_title("Threshold sensitivity", loc="left", pad=4, fontsize=7.1, fontweight="normal")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5.5%", pad=0.055)
    cbar = ax.figure.colorbar(image, cax=cax, orientation="vertical")
    cbar.ax.yaxis.set_ticks_position("right")
    cbar.ax.set_title("Zero\n(%)", fontsize=4.9, pad=2.0)
    cbar.ax.tick_params(labelsize=4.9, length=1.8)
    cbar.outline.set_linewidth(0.45)


def main() -> None:
    configure()
    points, intervals, network, maps = load_data()
    fig = plt.figure(figsize=(183 / 25.4, 150 / 25.4), constrained_layout=False)
    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=[1.14, 0.86],
        left=0.058,
        right=0.973,
        top=0.948,
        bottom=0.105,
        hspace=0.25,
    )
    top = outer[0].subgridspec(1, 3, wspace=0.13)
    map_axes = []
    for index, year in enumerate(YEARS):
        cell = top[index].subgridspec(1, 2, width_ratios=[1.0, 0.047], wspace=0.02)
        ax = fig.add_subplot(cell[0])
        cax = fig.add_subplot(cell[1])
        map_panel(ax, cax, maps[year], year)
        box = cax.get_position()
        cax.set_position([box.x0, box.y0 + .170 * box.height, box.width, .680 * box.height])
        # Keep panel letters inside their own map cells rather than over the
        # preceding map's right-hand colour bar.
        panel_label(ax, chr(ord("a") + index), x=0.025, y=1.095)
        map_axes.append(ax)
    fig.legend(
        handles=[
            Patch(
                facecolor="#E7EEF2",
                edgecolor="#CDD6DB",
                linewidth=0.5,
                label="Quality gate unmet",
            ),
            Patch(facecolor="#E9EDF0", edgecolor="#C9D0D4", linewidth=0.5, label="No low-price access"),
            Patch(facecolor="#FFFFFF", edgecolor="#AEB8BE", linewidth=0.5, hatch="////", label="Network unavailable"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.50, 0.548),
        ncol=3,
        frameon=False,
        fontsize=5.35,
        handlelength=1.2,
        columnspacing=0.9,
    )

    bottom = outer[1].subgridspec(
        1,
        5,
        width_ratios=[1.14, 0.82, 0.68, 0.85, 1.64],
        wspace=0.40,
    )
    ax_d = fig.add_subplot(bottom[0])
    opportunity_funnel(ax_d, points, network)
    panel_label(ax_d, "d", x=-0.09, y=1.04)

    metric_specs = (
        ("zero_access_population_share", "Zero access (%)", 100.0, False),
        ("weighted_gini", "Gini", 1.0, False),
        ("concentration_index", "Income\nconcentration", 1.0, True),
    )
    for position, (metric, title, scale, zero) in enumerate(metric_specs, start=1):
        ax = fig.add_subplot(bottom[position])
        coefficient_panel(ax, intervals, metric, title, scale=scale, zero=zero)
        if position == 1:
            panel_label(ax, "e", x=-0.17, y=1.04)
        else:
            ax.set_yticklabels([])

    ax_f = fig.add_subplot(bottom[4])
    sensitivity_tile(ax_f, points)
    panel_label(ax_f, "f", x=-0.12, y=1.04)

    for label in fig.findobj(mpl.text.Text):
        label.set_color(COLORS["ink"])
    save_bundle(fig, OUT, STEM, preview_dpi=320)
    plt.close(fig)


if __name__ == "__main__":
    main()

