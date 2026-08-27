"""Reference-driven Fig. 2 redraw for the Cities reject-and-resubmit.

The layout follows the locked Nature-PDF figure contract: four equal maps,
independent cartographic furniture and vertical keys, followed by three
quantitative panels that show uncertainty, component effects and coverage.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
if str(FIGURES) not in sys.path:
    sys.path.insert(0, str(FIGURES))

from v5_cities_visual_system import (  # noqa: E402
    COLORS,
    COMPONENT_COLORS,
    MM,
    YEAR_COLORS,
    clean_axis,
    configure,
    north_arrow,
    outer_boundary,
    panel_label,
    save_bundle,
    segmented_scale_bar,
)


YEARS = [2011, 2016, 2021, 2024]
FIELD = "equal_six_strict"

# Matplotlib/ColorBrewer sequential families sampled at matched lightness.
# Breakpoints remain identical across years; hue identifies time, while
# lightness encodes SDI magnitude within every map.
YEAR_RAMPS = {
    # The first valid class is deliberately chromatic rather than near-white.
    # White/near-white is reserved for missing or no-outlet polygons.
    2011: ["#F1D4E5", "#D9A9CE", "#C77DBB", "#B14C9A", "#7A1F68"],
    2016: ["#D8EAF5", "#AACFE6", "#72ADD2", "#3D83B7", "#1D527D"],
    2021: ["#DCEED7", "#AED7A7", "#78BB78", "#439956", "#1F6B3B"],
    2024: ["#FBE2C8", "#F7BE82", "#ED9550", "#D86B29", "#9F3F17"],
}

COMPONENT_LABELS = {
    "nutrition_score": "Nutrition",
    "carbon_score": "Carbon",
    "sustainability_score": "Environmental sustainability",
    "hygiene_score": "Hygiene",
    "practice_score": "Practice",
    "diversity_score": "Cuisine diversity",
}

COMPONENT_COLOUR_KEYS = {
    "Environmental sustainability": "Sustainability",
}


def load_maps() -> dict[int, gpd.GeoDataFrame]:
    data_dir = ROOT / "source_data" / "fig_v4_four_year"
    frames: dict[int, gpd.GeoDataFrame] = {}
    for year in YEARS:
        frame = gpd.read_file(data_dir / f"{year}_lsbg_components.gpkg").to_crs(2326)
        if FIELD not in frame.columns and "sdi_equal_arithmetic" in frame.columns:
            frame[FIELD] = frame["sdi_equal_arithmetic"]
        frame[FIELD] = pd.to_numeric(frame[FIELD], errors="coerce")
        frame["restaurant_n"] = pd.to_numeric(frame["restaurant_n"], errors="coerce").fillna(0)
        frame["geometry"] = frame.geometry.simplify(10, preserve_topology=True)
        frames[year] = frame
    return frames


def pooled_quintiles(frames: dict[int, gpd.GeoDataFrame]) -> np.ndarray:
    values = pd.concat(
        [f.loc[f["restaurant_n"].gt(0), FIELD] for f in frames.values()],
        ignore_index=True,
    ).dropna()
    breaks = values.quantile(np.linspace(0, 1, 6)).to_numpy(dtype=float, copy=True)
    if len(np.unique(np.round(breaks, 10))) != 6:
        raise ValueError("Strict SDI pooled quintiles are not unique")
    breaks[0] = np.nextafter(breaks[0], -np.inf)
    breaks[-1] = np.nextafter(breaks[-1], np.inf)
    return breaks


def draw_map(
    ax: plt.Axes,
    cax: plt.Axes,
    frame: gpd.GeoDataFrame,
    year: int,
    label: str,
    breaks: np.ndarray,
    common_bounds: np.ndarray,
) -> None:
    cmap = ListedColormap(YEAR_RAMPS[year], name=f"sdi_{year}")
    norm = BoundaryNorm(breaks, cmap.N)

    def layer(data: gpd.GeoDataFrame, **kwargs: object) -> None:
        before = len(ax.collections)
        data.plot(ax=ax, **kwargs)
        for collection in ax.collections[before:]:
            collection.set_rasterized(True)

    # Coverage states form a quiet context layer; they must not resemble low SDI.
    layer(frame, color="#F3F5F6", edgecolor="#C2CACE", linewidth=0.075, alpha=0.32)
    no_outlet = frame["restaurant_n"].le(0)
    layer(frame.loc[no_outlet], color="#FFFFFF", edgecolor="#D2D8DB", linewidth=0.055, alpha=0.40)
    observed = frame[FIELD].notna() & frame["restaurant_n"].gt(0)
    layer(
        frame.loc[observed],
        column=FIELD,
        cmap=cmap,
        norm=norm,
        edgecolor="white",
        linewidth=0.025,
        alpha=0.91,
    )

    xmin, ymin, xmax, ymax = common_bounds
    dx, dy = xmax - xmin, ymax - ymin
    # Dedicated bottom and right margins keep furniture away from polygons.
    ax.set_xlim(xmin - 0.010 * dx, xmax + 0.038 * dx)
    ax.set_ylim(ymin - 0.105 * dy, ymax + 0.022 * dy)
    ax.set_aspect("equal")
    ax.set_axis_off()
    panel_label(ax, label, x=-0.015, y=1.015)
    ax.text(
        0.500,
        1.015,
        str(year),
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8.0,
        fontweight="bold",
        color=COLORS["ink"],
        clip_on=False,
    )
    outer_boundary(ax, frame, linewidth=0.20)
    # Left placement keeps the arrow clear of the right-side colour scale.
    north_arrow(ax, x=0.095, y=0.940, height=0.055)
    segmented_scale_bar(ax, length_km=10, x=0.055, y=0.012)

    cbar = plt.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cax,
        orientation="vertical",
        boundaries=breaks,
        ticks=breaks,
        spacing="uniform",
    )
    cbar.ax.set_yticklabels([f"{0.0 if abs(v) < 0.0005 else v:.2f}" for v in breaks])
    cbar.ax.tick_params(labelsize=5.3, length=1.8, width=0.5, pad=1.3)
    cbar.ax.set_title("SDI", fontsize=5.8, fontweight="bold", pad=2.5)
    cbar.outline.set_linewidth(0.45)


def trajectory_panel(ax: plt.Axes, annual: pd.DataFrame, letter: str = "e") -> None:
    work = annual.loc[annual["metric"].eq(FIELD)].set_index("year").loc[YEARS]
    x = np.asarray(YEARS, dtype=float)
    mean = work["population_weighted_mean"].to_numpy(float)
    lo = work["ci_low"].to_numpy(float)
    hi = work["ci_high"].to_numpy(float)

    ax.fill_between(x, lo, hi, color="#DCE7EE", alpha=0.95, linewidth=0, zorder=1)
    segments = np.stack([np.column_stack([x[:-1], mean[:-1]]), np.column_stack([x[1:], mean[1:]])], axis=1)
    ax.add_collection(LineCollection(segments, colors=[YEAR_COLORS[y] for y in YEARS[1:]], linewidths=1.65, zorder=3))
    for year, xx, yy in zip(YEARS, x, mean):
        ax.scatter(xx, yy, s=28, color=YEAR_COLORS[year], edgecolor="white", linewidth=0.55, zorder=4)
        ax.text(xx, hi[list(x).index(xx)] + 0.0030, f"{yy:.3f}", ha="center", va="bottom", fontsize=5.8, color=COLORS["ink"])
    ax.axhline(mean[0], color="#AEB9BE", lw=0.55, ls=(0, (3, 2)), zorder=0)
    ax.set_xticks(YEARS)
    ax.set_xlim(2010.2, 2024.8)
    ax.set_ylim(0.365, 0.462)
    ax.set_ylabel("Population-weighted SDI")
    ax.set_xlabel("Year")
    clean_axis(ax)
    panel_label(ax, letter, x=-0.11, y=1.07)
    ax.text(0, 1.07, "Temporal change", transform=ax.transAxes, ha="left", va="top", fontsize=7.8, fontweight="bold")
    ax.text(0.99, 0.02, "95% DCCA-block CI", transform=ax.transAxes, ha="right", va="bottom", fontsize=5.4, color=COLORS["muted"])


def component_panel(ax: plt.Axes, changes: pd.DataFrame, letter: str = "f") -> None:
    order = [
        "sustainability_score",
        "diversity_score",
        "practice_score",
        "hygiene_score",
        "nutrition_score",
        "carbon_score",
    ]
    work = changes.loc[(changes["year"].eq(2024)) & changes["metric"].isin(order)].set_index("metric").loc[order]
    y = np.arange(len(order))[::-1]
    for yy, metric in zip(y, order):
        row = work.loc[metric]
        value, lo, hi = float(row["difference"]), float(row["ci_low"]), float(row["ci_high"])
        label = COMPONENT_LABELS[metric]
        colour = COMPONENT_COLORS[COMPONENT_COLOUR_KEYS.get(label, label)]
        ax.plot([lo, hi], [yy, yy], color=colour, lw=2.2, alpha=0.38, solid_capstyle="round", zorder=1)
        ax.plot([lo, hi], [yy, yy], color=colour, lw=0.75, solid_capstyle="round", zorder=2)
        ax.scatter(value, yy, s=28, color=colour, edgecolor="white", linewidth=0.55, zorder=3)
        ax.text(hi + 0.007, yy, f"{value:+.3f}", ha="left", va="center", fontsize=5.7, color=colour)
    ax.axvline(0, color="#7E8B91", lw=0.65, zorder=0)
    ax.set_yticks(y, [COMPONENT_LABELS[m] for m in order])
    ax.set_xlim(-0.07, 0.205)
    ax.set_xticks([-0.05, 0.00, 0.05, 0.10, 0.15])
    ax.set_xlabel("Change in score, 2024 minus 2011")
    ax.tick_params(axis="y", length=0, pad=2)
    clean_axis(ax)
    panel_label(ax, letter, x=-0.16, y=1.07)
    ax.text(0, 1.07, "Component-specific change", transform=ax.transAxes, ha="left", va="top", fontsize=7.8, fontweight="bold")
    ax.text(0.99, 0.02, "95% DCCA-block CI", transform=ax.transAxes, ha="right", va="bottom", fontsize=5.4, color=COLORS["muted"])


def coverage_panel(ax: plt.Axes, coverage: pd.DataFrame, letter: str = "g") -> None:
    strict = coverage.loc[coverage["metric"].eq(FIELD)].set_index("year").loc[YEARS]
    high = coverage.loc[coverage["metric"].eq("equal_six_highcoverage")].set_index("year").loc[YEARS]
    x = np.asarray(YEARS)
    y_strict = 100 * strict["population_coverage_share"].to_numpy(float)
    y_high = 100 * high["population_coverage_share"].to_numpy(float)
    ax.fill_between(x, y_high, y_strict, color="#E7DDF1", alpha=0.65, linewidth=0, label="Completeness gap")
    ax.plot(x, y_strict, color="#173F5F", lw=1.45, marker="o", ms=3.8, label="Six-component coverage")
    ax.plot(x, y_high, color="#7566A8", lw=1.45, marker="s", ms=3.5, label="High-coverage subset")
    ax.axhline(50, color="#B7C0C4", lw=0.55, ls=(0, (3, 2)))
    ax.text(2024.25, y_strict[-1], f"{y_strict[-1]:.1f}%", ha="left", va="center", fontsize=5.8, color="#173F5F")
    ax.text(2024.25, y_high[-1], f"{y_high[-1]:.1f}%", ha="left", va="center", fontsize=5.8, color="#7566A8")
    ax.set_xticks(YEARS)
    ax.set_xlim(2010.2, 2026.4)
    ax.set_ylim(0, 78)
    ax.set_yticks([0, 25, 50, 75])
    ax.set_ylabel("Population covered (%)")
    ax.set_xlabel("Year")
    clean_axis(ax)
    panel_label(ax, letter, x=-0.13, y=1.07)
    ax.text(0, 1.07, "Coverage audit", transform=ax.transAxes, ha="left", va="top", fontsize=7.8, fontweight="bold")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.02), handlelength=1.5, fontsize=5.6)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "release" / "v10_reference_driven")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure()

    frames = load_maps()
    breaks = pooled_quintiles(frames)
    annual = pd.read_csv(ROOT / "source_data" / "figS_temporal_uncertainty_v4" / "annual_block_bootstrap_intervals.csv")
    changes = pd.read_csv(ROOT / "source_data" / "figS_temporal_uncertainty_v4" / "change_from_2011_intervals.csv")
    coverage = pd.read_csv(ROOT / "source_data" / "figS_temporal_uncertainty_v4" / "metric_coverage_audit.csv")
    all_bounds = np.array([frames[y].total_bounds for y in YEARS])
    common_bounds = np.array([all_bounds[:, 0].min(), all_bounds[:, 1].min(), all_bounds[:, 2].max(), all_bounds[:, 3].max()])

    fig = plt.figure(figsize=(183 * MM, 190 * MM), facecolor="white")
    outer = fig.add_gridspec(
        3,
        1,
        height_ratios=[1.72, 0.10, 0.72],
        left=0.055,
        right=0.952,
        top=0.982,
        bottom=0.075,
        hspace=0.045,
    )

    maps = outer[0].subgridspec(2, 2, wspace=0.10, hspace=0.095)
    for idx, (year, letter) in enumerate(zip(YEARS, "abcd")):
        row, col = divmod(idx, 2)
        cell = maps[row, col].subgridspec(1, 2, width_ratios=[1.0, 0.046], wspace=0.035)
        ax = fig.add_subplot(cell[0, 0])
        cax = fig.add_subplot(cell[0, 1])
        draw_map(ax, cax, frames[year], year, letter, breaks, common_bounds)

    key_ax = fig.add_subplot(outer[1])
    key_ax.set_axis_off()
    key_ax.legend(
        handles=[
            Patch(facecolor="#F3F5F6", edgecolor="#C2CACE", lw=0.45, label="Incomplete six-component data"),
            Patch(facecolor="#FFFFFF", edgecolor="#D2D8DB", lw=0.45, label="No platform-listed outlet"),
        ],
        loc="center",
        ncol=2,
        frameon=False,
        fontsize=5.5,
        handlelength=1.2,
        columnspacing=1.0,
    )

    bottom = outer[2].subgridspec(1, 3, width_ratios=[0.95, 1.25, 0.95], wspace=0.42)
    trajectory_panel(fig.add_subplot(bottom[0, 0]), annual)
    component_panel(fig.add_subplot(bottom[0, 1]), changes)
    coverage_panel(fig.add_subplot(bottom[0, 2]), coverage)

    pd.DataFrame({"pooled_break": breaks}).to_csv(args.output_dir / "Fig2_v10_map_breaks.csv", index=False)
    stem = "Fig2_FourYear_Strict_SDI_v10_REFERENCE_DRIVEN"
    save_bundle(fig, args.output_dir, stem, preview_dpi=320)
    plt.close(fig)
    print(args.output_dir / f"{stem}.png")


if __name__ == "__main__":
    main()
