from __future__ import annotations

import os
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
from matplotlib.patches import Polygon, Rectangle

from v5_nature_theme import (
    COMPONENT_COLORS,
    PALETTE,
    configure as configure_nature_theme,
    sequential_component,
    sequential_sdi,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DATA_DIR = REPO_ROOT / "source_data" / "fig_v4_four_year"
DATA_DIR = (
    PUBLIC_DATA_DIR
    if (PUBLIC_DATA_DIR / "2011_lsbg_components.gpkg").exists()
    else REPO_ROOT / "outputs" / "restricted" / "v4_four_year_dual_scale_components"
)
OUT_DIR = Path(
    os.environ.get(
        "CITIES_FIGURE_DIR",
        str(REPO_ROOT / "release" / "v4_figures"),
    )
)
SOURCE_DIR = REPO_ROOT / "source_data" / "fig_v4_four_year"
LAND_MASK_FILE = REPO_ROOT / "source_data" / "figs02" / "panel_a_land_mask.geojson"

YEARS = (2011, 2016, 2021, 2024)
MM = 1 / 25.4

COLORS = {
    "SDI": PALETTE["navy"],
    "Nutrition": COMPONENT_COLORS["Nutrition"],
    "Carbon": COMPONENT_COLORS["Carbon"],
    "Diversity": COMPONENT_COLORS["Cuisine diversity"],
    "Sustainability": COMPONENT_COLORS["Sustainability"],
    "Hygiene": COMPONENT_COLORS["Hygiene"],
    "Practice": COMPONENT_COLORS["Practice"],
    "ink": PALETTE["ink"],
    "muted": PALETTE["muted"],
    "grid": PALETTE["grid"],
    "no_restaurant": PALETTE["context"],
    "missing": PALETTE["missing"],
}

COMPONENTS = [
    ("nutrition_score", "Nutrition", COLORS["Nutrition"]),
    ("carbon_score", "Carbon", COLORS["Carbon"]),
    ("diversity_score", "Cuisine diversity", COLORS["Diversity"]),
    ("sustainability_score", "Sustainability", COLORS["Sustainability"]),
    ("hygiene_score", "Hygiene", COLORS["Hygiene"]),
    ("practice_score", "Practice", COLORS["Practice"]),
]


def configure_matplotlib() -> None:
    configure_nature_theme()


def load_frames(scale: str) -> dict[int, gpd.GeoDataFrame]:
    frames = {}
    for year in YEARS:
        frame = gpd.read_file(DATA_DIR / f"{year}_{scale.lower()}_components.gpkg")
        if frame.crs is None:
            raise ValueError(f"Missing CRS for {year} {scale}")
        frame = frame.to_crs(2326)
        # Plot-only simplification; archived source geometry remains unchanged.
        frame["geometry"] = frame.geometry.simplify(18.0, preserve_topology=True)
        frames[year] = frame
    return frames


def sequential_map(color: str, name: str) -> LinearSegmentedColormap:
    return sequential_component(color, name)


def pooled_quantile_breaks(
    frames: dict[int, gpd.GeoDataFrame],
    field: str,
    quantiles: int = 5,
    positive_only: bool = False,
) -> np.ndarray:
    values = pd.concat(
        [
            pd.to_numeric(frame.loc[frame["restaurant_n"].fillna(0).gt(0), field], errors="coerce")
            for frame in frames.values()
        ],
        ignore_index=True,
    ).dropna()
    if positive_only:
        values = values.loc[values.gt(0)]
    breaks = np.unique(values.quantile(np.linspace(0, 1, quantiles + 1)).to_numpy(dtype=float))
    if len(breaks) < 3:
        low, high = values.min(), values.max()
        breaks = np.linspace(low, high if high > low else low + 1, quantiles + 1)
    breaks[0] = np.nextafter(breaks[0], -np.inf)
    breaks[-1] = np.nextafter(breaks[-1], np.inf)
    return breaks


def format_break(value: float) -> str:
    """Format display breaks without exposing a floating-point negative zero."""
    value = 0.0 if abs(float(value)) < 5e-4 else float(value)
    return f"{value:.2f}"


def add_panel_label(ax: plt.Axes, label: str, x: float = -0.05, y: float = 1.03) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="bottom",
        ha="left",
        color="#111111",
        clip_on=False,
    )


def add_north_arrow(ax: plt.Axes, x: float = 0.09, y: float = 0.92, height: float = 0.065) -> None:
    width = height * 0.44
    left = np.array([[x, y], [x - width / 2, y - height], [x, y - height * 0.72]])
    right = np.array([[x, y], [x + width / 2, y - height], [x, y - height * 0.72]])
    ax.add_patch(
        Polygon(left, closed=True, transform=ax.transAxes, facecolor="white", edgecolor=COLORS["ink"], lw=0.65, zorder=10)
    )
    ax.add_patch(
        Polygon(right, closed=True, transform=ax.transAxes, facecolor=COLORS["ink"], edgecolor=COLORS["ink"], lw=0.65, zorder=10)
    )
    ax.text(x, y + 0.018, "N", transform=ax.transAxes, ha="center", va="bottom", fontsize=7.5, fontweight="bold", color=COLORS["ink"])


def add_scale_bar(ax: plt.Axes, length_km: int = 10, segments: int = 2) -> None:
    from v5_cities_visual_system import segmented_scale_bar
    segmented_scale_bar(ax, length_km=length_km, x=.07, y=.014)
    return
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    length = length_km * 1000.0
    segment = length / segments
    x0 = xmin + (xmax - xmin) * 0.07
    # Keep every label inside the deliberately reserved strip below the
    # southernmost mapped geometry. Previously the "5" label occasionally
    # landed on a small island, which made the scale look like map data.
    y0 = ymin + (ymax - ymin) * 0.014
    height = (ymax - ymin) * 0.012
    for i in range(segments):
        ax.add_patch(
            Rectangle(
                (x0 + i * segment, y0),
                segment,
                height,
                facecolor=COLORS["ink"] if i % 2 == 0 else "white",
                edgecolor=COLORS["ink"],
                lw=0.6,
                zorder=12,
            )
        )
    for i in range(segments + 1):
        x = x0 + i * segment
        ax.plot([x, x], [y0, y0 + height * 2.0], color=COLORS["ink"], lw=0.65, zorder=13)
        if i in (0, segments):
            ax.text(
                x,
                y0 + height * 2.25,
                str(int(i * length_km / segments)),
                ha="center",
                va="bottom",
                fontsize=5.5,
                color=COLORS["ink"],
            )
    ax.text(
        x0 + length + segment * 0.17,
        y0 + height * 0.35,
        "km",
        ha="left",
        va="center",
        fontsize=5.5,
        color=COLORS["ink"],
        clip_on=False,
    )


def plot_choropleth(
    ax: plt.Axes,
    frame: gpd.GeoDataFrame,
    field: str,
    cmap: mpl.colors.Colormap,
    norm: mpl.colors.Normalize,
    title: str | None = None,
    show_cartography: bool = False,
    positive_only: bool = False,
) -> None:
    def draw(layer: gpd.GeoDataFrame, **kwargs) -> None:
        before = len(ax.collections)
        layer.plot(ax=ax, **kwargs)
        # Mixed vector/raster export: map paths are rasterised at export DPI;
        # labels, axes and statistics remain editable vectors.
        for collection in ax.collections[before:]:
            collection.set_rasterized(True)

    draw(frame, color=COLORS["missing"], edgecolor=PALETTE["boundary"], linewidth=0.14, alpha=0.34)
    no_restaurant = frame["restaurant_n"].fillna(0).eq(0)
    draw(
        frame.loc[no_restaurant],
        color=COLORS["no_restaurant"],
        edgecolor="#C8D1D4",
        linewidth=0.12,
        alpha=0.40,
    )
    numeric = pd.to_numeric(frame[field], errors="coerce")
    observed = numeric.notna() & ~no_restaurant
    if positive_only:
        observed &= numeric.gt(0)
    draw(
        frame.loc[observed],
        column=field,
        cmap=cmap,
        norm=norm,
        edgecolor="#F7FAFA",
        linewidth=0.075,
        alpha=0.91,
    )
    try:
        frame.dissolve().boundary.plot(ax=ax, color="#46525A", linewidth=0.18, zorder=18)
    except Exception:
        frame.boundary.plot(ax=ax, color="#46525A", linewidth=0.10, zorder=18)
    bounds = frame.total_bounds
    xpad = (bounds[2] - bounds[0]) * 0.025
    # Reserve a narrow strip under the map for the scale bar.
    ypad_top = (bounds[3] - bounds[1]) * 0.025
    # Keep cartographic furniture outside the geography without shrinking the
    # already compact four-year map panels more than necessary.
    ypad_bottom = (bounds[3] - bounds[1]) * 0.085
    ax.set_xlim(bounds[0] - xpad, bounds[2] + xpad)
    ax.set_ylim(bounds[1] - ypad_bottom, bounds[3] + ypad_top)
    ax.set_aspect("equal")
    ax.set_axis_off()
    if title:
        ax.set_title(title, loc="center", pad=2.5, fontweight="bold", color=COLORS["ink"])
    if show_cartography:
        add_north_arrow(ax, x=0.09, y=0.940, height=0.075)
        add_scale_bar(ax)


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for label in fig.findobj(mpl.text.Text):
        label.set_color("#151515")
    fig.savefig(OUT_DIR / f"{stem}.svg", dpi=600, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{stem}.pdf", dpi=600, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{stem}_PREVIEW.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{stem}_600dpi.png", dpi=600, bbox_inches="tight")


def plot_main_figure(lsbg: dict[int, gpd.GeoDataFrame], summary: pd.DataFrame) -> None:
    field = "sdi_equal_arithmetic"
    breaks = pooled_quantile_breaks(lsbg, field, 5, positive_only=True)
    cmap = sequential_sdi("sdi_main")
    norm = BoundaryNorm(breaks, cmap.N)

    fig = plt.figure(figsize=(183 * MM, 128 * MM), constrained_layout=False)
    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=[1.10, 0.90],
        left=0.035,
        right=0.985,
        top=0.965,
        bottom=0.105,
        hspace=0.20,
    )
    maps = outer[0].subgridspec(1, 4, wspace=0.055)
    letters = "abcd"
    for index, year in enumerate(YEARS):
        ax = fig.add_subplot(maps[0, index])
        plot_choropleth(
            ax,
            lsbg[year],
            field,
            cmap,
            norm,
            title=str(year),
            show_cartography=(year == 2024),
            positive_only=True,
        )
        add_panel_label(ax, letters[index], x=-0.04)

    cax = fig.add_axes([0.295, 0.552, 0.41, 0.007])
    cb = mpl.colorbar.ColorbarBase(
        cax,
        cmap=cmap,
        norm=norm,
        boundaries=breaks,
        ticks=breaks,
        orientation="horizontal",
        spacing="proportional",
        drawedges=False,
    )
    cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(length=1.8, pad=1.0, labelsize=5.1)
    cb.set_label("SDI · pooled positive-value LSBG quintiles", labelpad=2.0)
    cb.ax.set_xticklabels([format_break(value) for value in breaks])

    bottom = outer[1].subgridspec(1, 3, width_ratios=[1.45, 0.92, 0.83], wspace=0.52)

    # Component evolution.
    ax_e = fig.add_subplot(bottom[0, 0])
    ls = summary.loc[summary["scale"].eq("LSBG")].set_index("year").loc[list(YEARS)]
    for field_name, label, color in COMPONENTS:
        values = ls[f"population_weighted_{field_name}"].to_numpy(dtype=float)
        ax_e.plot(YEARS, values, marker="o", ms=3.1, color=color, lw=1.15)
        ax_e.text(2024.35, values[-1], label, color=color, fontsize=5.8, va="center", clip_on=False)
    ax_e.set_xlim(2010.3, 2027.0)
    ax_e.set_xticks(YEARS)
    ax_e.set_ylim(0.05, 0.66)
    ax_e.set_ylabel("Population-weighted component score")
    ax_e.set_title("Component trajectories", loc="left", pad=3, fontweight="bold")
    ax_e.grid(axis="y", color=COLORS["grid"], lw=0.5)
    ax_e.spines[["top", "right"]].set_visible(False)
    add_panel_label(ax_e, "e", x=-0.13, y=1.02)

    # Supply and missing-local-opportunity trajectory.
    ax_f = fig.add_subplot(bottom[0, 1])
    supply = ls["population_weighted_supply_per_1000"].to_numpy(dtype=float)
    no_rest = 100 * ls["population_without_restaurant_share"].to_numpy(dtype=float)
    ax_f.plot(YEARS, supply, marker="o", ms=3.5, color=COLORS["SDI"], lw=1.3)
    ax_f.set_ylabel("Restaurants per 1,000 residents", color=COLORS["SDI"])
    ax_f.tick_params(axis="y", colors=COLORS["SDI"])
    ax_f.set_xticks(YEARS)
    ax_f.set_xlim(2010.3, 2024.7)
    ax_f.set_ylim(1.3, 3.8)
    ax_f2 = ax_f.twinx()
    ax_f2.plot(YEARS, no_rest, marker="s", ms=3.2, color=PALETTE["orange"], lw=1.2)
    ax_f2.set_ylabel("Population without a local outlet (%)", color=PALETTE["orange"], labelpad=2)
    ax_f2.tick_params(axis="y", colors=PALETTE["orange"])
    ax_f2.set_ylim(15, 23)
    ax_f.grid(axis="x", color=COLORS["grid"], lw=0.45)
    ax_f.spines["top"].set_visible(False)
    ax_f2.spines["top"].set_visible(False)
    ax_f.set_title("Supply and local gaps", loc="left", pad=3, fontweight="bold")
    add_panel_label(ax_f, "f", x=-0.20, y=1.02)

    # Scale comparison with common LSBG calibration.
    ax_g = fig.add_subplot(bottom[0, 2])
    lvals = summary.loc[summary["scale"].eq("LSBG")].set_index("year")[
        "population_weighted_sdi_equal_arithmetic"
    ]
    dvals = summary.loc[summary["scale"].eq("DCCA")].set_index("year")[
        "population_weighted_sdi_equal_arithmetic"
    ]
    y_positions = np.arange(len(YEARS))[::-1]
    for y_pos, year in zip(y_positions, YEARS):
        ax_g.plot([lvals[year], dvals[year]], [y_pos, y_pos], color="#BCC5CA", lw=1.1, zorder=1)
        ax_g.scatter(lvals[year], y_pos, s=22, color=COLORS["SDI"], edgecolor="white", lw=0.4, zorder=2)
        ax_g.scatter(dvals[year], y_pos, s=22, color=COLORS["Sustainability"], edgecolor="white", lw=0.4, zorder=2)
    ax_g.set_yticks(y_positions, [str(year) for year in YEARS])
    ax_g.set_xlabel("Population-weighted SDI")
    ax_g.set_xlim(0.35, 0.49)
    ax_g.grid(axis="x", color=COLORS["grid"], lw=0.5)
    ax_g.spines[["top", "right", "left"]].set_visible(False)
    ax_g.tick_params(axis="y", length=0)
    ax_g.set_title("Scale sensitivity", loc="left", pad=3, fontweight="bold")
    ax_g.scatter([], [], color=COLORS["SDI"], label="LSBG", s=20)
    ax_g.scatter([], [], color=COLORS["Sustainability"], label="DCCA", s=20)
    ax_g.legend(
        frameon=False,
        loc="lower right",
        bbox_to_anchor=(1.0, 1.055),
        ncol=2,
        columnspacing=0.8,
        handletextpad=0.3,
        borderpad=0.1,
    )
    add_panel_label(ax_g, "g", x=-0.22, y=1.02)

    save_figure(fig, "Fig2_FourYear_SDI_LSBG_Cities_RR_v4_Nature")
    plt.close(fig)


def plot_component_atlas(lsbg: dict[int, gpd.GeoDataFrame]) -> None:
    fig, axes = plt.subplots(
        6,
        4,
        figsize=(183 * MM, 236 * MM),
        gridspec_kw={"hspace": 0.02, "wspace": 0.025, "left": 0.12, "right": 0.985, "top": 0.97, "bottom": 0.045},
    )
    letters = iter("abcdefghijklmnopqrstuvwxyz")
    for row, (field, label, color) in enumerate(COMPONENTS):
        breaks = pooled_quantile_breaks(lsbg, field, 5)
        cmap = sequential_map(color, f"{field}_map")
        norm = BoundaryNorm(breaks, cmap.N)
        for col, year in enumerate(YEARS):
            ax = axes[row, col]
            plot_choropleth(ax, lsbg[year], field, cmap, norm, show_cartography=False)
            if row == 0:
                ax.set_title(str(year), loc="center", pad=1.5, fontweight="bold")
            if col == 0:
                ax.text(-0.08, 0.50, label, transform=ax.transAxes, rotation=90, ha="right", va="center", fontweight="bold", color=color, fontsize=7.3)
            add_panel_label(ax, next(letters), x=0.01, y=0.91)
        # One compact, row-specific scale in the right margin.
        cax = fig.add_axes([0.989, 0.855 - row * 0.154, 0.009, 0.085])
        cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, boundaries=breaks, orientation="vertical")
        cb.outline.set_linewidth(0.35)
        cb.set_ticks([breaks[0], breaks[-1]])
        cb.set_ticklabels([f"{max(0.0, breaks[0]):.2f}", f"{breaks[-1]:.2f}"])
        cb.ax.tick_params(length=1.5, pad=1, labelsize=4.8)

    fig.text(0.12, 0.018, "Grey: no platform-listed restaurant; white: component unavailable", fontsize=5.8, color=COLORS["muted"], ha="left")
    save_figure(fig, "FigS_Component_Atlas_4Years_LSBG_Cities_RR_v4_Nature")
    plt.close(fig)


def plot_maup_figure(
    lsbg: dict[int, gpd.GeoDataFrame], dcca: dict[int, gpd.GeoDataFrame]
) -> None:
    # DCCA census polygons include marine allocation. Clip them to the
    # corresponding LSBG land mask for a land-only cartographic comparison.
    dcca_land: dict[int, gpd.GeoDataFrame] = {}
    official_land_mask = gpd.read_file(LAND_MASK_FILE).to_crs(2326).geometry.union_all()
    for year in YEARS:
        clipped = dcca[year].copy()
        clipped["geometry"] = clipped.geometry.intersection(official_land_mask)
        clipped = clipped.loc[~clipped.geometry.is_empty].copy()
        dcca_land[year] = clipped
    values = pd.concat(
        [frame["sdi_equal_arithmetic"] for frame in list(lsbg.values()) + list(dcca_land.values())],
        ignore_index=True,
    ).dropna()
    values = values.loc[values.gt(0)]
    breaks = np.unique(values.quantile(np.linspace(0, 1, 6)).to_numpy(dtype=float))
    breaks[0] = np.nextafter(breaks[0], -np.inf)
    breaks[-1] = np.nextafter(breaks[-1], np.inf)
    year_colours = {
        2011: "#C94157",
        2016: "#347FC0",
        2021: "#419B58",
        2024: "#E69736",
    }

    fig, axes = plt.subplots(
        4,
        2,
        figsize=(183 * MM, 209 * MM),
        gridspec_kw={"hspace": .12, "wspace": .035, "left": .045,
                     "right": .965, "top": .965, "bottom": .035},
    )
    for col, (scale, frames) in enumerate((("LSBG", lsbg), ("DCCA", dcca_land))):
        for row, year in enumerate(YEARS):
            ax = axes[row, col]
            cmap = sequential_map(year_colours[year], f"maup_{year}")
            norm = BoundaryNorm(breaks, cmap.N)
            plot_choropleth(
                ax,
                frames[year],
                "sdi_equal_arithmetic",
                cmap,
                norm,
                show_cartography=True,
                positive_only=True,
            )
            ax.set_title(f"{year}  ·  {scale}", loc="left", pad=2.3,
                         fontsize=7.4, fontweight="normal")
            cax = ax.inset_axes([1.006, .095, .032, .81], transform=ax.transAxes)
            cb = mpl.colorbar.ColorbarBase(
                cax,
                cmap=cmap,
                norm=norm,
                boundaries=breaks,
                ticks=breaks,
                orientation="vertical",
            )
            cb.outline.set_linewidth(0.42)
            cb.ax.yaxis.set_ticks_position("right")
            cb.ax.tick_params(length=1.5, pad=1.0, labelsize=5.05)
            cb.ax.set_yticklabels([format_break(value) for value in breaks])
            add_panel_label(ax, chr(ord("a") + row * 2 + col), x=-.055, y=1.00)
    save_figure(fig, "FigS_MAUP_LSBG_DCCA_4Years_Cities_RR_v5_RIGHT_CBAR")
    plt.close(fig)


def main() -> None:
    configure_matplotlib()
    lsbg = load_frames("LSBG")
    dcca = load_frames("DCCA")
    summary = pd.read_csv(DATA_DIR / "population_weighted_summary.csv")
    plot_main_figure(lsbg, summary)
    plot_component_atlas(lsbg)
    plot_maup_figure(lsbg, dcca)
    print(OUT_DIR)


if __name__ == "__main__":
    main()
