"""Render a PNG-only Nature-style preview for the R2.2 supply benchmark.

Figure contract
---------------
Core conclusion: The main fine-scale spatial pattern of restaurant-count
accessibility is robust to replacing OpenRice with the official FEHD licensed
supply frame, while the benchmark does not validate SDI quality attributes.
Archetype: asymmetric mixed-modality quantitative figure.
Panels: (a,b) matched-scale 15-min maps; (c) LSBG agreement at 10/15 min;
(d) compact robustness metrics.
Export: 183 x 150 mm, with vector PDF/SVG and high-resolution PNG.
Reviewer risk: FEHD coordinate missingness is district-calibrated sensitivity;
informal dining and quality-selection bias remain outside this figure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
import geopandas as gpd
import numpy as np
import pandas as pd

from v5_cities_visual_system import (
    FONT_FAMILY,
    outer_boundary,
    segmented_scale_bar,
)


# Mandatory editable-text rules for the later SVG/PDF approval export.
plt.rcParams["font.family"] = FONT_FAMILY
plt.rcParams["font.sans-serif"] = [FONT_FAMILY, "Helvetica", "Nimbus Sans", "DejaVu Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams.update(
    {
        "font.size": 7.2,
        "axes.titlesize": 7.7,
        "axes.labelsize": 7.2,
        "axes.linewidth": 0.65,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "legend.fontsize": 6.8,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


WORKSPACE = Path(__file__).resolve().parents[1]
SOURCE_BUNDLE = WORKSPACE / "source_data" / "figS_openrice_fehd_v4"
FIGURE_DIR = WORKSPACE / "figures"
RAW_LSBG = SOURCE_BUNDLE / "lsbg_2021_simplified.gpkg"
PALETTE_DEFINITIONS = {
    "nature_blue": ["#E8F1F7", "#D2E4EE", "#ACD0E2", "#75B0CF", "#3787B3", "#155C8A"],
    "teal_violet": ["#EAF6F4", "#BCE4DF", "#78CBC8", "#3E9DB8", "#3F6FAC", "#5A43A6"],
    "cool": [mpl.colors.to_hex(color) for color in mpl.colormaps["cool"](np.linspace(0.06, 0.94, 6))],
    "rd_bu": [mpl.colors.to_hex(color) for color in mpl.colormaps["RdBu_r"](np.linspace(0.08, 0.92, 6))],
    "cool_soft": ["#E7F7F3", "#B7E7DF", "#6BCBC4", "#5E8CC5", "#9457B8", "#D83C91"],
    "rd_bu_soft": ["#EAF3F8", "#B9D8E9", "#79ACCC", "#F4C4B5", "#E47A63", "#B51F35"],
}
SOURCE_MAP_PALETTES = {
    "OpenRice": ["#FDE7EA", "#F7BFC8", "#ED91A0", "#DC6178", "#C23D56", "#92223B"],
    "FEHD": ["#E5F2FC", "#BAD8F3", "#86BDE6", "#529ED3", "#2B78B5", "#17568D"],
}
TEAL = "#347FC0"
VIOLET = "#C94157"
BLUE = "#347FC0"
CORAL = "#C94157"
INK = "#24323A"
MUTED = "#6F7C82"
GRID = "#DDE4E6"
MISSING = "#F2F3F1"


def aligned_panel_heading(ax, label: str, title: str) -> None:
    """Place every panel letter and title on one shared local baseline."""
    heading_y = 1.035
    ax.text(
        -0.055,
        heading_y,
        label,
        transform=ax.transAxes,
        ha="right",
        va="baseline",
        fontsize=9.0,
        fontweight="bold",
        color="#111111",
        clip_on=False,
    )
    ax.text(
        0.0,
        heading_y,
        title,
        transform=ax.transAxes,
        ha="left",
        va="baseline",
        fontsize=7.7,
        fontweight="bold",
        color="#111111",
        clip_on=False,
    )


def draw_scale_bar(ax, bounds: np.ndarray) -> None:
    """Draw a two-segment scale with raised ticks outside mapped data."""
    minx, miny, maxx, maxy = bounds
    width = maxx - minx
    length_m = 10_000.0
    frac = length_m / width
    x0, y0 = 0.065, 0.28
    x1 = x0 + frac / 2
    x2 = x0 + frac
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.plot(
        [x0, x1],
        [y0, y0],
        transform=ax.transAxes,
        color=INK,
        lw=1.55,
        solid_capstyle="butt",
    )
    ax.plot(
        [x1, x2],
        [y0, y0],
        transform=ax.transAxes,
        color=INK,
        lw=1.55,
        solid_capstyle="butt",
    )
    for xx in (x0, x1, x2):
        ax.plot(
            [xx, xx],
            [y0, y0 + 0.22],
            transform=ax.transAxes,
            color=INK,
            lw=0.60,
        )
    for xx, label in ((x0, "0"), (x1, "5"), (x2, "10 km")):
        ax.text(
            xx,
            y0 + 0.28,
            label,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=5.8,
            color=INK,
        )


def draw_north_arrow(ax) -> None:
    """Black/white split triangular north arrow following the author reference."""
    # Left placement separates the arrow from the right-side colour scale.
    nx, ny = 0.080, 0.887
    top = (nx, ny + 0.042)
    joint = (nx, ny - 0.017)
    left = (nx - 0.022, ny - 0.054)
    right = (nx + 0.022, ny - 0.054)
    left_half = Polygon(
        [top, left, joint],
        closed=True,
        transform=ax.transAxes,
        facecolor="white",
        edgecolor=INK,
        linewidth=0.50,
        zorder=9,
        clip_on=False,
    )
    right_half = Polygon(
        [top, right, joint],
        closed=True,
        transform=ax.transAxes,
        facecolor=INK,
        edgecolor=INK,
        linewidth=0.50,
        zorder=9,
        clip_on=False,
    )
    ax.add_patch(left_half)
    ax.add_patch(right_half)
    ax.text(
        nx,
        ny + 0.055,
        "N",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=6.2,
        fontweight="normal",
        color=INK,
    )


def draw_map(
    ax,
    geo: gpd.GeoDataFrame,
    column: str,
    title: str,
    norm: BoundaryNorm,
    cmap: ListedColormap,
) -> None:
    observed = geo[column].notna()
    geo.loc[~observed].plot(
        ax=ax,
        facecolor=MISSING,
        edgecolor="#8F9694",
        linewidth=0.35,
        hatch="////",
        alpha=0.34,
        zorder=1,
    )
    geo.loc[observed].plot(
        ax=ax,
        column=column,
        cmap=cmap,
        norm=norm,
        edgecolor="#FFFFFF",
        linewidth=0.08,
        alpha=0.91,
        zorder=2,
    )
    geo.boundary.plot(ax=ax, color="#737A78", linewidth=0.12, alpha=0.70, zorder=3)
    bounds = geo.total_bounds
    padx = 0.018 * (bounds[2] - bounds[0])
    pady = 0.018 * (bounds[3] - bounds[1])
    ax.set_xlim(bounds[0] - padx, bounds[2] + padx)
    ax.set_ylim(bounds[1] - .082 * (bounds[3] - bounds[1]), bounds[3] + pady)
    ax.set_aspect("equal")
    ax.axis("off")
    outer_boundary(ax, geo, linewidth=.28)
    draw_north_arrow(ax)
    segmented_scale_bar(ax, length_km=10, x=.060, y=.012)


def draw_scatter(ax, data: pd.DataFrame, summary: dict) -> None:
    max_value = 0.0
    for threshold, color, label in ((10, TEAL, "10 min"), (15, VIOLET, "15 min")):
        group = data.loc[data["threshold_min"].eq(float(threshold))]
        x = group["openrice_per_1000_source"]
        y = group["fehd_calibrated_per_1000_source"]
        max_value = max(max_value, float(x.max()), float(y.max()))
        ax.scatter(
            x,
            y,
            s=3.4,
            color=color,
            alpha=1.0,
            linewidth=0,
            rasterized=True,
            label=label,
        )
    limit = np.ceil(max_value / 10) * 10
    ax.plot([0, limit], [0, limit], ls="--", color="#8A9498", lw=0.8, zorder=0)
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    # Use the same rectangular plotting area as d/e. Identical numeric ranges
    # and the identity reference remain explicit; no observations are changed.
    ax.set_aspect("auto")
    ax.set_xlabel("OpenRice opportunity\n(per 1,000 outlets)")
    ax.set_ylabel("FEHD opportunity\n(per 1,000 outlets)")
    ax.legend(loc="upper left", handletextpad=0.4, borderaxespad=0.2)


def draw_robustness(ax, summary: dict) -> None:
    rows = [
        ("Pearson $r$", "pearson"),
        ("Spearman $\\rho$", "spearman"),
        ("Population\nagreement", "population_weighted_flag_agreement"),
        ("Low-access\nset overlap", "jaccard"),
    ]
    y = np.arange(len(rows))[::-1]
    for threshold, color, offset in ((10, TEAL, 0.10), (15, VIOLET, -0.10)):
        values = []
        for _, key in rows:
            if key in {"pearson", "spearman"}:
                value = summary[str(threshold)]["correlation"]["fehd_calibrated"][key]
            else:
                value = summary[str(threshold)]["low_access_comparison"][
                    "fehd_calibrated"
                ][key]
            values.append(float(value))
        yy = y + offset
        ax.scatter(values, yy, s=24, color=color, edgecolor="white", linewidth=0.55, zorder=3)
        for value, y_value in zip(values, yy):
            ax.text(
                min(value + 0.006, 1.005),
                y_value,
                f"{value:.3f}",
                ha="left",
                va="center",
                fontsize=6.2,
                color=INK,
            )
    ax.set_yticks(y)
    ax.set_yticklabels([label for label, _ in rows])
    for label in ax.get_yticklabels():
        label.set_fontsize(6.4)
    ax.set_xlim(0.78, 1.025)
    ax.set_xticks([0.8, 0.9, 1.0])
    ax.set_xlabel("Agreement")
    ax.tick_params(axis="y", length=0)
    handles = [
        Line2D([0], [0], marker="o", lw=0, color=TEAL, label="10 min", markersize=4.5),
        Line2D([0], [0], marker="o", lw=0, color=VIOLET, label="15 min", markersize=4.5),
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.89),
        ncol=2,
        handletextpad=0.3,
        columnspacing=0.8,
    )


def draw_bland_altman(ax, data: pd.DataFrame) -> None:
    """Audit proportional agreement without treating correlation as identity."""
    bias_handles = []
    for threshold, color, marker, label in (
        (10, TEAL, "o", "10 min"),
        (15, VIOLET, "s", "15 min"),
    ):
        group = data.loc[data["threshold_min"].eq(float(threshold))]
        left = group["openrice_per_1000_source"].to_numpy(float)
        right = group["fehd_calibrated_per_1000_source"].to_numpy(float)
        mean = (left + right) / 2
        difference = right - left
        bias = float(np.nanmean(difference))
        sd = float(np.nanstd(difference, ddof=1))
        ax.scatter(mean, difference, s=2.8, color=color, alpha=1.0,
                   linewidth=0, marker=marker, rasterized=True)
        ax.axhline(bias, color=color, lw=1.15, zorder=3)
        ax.axhline(bias - 1.96 * sd, color=color, lw=.65,
                   linestyle=(0, (3, 2)), alpha=.72)
        ax.axhline(bias + 1.96 * sd, color=color, lw=.65,
                   linestyle=(0, (3, 2)), alpha=.72)
        bias_handles.append(
            Line2D([0], [0], color=color, lw=1.15,
                   label=label)
        )
    ax.axhline(0, color="#7E8A90", lw=.55, zorder=0)
    ax.set_xlabel("Mean normalized accessibility")
    ax.set_ylabel("FEHD minus OpenRice")
    ax.legend(handles=bias_handles, loc="upper left", frameon=False,
              handlelength=1.5, handletextpad=.45, borderaxespad=.2,
              fontsize=5.3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison",
        type=Path,
        default=SOURCE_BUNDLE / "lsbg_openrice_fehd_count_access.csv",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=SOURCE_BUNDLE / "comparison_summary.json",
    )
    parser.add_argument("--lsbg", type=Path, default=RAW_LSBG)
    parser.add_argument(
        "--palette",
        choices=tuple(PALETTE_DEFINITIONS),
        default="nature_blue",
        help="Map palette; non-default palettes are written as separate comparison previews.",
    )
    parser.add_argument(
        "--output-base",
        type=Path,
        default=FIGURE_DIR
        / "FigS_OpenRice_FEHD_Supply_Benchmark_2021_v13_DISTINCT_SOURCES",
    )
    args = parser.parse_args()
    data = pd.read_csv(args.comparison, dtype={"lsbg_id": str})
    summary = json.loads(args.summary.read_text(encoding="utf-8"))["summary"]
    geo = gpd.read_file(args.lsbg)[["lsbg", "geometry"]].copy()
    geo["lsbg_id"] = geo["lsbg"].astype(str).str.replace(r"\.0$", "", regex=True)
    map_data = data.loc[data["threshold_min"].eq(15.0)].copy()
    geo = geo.merge(map_data, on="lsbg_id", how="left", validate="one_to_one")

    map_columns = ["openrice_per_1000_source", "fehd_calibrated_per_1000_source"]
    pooled = np.concatenate(
        [geo[column].dropna().to_numpy(dtype=float) for column in map_columns]
    )
    # One set of pooled quantile boundaries is shared by both maps. This gives
    # each colour class similar visual occupancy without letting the maps use
    # different class definitions (which would invalidate direct comparison).
    access_colors = PALETTE_DEFINITIONS[args.palette]
    quantile_probs = np.linspace(0.0, 1.0, len(access_colors) + 1)
    boundaries = np.quantile(pooled, quantile_probs)
    boundaries[0] = min(0.0, float(boundaries[0]))
    boundaries[-1] = float(np.nextafter(boundaries[-1], np.inf))
    if np.any(np.diff(boundaries) <= 0):
        raise ValueError(f"Pooled quantile boundaries are not strictly increasing: {boundaries}")
    source_cmaps = {
        key: ListedColormap(colors, name=f"{key.lower()}_quantiles")
        for key, colors in SOURCE_MAP_PALETTES.items()
    }
    norm = BoundaryNorm(boundaries, ncolors=len(access_colors), clip=True)

    mm = 1 / 25.4
    fig = plt.figure(figsize=(183 * mm, 119 * mm), constrained_layout=False)
    # Explicit, compact rows avoid empty GridSpec space after equal-aspect
    # geography. All three diagnostic panels have identical physical sizes.
    ax_a = fig.add_axes([.055, .475, .377, .50])
    cax_a = fig.add_axes([.439, .475, .013, .50])
    ax_b = fig.add_axes([.532, .475, .377, .50])
    cax_b = fig.add_axes([.916, .475, .013, .50])
    lower_y, lower_h, lower_w = .125, .285, .225
    ax_c = fig.add_axes([.092, lower_y, lower_w, lower_h])
    ax_d = fig.add_axes([.402, lower_y, lower_w, lower_h])
    ax_e = fig.add_axes([.758, lower_y, lower_w, lower_h])

    draw_map(
        ax_a,
        geo,
        map_columns[0],
        "OpenRice, 15-min count accessibility",
        norm,
        source_cmaps["OpenRice"],
    )
    draw_map(
        ax_b,
        geo,
        map_columns[1],
        "Official FEHD benchmark, district calibrated",
        norm,
        source_cmaps["FEHD"],
    )
    fig.canvas.draw()
    for cax, source, map_ax in ((cax_a, "OpenRice", ax_a), (cax_b, "FEHD", ax_b)):
        # Hong Kong is much wider than it is tall.  Equal-aspect map drawing
        # therefore leaves internal vertical whitespace inside the map axes;
        # a full-height colourbar looks substantially taller than the mapped
        # boundary.  Match the bar to the visible boundary height instead.
        cbox = cax.get_position()
        ymin, ymax = geo.total_bounds[[1, 3]]
        corners = fig.transFigure.inverted().transform(
            map_ax.transData.transform([[geo.total_bounds[0], ymin],
                                        [geo.total_bounds[0], ymax]]))
        cax.set_position([cbox.x0, corners[0, 1], cbox.width,
                          corners[1, 1] - corners[0, 1]])
        cb = fig.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=source_cmaps[source]),
            cax=cax,
            orientation="vertical",
            boundaries=boundaries,
            ticks=boundaries,
            spacing="uniform",
        )
        cb.set_label("per 1,000 outlets", fontsize=5.5, labelpad=2.5)
        cb.ax.yaxis.set_label_position("right")
        cb.ax.yaxis.set_ticks_position("right")
        cb.outline.set_linewidth(0.45)
        cb.ax.tick_params(length=1.8, width=0.45, pad=1.2, labelsize=5.3)
        cb.ax.set_yticklabels(
            [f"{value:.1f}" if value < 20 else f"{value:.0f}" for value in boundaries]
        )

    draw_scatter(ax_c, data, summary)
    draw_bland_altman(ax_d, data)
    draw_robustness(ax_e, summary)
    headings = (
        (ax_a, "a", "OpenRice supply"),
        (ax_b, "b", "Official FEHD benchmark"),
        (ax_c, "c", "LSBG agreement"),
        (ax_d, "d", "Bias and agreement limits"),
        (ax_e, "e", "Availability diagnostics"),
    )
    for ax, label, title in headings:
        aligned_panel_heading(ax, label, title)

    args.output_base.parent.mkdir(parents=True, exist_ok=True)
    for label in fig.findobj(mpl.text.Text):
        label.set_color("#151515")
    variant_suffix = "" if args.palette == "nature_blue" else f"_{args.palette}"
    base = args.output_base.with_name(args.output_base.name + variant_suffix)
    preview = base.with_name(base.name + "_PREVIEW").with_suffix(".png")
    fig.savefig(base.with_suffix(".svg"), facecolor="white")
    fig.savefig(base.with_suffix(".pdf"), facecolor="white")
    fig.savefig(base.with_name(base.name + "_600dpi").with_suffix(".png"), dpi=600, facecolor="white")
    fig.savefig(preview, dpi=180, facecolor="white")
    contract = {
        "status": "FINAL_EXPORT_CANDIDATE",
        "core_conclusion": (
            "Fine-scale restaurant-count availability is robust to the official FEHD "
            "licensed-supply comparator; quality-selection bias remains unresolved."
        ),
        "archetype": "asymmetric mixed-modality quantitative figure",
        "backend": "Python",
        "final_size_mm": [183, 119],
        "map_classification": "shared pooled quantiles",
        "palette": "source-distinct red/blue sequential families",
        "palette_colors": SOURCE_MAP_PALETTES,
        "shared_map_boundaries": [float(value) for value in boundaries],
        "figure_title_inside_canvas": False,
        "caption_location": "Word caption below figure",
        "cartographic_elements": {
            "north_arrow": "black-white split triangle",
            "scale_bar": "two segments with raised 0/5/10-km ticks outside mapped data",
        },
        "preview": str(preview),
        "source_data": "source_data/figS_openrice_fehd_v4",
        "final_exports_generated": True,
        "public_redistribution_status": "PENDING_DISCLOSURE_REVIEW",
    }
    contract_name = (
        "FigS_OpenRice_FEHD_supply_benchmark_contract.json"
        if not variant_suffix
        else f"FigS_OpenRice_FEHD_supply_benchmark{variant_suffix}_contract.json"
    )
    (args.output_base.parent / contract_name).write_text(
        json.dumps(contract, indent=2), encoding="utf-8"
    )
    plt.close(fig)
    print(json.dumps(contract, indent=2))


if __name__ == "__main__":
    main()
