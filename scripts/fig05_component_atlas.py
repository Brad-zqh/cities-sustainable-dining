# Academic Figure Skill Asset Confirmation (verified against assets/figures/)
# (a-x) small-multiple choropleth atlas -> project cartographic production assets -> param inherit
# RULE: "native run" = load pre-rendered PNG via Image.open().ax.imshow().
#       "param inherit" = drawing function below that copies Class A/B/C values.
#       If a panel says "native run" and you write a drawing function, you broke the contract.

"""Four-year six-component atlas using only evidence-locked LSBG data.

No simulated values, imputation, downsampling or row selection is used.  Each
component uses pooled four-year breaks so maps remain comparable through time.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib as mpl

from v5_cities_visual_system import (
    FONT_FAMILY,
    north_arrow,
    outer_boundary,
    segmented_scale_bar,
)

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
    for label in fig.findobj(mpl.text.Text):
        label.set_color("#151515")
    fig.savefig(f"{filename}.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(f"{filename}.png", bbox_inches="tight", dpi=300)


import matplotlib

matplotlib.use("Agg")
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "source_data" / "fig_v4_four_year"
OUT = Path(os.environ.get("CITIES_FIGURE_DIR", str(ROOT / "figures")))
STEM = "FigS1_FourYear_Component_Atlas_v27_INDEPENDENT_RIGHT_CBAR"
YEARS = (2011, 2016, 2021, 2024)
CONTEXT = "#EEF2F4"
EDGE = "#C6CED2"

COMPONENTS = [
    ("nutrition_score", "Nutrition", "#E69736", ["#FFF1D8", "#FAD49C", "#F3B96B", "#E69736", "#B56816"]),
    ("carbon_score", "Carbon", "#347FC0", ["#E6F2FC", "#BAD8F1", "#80B8E1", "#347FC0", "#1B568F"]),
    ("diversity_score", "Cuisine\ndiversity", "#8060B2", ["#F2EBFA", "#D9C9EE", "#B798D8", "#8060B2", "#594083"]),
    ("sustainability_score", "Environmental\nsustainability", "#419B58", ["#ECF7EB", "#BFE3BD", "#84C98B", "#419B58", "#286C3B"]),
    ("hygiene_score", "Hygiene", "#C94157", ["#FCE9EC", "#F3BBC3", "#E88495", "#C94157", "#96243A"]),
    ("practice_score", "Practice", "#159B87", ["#E7F7F3", "#B7E4DA", "#75CABA", "#159B87", "#096A5F"]),
]


def strict_edges(values: pd.Series, zero_heavy: bool) -> np.ndarray:
    clean = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if clean.empty:
        raise ValueError("Component has no observed values")
    if zero_heavy:
        positive = clean.loc[clean.gt(0)]
        edges = np.r_[0.0, positive.quantile([.25, .50, .75, .90, 1.0]).to_numpy(float)]
    else:
        edges = clean.quantile([0, .20, .40, .60, .80, 1.0]).to_numpy(float)
    span = max(float(clean.max() - clean.min()), 1.0)
    for i in range(1, len(edges)):
        if not np.isfinite(edges[i]) or edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + span * 1e-6
    return edges


def fmt_tick(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


def panel_label(ax: plt.Axes, letter: str) -> None:
    ax.text(-.02, 1.095, letter, transform=ax.transAxes, ha="right", va="top",
            fontsize=7.7, fontweight="bold", color=BLACK, clip_on=False)


def north_and_scale(ax: plt.Axes) -> None:
    north_arrow(ax, x=.095, y=.940, height=.058)
    segmented_scale_bar(ax, length_km=10, x=.055, y=.014)


def draw_map(ax: plt.Axes, frame: gpd.GeoDataFrame, column: str,
             colours: list[str], edges: np.ndarray, bounds: np.ndarray,
             letter: str, show_cartography: bool) -> tuple[int, int]:
    values = pd.to_numeric(frame[column], errors="coerce")
    observed = frame.loc[values.notna()].copy()
    observed[column] = values.loc[values.notna()].astype(float)
    cmap = ListedColormap(colours)
    norm = BoundaryNorm(edges, len(colours), clip=True)
    frame.plot(ax=ax, color=CONTEXT, edgecolor=EDGE, linewidth=.045, alpha=.32)
    observed.plot(ax=ax, column=column, cmap=cmap, norm=norm,
                  edgecolor="white", linewidth=.022, alpha=.91)
    xmin, ymin, xmax, ymax = bounds
    dx, dy = xmax - xmin, ymax - ymin
    ax.set_xlim(xmin - .008 * dx, xmax + .016 * dx)
    ax.set_ylim(ymin - .065 * dy, ymax + .018 * dy)
    ax.set_aspect("equal"); ax.set_axis_off()
    outer_boundary(ax, frame, linewidth=.22)
    panel_label(ax, letter)
    if show_cartography:
        north_and_scale(ax)
    # Each map is independently interpretable while retaining the same pooled
    # four-year breaks within a component row.  The colour scale is deliberately
    # right-aligned and compact so it reads as part of the map, not as a detached
    # row-level legend.
    cax = ax.inset_axes([1.008, .095, .030, .815], transform=ax.transAxes)
    cb = plt.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
        orientation="vertical", boundaries=edges,
        ticks=edges,
    )
    cb.ax.yaxis.set_ticks_position("right")
    cb.ax.set_yticklabels([fmt_tick(value) for value in edges])
    cb.ax.tick_params(labelsize=4.7, length=1.5, width=.42, pad=1.0)
    cb.outline.set_linewidth(.42)
    return len(observed), int(values.isna().sum())


def main() -> int:
    frames = {year: gpd.read_file(DATA / f"{year}_lsbg_components.gpkg") for year in YEARS}
    if set(frames) != set(YEARS):
        raise AssertionError("Four-year component grid incomplete")
    bounds = np.array([
        min(f.total_bounds[0] for f in frames.values()),
        min(f.total_bounds[1] for f in frames.values()),
        max(f.total_bounds[2] for f in frames.values()),
        max(f.total_bounds[3] for f in frames.values()),
    ])

    fig = plt.figure(figsize=(183 / 25.4, 235 / 25.4), facecolor="white")
    gs = fig.add_gridspec(
        6, 4,
        left=.130, right=.957, top=.967, bottom=.047,
        wspace=.18, hspace=.13,
    )
    audit = []
    letter_index = 0
    for row, (column, label, label_colour, colours) in enumerate(COMPONENTS):
        pooled = pd.concat([pd.to_numeric(frames[y][column], errors="coerce") for y in YEARS], ignore_index=True)
        edges = strict_edges(pooled, float(pooled.dropna().eq(0).mean()) >= .10)
        row_first_ax = None
        for col, year in enumerate(YEARS):
            ax = fig.add_subplot(gs[row, col])
            if col == 0:
                row_first_ax = ax
            observed_n, missing_n = draw_map(
                ax, frames[year], column, colours, edges, bounds,
                # Every map panel must be independently interpretable.
                chr(ord("a") + letter_index), True,
            )
            if row == 0:
                ax.set_title(str(year), fontsize=8.2, fontweight="normal", pad=4)
            audit.append({
                "component": column, "year": year,
                "observed_lsbg_n": observed_n, "missing_lsbg_n": missing_n,
                **{f"pooled_break_{i}": float(v) for i, v in enumerate(edges)},
            })
            letter_index += 1
        if row_first_ax is None:
            raise AssertionError("Missing first map axis for component row")
        row_first_ax.text(-.12, .50, label, transform=row_first_ax.transAxes,
                          ha="right", va="center", fontsize=6.3, fontweight="normal",
                          color=BLACK, clip_on=False)

    fig.legend(
        handles=[Patch(facecolor=CONTEXT, edgecolor=EDGE, label="Component unavailable / no observed evidence")],
        loc="lower center", bbox_to_anchor=(.53, .012), frameon=False,
        fontsize=6.0, handlelength=1.0,
    )

    OUT.mkdir(parents=True, exist_ok=True)
    base = OUT / STEM
    save_cns_figure(fig, str(base))
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight",
                pil_kwargs={"compression": "tiff_lzw"})
    pd.DataFrame(audit).to_csv(OUT / f"{STEM}_source_audit.csv", index=False)
    plt.close(fig)
    print(base.with_suffix(".png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
