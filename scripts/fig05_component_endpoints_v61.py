"""Main-text endpoint atlas for the six evidence-locked SDI components.

The full 2011/2016/2021/2024 atlas remains supplementary evidence.  This
main-text plate shows 2011 and 2024 only, while retaining class breaks pooled
over all four study years.  No values are imputed, simulated or reclassified.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

from fig05_component_atlas import COMPONENTS, fmt_tick, strict_edges
from v5_cities_visual_system import FONT_FAMILY, north_arrow, outer_boundary, segmented_scale_bar


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "source_data" / "fig_v4_four_year"
OUT = ROOT / "figures" / "v61_nature_visual_review"
STEM = "Fig05_Component_Endpoints_2011_2024"
ALL_YEARS = (2011, 2016, 2021, 2024)
DISPLAY_YEARS = (2011, 2024)
CONTEXT = "#EEF2F4"
EDGE = "#C6CED2"
INK = "#151515"


mpl.rcParams.update(
    {
        "font.family": FONT_FAMILY,
        "font.sans-serif": [FONT_FAMILY, "Helvetica", "Nimbus Sans", "DejaVu Sans"],
        "font.size": 8.4,
        "axes.titlesize": 9.2,
        "axes.labelsize": 8.4,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "legend.fontsize": 8.0,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


def panel_label(ax: plt.Axes, letter: str) -> None:
    ax.text(
        -0.035,
        1.065,
        letter,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9.0,
        fontweight="bold",
        color=INK,
        clip_on=False,
    )


def draw_map(
    ax: plt.Axes,
    frame: gpd.GeoDataFrame,
    column: str,
    colours: list[str],
    edges: np.ndarray,
    bounds: np.ndarray,
    letter: str,
) -> tuple[int, int]:
    values = pd.to_numeric(frame[column], errors="coerce")
    observed = frame.loc[values.notna()].copy()
    observed[column] = values.loc[values.notna()].astype(float)
    cmap = ListedColormap(colours)
    norm = BoundaryNorm(edges, len(colours), clip=True)

    frame.plot(ax=ax, color=CONTEXT, edgecolor=EDGE, linewidth=0.055, alpha=0.36)
    observed.plot(
        ax=ax,
        column=column,
        cmap=cmap,
        norm=norm,
        edgecolor="white",
        linewidth=0.027,
        alpha=0.96,
    )
    xmin, ymin, xmax, ymax = bounds
    dx, dy = xmax - xmin, ymax - ymin
    ax.set_xlim(xmin - 0.008 * dx, xmax + 0.014 * dx)
    ax.set_ylim(ymin - 0.060 * dy, ymax + 0.020 * dy)
    ax.set_aspect("equal")
    ax.set_axis_off()
    outer_boundary(ax, frame, linewidth=0.27)
    panel_label(ax, letter)
    north_arrow(ax, x=0.090, y=0.925, height=0.054)
    segmented_scale_bar(ax, length_km=10, x=0.055, y=0.014)

    cax = ax.inset_axes([1.004, 0.075, 0.026, 0.850], transform=ax.transAxes)
    cb = plt.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cax,
        orientation="vertical",
        boundaries=edges,
        ticks=edges,
    )
    cb.ax.yaxis.set_ticks_position("right")
    cb.ax.set_yticklabels([fmt_tick(value) for value in edges])
    cb.ax.tick_params(labelsize=6.5, length=2.0, width=0.45, pad=1.0)
    cb.outline.set_linewidth(0.48)
    return len(observed), int(values.isna().sum())


def save(fig: plt.Figure, base: Path) -> None:
    for label in fig.findobj(mpl.text.Text):
        label.set_color(INK)
    fig.savefig(base.with_suffix(".png"), dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".pdf"), dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(
        base.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )


def main() -> int:
    frames = {year: gpd.read_file(DATA / f"{year}_lsbg_components.gpkg") for year in ALL_YEARS}
    bounds = np.array(
        [
            min(frame.total_bounds[0] for frame in frames.values()),
            min(frame.total_bounds[1] for frame in frames.values()),
            max(frame.total_bounds[2] for frame in frames.values()),
            max(frame.total_bounds[3] for frame in frames.values()),
        ]
    )

    fig = plt.figure(figsize=(183 / 25.4, 222 / 25.4), facecolor="white")
    gs = fig.add_gridspec(
        6,
        2,
        left=0.115,
        right=0.965,
        top=0.976,
        bottom=0.052,
        wspace=0.155,
        hspace=0.080,
    )
    audit: list[dict] = []
    letter_index = 0
    for row, (column, component, _, colours) in enumerate(COMPONENTS):
        pooled = pd.concat(
            [pd.to_numeric(frames[year][column], errors="coerce") for year in ALL_YEARS],
            ignore_index=True,
        )
        edges = strict_edges(pooled, float(pooled.dropna().eq(0).mean()) >= 0.10)
        first_ax: plt.Axes | None = None
        for col, year in enumerate(DISPLAY_YEARS):
            ax = fig.add_subplot(gs[row, col])
            if first_ax is None:
                first_ax = ax
            observed_n, missing_n = draw_map(
                ax,
                frames[year],
                column,
                colours,
                edges,
                bounds,
                chr(ord("a") + letter_index),
            )
            if row == 0:
                ax.set_title(str(year), pad=4.0, fontweight="normal")
            audit.append(
                {
                    "component": column,
                    "year": year,
                    "observed_lsbg_n": observed_n,
                    "missing_lsbg_n": missing_n,
                    **{f"pooled_break_{i}": float(value) for i, value in enumerate(edges)},
                }
            )
            letter_index += 1
        if first_ax is None:
            raise AssertionError("missing component row")
        first_ax.text(
            -0.085,
            0.50,
            component,
            transform=first_ax.transAxes,
            ha="right",
            va="center",
            fontsize=8.7,
            fontweight="normal",
            color=INK,
            clip_on=False,
        )

    fig.legend(
        handles=[Patch(facecolor=CONTEXT, edgecolor=EDGE, label="Component unavailable / no observed evidence")],
        loc="lower center",
        bbox_to_anchor=(0.54, 0.011),
        frameon=False,
        fontsize=7.7,
        handlelength=1.05,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    base = OUT / STEM
    save(fig, base)
    pd.DataFrame(audit).to_csv(OUT / f"{STEM}_source_audit.csv", index=False)
    plt.close(fig)
    print(base.with_suffix(".png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
