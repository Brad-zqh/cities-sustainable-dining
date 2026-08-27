"""Shared visual system for Cities R&R author-review figures.

The module fixes semantic color roles, typography, panel labels, map furniture
and export settings so that figures read as one manuscript rather than a set of
unrelated plotting templates.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


# Bundle the manuscript font with the repository so that every exported panel
# is rendered identically across machines. TeX Gyre Heros is metrically and
# visually close to Helvetica and is available under an open licence.
_FONT_DIR = Path(__file__).resolve().parents[1] / "assets" / "fonts"
_FONT_REGULAR = _FONT_DIR / "texgyreheros-regular.otf"
for _font_file in _FONT_DIR.glob("texgyreheros-*.otf") if _FONT_DIR.exists() else []:
    font_manager.fontManager.addfont(str(_font_file))
FONT_FAMILY = (
    font_manager.FontProperties(fname=str(_FONT_REGULAR)).get_name()
    if _FONT_REGULAR.exists()
    else "DejaVu Sans"
)
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Polygon, Rectangle


MM = 1 / 25.4

COLORS = {
    "ink": "#171A1D",
    "muted": "#69757D",
    "boundary": "#AAB6BC",
    "outer_boundary": "#46525A",
    "grid": "#E7ECEE",
    "context": "#EEF2F3",
    "missing": "#FFFFFF",
    "blue_dark": "#184A78",
    "blue": "#347FC0",
    "blue_light": "#BCD9F0",
    "teal": "#159B87",
    "teal_light": "#A9DFD3",
    "coral": "#C94157",
    "coral_light": "#F0B8BE",
    "gold": "#E6A43D",
    "violet": "#8060B2",
    "green": "#419B58",
}

# One project-wide multi-hue sequential ramp for unsigned spatial magnitudes.
# YlGnBu moves from warm light values through fresh teal to deep blue while
# preserving a monotonic luminance order; it is colourful without becoming an
# unordered rainbow.  The low end starts above paper-white so positive values
# remain distinct from missing/no-outlet polygons.
# The first positive class must remain visibly different from both paper white
# and the neutral no-data context.  A hand-tuned blue--teal sequence gives the
# manuscript one recognisable spatial language while preserving monotonic
# luminance and colour-blind legibility.
QUINTILE_COLORS = [
    "#D5ECF0",
    "#A9D7DE",
    "#79BDC8",
    "#4B9EAF",
    "#337D9B",
    "#225B7E",
    "#163B5C",
]

# A restrained multi-hue set is reserved for categorical time.  Each hue is
# sampled from a classic Matplotlib sequential family at comparable lightness,
# so years remain distinguishable without turning maps into rainbow scales.
YEAR_COLORS = {
    2011: "#C94157",
    2016: "#347FC0",
    2021: "#419B58",
    2024: "#E69736",
}
COMPONENT_COLORS = {
    "Nutrition": "#C94157",
    "Carbon": "#347FC0",
    "Cuisine diversity": "#8060B2",
    "Sustainability": "#159B87",
    "Hygiene": "#E69736",
    "Practice": "#419B58",
}


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "font.sans-serif": [FONT_FAMILY, "Helvetica", "Nimbus Sans", "DejaVu Sans"],
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "axes.titlesize": 9.0,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "text.color": "#171A1D",
            "axes.labelcolor": "#171A1D",
            "xtick.color": "#171A1D",
            "ytick.color": "#171A1D",
            "font.weight": "normal",
            "axes.titleweight": "normal",
        }
    )


def sdi_cmap(name: str = "cities_sdi") -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(name, QUINTILE_COLORS, N=256)


def panel_label(
    ax: plt.Axes,
    label: str,
    x: float = -0.065,
    y: float = 1.035,
    fontsize: float = 9.6,
) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=fontsize,
        fontweight="bold",
        color="black",
        clip_on=False,
    )


def clean_axis(ax: plt.Axes, grid: str | None = None) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_color(COLORS["ink"])
    ax.spines["bottom"].set_color(COLORS["ink"])
    ax.tick_params(colors=COLORS["ink"])
    # Nature's figure guide asks authors to avoid background gridlines. Sparse
    # reference lines must therefore be added explicitly by the calling panel.
    ax.grid(False)


def north_arrow(ax: plt.Axes, x: float = 0.91, y: float = 0.87, height: float = 0.078) -> None:
    """Compact two-tone north arrow in axes coordinates."""
    width = height * 0.72
    apex = (x, y)
    base_y = y - height
    center_y = y - height * 0.67
    left = np.array([apex, (x - width / 2, base_y), (x, center_y)])
    right = np.array([apex, (x + width / 2, base_y), (x, center_y)])
    ax.add_patch(
        Polygon(left, closed=True, transform=ax.transAxes, facecolor="white", edgecolor=COLORS["ink"], lw=0.55, zorder=20)
    )
    ax.add_patch(
        Polygon(right, closed=True, transform=ax.transAxes, facecolor=COLORS["ink"], edgecolor=COLORS["ink"], lw=0.55, zorder=20)
    )
    ax.text(x, y + 0.012, "N", transform=ax.transAxes, ha="center", va="bottom", fontsize=7.2, fontweight="normal")


def segmented_scale_bar(ax: plt.Axes, length_km: int = 10, x: float = 0.07, y: float = 0.018) -> None:
    """Compact 0–10 km scale bar with an unlabelled midpoint tick."""
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    length = length_km * 1000.0
    x0 = xmin + (xmax - xmin) * x
    y0 = ymin + (ymax - ymin) * y
    seg = length / 2
    h = (ymax - ymin) * 0.010
    ax.plot([x0, x0 + length], [y0, y0], color=COLORS["ink"], lw=.85, zorder=20, solid_capstyle="butt")
    for i, label in enumerate((0, None, length_km)):
        xx = x0 + i * seg
        ax.plot([xx, xx], [y0, y0 + h * 1.9], color=COLORS["ink"], lw=0.55, zorder=21)
        if label is not None:
            ax.text(xx, y0 + h * 2.30, str(label), ha="center", va="bottom", fontsize=5.2, color=COLORS["ink"])
    ax.text(x0 + length + 0.012 * (xmax - xmin), y0, "km", ha="left", va="center", fontsize=5.2, color=COLORS["ink"])


def outer_boundary(ax: plt.Axes, frame, linewidth: float = 0.34) -> None:
    """Draw a restrained, legible exterior outline without adding data."""
    try:
        frame.dissolve().boundary.plot(
            ax=ax, color=COLORS["outer_boundary"], linewidth=linewidth, zorder=18
        )
    except Exception:
        frame.boundary.plot(
            ax=ax, color=COLORS["outer_boundary"],
            linewidth=max(0.10, linewidth * 0.55), zorder=18
        )


def grey_osm_basemap(
    ax: plt.Axes,
    crs,
    base_zoom: int = 11,
    label_zoom: int = 10,
    base_alpha: float = 0.42,
    label_alpha: float = 0.42,
) -> None:
    """Add one restrained CARTO Positron/OSM context layer.

    The layer is cartographic context only.  It is deliberately muted and is
    never used as a historical observation or analytical covariate.  Figure
    captions carry the required OSM/CARTO attribution and retrieval date.
    """
    import contextily as ctx

    if crs is None:
        raise ValueError("A projected CRS is required for the shared OSM basemap")
    ctx.add_basemap(
        ax,
        source=ctx.providers.CartoDB.PositronNoLabels,
        zoom=base_zoom,
        crs=crs,
        attribution=False,
        reset_extent=True,
        interpolation="bilinear",
        zorder=0,
        alpha=base_alpha,
    )
    ctx.add_basemap(
        ax,
        source=ctx.providers.CartoDB.PositronOnlyLabels,
        zoom=label_zoom,
        crs=crs,
        attribution=False,
        reset_extent=True,
        interpolation="bilinear",
        zorder=1,
        alpha=label_alpha,
    )


def save_bundle(fig: plt.Figure, output_dir: Path, stem: str, preview_dpi: int = 260) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for label in fig.findobj(mpl.text.Text):
        label.set_color("#171A1D")
    fig.savefig(output_dir / f"{stem}.svg")
    fig.savefig(output_dir / f"{stem}.pdf")
    fig.savefig(output_dir / f"{stem}.tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(output_dir / f"{stem}.png", dpi=preview_dpi)
