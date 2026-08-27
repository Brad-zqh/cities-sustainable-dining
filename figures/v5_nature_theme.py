"""Shared visual grammar for the Cities revision figure system.

The module changes presentation only.  It deliberately contains no data
transformations, statistical calculations, or figure-specific claims.
"""

from __future__ import annotations

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

from v5_cities_visual_system import FONT_FAMILY


PALETTE = {
    # Editorial neutrals.
    "ink": "#151515",
    "muted": "#66767E",
    "soft_text": "#829097",
    "grid": "#DDE5E7",
    "context": "#F3F6F6",
    "missing": "#FFFFFF",
    "boundary": "#AEBBC0",
    # Recurrent semantic signals.
    "navy": "#245477",
    "blue": "#347FC0",
    "teal": "#159B87",
    "green": "#419B58",
    "gold": "#E6A43D",
    "orange": "#E69736",
    "coral": "#C94157",
    "rose": "#C9838B",
    "violet": "#8060B2",
    "slate": "#5F7078",
}


YEAR_COLORS = {
    2011: "#C94157",
    2016: "#347FC0",
    2021: "#419B58",
    2024: "#E69736",
}


COMPONENT_COLORS = {
    "Nutrition": "#E6A43D",
    "Carbon": "#347FC0",
    "Cuisine diversity": "#8060B2",
    "Sustainability": "#159B87",
    "Hygiene": "#C94157",
    "Practice": "#419B58",
}


def configure() -> None:
    """Apply a compact, journal-oriented Matplotlib style."""
    mpl.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "font.sans-serif": [FONT_FAMILY, "Helvetica", "Nimbus Sans", "DejaVu Sans"],
            "font.size": 6.8,
            "axes.titlesize": 7.4,
            "axes.titleweight": "normal",
            "axes.labelsize": 6.8,
            "axes.labelcolor": PALETTE["ink"],
            "axes.edgecolor": PALETTE["ink"],
            "axes.linewidth": 0.55,
            "xtick.labelsize": 6.1,
            "ytick.labelsize": 6.1,
            "xtick.color": PALETTE["ink"],
            "ytick.color": PALETTE["ink"],
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.major.size": 2.4,
            "ytick.major.size": 2.4,
            "legend.fontsize": 5.9,
            "legend.frameon": False,
            "lines.linewidth": 1.2,
            "lines.solid_capstyle": "round",
            "text.color": PALETTE["ink"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def sequential_sdi(name: str = "sdi") -> LinearSegmentedColormap:
    """Shared cool sequential scale used by SDI and access maps."""
    return LinearSegmentedColormap.from_list(
        name,
        ["#F1F6F5", "#D4E8E5", "#A5D2CC", "#67AEA9", "#397F91", "#244D72"],
        N=256,
    )


def sequential_component(color: str, name: str) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        name,
        ["#F6F8F7", "#E4ECEB", color],
        N=256,
    )


def clean_axis(ax, grid_axis: str = "y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=grid_axis, color=PALETTE["grid"], linewidth=0.45, alpha=0.82)
    ax.set_axisbelow(True)
