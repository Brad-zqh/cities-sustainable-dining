"""Times New Roman companion style for the Cities figure system.

This keeps the project's semantic colours, map furniture and export rules,
changing typography only so the manuscript can be compared in a serif style.
"""
from v5_cities_visual_system import *  # noqa: F401,F403
import matplotlib as mpl

FONT_FAMILY = "Times New Roman"

def configure() -> None:
    # Start from the validated Cities system, then change only typography.
    mpl.rcParams.update({
        "font.family": FONT_FAMILY,
        "font.serif": [FONT_FAMILY, "Times", "DejaVu Serif"],
        "font.size": 8.0,
        "axes.labelsize": 8.0,
        "axes.titlesize": 9.0,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "legend.fontsize": 7.0,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
