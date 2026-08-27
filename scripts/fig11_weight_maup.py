"""Nature-style weighting and MAUP sensitivity figure from public aggregates."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
if str(FIGURES) not in sys.path:
    sys.path.insert(0, str(FIGURES))
from v5_cities_visual_system import COLORS, FONT_FAMILY, YEAR_COLORS, configure, panel_label, save_bundle

SOURCE = ROOT / "source_data" / "figS_weight_sensitivity_v4"
OUT = ROOT / "figures"
STEM = "Fig11_Weighting_MAUP_Sensitivity_v15_HEATMAP"

INK = COLORS["ink"]
MUTED = COLORS["muted"]
GRID = COLORS["grid"]
BLUE = COLORS["blue"]
TEAL = COLORS["teal"]
ORANGE = COLORS["gold"]
VIOLET = COLORS["violet"]
GREEN = COLORS["green"]
RED = COLORS["coral"]

VARIANTS = [
    "Equal five (no practice)",
    "Geometric six",
    "Absolute-PC1 weights",
    "Entropy weights",
]
VARIANT_COLORS = {
    "Equal six (strict)": INK,
    "Equal five (no practice)": GREEN,
    "Geometric six": VIOLET,
    "Absolute-PC1 weights": BLUE,
    "Entropy weights": ORANGE,
}
SHORT = {
    "Equal five (no practice)": "Equal five",
    "Geometric six": "Geometric",
    "Absolute-PC1 weights": "Absolute PC1",
    "Entropy weights": "Entropy",
}


def style() -> None:
    configure()
    mpl.rcParams.update({
        "font.family": FONT_FAMILY,
        "font.sans-serif": [FONT_FAMILY, "Helvetica", "Nimbus Sans", "DejaVu Sans"],
        "font.size": 8.0,
        "axes.titlesize": 9.0,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "legend.fontsize": 6.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.65,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.dpi": 360,
        "savefig.bbox": "tight",
    })


def panel(ax: plt.Axes, letter: str, title: str) -> None:
    panel_label(ax, letter, x=-0.10, y=1.06, fontsize=10.0)
    ax.set_title(title, loc="left", pad=5, fontweight="bold", color=INK)


def finish(ax: plt.Axes, grid: str = "x") -> None:
    ax.grid(axis=grid, color=GRID, lw=0.55)
    ax.spines["left"].set_color(INK)
    ax.spines["bottom"].set_color(INK)
    ax.tick_params(colors=INK, width=0.6, length=3)


def main() -> int:
    style()
    OUT.mkdir(parents=True, exist_ok=True)
    weights = pd.read_csv(SOURCE / "weight_schemes.csv")
    agreement = pd.read_csv(SOURCE / "variant_agreement.csv")
    maup = pd.read_csv(SOURCE / "maup_agreement.csv")

    fig, axes = plt.subplots(2, 2, figsize=(183 / 25.4, 138 / 25.4), facecolor="white")
    fig.subplots_adjust(left=0.13, right=0.975, top=0.945, bottom=0.105, wspace=0.38, hspace=0.52)
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    # a: empirically derived alternatives against the transparent equal-weight reference.
    panel(ax_a, "a", "Alternative component weights")
    component_order = ["Nutrition", "Carbon", "Cuisine diversity", "Hygiene", "Practice tag", "Sustainability signal"]
    component_display = ["Nutrition", "Carbon", "Cuisine diversity", "Hygiene", "Practice", "Environmental sustainability"]
    method_order = ["Equal six", "Absolute PC1 loading", "Entropy"]
    method_colors = {"Equal six": INK, "Absolute PC1 loading": BLUE, "Entropy": ORANGE}
    method_markers = {"Equal six": "o", "Absolute PC1 loading": "s", "Entropy": "^"}
    ypos = np.arange(len(component_order))[::-1]
    offsets = {"Equal six": 0.18, "Absolute PC1 loading": 0.0, "Entropy": -0.18}
    for method in method_order:
        frame = weights.loc[weights["scheme"].eq(method)].set_index("component_label")
        vals = np.array([frame.loc[name, "weight"] for name in component_order], dtype=float)
        ax_a.scatter(vals, ypos + offsets[method], s=31, marker=method_markers[method],
                     color=method_colors[method], edgecolor="white", lw=0.5, label=method, zorder=3)
    ax_a.set_yticks(ypos, component_display)
    ax_a.set_xlim(0, 0.50)
    ax_a.set_xlabel("Weight")
    ax_a.tick_params(axis="y", length=0)
    ax_a.legend(loc="lower right", frameon=False, fontsize=5.6, handletextpad=0.35,
                labelspacing=0.35)
    finish(ax_a)

    # b: multiyear LSBG rank agreement with equal arithmetic SDI.
    panel(ax_b, "b", "Rank agreement with equal weighting")
    y = np.arange(len(VARIANTS))[::-1]
    offsets = {2011: 0.24, 2016: 0.08, 2021: -0.08, 2024: -0.24}
    for year in [2011, 2016, 2021, 2024]:
        frame = agreement.loc[
            agreement["year"].eq(year) & agreement["scale"].eq("LSBG")
            & agreement["variant"].isin(VARIANTS)
        ].set_index("variant")
        vals = np.array([frame.loc[v, "spearman_vs_equal"] for v in VARIANTS])
        ax_b.scatter(vals, y + offsets[year], s=27, color=YEAR_COLORS[year],
                     edgecolor="white", lw=0.5, label=str(year), zorder=3)
    ax_b.set_yticks(y, [SHORT[v] for v in VARIANTS])
    ax_b.set_xlim(0.35, 1.02)
    ax_b.axvline(1, color=MUTED, ls="--", lw=0.65)
    ax_b.set_xlabel("Spearman ρ (LSBG)")
    ax_b.tick_params(axis="y", length=0)
    ax_b.legend(loc="upper left", ncol=2, frameon=False, fontsize=5.5,
                handletextpad=0.3, columnspacing=0.7)
    finish(ax_b)

    # c: the same global correlation can conceal changes at policy-relevant tails.
    panel(ax_c, "c", "2021 tail-membership stability")
    frame = agreement.loc[
        agreement["year"].eq(2021) & agreement["scale"].eq("LSBG")
        & agreement["variant"].isin(VARIANTS)
    ].set_index("variant")
    for yy, variant in zip(y, VARIANTS):
        lo = float(frame.loc[variant, "low_quintile_jaccard"])
        hi = float(frame.loc[variant, "high_quintile_jaccard"])
        ax_c.plot([lo, hi], [yy, yy], color="#AAB5BA", lw=1.4, zorder=1)
        ax_c.scatter(lo, yy, s=31, color=BLUE, edgecolor="white", lw=0.5, zorder=3)
        ax_c.scatter(hi, yy, s=31, color=RED, edgecolor="white", lw=0.5, zorder=3)
    ax_c.set_yticks(y, [SHORT[v] for v in VARIANTS])
    ax_c.set_xlim(0, 1.03)
    ax_c.set_xlabel("Jaccard overlap with equal weighting")
    ax_c.tick_params(axis="y", length=0)
    ax_c.scatter([], [], color=BLUE, s=28, label="Low quintile")
    ax_c.scatter([], [], color=RED, s=28, label="High quintile")
    ax_c.legend(loc="lower right", frameon=False, fontsize=5.6, handletextpad=0.3)
    finish(ax_c)

    # d: direct LSBG vs population-aggregated DCCA agreement (MAUP diagnostic).
    # A value-labelled heatmap avoids another generic line chart and makes
    # year-by-specification sensitivity immediately comparable.
    panel(ax_d, "d", "Cross-scale agreement (LSBG to DCCA)")
    years = [2011, 2016, 2021, 2024]
    variants = ["Equal six (strict)", *VARIANTS]
    matrix = np.array([
        maup.loc[maup["variant"].eq(variant)].set_index("year").loc[years, "spearman"].to_numpy(float)
        for variant in variants
    ])
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "cities_agreement_blue_red",
        ["#2166AC", "#A6CEE3", "#F7F7F7", "#F4A582", "#B2182B"],
    )
    image = ax_d.imshow(matrix, aspect="auto", vmin=0.58, vmax=0.84, cmap=cmap)
    ax_d.set_xticks(np.arange(len(years)), years)
    ax_d.set_yticks(
        np.arange(len(variants)),
        ["Equal six", *[SHORT[v] for v in VARIANTS]],
    )
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            ax_d.text(
                col, row, f"{value:.2f}", ha="center", va="center",
                fontsize=6.0, fontweight="bold",
                color=INK,
            )
    cbar = fig.colorbar(image, ax=ax_d, orientation="horizontal", fraction=0.075, pad=0.14)
    cbar.set_label("Spearman ρ", fontsize=6.4)
    cbar.ax.tick_params(labelsize=5.8, length=2)
    cbar.outline.set_linewidth(0.45)
    ax_d.tick_params(axis="y", length=0)
    for spine in ax_d.spines.values():
        spine.set_visible(False)

    save_bundle(fig, OUT, STEM, preview_dpi=360)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
