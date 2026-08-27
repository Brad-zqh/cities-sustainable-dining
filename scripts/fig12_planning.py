# Academic Figure Skill Asset Confirmation (verified against assets/figures/)
# (a) spatial planning map → cross-type inherit + project cartographic asset → param inherit
# (b) K=10 trade-off scatter → assets/figures/basic-plots/ → param inherit
# (c) strategy × programme-size outcome matrix → assets/figures/MarkerGeneDotPlot → param inherit
# RULE: "native run" = load pre-rendered PNG via Image.open().ax.imshow().
#       "param inherit" = drawing function below that copies Class A/B/C values.
#       If a panel says "native run" and you write a drawing function, you broke the contract.

# Academic Figure Skill Typography Baseline — COPY VERBATIM, place at TOP of script
import matplotlib as mpl
from v5_cities_visual_system import (
    FONT_FAMILY,
    north_arrow as shared_north_arrow,
    outer_boundary,
    segmented_scale_bar,
)
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
    "pdf.fonttype": 42,         # TrueType font embedding
    "svg.fonttype": "none",     # editable text in SVG
    "savefig.bbox": "tight",    # trim whitespace
    "savefig.dpi": 300,
})

def save_cns_figure(fig, filename):
    """Standard Academic Figure Skill export: vector PDF + 300dpi PNG preview."""
    for label in fig.findobj(mpl.text.Text):
        label.set_color("#151515")
    fig.savefig(f"{filename}.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(f"{filename}.png", bbox_inches="tight", dpi=300)


from pathlib import Path
import os

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyogrio
from matplotlib.colors import BoundaryNorm, ListedColormap, to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "source_data" / "fig04_equity_siting_v7"
JOINT_SRC = ROOT / "source_data" / "fig14_joint_subgroup_planning_v1"
OUT = Path(os.environ.get("CITIES_FIGURE_DIR", str(ROOT / "figures")))
STEM = "Fig12_Joint_Equity_Planning_Strategies_v7_NATURE"

INK = "#263238"
MUTED = "#6F7C82"
LIGHT_LINE = "#D7DEE1"
CONTEXT = "#EFF2F3"

STRATEGIES = [
    "zero_affordability_gap",
    "equity_joint",
    "access_deficit",
    "population_reach",
]
COLORS = {
    "zero_affordability_gap": "#BA4051",
    "equity_joint": "#E49A38",
    "access_deficit": "#397FAB",
    "population_reach": "#359065",
}
TITLES = {
    "zero_affordability_gap": "Low-income zero-gap",
    "equity_joint": "Joint disadvantage",
    "access_deficit": "Low access",
    "population_reach": "Population reach",
}
SHORT = {
    "zero_affordability_gap": "LZG",
    "equity_joint": "JD",
    "access_deficit": "LA",
    "population_reach": "POP",
}
MARKERS = {
    "zero_affordability_gap": "o",
    "equity_joint": "s",
    "access_deficit": "^",
    "population_reach": "D",
}

METRICS = [
    ("population_within_selected_catchments", "Catchment population", "M", 1 / 1_000_000, (0, 2.8)),
    ("zero_affordability_priority_reached_share", "Strict-priority reach", "%", 100, (0, 20)),
    ("joint_priority_reached_share", "Joint-priority reach", "%", 100, (0, 60)),
]


def panel_label(ax: plt.Axes, label: str, x: float = -0.045, y: float = 1.04) -> None:
    ax.text(x, y, label, transform=ax.transAxes, ha="right", va="top",
            fontsize=9, fontweight="bold", color=BLACK, clip_on=False)


def load_data():
    results = pd.read_csv(JOINT_SRC / "scenario_results.csv")
    selections = pd.read_csv(JOINT_SRC / "selected_planning_nodes.csv", dtype={"site_id": str})
    sites = pd.read_csv(SRC / "candidate_planning_nodes.csv", dtype={"site_id": str})
    demand = pd.read_csv(JOINT_SRC / "priority_population_definition.csv", dtype={"lsbg_id": str})
    lsbg = pyogrio.read_dataframe(SRC / "lsbg_scenario_map.gpkg").to_crs("EPSG:2326")
    lsbg["lsbg_id"] = lsbg["lsbg_id"].astype(str)
    duplicated = [c for c in demand.columns if c != "lsbg_id" and c in lsbg.columns]
    lsbg = lsbg.drop(columns=duplicated).merge(demand, on="lsbg_id", how="left", validate="one_to_one")
    sites_gdf = gpd.GeoDataFrame(
        sites,
        geometry=gpd.points_from_xy(sites.longitude, sites.latitude, crs="EPSG:4326"),
        crs="EPSG:4326",
    ).to_crs(lsbg.crs)
    active = results.loc[results["scenario"].isin(STRATEGIES)].copy()
    expected = {(s, k) for s in STRATEGIES for k in (5, 10, 20)}
    observed = set(zip(active["scenario"], active["budget_sites"]))
    if observed != expected:
        raise ValueError(f"Scenario grid incomplete: missing={sorted(expected-observed)}, extra={sorted(observed-expected)}")
    return results, active, selections, lsbg, sites_gdf


def north_arrow(ax: plt.Axes, x: float = 0.085, y: float = 0.940) -> None:
    shared_north_arrow(ax, x=x, y=y, height=.062)


def scale_bar(ax: plt.Axes, length_km: int = 10, x: float = 0.045, y: float = 0.018) -> None:
    segmented_scale_bar(ax, length_km=length_km, x=x, y=y)


def draw_map(ax: plt.Axes, cax: plt.Axes, lsbg: gpd.GeoDataFrame,
             sites: gpd.GeoDataFrame, selections: pd.DataFrame) -> None:
    frame = lsbg.copy()
    frame["priority_share"] = (
        frame["zero_affordability_priority_population"] / frame["t_pop"].replace(0, np.nan)
    ).fillna(0)
    positive = frame.loc[frame["priority_share"].gt(0), "priority_share"]
    breaks = np.unique(np.quantile(positive, [0, .25, .50, .75, 1]))
    if len(breaks) < 5:
        breaks = np.linspace(0, max(float(positive.max()), .01), 5)
    cmap = ListedColormap(["#E4F1F3", "#AEDAD8", "#65BBAF", "#277C91"])
    norm = BoundaryNorm(breaks, cmap.N)
    frame.plot(ax=ax, color=CONTEXT, edgecolor="white", linewidth=.05, alpha=.32)
    frame.loc[frame["priority_share"].gt(0)].plot(
        ax=ax, column="priority_share", cmap=cmap, norm=norm,
        edgecolor="white", linewidth=.055, alpha=.90,
    )
    sites.plot(ax=ax, color="#8B989D", markersize=2.2, alpha=.23, zorder=3)
    for strategy in STRATEGIES:
        ids = set(selections.loc[
            selections["scenario"].eq(strategy) & selections["budget_sites"].eq(10), "site_id"
        ])
        selected = sites.loc[sites["site_id"].isin(ids)]
        selected.plot(ax=ax, marker=MARKERS[strategy], facecolor="white",
                      edgecolor="white", linewidth=1.8, markersize=34, zorder=5)
        selected.plot(ax=ax, marker=MARKERS[strategy], facecolor=COLORS[strategy],
                      edgecolor="white", linewidth=.55, markersize=23, alpha=.95, zorder=6)
    xmin, ymin, xmax, ymax = frame.total_bounds
    dx, dy = xmax - xmin, ymax - ymin
    ax.set_xlim(xmin - .01 * dx, xmax + .02 * dx)
    ax.set_ylim(ymin - .11 * dy, ymax + .025 * dy)
    ax.set_aspect("equal"); ax.set_axis_off()
    outer_boundary(ax, frame, linewidth=.30)
    ax.set_title("Selected public-housing planning nodes (K = 10)", loc="left", x=.035,
                 fontsize=8.15, fontweight="normal", pad=4)
    panel_label(ax, "a", x=-.02, y=1.02)
    north_arrow(ax); scale_bar(ax)
    handles = [
        Line2D([0], [0], marker=MARKERS[strategy], linestyle="none",
               markerfacecolor=COLORS[strategy], markeredgecolor="white",
               markeredgewidth=.5, markersize=5.4,
               label=f"{SHORT[strategy]}  {TITLES[strategy]}")
        for strategy in STRATEGIES
    ]
    # The adjacent scorecard defines all four symbols; do not repeat a tiny
    # legend across the coastline and scale bar.
    cb = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                      orientation="vertical", spacing="uniform")
    cb.set_ticks(breaks)
    cb.set_ticklabels([f"{100*v:.0f}%" for v in breaks])
    cb.ax.yaxis.set_ticks_position("right")
    cb.ax.yaxis.set_label_position("right")
    cb.ax.tick_params(labelleft=False, labelright=True)
    cb.ax.tick_params(labelsize=5.5, length=2, pad=1.5)
    cb.ax.set_title("Priority\nshare", fontsize=5.6, loc="center", pad=3)
    cb.outline.set_linewidth(.45)


def draw_tradeoff(ax: plt.Axes, active: pd.DataFrame) -> None:
    """Compact K=10 scorecard with exact, non-overlapping outcome values."""
    k10 = active.loc[active["budget_sites"].eq(10)].copy()
    ax.set_title("Planning outcomes · K=10", loc="left", fontsize=8.25,
                 fontweight="normal", pad=5)
    panel_label(ax, "b", x=-.10, y=1.02)
    ax.set_xlim(0, 1); ax.set_ylim(0, 5); ax.set_axis_off()
    columns = (.55, .75, .93)
    for x, title in zip(columns, ("Residents\n(M)", "Strict\n(%)", "Joint\n(%)")):
        ax.text(x, 4.45, title, ha="center", va="center", fontsize=6.3, color=INK)
    for index, strategy in enumerate(STRATEGIES):
        y = 3.72 - index * .88
        colour = COLORS[strategy]
        ax.add_patch(FancyBboxPatch(
            (.015, y - .33), .97, .66, boxstyle="round,pad=.012,rounding_size=.018",
            facecolor=blend_with_white(colour, .055), edgecolor=blend_with_white(colour, .30),
            linewidth=.55, zorder=0,
        ))
        ax.scatter(.055, y, s=34, marker=MARKERS[strategy], color=colour,
                   edgecolor="white", linewidth=.5, zorder=2)
        ax.text(.095, y + .08, SHORT[strategy], ha="left", va="center",
                fontsize=7.0, color=colour, fontweight="normal")
        title = {"zero_affordability_gap":"Low-income\nzero-gap",
                 "equity_joint":"Joint\ndisadvantage",
                 "access_deficit":"Low access", "population_reach":"Population\nreach"}[strategy]
        ax.text(.095, y - .15, title, ha="left", va="center",
                fontsize=6.0, color=INK, linespacing=1.05)
        row = k10.loc[k10["scenario"].eq(strategy)].iloc[0]
        values = (
            float(row["population_within_selected_catchments"]) / 1_000_000,
            100 * float(row["zero_affordability_priority_reached_share"]),
            100 * float(row["joint_priority_reached_share"]),
        )
        formats = ("{:.2f}", "{:.1f}", "{:.1f}")
        for x, value, fmt in zip(columns, values, formats):
            ax.text(x, y, fmt.format(value), ha="center", va="center",
                    fontsize=7.0, color=INK, fontweight="normal")
    # Design scope is stated in the manuscript caption.


def fmt_value(value: float, unit: str) -> str:
    if unit == "M":
        return f"{value:.2f}"
    return f"{value:.1f}"


def blend_with_white(colour: str, strength: float) -> tuple[float, float, float]:
    rgb = np.asarray(mpl.colors.to_rgb(colour))
    return tuple((1 - strength) * np.ones(3) + strength * rgb)


def draw_strategy_cards(ax: plt.Axes, active: pd.DataFrame) -> None:
    """Four strategy cards with exact K=5/10/20 horizontal bullet bars."""
    metrics = [
        ("population_within_selected_catchments", "Residents in catchments", "M", 1 / 1_000_000),
        ("zero_affordability_priority_reached_share", "Strict-priority reach", "%", 100),
        ("joint_priority_reached_share", "Joint-priority reach", "%", 100),
    ]
    ks = (5, 10, 20)
    shades = {5: .34, 10: .62, 20: .94}
    maxima = {
        column: 1.08 * float((active[column] * scale).max())
        for column, _, _, scale in metrics
    }
    ax.set_xlim(0, 1); ax.set_ylim(0, 4.55); ax.set_axis_off()
    panel_label(ax, "c", x=-.012, y=1.035)
    ax.text(0, 1.035, "Programme size and planning outcomes",
            transform=ax.transAxes, ha="left", va="top", fontsize=8.15, color=INK)
    starts = (.285, .525, .765)
    track_width = .14
    for start, (_, title, unit, _) in zip(starts, metrics):
        ax.text(start + track_width / 2, 4.18, f"{title}\n({unit})",
                ha="center", va="center", fontsize=6.0, color=INK)

    centres = (3.55, 2.62, 1.69, .76)
    for centre, strategy in zip(centres, STRATEGIES):
        colour = COLORS[strategy]
        ax.add_patch(FancyBboxPatch(
            (.004, centre - .39), .985, .78,
            boxstyle="round,pad=.010,rounding_size=.018",
            facecolor=blend_with_white(colour, .045),
            edgecolor=blend_with_white(colour, .28), linewidth=.55,
        ))
        ax.scatter(.025, centre + .08, s=30, marker=MARKERS[strategy],
                   color=colour, edgecolor="white", linewidth=.5, zorder=3)
        ax.text(.052, centre + .13, SHORT[strategy], ha="left", va="center",
                fontsize=6.7, color=colour, fontweight="normal")
        title = {"zero_affordability_gap":"Low-income\nzero-gap",
                 "equity_joint":"Joint\ndisadvantage",
                 "access_deficit":"Low access", "population_reach":"Population\nreach"}[strategy]
        ax.text(.052, centre - .12, title, ha="left", va="center",
                fontsize=6.0, color=INK, linespacing=1.05)
        for offset, k in zip((.18, 0, -.18), ks):
            ax.text(.265, centre + offset, f"K={k}", ha="right", va="center",
                    fontsize=5.8, color=INK)
        rows = active.loc[active["scenario"].eq(strategy)].set_index("budget_sites")
        for start, (column, _, unit, scale) in zip(starts, metrics):
            for offset, k in zip((.18, 0, -.18), ks):
                value = float(rows.loc[k, column]) * scale
                width = track_width * np.clip(value / maxima[column], 0, 1)
                y = centre + offset - .045
                ax.add_patch(Rectangle((start, y), track_width, .09,
                                       facecolor="#E9EDEF", edgecolor="none", zorder=0))
                ax.add_patch(Rectangle((start, y), width, .09,
                                       facecolor=blend_with_white(colour, shades[k]),
                                       edgecolor="none", zorder=1))
                label = f"{value:.2f}" if unit == "M" else f"{value:.1f}"
                ax.text(start + track_width + .009, y + .045, label,
                        ha="left", va="center", fontsize=6.0, color=INK)
    # Within-outcome normalization is explained in the manuscript caption.


def draw_selection_overlap(ax: plt.Axes, selections: pd.DataFrame) -> None:
    """Show whether distinct K=10 objectives actually select the same nodes."""
    sets = {
        strategy: set(selections.loc[
            selections["scenario"].eq(strategy)
            & selections["budget_sites"].eq(10),
            "site_id",
        ])
        for strategy in STRATEGIES
    }
    overlap = np.zeros((len(STRATEGIES), len(STRATEGIES)), dtype=float)
    for row, left in enumerate(STRATEGIES):
        for column, right in enumerate(STRATEGIES):
            union = sets[left] | sets[right]
            overlap[row, column] = len(sets[left] & sets[right]) / len(union) if union else np.nan

    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "overlap", ["#F3F8F7", "#C5E5DD", "#75BDB5", "#377EAB"], N=256
    )
    image = ax.imshow(overlap, vmin=0, vmax=1, cmap=cmap, interpolation="nearest")
    labels = [SHORT[strategy] for strategy in STRATEGIES]
    ax.set_xticks(range(4), labels=labels)
    ax.set_yticks(range(4), labels=labels)
    ax.tick_params(length=0, pad=2)
    ax.xaxis.tick_top()
    for row in range(4):
        for column in range(4):
            value = overlap[row, column]
            ax.text(column, row, f"{value:.2f}", ha="center", va="center",
                    fontsize=6.5, fontweight="normal",
                    color="#151515")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("K=10 node-set overlap", loc="left", fontsize=7.35,
                 fontweight="normal", pad=16)
    panel_label(ax, "d", x=-.22, y=1.11)
    colourbar = plt.colorbar(image, ax=ax, orientation="horizontal", fraction=.07,
                             pad=.18, aspect=22)
    colourbar.set_ticks([0, .5, 1])
    colourbar.set_label("Jaccard overlap", fontsize=6.0, labelpad=2)
    colourbar.ax.tick_params(labelsize=6.0, length=2, pad=1)
    colourbar.outline.set_linewidth(.4)


def validate_signal(active: pd.DataFrame) -> None:
    for column, _, _, scale, _ in METRICS:
        values = active[column].to_numpy(float) * scale
        mean = np.mean(values)
        ratio = (np.max(values) - np.min(values)) / max(abs(mean), 1e-12)
        if ratio <= .05:
            raise ValueError(f"Insufficient plotted signal for {column}: range/mean={ratio:.3f}")


def main() -> int:
    results, active, selections, lsbg, sites = load_data()
    validate_signal(active)
    OUT.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(183 / 25.4, 172 / 25.4), facecolor="white")
    outer = fig.add_gridspec(
        2, 2, height_ratios=[1.06, .94], width_ratios=[1.42, .78],
        left=.057, right=.973, top=.952, bottom=.088,
        wspace=.22, hspace=.18,
    )
    map_cell = outer[0, 0].subgridspec(1, 2, width_ratios=[1, .042], wspace=.016)
    ax_map = fig.add_subplot(map_cell[0, 0]); cax = fig.add_subplot(map_cell[0, 1])
    draw_map(ax_map, cax, lsbg, sites, selections)
    box = cax.get_position()
    cax.set_position([box.x0, box.y0 + .105 * box.height, box.width, .77 * box.height])
    draw_tradeoff(fig.add_subplot(outer[0, 1]), active)

    lower = outer[1, :].subgridspec(1, 2, width_ratios=[1.72, .48], wspace=.24)
    draw_strategy_cards(fig.add_subplot(lower[0, 0]), active)
    draw_selection_overlap(fig.add_subplot(lower[0, 1]), selections)

    base = OUT / STEM
    for label in fig.findobj(mpl.text.Text):
        label.set_color(INK)
    save_cns_figure(fig, str(base))
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight",
                pil_kwargs={"compression": "tiff_lzw"})
    active.to_csv(OUT / "Fig12_Joint_Equity_Planning_Strategies_v7_source_data.csv", index=False)
    results.loc[results["scenario"].eq("BAU")].to_csv(
        OUT / "Fig12_Joint_Equity_Planning_Strategies_v7_BAU_reference.csv", index=False
    )
    plt.close(fig)
    print(base.with_suffix(".png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
