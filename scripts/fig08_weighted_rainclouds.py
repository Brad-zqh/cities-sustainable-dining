"""Population-weighted raincloud distributions and uncertainty trajectories.

This figure contrasts the low-price comparator with the actual primary
quality-context-qualified affordable walking estimand.  Split population-
weighted density silhouettes and weighted box statistics expose the
distribution within household-income quintiles.  The uncertainty ribbons are
the audited 999-replicate DCCA-block-bootstrap intervals, not panel shading.
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
if str(FIGURES) not in sys.path:
    sys.path.insert(0, str(FIGURES))

from v5_cities_visual_system import COLORS, clean_axis, configure, panel_label, save_bundle  # noqa: E402


DATA = ROOT / "source_data" / "fig13_joint_quality_affordable_access_v1"
OUT = ROOT / "figures"
STEM = "Fig8_Joint_Access_Weighted_Rainclouds_Inequality_v3"
YEARS = [2016, 2021, 2024]
OUTCOMES = [
    ("low_price_access", "≤HK$100 comparator"),
    ("joint_access", "Quality + walk + price"),
]
YEAR_FAMILIES = {
    2016: ("#E8A5AD", "#AA3045"),
    2021: ("#9FC7E5", "#285F8C"),
    2024: ("#A7D6AC", "#337343"),
}
METRIC_FAMILIES = {
    "weighted_gini": ("#EBC093", "#C56A2C"),
    "zero_access_population_share": ("#C8B8E0", "#70509A"),
    "concentration_index": ("#9DD8CE", "#247C72"),
}
INK = COLORS["ink"]
MUTED = COLORS["muted"]
LIGHT = "#D9E1E5"


def weighted_quantile(values: np.ndarray, weights: np.ndarray, probs: list[float]) -> np.ndarray:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values = values[valid]
    weights = weights[valid]
    if not len(values):
        return np.full(len(probs), np.nan)
    order = np.argsort(values, kind="mergesort")
    values = values[order]
    weights = weights[order]
    cdf = (np.cumsum(weights) - 0.5 * weights) / weights.sum()
    return np.interp(probs, cdf, values, left=values[0], right=values[-1])


def population_quintile_slices(frame: pd.DataFrame) -> dict[int, pd.DataFrame]:
    data = frame.loc[
        frame["network_available"].astype(bool),
        ["ma_hh", "t_pop", "low_price_access", "joint_access"],
    ].replace([np.inf, -np.inf], np.nan).dropna(subset=["ma_hh", "t_pop"])
    data = data.loc[data["ma_hh"].gt(0) & data["t_pop"].gt(0)].sort_values("ma_hh", kind="mergesort")
    total = float(data["t_pop"].sum())
    edges = np.linspace(0.0, total, 6)
    rows: dict[int, list[dict]] = {q: [] for q in range(1, 6)}
    cursor = 0.0
    for income, group in data.groupby("ma_hh", sort=False):
        group_weight = float(group["t_pop"].sum())
        group_end = cursor + group_weight
        overlaps = np.maximum(0.0, np.minimum(group_end, edges[1:]) - np.maximum(cursor, edges[:-1]))
        for q, overlap in enumerate(overlaps, start=1):
            if overlap <= 0:
                continue
            scale = overlap / group_weight
            for row in group.itertuples(index=False):
                rows[q].append(
                    {
                        "weight": float(row.t_pop) * scale,
                        "low_price_access": float(row.low_price_access),
                        "joint_access": float(row.joint_access),
                    }
                )
        cursor = group_end
    return {q: pd.DataFrame(items) for q, items in rows.items()}


def weighted_box_stats(frame: pd.DataFrame, outcome: str) -> dict:
    values = np.log10(1.0 + pd.to_numeric(frame[outcome], errors="coerce").to_numpy(float))
    weights = frame["weight"].to_numpy(float)
    q05, q25, q50, q75, q95 = weighted_quantile(values, weights, [0.05, 0.25, 0.50, 0.75, 0.95])
    return {"whislo": q05, "q1": q25, "med": q50, "q3": q75, "whishi": q95, "fliers": []}


def draw_weighted_half_violin(
    ax: plt.Axes,
    frame: pd.DataFrame,
    outcome: str,
    center: float,
    direction: float,
    color: str,
) -> None:
    """Draw a population-weighted half violin in log1p opportunity space."""
    values = np.log10(1.0 + pd.to_numeric(frame[outcome], errors="coerce").to_numpy(float))
    weights = pd.to_numeric(frame["weight"], errors="coerce").to_numpy(float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values, weights = values[valid], weights[valid]
    if len(np.unique(values)) < 2:
        return
    grid = np.linspace(max(-0.02, float(values.min()) - 0.04), float(values.max()) + 0.04, 220)
    density = gaussian_kde(values, weights=weights, bw_method="scott")(grid)
    density = density / max(float(density.max()), 1e-12) * 0.31
    edge = center + direction * density
    ax.fill_betweenx(
        grid, center, edge, facecolor=color, edgecolor="none",
        alpha=0.20, zorder=1,
    )
    ax.plot(edge, grid, color=color, lw=0.52, alpha=0.72, zorder=1.4)


def box_panel(ax: plt.Axes, access: pd.DataFrame, year: int, letter: str) -> list[dict]:
    slices = population_quintile_slices(access.loc[access["year"].eq(year)])
    prepared = []
    offsets = [-0.075, 0.075]
    family = YEAR_FAMILIES[year]
    for ((outcome, label), offset), color in zip(zip(OUTCOMES, offsets), family):
        stats = []
        positions = []
        for q in range(1, 6):
            stat = weighted_box_stats(slices[q], outcome)
            stat["label"] = ""
            stats.append(stat)
            positions.append(q + offset)
            prepared.append({"year": year, "income_quintile": q, "outcome": outcome, **stat})
            draw_weighted_half_violin(
                ax, slices[q], outcome, q - 0.02 if offset < 0 else q + 0.02,
                -1.0 if offset < 0 else 1.0, color,
            )
        artists = ax.bxp(
            stats,
            positions=positions,
            widths=0.13,
            patch_artist=True,
            showfliers=False,
            manage_ticks=False,
            zorder=3,
        )
        for box in artists["boxes"]:
            box.set(facecolor=color, edgecolor=color, alpha=0.58, linewidth=0.80)
        for med in artists["medians"]:
            med.set(color=INK, linewidth=1.20)
        for key in ("whiskers", "caps"):
            for artist in artists[key]:
                artist.set(color=color, linewidth=0.85)

    ticks = [0, 10, 50, 100, 250, 500, 1000, 1500]
    ax.set_ylim(-0.04, np.log10(1501) + 0.06)
    ax.set_yticks(np.log10(1 + np.asarray(ticks)), [str(v) if v < 1000 else f"{v/1000:.1f}k" for v in ticks])
    ax.set_xticks(range(1, 6), ["Q1\nlowest", "Q2", "Q3", "Q4", "Q5\nhighest"])
    ax.set_xlim(0.55, 5.45)
    ax.set_title(str(year), loc="left", fontweight="bold", pad=5)
    ax.set_xlabel("Household-income population quintile")
    clean_axis(ax)
    ax.axhline(np.log10(1), color=LIGHT, lw=0.7, zorder=0)
    panel_label(ax, letter, x=-0.14, y=1.055)
    return prepared


def read_point(points: pd.DataFrame, year: int, outcome: str, metric: str) -> float:
    row = points.loc[
        points["year"].eq(year)
        & points["outcome"].eq(outcome)
    ].iloc[0]
    return float(row[metric])


def read_ci(intervals: pd.DataFrame, year: int, outcome: str, metric: str) -> tuple[float, float]:
    row = intervals.loc[
        intervals["year"].eq(year)
        & intervals["outcome"].eq(outcome)
        & intervals["metric"].eq(metric)
    ].iloc[0]
    return float(row["ci_low"]), float(row["ci_high"])


def ribbon_panel(
    ax: plt.Axes,
    points: pd.DataFrame,
    intervals: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    letter: str,
    scale: float = 1.0,
    zero_line: bool = False,
) -> list[dict]:
    prepared = []
    x = np.arange(len(YEARS), dtype=float)
    family = METRIC_FAMILIES[metric]
    for (outcome, label), color in zip(OUTCOMES, family):
        estimate = np.array([read_point(points, y, outcome, metric) for y in YEARS]) * scale
        bounds = np.array([read_ci(intervals, y, outcome, metric) for y in YEARS]) * scale
        ax.fill_between(x, bounds[:, 0], bounds[:, 1], color=color, alpha=0.16, linewidth=0, zorder=1)
        ax.plot(x, estimate, color=color, lw=1.65, marker="o", ms=4.2,
                markeredgecolor="white", markeredgewidth=0.55, zorder=3)
        short_label = "Joint" if outcome == "joint_access" else "≤HK$100"
        ax.text(x[-1] + 0.08, estimate[-1], short_label, color=color, va="center", ha="left",
                fontsize=6.0, fontweight="bold")
        for year, value, (lo, hi) in zip(YEARS, estimate, bounds):
            prepared.append({"metric": metric, "outcome": outcome, "year": year,
                             "estimate": value, "ci_low": lo, "ci_high": hi})
    if zero_line:
        ax.axhline(0, color=MUTED, lw=0.75, ls=(0, (3, 2)), zorder=0)
    ax.set_xlim(-0.08, 2.53)
    ax.set_xticks(x, YEARS)
    ax.set_title(title, loc="left", fontweight="bold", pad=5)
    ax.set_ylabel(ylabel)
    clean_axis(ax)
    panel_label(ax, letter, x=-0.17, y=1.055)
    return prepared


def main() -> int:
    configure()
    plt.rcParams.update({
        "font.size": 7.0,
        "axes.titlesize": 8.2,
        "axes.labelsize": 7.2,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
    })
    access = pd.read_csv(DATA / "lsbg_joint_quality_affordable_access.csv")
    points = pd.read_csv(DATA / "joint_access_point_estimates.csv")
    points = points.loc[points["outcome"].isin(["low_price_access", "joint_access"])].copy()
    intervals = pd.read_csv(DATA / "joint_access_dcca_block_intervals.csv")

    fig = plt.figure(figsize=(183 / 25.4, 125 / 25.4), facecolor="white")
    outer = fig.add_gridspec(
        2, 3, height_ratios=[0.90, 0.76],
        left=0.075, right=0.965, top=0.895, bottom=0.11,
        wspace=0.34, hspace=0.64,
    )
    box_rows = []
    for idx, year in enumerate(YEARS):
        ax = fig.add_subplot(outer[0, idx])
        box_rows.extend(box_panel(ax, access, year, chr(ord("a") + idx)))
        if idx == 0:
            ax.set_ylabel("Reachable restaurant opportunities\n(15 min; log-scaled axis)")
        else:
            ax.set_yticklabels([])
            ax.set_ylabel("")
    fig.legend(
        handles=[
            Patch(facecolor="#D7DCE0", edgecolor="#A9B1B7", label=OUTCOMES[0][1]),
            Patch(facecolor="#59636B", edgecolor="#59636B", label=OUTCOMES[1][1]),
        ],
        loc="upper center", bbox_to_anchor=(0.50, 0.985), ncol=2,
        frameon=False, handlelength=1.2, columnspacing=1.1,
    )

    trajectory_rows = []
    specs = [
        ("weighted_gini", "Gini coefficient", "Gini", "d", 1.0, False),
        ("zero_access_population_share", "Zero-access population", "Population (%)", "e", 100.0, False),
        ("concentration_index", "Income concentration", "Concentration index", "f", 1.0, True),
    ]
    for idx, spec in enumerate(specs):
        ax = fig.add_subplot(outer[1, idx])
        trajectory_rows.extend(ribbon_panel(ax, points, intervals, *spec))

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(box_rows).to_csv(OUT / "fig8_joint_weighted_box_statistics.csv", index=False)
    pd.DataFrame(trajectory_rows).to_csv(OUT / "fig8_joint_inequality_trajectory_intervals.csv", index=False)
    for label in fig.findobj(matplotlib.text.Text):
        label.set_color(COLORS["ink"])
    save_bundle(fig, OUT, STEM, preview_dpi=420)
    plt.close(fig)
    print(OUT / f"{STEM}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
