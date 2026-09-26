# Academic Figure Skill Asset Confirmation (verified against assets/figures/)
# (a-e) grouped bars -> GroupedBarChart -> param inherit
# (g) same-year domain contrasts -> Forest -> param inherit
# RULE: all plotted values are loaded from the verified subgroup source bundle.

"""Within-year inequality in quality-qualified affordable walking access.

Core conclusion:
    Same-year socioeconomic and sex groups differ in the probability of having
    no opportunity that jointly satisfies the quality, walking and price gates.
Archetype:
    Seven-panel quantitative figure (six grouped social-domain comparisons plus
    one explicitly interpreted 2024 within-domain contrast forest).
Statistics:
    Paired DCCA spatial-block bootstrap contrasts against one prespecified
    reference group within each domain and year; BH correction within each
    domain-year family.
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from v5_cities_visual_system import FONT_FAMILY


mpl.rcParams.update({
    "font.family": FONT_FAMILY,
    "font.sans-serif": [FONT_FAMILY, "Helvetica", "Nimbus Sans", "DejaVu Sans"],
    "font.size": 7,
    "axes.titlesize": 8,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 5.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.7,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.bbox": "tight",
})


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source_data" / "fig14_joint_subgroup_planning_v1"
OUT = ROOT / "figures"
STEM = "Fig17_SameYear_Socioeconomic_Zero_Joint_Opportunity"
YEARS = (2016, 2021, 2024)

DOMAIN_PALETTES = {
    "Ethnicity": {2016: "#F4C7C4", 2021: "#DF7772", 2024: "#A83438"},
    "Occupation": {2016: "#C4DDF0", 2021: "#6EA8D1", 2024: "#28608F"},
    "Education": {2016: "#C8E4CA", 2021: "#74B77E", 2024: "#2F7445"},
    "Household income": {2016: "#F8D7AD", 2021: "#E9A458", 2024: "#B5651D"},
    "Age": {2016: "#D9CDEA", 2021: "#A489C5", 2024: "#67488D"},
    "Sex": {2016: "#C9E9E6", 2021: "#72BEB8", 2024: "#187C78"},
}
DOMAIN_DARK = {domain: palette[2024] for domain, palette in DOMAIN_PALETTES.items()}
DOMAIN_ORDER = {
    "Ethnicity": ["Chinese", "Filipino", "Indonesian", "White", "Other"],
    "Occupation": ["Managers", "Professionals", "Clerical", "Service/sales", "Elementary"],
    "Education": ["Primary or below", "Secondary", "Post-secondary"],
    "Household income": ["<HK$10k", "HK$10–20k", "HK$20–40k", "≥HK$40k"],
    "Age": ["15–24", "25–44", "45–64", "65+"],
    "Sex": ["Female", "Male"],
}
REFERENCES = {
    "Ethnicity": "Chinese",
    "Occupation": "Managers",
    "Education": "Post-secondary",
    "Household income": "≥HK$40k",
    "Age": "25–44",
    "Sex": "Female",
}
DISPLAY = {
    "Primary or below": "Primary\nor below",
    "Post-secondary": "Post-\nsecondary",
    "Service/sales": "Service/\nsales",
    "HK$10–20k": "HK$10–20k",
    "HK$20–40k": "HK$20–40k",
}


def bh_adjust(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = np.minimum.accumulate(
        (ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1]
    )[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def within_year_contrasts(replicates: pd.DataFrame, points: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    point_lookup = points.set_index(["year", "domain", "group"])
    for domain, groups in DOMAIN_ORDER.items():
        reference = REFERENCES[domain]
        for year in YEARS:
            frame = replicates.loc[
                replicates["domain"].eq(domain) & replicates["year"].eq(year)
            ]
            ref = frame.loc[frame["group"].eq(reference), ["replicate", "zero_joint_access_share"]].rename(
                columns={"zero_joint_access_share": "reference_value"}
            )
            family_idx: list[int] = []
            family_p: list[float] = []
            for group in groups:
                if group == reference:
                    continue
                test = frame.loc[frame["group"].eq(group), ["replicate", "zero_joint_access_share"]].rename(
                    columns={"zero_joint_access_share": "group_value"}
                )
                paired = test.merge(ref, on="replicate", how="inner", validate="one_to_one")
                if len(paired) < 2:
                    raise ValueError(f"Insufficient paired replicates: {domain}, {year}, {group}")
                diff = (paired["group_value"] - paired["reference_value"]).to_numpy(float)
                p = 2.0 * min(float(np.mean(diff <= 0.0)), float(np.mean(diff >= 0.0)))
                p = min(1.0, max(p, 2.0 / (len(diff) + 1.0)))
                point_diff = (
                    float(point_lookup.loc[(year, domain, group), "zero_joint_access_share"])
                    - float(point_lookup.loc[(year, domain, reference), "zero_joint_access_share"])
                )
                rows.append({
                    "year": year,
                    "domain": domain,
                    "reference_group": reference,
                    "comparison_group": group,
                    "difference_percentage_points": point_diff * 100.0,
                    "lower_95_percentage_points": float(np.quantile(diff, 0.025) * 100.0),
                    "upper_95_percentage_points": float(np.quantile(diff, 0.975) * 100.0),
                    "p_two_sided": p,
                    "paired_replicates": len(diff),
                    "q_bh_within_domain_year": np.nan,
                })
                family_idx.append(len(rows) - 1)
                family_p.append(p)
            adjusted = bh_adjust(np.asarray(family_p))
            for index, q in zip(family_idx, adjusted):
                rows[index]["q_bh_within_domain_year"] = float(q)
    return pd.DataFrame(rows)


def fmt_n(value: float) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}m"
    return f"{value / 1000:.0f}k" if value >= 1000 else f"{value:.0f}"


def fmt_q(value: float) -> str:
    if value < 0.001:
        return "$q$<.001"
    if value < 0.10:
        return f"$q$={value:.3f}".replace("0.", ".")
    return f"$q$={value:.2f}".replace("0.", ".")


def significance_stars(q_value: float) -> str:
    if q_value < .001:
        return "***"
    if q_value < .01:
        return "**"
    if q_value < .05:
        return "*"
    return ""


def clean_axis(ax: plt.Axes) -> None:
    ax.spines["left"].set_color("#27343A")
    ax.spines["bottom"].set_color("#27343A")
    ax.tick_params(color="#27343A", labelcolor="#27343A", width=0.6, length=2.5)
    ax.grid(False)


def draw_domain(
    ax: plt.Axes,
    domain: str,
    points: pd.DataFrame,
    intervals: pd.DataFrame,
    contrasts: pd.DataFrame,
    show_ylabel: bool,
) -> None:
    groups = DOMAIN_ORDER[domain]
    reference = REFERENCES[domain]
    year_positions = np.arange(len(YEARS), dtype=float)
    group_width = min(.145, .76 / len(groups))
    offsets = (np.arange(len(groups), dtype=float) - (len(groups) - 1) / 2) * group_width
    base = np.asarray(mpl.colors.to_rgb(DOMAIN_DARK[domain]))
    strengths = np.linspace(.40, .96, len(groups))
    group_colours = {
        group: tuple(np.ones(3) * (1 - strength) + base * strength)
        for group, strength in zip(groups, strengths)
    }
    all_highs: list[float] = []
    bracket_highs: list[float] = []

    for year_position, year in zip(year_positions, YEARS):
        p = points.loc[(points["domain"] == domain) & (points["year"] == year)].set_index("group")
        ci = intervals.loc[
            (intervals["domain"] == domain)
            & (intervals["year"] == year)
            & (intervals["metric"] == "zero_joint_access_share")
        ].set_index("group")
        values = np.array([float(p.loc[group, "zero_joint_access_share"]) * 100 for group in groups])
        low = np.array([float(ci.loc[group, "lower_95"]) * 100 for group in groups])
        high = np.array([float(ci.loc[group, "upper_95"]) * 100 for group in groups])
        all_highs.extend(high.tolist())
        yerr = np.vstack([values - low, high - values])
        for index, group in enumerate(groups):
            position = year_position + offsets[index]
            colour = group_colours[group]
            ax.bar(position, values[index], width=group_width * .88,
                   color=colour, edgecolor="white", linewidth=.36, zorder=2)
            ax.errorbar(position, values[index],
                        yerr=[[yerr[0, index]], [yerr[1, index]]], fmt="none",
                        ecolor=colour, elinewidth=3.1, capsize=2.2,
                        capthick=2.0, alpha=.16, zorder=2.4)
            ax.errorbar(position, values[index],
                        yerr=[[yerr[0, index]], [yerr[1, index]]], fmt="none",
                        ecolor=DOMAIN_DARK[domain], elinewidth=.62,
                        capsize=1.5, capthick=.6, alpha=.90, zorder=3)
        # Draw only evidence-supported same-year comparisons against the
        # prespecified reference. Bracket levels are data-driven and no mark
        # is drawn when the BH-adjusted q value is not below .05.
        reference_index = groups.index(reference)
        significant_rows = []
        for group in groups:
            if group == reference:
                continue
            row = contrasts.loc[
                contrasts["domain"].eq(domain)
                & contrasts["year"].eq(year)
                & contrasts["comparison_group"].eq(group)
            ].iloc[0]
            q = float(row["q_bh_within_domain_year"])
            if q < .05:
                significant_rows.append((groups.index(group), q))
        significant_rows.sort(key=lambda item: abs(item[0] - reference_index))
        previous_bracket_top = -np.inf
        for level, (comparison_index, q) in enumerate(significant_rows):
            x_ref = year_position + offsets[reference_index]
            x_cmp = year_position + offsets[comparison_index]
            # Nested comparisons must not merge their stars visually. Clear
            # every intervening interval, then reserve a separate text row.
            left, right = sorted((reference_index, comparison_index))
            y = max(float(np.max(high[left:right + 1])) + 1.25,
                    previous_bracket_top + 3.8)
            h = .58
            ax.plot([x_ref, x_ref, x_cmp, x_cmp], [y, y + h, y + h, y],
                    color="#59656B", lw=.55, clip_on=False, zorder=4)
            ax.text((x_ref + x_cmp) / 2, y + h + .22, significance_stars(q),
                    ha="center", va="bottom", fontsize=6.2, color="#151515",
                    clip_on=False)
            bracket_highs.append(y + h + 1.0)
            previous_bracket_top = y + h

    p21 = points.loc[(points["domain"] == domain) & points["year"].eq(2021)].set_index("group")
    ax.set_xticks(year_positions, [str(year) for year in YEARS])
    ax.tick_params(axis="x", labelsize=6.2, pad=2)
    upper_signal = max(bracket_highs) if bracket_highs else max(all_highs)
    upper = max(46.0, np.ceil((max(max(all_highs), upper_signal) + 4.0) / 5.0) * 5.0)
    ax.set_ylim(0, upper)
    ax.set_yticks(np.arange(0, upper + 0.1, 10))
    ax.set_ylabel("Zero joint opportunity (%)" if show_ylabel else "")
    ax.set_title(domain, loc="left", fontweight="normal", color="#151515", pad=7)
    handles = []
    for group in groups:
        marker = "†" if group == reference else ""
        label = DISPLAY.get(group, group).replace("\n", " ")
        n = fmt_n(float(p21.loc[group, "group_denominator"]))
        handles.append(mpl.patches.Patch(facecolor=group_colours[group], edgecolor="none",
                                         label=f"{label}{marker} · {n}"))
    ax.legend(handles=handles, title="Subgroup · $N$", ncol=1, loc="upper right",
              bbox_to_anchor=((.70, .995) if domain.lower() == 'ethnicity' else (.995, .995)), fontsize=4.8, title_fontsize=5.15,
              handlelength=.90, handletextpad=.36, labelspacing=.18,
              borderaxespad=.12)
    clean_axis(ax)
    ax.spines["left"].set_visible(show_ylabel)
    ax.tick_params(axis="y", left=show_ylabel, labelleft=show_ylabel)
    ax.axhline(0, color="#27343A", lw=0.7, zorder=1)


def draw_same_year_forest(ax: plt.Axes, contrasts: pd.DataFrame) -> None:
    """Show the strongest observed 2024 comparison in each prespecified domain."""
    selected = []
    for domain in DOMAIN_ORDER:
        family = contrasts.loc[
            contrasts["domain"].eq(domain) & contrasts["year"].eq(2024)
        ].copy()
        row = family.iloc[family["difference_percentage_points"].abs().argmax()]
        selected.append(row)

    positions = np.arange(len(selected))[::-1]
    minimum = min(float(row["lower_95_percentage_points"]) for row in selected)
    maximum = max(float(row["upper_95_percentage_points"]) for row in selected)
    span = max(maximum - minimum, 1.0)
    ax.axvline(0, color="#849097", lw=.7, ls=(0, (3, 2)), zorder=0)
    labels = []
    for y, row in zip(positions, selected):
        domain = str(row["domain"])
        colour = DOMAIN_DARK[domain]
        x = float(row["difference_percentage_points"])
        lower = float(row["lower_95_percentage_points"])
        upper = float(row["upper_95_percentage_points"])
        significant = float(row["q_bh_within_domain_year"]) < .05
        ax.plot([lower, upper], [y, y], color=colour, lw=5.2,
                alpha=.14, solid_capstyle="round", zorder=1)
        ax.plot([lower, upper], [y, y], color=colour, lw=1.0,
                solid_capstyle="round", zorder=2)
        ax.scatter(x, y, s=48, marker="D" if significant else "o",
                   facecolor="white", edgecolor=colour, linewidth=1.1, zorder=3)
        comparison = str(row["comparison_group"]).replace("$", r"\$")
        reference = str(row["reference_group"]).replace("$", r"\$")
        labels.append(f"{domain}: {comparison} vs {reference}")
        ax.text(maximum + .045 * span, y,
                f"{x:+.1f} pp   {fmt_q(float(row['q_bh_within_domain_year']))}",
                va="center", ha="left", fontsize=6.7, color="#151515")

    ax.set_yticks(positions, labels=labels)
    ax.tick_params(axis="y", labelsize=6.5, length=0, pad=4)
    ax.set_xlim(minimum - .07 * span, maximum + .34 * span)
    ax.set_ylim(-.65, len(selected) - .35)
    ax.set_title("Largest same-year within-domain gap · 2024", loc="left",
                 fontweight="normal", color="#151515", pad=5)
    ax.set_xlabel("Zero joint-opportunity difference vs same-year reference (percentage points)")
    clean_axis(ax)
    ax.spines["left"].set_visible(False)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.075, 1.11, label, transform=ax.transAxes, fontsize=9,
            fontweight="bold", ha="left", va="top", color="#111111")


def main() -> int:
    points = pd.read_csv(SOURCE / "joint_subgroup_point_estimates.csv")
    intervals = pd.read_csv(SOURCE / "joint_subgroup_dcca_block_intervals.csv")
    replicates = pd.read_csv(SOURCE / "joint_subgroup_dcca_block_replicates.csv")
    contrasts = within_year_contrasts(replicates, points)

    if len(points) != 69 or len(intervals) != 138 or len(replicates) != 68931:
        raise ValueError("Unexpected source-data dimensions")
    if contrasts["paired_replicates"].min() != 999:
        raise ValueError("Every within-year contrast must retain 999 paired DCCA replicates")

    fig = plt.figure(figsize=(183 / 25.4, 198 / 25.4), facecolor="white")
    gs = fig.add_gridspec(
        3, 12, left=0.082, right=0.98, top=0.955, bottom=0.072,
        height_ratios=[1.0, 1.0, 1.16], wspace=0.68, hspace=.46,
    )
    placements = [
        ("Ethnicity", (0, slice(0, 4))),
        ("Occupation", (0, slice(4, 9))),
        ("Education", (0, slice(9, 12))),
        ("Household income", (1, slice(0, 4))),
        ("Age", (1, slice(4, 8))),
        ("Sex", (1, slice(8, 12))),
    ]
    for index, (domain, place) in enumerate(placements):
        ax = fig.add_subplot(gs[place[0], place[1]])
        draw_domain(ax, domain, points, intervals, contrasts, show_ylabel=index in (0, 3))
        panel_label(ax, chr(ord("a") + index))
    ax_g = fig.add_subplot(gs[2, 2:11])
    draw_same_year_forest(ax_g, contrasts)
    panel_label(ax_g, "g")

    # Statistical assumptions and the fixed-2021 denominator are stated in
    # the manuscript caption; duplicating them as a miniature in-figure note
    # reduced print legibility without adding a second scientific result.

    OUT.mkdir(parents=True, exist_ok=True)
    base = OUT / STEM
    contrasts.to_csv(base.with_name(base.name + "_within_year_contrasts.csv"), index=False)
    for text in fig.findobj(mpl.text.Text):
        text.set_color("#151515")
    fig.savefig(base.with_suffix(".svg"))
    fig.savefig(base.with_suffix(".pdf"))
    fig.savefig(base.with_suffix(".png"), dpi=600)
    fig.savefig(base.with_suffix(".tiff"), dpi=600)
    plt.close(fig)

    significant = contrasts.loc[contrasts["q_bh_within_domain_year"].lt(0.05)]
    print(base.with_suffix(".png"))
    print(f"within_year_contrasts={len(contrasts)}; q<0.05={len(significant)}")
    print(significant[["year", "domain", "comparison_group", "reference_group", "difference_percentage_points", "q_bh_within_domain_year"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
