# Academic Figure Skill Asset Confirmation (verified against assets/figures/)
# a-f grouped weighted boxplots -> GroupedViolin/GroupedBarChart -> param inherit
# g signed subgroup contrast matrix -> CorrelationMatrix -> param inherit
# h temporal income-concentration coefficient plot -> LineTrend -> param inherit

"""Recover component-level social inequality using current, auditable data.

The original submission contained component-specific boxplots and correlations,
but those legacy panels used a superseded component definition and 2006 census
data.  This figure instead recomputes population-weighted distributions from
the revised 2016, 2021 and 2024 area-component files, while the 2024 subgroup
contrasts explicitly use the fixed 2021 census composition.  The concentration
coefficients are reused, without modification, from the audited release.

No individual outcomes, synthetic observations, significance tests or inferred
restaurant-level sustainability scores are introduced.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib as mpl

from v5_cities_visual_system import FONT_FAMILY

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

CATEGORICAL = ["#2166AC", "#B2182B", "#1B7837", "#F1A340", "#762A83", "#666666"]
CATEGORICAL_EXTENDED = [
    "#2166AC", "#B2182B", "#1B7837", "#F1A340", "#762A83", "#666666",
    "#4393C3", "#D6604D", "#5AAE61", "#B35806", "#9970AB", "#999999",
]
DIVERGING = ["#2166AC", "#F7F7F7", "#B2182B"]
SEQUENTIAL = ["#F7FBFF", "#6BAED6", "#08306B"]
ACCENT_RED = "#B2182B"
GREY = "#999999"
BLACK = "#222222"

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


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyogrio
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, to_rgb
from matplotlib.lines import Line2D

from compute_joint_subgroups_planning import GROUPS, census_paths


ROOT = Path(__file__).resolve().parents[1]
COMPONENT_DATA = ROOT / "source_data" / "fig_v4_four_year"
CONCENTRATION_DATA = (
    ROOT / "source_data" / "figS_sdi_structural_inequality_v4" / "component_income_concentration.csv"
)
SOURCE_OUT = ROOT / "source_data" / "fig13_component_social_decomposition_v1"
FIGURE_OUT = Path(os.environ.get("CITIES_FIGURE_DIR", str(ROOT / "figures")))
STEM = "Fig13_Component_Social_Inequality_Weighted_Boxplots_v1_NATURE"
YEARS = (2016, 2021, 2024)

COMPONENTS = [
    ("nutrition_score", "Nutrition", "#D9822B"),
    ("carbon_score", "Carbon", "#2166AC"),
    ("diversity_score", "Cuisine diversity", "#762A83"),
    ("sustainability_score", "Environmental sustainability", "#1B7837"),
    ("hygiene_score", "Hygiene", "#B2182B"),
    ("practice_score", "Practice", "#008B8B"),
]

FOCAL_CONTRASTS = [
    ("Ethnicity", "Filipino", "Chinese", "Filipino − Chinese"),
    ("Occupation", "Elementary", "Managers", "Elementary − Managers"),
    ("Education", "Primary or below", "Post-secondary", "Primary − Post-secondary"),
    ("Household income", "<HK$10k", "≥HK$40k", "<HK$10k − ≥HK$40k"),
    ("Age", "65+", "25–44", "65+ − 25–44"),
    ("Sex", "Female", "Male", "Female − Male"),
]


def blend_with_white(colour: str, strength: float) -> tuple[float, float, float]:
    base = np.asarray(to_rgb(colour), dtype=float)
    return tuple((1.0 - strength) + strength * base)


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantiles: list[float]) -> np.ndarray:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values = values[valid]
    weights = weights[valid]
    if values.size == 0:
        return np.full(len(quantiles), np.nan)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    sorted_weights = weights[order]
    centres = (np.cumsum(sorted_weights) - 0.5 * sorted_weights) / sorted_weights.sum()
    return np.interp(quantiles, centres, sorted_values, left=sorted_values[0], right=sorted_values[-1])


def assign_population_income_quintiles(frame: pd.DataFrame) -> pd.Series:
    income = pd.to_numeric(frame["ma_hh"], errors="coerce")
    population = pd.to_numeric(frame["t_pop"], errors="coerce")
    valid = income.notna() & population.gt(0)
    ordered = frame.loc[valid].copy()
    ordered["_income"] = income.loc[valid].to_numpy(float)
    ordered["_population"] = population.loc[valid].to_numpy(float)
    ordered = ordered.sort_values(["_income", "lsbg"], kind="mergesort")
    rank = (ordered["_population"].cumsum() - 0.5 * ordered["_population"]) / ordered["_population"].sum()
    quintiles = np.minimum((rank * 5).astype(int), 4) + 1
    return pd.Series(quintiles.to_numpy(int), index=ordered.index, dtype="Int64").reindex(frame.index)


def build_distributions() -> pd.DataFrame:
    records = []
    for year in YEARS:
        frame = pd.read_csv(COMPONENT_DATA / f"{year}_lsbg_components.csv", dtype={"lsbg": str})
        frame["income_quintile"] = assign_population_income_quintiles(frame)
        for column, label, _ in COMPONENTS:
            for quintile in range(1, 6):
                subset = frame.loc[frame["income_quintile"].eq(quintile)].copy()
                values = pd.to_numeric(subset[column], errors="coerce").to_numpy(float)
                weights = pd.to_numeric(subset["t_pop"], errors="coerce").to_numpy(float)
                valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
                q05, q25, q50, q75, q95 = weighted_quantile(values, weights, [.05, .25, .50, .75, .95])
                records.append({
                    "year": year,
                    "population_reference_year": 2016 if year == 2016 else 2021,
                    "component": label,
                    "component_column": column,
                    "income_quintile": quintile,
                    "observed_lsbg_n": int(valid.sum()),
                    "population_observed": float(weights[valid].sum()),
                    "q05": q05,
                    "q25": q25,
                    "median": q50,
                    "q75": q75,
                    "q95": q95,
                    "weighted_mean": float(np.average(values[valid], weights=weights[valid])) if valid.any() else np.nan,
                })
    result = pd.DataFrame.from_records(records)
    if len(result) != len(YEARS) * len(COMPONENTS) * 5:
        raise AssertionError("Component/year/quintile combinations were lost")
    if result[["q05", "q25", "median", "q75", "q95"]].isna().any().any():
        raise AssertionError("At least one reported income quintile has no observed component values")
    return result


def build_group_means() -> pd.DataFrame:
    census_file = census_paths()[2021]
    released_means = SOURCE_OUT / "component_domain_group_means_2024.csv"
    if not census_file.is_file():
        if not released_means.is_file():
            raise FileNotFoundError(
                "The restricted 2021 census geography is unavailable and the released "
                "aggregate subgroup-component table is missing."
            )
        return pd.read_csv(released_means)

    frame = pd.read_csv(COMPONENT_DATA / "2024_lsbg_components.csv", dtype={"lsbg": str})
    raw_fields = sorted({field for members in GROUPS.values() for field in members.values()
                         if not field.startswith("derived_")})
    census = pyogrio.read_dataframe(
        census_file, columns=["lsbg", "t_pop", "male_p", "female_p", *raw_fields],
        read_geometry=False,
    )
    census["lsbg"] = census["lsbg"].astype(str)
    census = census.rename(columns={"t_pop": "census_t_pop"})
    census["derived_male_count"] = pd.to_numeric(census["census_t_pop"]) * pd.to_numeric(census["male_p"])
    census["derived_female_count"] = pd.to_numeric(census["census_t_pop"]) * pd.to_numeric(census["female_p"])
    merged = frame.merge(census, on="lsbg", how="inner", validate="one_to_one")
    if len(merged) != len(frame):
        raise AssertionError("2024 component geography does not exactly match the 2021 census geography")
    if not np.allclose(merged["t_pop"], merged["census_t_pop"], rtol=0, atol=1, equal_nan=True):
        raise AssertionError("Component population and fixed 2021 census population differ")

    records = []
    for domain, groups in GROUPS.items():
        for group, field in groups.items():
            weights = pd.to_numeric(merged[field], errors="coerce").fillna(0).to_numpy(float)
            for column, label, _ in COMPONENTS:
                values = pd.to_numeric(merged[column], errors="coerce").to_numpy(float)
                valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
                if not valid.any():
                    raise AssertionError(f"No observed values for {domain}/{group}/{label}")
                records.append({
                    "restaurant_evidence_year": 2024,
                    "population_reference_year": 2021,
                    "domain": domain,
                    "group": group,
                    "component": label,
                    "source_weight_field": field,
                    "weighted_component_mean": float(np.average(values[valid], weights=weights[valid])),
                    "observed_group_weight": float(weights[valid].sum()),
                    "observed_lsbg_n": int(valid.sum()),
                })
    return pd.DataFrame.from_records(records)


def build_contrasts(means: pd.DataFrame) -> pd.DataFrame:
    records = []
    for domain, focal, reference, label in FOCAL_CONTRASTS:
        for _, component, _ in COMPONENTS:
            left = means.loc[(means["domain"] == domain) & (means["group"] == focal)
                             & (means["component"] == component), "weighted_component_mean"]
            right = means.loc[(means["domain"] == domain) & (means["group"] == reference)
                              & (means["component"] == component), "weighted_component_mean"]
            if len(left) != 1 or len(right) != 1:
                raise AssertionError(f"Missing contrast for {domain}/{component}")
            records.append({
                "year": 2024,
                "population_reference_year": 2021,
                "domain": domain,
                "focal_group": focal,
                "reference_group": reference,
                "contrast_label": label,
                "component": component,
                "focal_mean": float(left.iloc[0]),
                "reference_mean": float(right.iloc[0]),
                "difference": float(left.iloc[0] - right.iloc[0]),
            })
    return pd.DataFrame.from_records(records)


def panel_label(ax: plt.Axes, letter: str) -> None:
    ax.text(-.135, 1.07, letter, transform=ax.transAxes, fontsize=9.4,
            fontweight="bold", color=BLACK, ha="left", va="top", clip_on=False)


def draw_box_panel(ax: plt.Axes, data: pd.DataFrame, component: str, colour: str, letter: str) -> None:
    panel_label(ax, letter)
    ax.set_title(component, loc="left", fontweight="normal", color=BLACK, pad=5)
    positions = []
    stats = []
    colours = []
    for quintile in range(1, 6):
        for idx, year in enumerate(YEARS):
            row = data.loc[(data["component"] == component) & (data["year"] == year)
                           & (data["income_quintile"] == quintile)].iloc[0]
            positions.append(quintile + (idx - 1) * .235)
            colours.append(blend_with_white(colour, (.32, .58, .88)[idx]))
            stats.append({"med": row["median"], "q1": row["q25"], "q3": row["q75"],
                          "whislo": row["q05"], "whishi": row["q95"], "fliers": []})

    artists = ax.bxp(stats, positions=positions, widths=.20, patch_artist=True,
                     showfliers=False, manage_ticks=False,
                     medianprops={"color": BLACK, "linewidth": .75},
                     whiskerprops={"color": "#48535A", "linewidth": .52},
                     capprops={"color": "#48535A", "linewidth": .52})
    for box, face in zip(artists["boxes"], colours):
        box.set(facecolor=face, edgecolor=face, linewidth=.65, alpha=.96)

    ax.set_xlim(.53, 5.47)
    ax.set_ylim(-.035, 1.035)
    ax.set_xticks(range(1, 6), ["Q1", "Q2", "Q3", "Q4", "Q5"])
    ax.set_yticks([0, .5, 1])
    ax.tick_params(axis="both", labelsize=6.1, pad=1.5)
    if letter in {"a", "d"}:
        ax.set_ylabel("Area score (0–1)", fontsize=6.8)
    if letter in {"d", "e", "f"}:
        ax.set_xlabel("Area-income quintile", fontsize=6.6, labelpad=2)
    handles = [Line2D([0], [0], marker="s", linestyle="", markersize=4.6,
                      markerfacecolor=blend_with_white(colour, strength), markeredgewidth=0,
                      label=str(year)) for year, strength in zip(YEARS, (.32, .58, .88))]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(.50, 1.02),
              ncol=3, fontsize=5.25, handlelength=.70, handletextpad=.25,
              columnspacing=.55, borderaxespad=.05, labelspacing=.20)


def draw_contrast_heatmap(ax: plt.Axes, contrasts: pd.DataFrame) -> None:
    panel_label(ax, "g")
    ax.set_title("2024 subgroup contrast in each quality component", loc="left",
                 fontsize=7.8, fontweight="normal", pad=5)
    labels = [item[3] for item in FOCAL_CONTRASTS]
    components = [item[1] for item in COMPONENTS]
    matrix = contrasts.pivot(index="contrast_label", columns="component", values="difference")
    matrix = matrix.reindex(index=labels, columns=components)
    maximum = max(.12, float(np.abs(matrix.to_numpy(float)).max()))
    cmap = LinearSegmentedColormap.from_list("subgroup_diverging", DIVERGING)
    image = ax.imshow(matrix.to_numpy(float), cmap=cmap,
                      norm=TwoSlopeNorm(vmin=-maximum, vcenter=0, vmax=maximum),
                      interpolation="none", aspect="auto")
    ax.set_xticks(range(len(components)), ["Nutrition", "Carbon", "Cuisine\ndiversity",
                                            "Environmental\nsustainability", "Hygiene", "Practice*"], fontsize=5.45)
    ax.set_yticks(range(len(labels)), labels, fontsize=5.8)
    ax.tick_params(axis="both", length=0, pad=3)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for iy in range(matrix.shape[0]):
        for ix in range(matrix.shape[1]):
            value = float(matrix.iloc[iy, ix])
            ax.text(ix, iy, f"{value:+.02f}", ha="center", va="center", fontsize=6.0,
                    color=BLACK, fontweight="bold" if abs(value) >= .07 else "normal")
    cax = ax.inset_axes([0, -.25, 1, .06])
    colorbar = plt.colorbar(image, cax=cax, orientation="horizontal")
    colorbar.set_ticks([-maximum, 0, maximum], labels=[f"−{maximum:.2f}", "0", f"+{maximum:.2f}"])
    colorbar.ax.tick_params(labelsize=5.6, length=2, pad=1)
    colorbar.outline.set_linewidth(.45)
    colorbar.set_label("Mean difference", fontsize=5.8, labelpad=2)


def draw_concentration(ax: plt.Axes) -> None:
    panel_label(ax, "h")
    ax.set_title("Income concentration across four years", loc="left",
                 fontsize=7.8, fontweight="normal", pad=5)
    concentration = pd.read_csv(CONCENTRATION_DATA)
    values = []
    for row, (_, component, colour) in enumerate(COMPONENTS):
        selection = concentration.loc[concentration["component"] == component]
        if component == "Environmental sustainability":
            selection = concentration.loc[concentration["component"] == "Sustainability text"]
        if component == "Practice":
            selection = concentration.loc[concentration["component"] == "Practice tag*"]
        if selection.empty:
            raise AssertionError(f"No audited income concentration rows for {component}")
        selection = selection.sort_values("year")
        xs = selection["concentration_index"].to_numpy(float)
        y = len(COMPONENTS) - row - 1
        ax.plot([xs.min(), xs.max()], [y, y], color=colour, lw=4.8, alpha=.13,
                solid_capstyle="round", zorder=1)
        ax.plot([xs.min(), xs.max()], [y, y], color=blend_with_white(colour, .62), lw=1.0,
                solid_capstyle="round", zorder=2)
        for index, record in enumerate(selection.itertuples()):
            marker = ("o", "s", "D", "^")[index]
            ax.scatter(record.concentration_index, y, s=17 + 3 * index, marker=marker,
                       color=blend_with_white(colour, .44 + .16 * index), edgecolor="white",
                       linewidth=.42, zorder=3)
        values.extend(xs.tolist())
    ax.axvline(0, color="#657279", lw=.6, linestyle=(0, (3, 2)), zorder=0)
    concentration_labels = [
        "Environmental\nsustainability" if item[1] == "Environmental sustainability" else item[1]
        for item in COMPONENTS
    ]
    ax.set_yticks(np.arange(len(COMPONENTS))[::-1], concentration_labels, fontsize=5.9)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.tick_params(axis="x", labelsize=5.8, pad=1)
    ax.set_xlabel("Income concentration index", fontsize=6.4, labelpad=2)
    ax.set_xlim(min(-.09, min(values) - .015), max(.12, max(values) + .02))
    handles = [Line2D([0], [0], marker=marker, linestyle="", markerfacecolor="#536069",
                      markeredgecolor="white", markeredgewidth=.4, markersize=4.2, label=str(year))
               for marker, year in zip(("o", "s", "D", "^"), (2011, 2016, 2021, 2024))]
    ax.legend(handles=handles, loc="upper right", ncol=2, fontsize=5.15,
              handlelength=.8, handletextpad=.27, columnspacing=.75, borderaxespad=.25)


def build_figure(distributions: pd.DataFrame, contrasts: pd.DataFrame) -> Path:
    mm = 1 / 25.4
    fig = plt.figure(figsize=(183 * mm, 166 * mm), facecolor="white")
    grid = fig.add_gridspec(3, 6, height_ratios=[1, 1, 1.22],
                            left=.076, right=.988, top=.956, bottom=.115,
                            wspace=.91, hspace=.57)
    for index, (_, label, colour) in enumerate(COMPONENTS):
        row, col = divmod(index, 3)
        ax = fig.add_subplot(grid[row, col * 2:(col + 1) * 2])
        draw_box_panel(ax, distributions, label, colour, "abcdef"[index])
    draw_contrast_heatmap(fig.add_subplot(grid[2, 0:4]), contrasts)
    concentration_axis = fig.add_subplot(grid[2, 4:6])
    position = concentration_axis.get_position()
    # Reserve a clean gutter for the two-line component label so panel h never
    # intrudes into the last heatmap column in panel g.
    concentration_axis.set_position([position.x0 + .032, position.y0, position.width - .032, position.height])
    draw_concentration(concentration_axis)
    # Provenance and statistical definitions are provided once in the figure
    # caption, avoiding a miniature duplicate note inside the artwork.
    FIGURE_OUT.mkdir(parents=True, exist_ok=True)
    base = FIGURE_OUT / STEM
    save_cns_figure(fig, str(base))
    fig.savefig(base.with_name(base.name + "_600dpi").with_suffix(".png"), dpi=600,
                bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return base.with_name(base.name + "_600dpi").with_suffix(".png")


def main() -> None:
    SOURCE_OUT.mkdir(parents=True, exist_ok=True)
    distributions = build_distributions()
    group_means = build_group_means()
    contrasts = build_contrasts(group_means)
    distributions.to_csv(SOURCE_OUT / "weighted_component_income_distributions.csv", index=False)
    group_means.to_csv(SOURCE_OUT / "component_domain_group_means_2024.csv", index=False)
    contrasts.to_csv(SOURCE_OUT / "component_domain_reference_contrasts_2024.csv", index=False)
    output = build_figure(distributions, contrasts)
    manifest = {
        "figure": str(output.relative_to(ROOT)),
        "component_files": [str((COMPONENT_DATA / f"{year}_lsbg_components.csv").relative_to(ROOT))
                            for year in YEARS],
        "census_composition": "2021 observed census composition for 2024 component-group contrasts",
        "income_concentration": str(CONCENTRATION_DATA.relative_to(ROOT)),
        "distribution_rows": int(len(distributions)),
        "group_mean_rows": int(len(group_means)),
        "reference_contrast_rows": int(len(contrasts)),
        "scientific_boundary": "Ecological area-component scores; no individual dining, expenditure, causal claim, fabricated p-value or 2006 legacy outcome.",
        "practice_provenance": "Undated snapshot; not interpreted as historical programme adoption.",
    }
    (SOURCE_OUT / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
