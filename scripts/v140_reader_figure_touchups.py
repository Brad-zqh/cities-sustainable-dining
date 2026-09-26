"""Restricted, presentation-only V140 redraws from the fixed V135 inputs.

This imports the archived/V138 renderers and changes panel geometry, heatmap
cell borders, key placement, and small-text treatment. It does not recompute
or replace any empirical estimate. Do not publish these restricted exports.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
from matplotlib.text import Text

import v138_reproduce_visual_polish as bridge


ROOT = bridge.ROOT
OUT = ROOT / "outputs/restricted/v140_reader_figure_touchups"
RAMP = ["#78A9C7", "#D3E4EC", "#FAFAFA", "#F1D0D2", "#E1848C"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def typography(fig: plt.Figure, floor: float = 6.7) -> None:
    """Keep only panel letters bold and raise the small-text floor."""
    for item in fig.findobj(Text):
        value = item.get_text().strip()
        if not value:
            continue
        panel_letter = len(value) == 1 and value in "abcdefghijkl" and item.get_fontweight() in ("bold", 700)
        item.set_fontweight("bold" if panel_letter else "normal")
        if item.get_fontsize() < floor:
            item.set_fontsize(floor)


def heatmap_grid(ax: plt.Axes, shape: tuple[int, int]) -> None:
    rows, cols = shape
    ax.set_xticks(np.arange(-.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-.5, rows, 1), minor=True)
    ax.grid(which="minor", color="#66747A", linewidth=.32, alpha=.62)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#253039")
        spine.set_linewidth(.55)


def key_left(image, bounds: tuple[float, float, float, float], *, title: str = "", label: str | None = None):
    fig = image.axes.figure
    previous = image.colorbar
    ticks = list(previous.get_ticks()) if previous is not None else None
    if previous is not None:
        if not title:
            title = previous.ax.get_title()
        if label is None:
            label = previous.ax.xaxis.label.get_text() or previous.ax.yaxis.label.get_text()
        previous.remove()
    cax = fig.add_axes(bounds)
    cb = fig.colorbar(image, cax=cax, orientation="vertical", ticks=ticks)
    cb.ax.yaxis.set_ticks_position("left")
    cb.ax.yaxis.set_label_position("left")
    cb.ax.tick_params(labelsize=6.7, length=2, width=.5, pad=1.5)
    cb.outline.set_visible(True)
    cb.outline.set_color("#253039")
    cb.outline.set_linewidth(.55)
    if title:
        cb.ax.set_title(title, fontsize=6.7, pad=3)
    if label:
        cb.set_label(label, fontsize=6.7, labelpad=4)
    return cb


def export(fig: plt.Figure, stem: str) -> dict[str, str]:
    OUT.mkdir(parents=True, exist_ok=True)
    results = {}
    for ext in ("png", "pdf", "svg"):
        target = OUT / f"{stem}.{ext}"
        fig.savefig(target, dpi=600, facecolor="white", bbox_inches=None)
        results[target.name] = sha256(target)
    return results


def render_15() -> dict:
    market, source = bridge.load_original("fig06_price_market")
    market.DATA_DIR = bridge.PRICE_MARKET
    market.configure_matplotlib()
    frames = market.load_frames()
    regression = pd.read_csv(market.DATA_DIR / "quality_market_decomposition.csv")
    summary = pd.read_csv(market.DATA_DIR / "price_market_summary.csv")
    breaks = market.pooled_quantile_breaks(frames, "low_price_share", quantiles=5)
    fig = plt.figure(figsize=(183 * market.MM, 238 * market.MM), facecolor="white")
    maps = fig.add_gridspec(2, 2, left=.070, right=.958, bottom=.500, top=.970,
                            wspace=.072, hspace=.055)
    for index, year in enumerate(market.YEARS):
        ax = fig.add_subplot(maps[index // 2, index % 2])
        cmap = LinearSegmentedColormap.from_list(f"price_share_{year}", market.YEAR_RAMPS[year], N=256)
        norm = BoundaryNorm(breaks, cmap.N)
        market.plot_choropleth(ax, frames[year], "low_price_share", cmap, norm,
                               title=str(year), show_cartography=True)
        cax = ax.inset_axes([1.006, .10, .030, .81], transform=ax.transAxes)
        cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                          orientation="vertical", boundaries=breaks, ticks=breaks)
        cb.ax.yaxis.set_ticks_position("right")
        cb.ax.set_yticklabels([f"{max(0, 100 * value):.0f}" for value in breaks])
        cb.ax.set_title("%", fontsize=6.7, pad=1.5)
        cb.ax.tick_params(labelsize=6.7, length=1.5, width=.45, pad=1)
        cb.outline.set_linewidth(.45)
        if index == 0:
            market.add_panel_label(ax, "a", x=-.09, y=1.02)
    ax_b = fig.add_axes([.100, .335, .345, .135])
    market.plot_income_attenuation(ax_b, regression)
    ax_b.legend(frameon=False, loc="upper center", bbox_to_anchor=(.50, -.50),
                ncol=2, fontsize=6.7, handletextpad=.4, columnspacing=.8)
    ax_c = fig.add_axes([.655, .335, .275, .135])
    market.plot_full_model_heatmap(ax_c, regression)
    image = ax_c.images[0]
    image.set_cmap(LinearSegmentedColormap.from_list("shared_heatmap", RAMP))
    ax_c.set_position([.655, .335, .275, .135])
    heatmap_grid(ax_c, image.get_array().shape)
    key_left(image, (.505, .335, .012, .135), label="Association with area quality score")
    count = fig.add_axes([.100, .055, .350, .160])
    share = fig.add_axes([.590, .055, .350, .160])
    market.plot_price_supply_trajectory(count, share, summary)
    count.tick_params(axis="x", labelbottom=True)
    count.set_xlabel("Year")
    share.set_title("Lower-price share", loc="left", pad=4, fontweight="normal")
    typography(fig)
    outputs = export(fig, "Fig15_Price_Composition_Socioeconomic_Associations")
    plt.close(fig)
    return {"renderer": str(source), "renderer_sha256": sha256(source),
            "input_sha256": {p.name: sha256(p) for p in [
                market.DATA_DIR / "quality_market_decomposition.csv",
                market.DATA_DIR / "price_market_summary.csv"]},
            "layout": "maps 2x2; b/c same row; d count/share side by side",
            "pooled_map_breaks": breaks.tolist(), "output_sha256": outputs}


def render_03() -> dict:
    study, source = bridge.load_original("render_study_benchmark_composite")
    study.shared.OUT = OUT
    from matplotlib.collections import PathCollection

    original_scatter = study.benchmark.draw_scatter
    original_bias = study.benchmark.draw_bland_altman
    original_robust = study.benchmark.draw_robustness
    original_export = study.shared.export

    def log_axis(ax, label):
        forward = lambda value: np.log1p(np.maximum(value, 0))
        inverse = lambda value: np.expm1(value)
        ax.set_xscale("function", functions=(forward, inverse))
        ax.set_xticks([0, 5, 20, 50, 100] if label == "agreement" else [0, 5, 20, 50, 80])
        if label == "agreement":
            ax.set_yscale("function", functions=(forward, inverse))
            ax.set_yticks([0, 5, 20, 50, 100])
            ax.set_xlabel("OpenRice opportunity\n(per 1,000; log-scaled axis)")
            ax.set_ylabel("FEHD opportunity\n(per 1,000; log-scaled axis)")
        else:
            ax.set_xlabel("Mean opportunity\n(log-scaled axis)")

    def presented_scatter(ax, data, summary):
        original_scatter(ax, data, summary)
        for collection in ax.collections:
            if isinstance(collection, PathCollection):
                collection.set_alpha(.28)
                collection.set_sizes([5.0])
        log_axis(ax, "agreement")
        ax.legend(loc="upper left", frameon=False, markerscale=2,
                  handletextpad=.35, borderaxespad=.25)

    def presented_bias(ax, data):
        original_bias(ax, data)
        previous_legend = ax.get_legend()
        legend_handles = previous_legend.legend_handles
        legend_labels = [item.get_text() for item in previous_legend.get_texts()]
        for collection in ax.collections:
            if isinstance(collection, PathCollection):
                collection.set_alpha(.24)
                collection.set_sizes([4.8])
        log_axis(ax, "bias")
        ax.legend(legend_handles, legend_labels, loc="upper left", frameon=False, fontsize=6.7,
                  handlelength=1.4, handletextpad=.35)

    def presented_robust(ax, summary):
        original_robust(ax, summary)
        rows = ["pearson", "spearman", "population_weighted_flag_agreement", "jaccard"]
        for y, key in zip(range(3, -1, -1), rows):
            def score(threshold):
                block = summary[str(threshold)]
                if key in ("pearson", "spearman"):
                    return float(block["correlation"]["fehd_calibrated"][key])
                return float(block["low_access_comparison"]["fehd_calibrated"][key])
            ax.plot([score(10), score(15)], [y + .10, y - .10],
                    color="#A7B1B8", linewidth=.75, zorder=1)
        ax.set_xlim(.78, 1.07)

    def presentation_export(fig, stem, content):
        panels = [next(ax for ax in fig.axes if ax.get_title(loc="left") == title)
                  for title in ("LSBG agreement", "Bias and agreement limits",
                                "Availability diagnostics")]
        for ax, position in zip(panels, ([.080, .070, .250, .245],
                                         [.370, .070, .250, .245],
                                         [.735, .070, .230, .245])):
            ax.set_position(position)
            ax.tick_params(labelsize=7.0)
            ax.title.set_fontsize(8.1)
            ax.xaxis.label.set_fontsize(7.0)
            ax.yaxis.label.set_fontsize(7.0)
        log_axis(panels[0], "agreement")
        log_axis(panels[1], "bias")
        typography(fig)
        return original_export(fig, stem, content)

    study.benchmark.draw_scatter = presented_scatter
    study.benchmark.draw_bland_altman = presented_bias
    study.benchmark.draw_robustness = presented_robust
    study.shared.export = presentation_export
    study.draw()
    name = study.STEM
    data = study.benchmark.SOURCE_BUNDLE / "lsbg_openrice_fehd_count_access.csv"
    summary = study.benchmark.SOURCE_BUNDLE / "comparison_summary.json"
    return {"renderer": str(source), "renderer_sha256": sha256(source),
            "benchmark_input_sha256": {p.name: sha256(p) for p in (data, summary)},
            "diagnostic_axis": "log(1+x) display only; underlying values and benchmark estimates unchanged",
            "output_sha256": {f"{name}.{ext}": sha256(OUT / f"{name}.{ext}")
                              for ext in ("png", "pdf", "svg")}}


def render_12_16(number: str) -> dict:
    evidence, source = bridge.load_original("render_evidence_composites")
    evidence.OUT = OUT
    original_export = evidence.export

    def presentation_export(fig, stem, content):
        ax = next(ax for ax in fig.axes if ax.images)
        image = ax.images[0]
        image.set_cmap(LinearSegmentedColormap.from_list("shared_heatmap", RAMP))
        if number == "12":
            ax.set_position([.650, .065, .255, .165])
            key_left(image, (.490, .065, .012, .165), title="Coverage (%)")
        else:
            fig.set_size_inches(183 / 25.4, 235 / 25.4, forward=True)
            ax.set_position([.640, .052, .260, .165])
            key_left(image, (.555, .052, .012, .165), title="Zero (%)")
            # Move the bottom-row screen up to use the space gained by the taller row.
            panel_j = next(a for a in fig.axes if a.get_title(loc="left") == "Sequential opportunity screen")
            panel_j.set_position([.105, .052, .350, .165])
            legends = fig.legends
            if legends:
                legends[0].set_bbox_to_anchor((.51, .744))
        heatmap_grid(ax, image.get_array().shape)
        typography(fig)
        return original_export(fig, stem, content)

    evidence.export = presentation_export
    if number == "12":
        evidence.sdi.base.YEAR_RAMPS.update(evidence.sdi.MAP_RAMPS)
        evidence.sdi.base.load_maps = lambda: _fixed_sdi_maps(evidence)
        evidence.ROOT = bridge.TemporalRoot(evidence.ROOT)
        evidence.sdi_composite()
    else:
        evidence.joint.DATA = bridge.JOINT
        evidence.rain.DATA = bridge.JOINT
        evidence.joint_composite()
    name = ("Fig12_Restaurant_Quality_SDI_Evolution_Coverage" if number == "12"
            else "Fig16_Joint_Quality_Walking_Price_Opportunity")
    return {"renderer": str(source), "renderer_sha256": sha256(source),
            "output_sha256": {f"{name}.{ext}": sha256(OUT / f"{name}.{ext}")
                              for ext in ("png", "pdf", "svg")}}


def _fixed_sdi_maps(evidence):
    import geopandas as gpd
    frames = {}
    for year in (2011, 2016, 2021, 2024):
        frame = gpd.read_file(bridge.AREA / f"{year}_lsbg_price_market.gpkg").to_crs(2326)
        field = evidence.sdi.base.FIELD
        if field not in frame and "sdi_equal_arithmetic" in frame:
            frame[field] = frame["sdi_equal_arithmetic"]
        frame[field] = pd.to_numeric(frame[field], errors="coerce")
        frame["restaurant_n"] = pd.to_numeric(frame["restaurant_n"], errors="coerce").fillna(0)
        frame["geometry"] = frame.geometry.simplify(10, preserve_topology=True)
        frames[year] = frame
    return frames


def render_18b() -> dict:
    scale, source = bridge.load_original("render_scale_weight_composite")
    scale.SOURCE = bridge.AGGREGATION
    scale.maps.DATA_DIR = bridge.VariantMapSource()
    scale.shared.OUT = OUT
    original_export = scale.shared.export

    def presentation_export(fig, stem, audit):
        if stem == "Fig9a-h_Scale_Maps_Two_Per_Row":
            return original_export(fig, "Fig18a_Spatial_Scale_Sensitivity_Maps", audit)
        if stem != "Fig9i-l_Scale_Diagnostics_Continued":
            raise AssertionError(stem)
        ax = next(a for a in fig.axes if a.images)
        image = ax.images[0]
        image.set_cmap(LinearSegmentedColormap.from_list("shared_heatmap", RAMP))
        ax.set_position([.615, .105, .325, .345])
        heatmap_grid(ax, image.get_array().shape)
        key_left(image, (.475, .105, .012, .345), label="Spearman ρ (LSBG–DCCA)")
        # Expand the right column so the Word image does not carry a blank strip.
        top_right = next(a for a in fig.axes if a.get_title(loc="left") == "Rank agreement")
        top_right.set_position([.560, .570, .385, .345])
        typography(fig, floor=7.4)
        return original_export(fig, "Fig18b_Weighting_And_Cross_Scale_Diagnostics", audit)

    scale.shared.export = presentation_export
    scale.build(paired_pages=True)
    name = "Fig18b_Weighting_And_Cross_Scale_Diagnostics"
    return {"renderer": str(source), "renderer_sha256": sha256(source),
            "output_sha256": {f"{name}.{ext}": sha256(OUT / f"{name}.{ext}")
                              for ext in ("png", "pdf", "svg")}}


def render_14() -> dict:
    composite, source = bridge.load_original("render_quality_inequality_composite")
    components = composite.comp
    import compute_joint_subgroups_planning as subgroup_source
    composite.STRUCT = ROOT / "outputs/restricted/v135_fixed_nutrition_structural_inequality_999"
    composite.DETAIL = ROOT / "outputs/restricted/v135_fixed_nutrition_component_social"
    composite.STEM = "Fig14_Restaurant_Quality_Inequality_FIXED_NUTRITION"
    components.COMPONENT_DATA = bridge.AREA
    components.CONCENTRATION_DATA = composite.STRUCT / "component_income_concentration.csv"
    components.SOURCE_OUT = composite.DETAIL
    components.census_paths = lambda: subgroup_source.census_paths(ROOT.parent)
    composite.shared.OUT = OUT
    original_export = composite.shared.export

    def presentation_export(fig, stem, content):
        fig.set_size_inches(183 / 25.4, 252 / 25.4, forward=True)
        panels = {ax.get_title(loc="left"): ax for ax in fig.axes if ax.images}
        if set(panels) != {"2024 subgroup contrasts", "Component concentration"}:
            raise AssertionError(f"Unexpected Fig14 matrices: {list(panels)}")
        for title, position, colorbar, label in (
            ("2024 subgroup contrasts", (.290, .090, .280, .180), (.050, .090, .012, .180),
             ""),
            ("Component concentration", (.750, .090, .220, .180), (.625, .090, .012, .180),
             "Income concentration index"),
        ):
            ax = panels[title]
            image = ax.images[0]
            image.set_cmap(LinearSegmentedColormap.from_list("shared_heatmap", RAMP))
            ax.set_position(position)
            heatmap_grid(ax, image.get_array().shape)
            key_left(image, colorbar, title="Mean difference" if not label else "", label=label)
        typography(fig)
        return original_export(fig, stem, content)

    composite.shared.export = presentation_export
    composite.render()
    name = composite.STEM
    return {"renderer": str(source), "renderer_sha256": sha256(source),
            "output_sha256": {f"{name}.{ext}": sha256(OUT / f"{name}.{ext}")
                              for ext in ("png", "pdf", "svg")}}


def render_19() -> dict:
    planning, source = bridge.load_original("render_planning_composite")
    planning.source.JOINT_SRC = bridge.DOWNSTREAM
    # 1645/1744 LSBGs have a true zero; give them a discernible pale tint.
    for strategy in planning.source.STRATEGIES:
        planning.MAP_RAMPS[strategy][0] = "#EDF1F3"
    original_overlap = planning.overlap

    def presented_overlap(fig, ax, cax, selections):
        matrix = original_overlap(fig, ax, cax, selections)
        image = ax.images[0]
        image.set_cmap(LinearSegmentedColormap.from_list("shared_heatmap", RAMP))
        ax.set_position([.615, .045, .275, .155])
        heatmap_grid(ax, image.get_array().shape)
        key_left(image, (.495, .045, .012, .155), title="Jaccard")
        for item in fig.texts:
            if item.get_text() == "h":
                item.set_x(.587)
            elif item.get_text() == "Selection overlap · K=10":
                item.set_x(.615)
        typography(fig)
        return matrix

    planning.overlap = presented_overlap
    old_argv = sys.argv
    try:
        sys.argv = [str(source), "--output-dir", str(OUT)]
        planning.main()
    finally:
        sys.argv = old_argv
    audit_path = OUT / "planning_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit.update(heatmap_internal_lines=True,
                 heatmap_colorbar_side="left",
                 zero_lsbgs=1645,
                 positive_lsbgs=99,
                 map_zero_style="visible pale neutral; common 0–100% numerical scale preserved")
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    name = "Fig19_Planning_Strategies"
    return {"renderer": str(source), "renderer_sha256": sha256(source),
            "selection_jaccard": audit["selection_jaccard"],
            "output_sha256": {f"{name}.{ext}": sha256(OUT / f"{name}.{ext}")
                              for ext in ("png", "pdf", "svg")}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=("03", "12", "14", "15", "16", "18b", "19"), required=True)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    before = bridge.assert_inputs()
    result = {"03": render_03, "12": lambda: render_12_16("12"), "14": render_14,
              "15": render_15,
              "16": lambda: render_12_16("16"), "18b": render_18b,
              "19": render_19}[args.only]()
    if before != bridge.assert_inputs():
        raise AssertionError("Fixed empirical inputs changed during V140 rendering")
    result.update(status="RESTRICTED_PRESENTATION_ONLY", figure=args.only,
                  fixed_input_sha256=before, public_release_authorized=False,
                  no_empirical_recomputation=True)
    (OUT / f"V140_FIG{args.only}_PRESENTATION_AUDIT.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
