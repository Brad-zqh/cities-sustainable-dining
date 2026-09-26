"""Re-render V134 empirical figure layouts with fixed-Nutrition V135 inputs.

This restricted/local bridge deliberately reuses the original V134 renderers,
including their panel geometry, colour systems and map breaks. It does not
recompute an empirical estimand or confer public-release approval.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import argparse
from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
HISTORICAL = Path(
    os.environ.get(
        "CITIES_V134_VISUAL_SCRIPTS",
        str(ROOT / "scripts"),
    )
)
OUT = ROOT / "outputs/restricted/v138_visual_polish"
POLISHED = ROOT / "scripts/v138_visual_polish"
JOINT = ROOT / "outputs/restricted/v134_fixed_nutrition_joint_access"
DOWNSTREAM = ROOT / "outputs/restricted/v134_fixed_nutrition_joint_downstream"
AREA = ROOT / "outputs/restricted/v134_fixed_nutrition_full_chain"
TEMPORAL = ROOT / "outputs/restricted/v135_fixed_nutrition_temporal_uncertainty"
PRICE_MARKET = ROOT / "outputs/restricted/v135_fixed_nutrition_price_market"
CONTRASTS = ROOT / "outputs/restricted/v135_fixed_nutrition_subgroup_contrasts"
AGGREGATION = ROOT / "outputs/restricted/v135_fixed_nutrition_aggregation_variants"
OLD_ATLAS_GEOMETRY = HISTORICAL.parent / "source_data/fig_v4_four_year"


class AreaSource:
    """Redirect only the V134 atlas filename to the fixed area GeoPackage."""

    def __truediv__(self, name: str) -> Path:
        if not name.endswith("_lsbg_components.gpkg"):
            raise ValueError(f"Unexpected V134 atlas input: {name}")
        return AREA / name.replace("_lsbg_components.gpkg", "_lsbg_price_market.gpkg")


class TemporalRoot:
    """Keep original renderer logic while redirecting its temporal CSV folder."""

    def __init__(self, original: Path):
        self.original = original

    def __truediv__(self, name: str) -> Path:
        if name == "source_data/figS_temporal_uncertainty_v4":
            return TEMPORAL
        return self.original / name


class VariantMapSource:
    """Map source required by the old diagnostics renderer's setup."""

    def __truediv__(self, name: str) -> Path:
        if not name.endswith("_components.gpkg"):
            raise ValueError(f"Unexpected scale diagnostics map input: {name}")
        return AGGREGATION / name.replace("_components.gpkg", "_sensitivity.gpkg")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_original(name: str):
    path = POLISHED / f"{name}.py" if (POLISHED / f"{name}.py").is_file() else HISTORICAL / f"{name}.py"
    if not path.is_file():
        raise FileNotFoundError(f"Original V134 renderer unavailable: {path}")
    if str(HISTORICAL) not in sys.path:
        sys.path.insert(0, str(HISTORICAL))
    historic_figures = HISTORICAL.parent / "figures"
    if historic_figures.is_dir() and str(historic_figures) not in sys.path:
        sys.path.insert(0, str(historic_figures))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def assert_inputs() -> dict[str, str]:
    required = [
        *[AREA / f"{year}_lsbg_price_market.gpkg" for year in (2011, 2016, 2021, 2024)],
        *[AREA / f"{year}_lsbg_components.csv" for year in (2011, 2016, 2021, 2024)],
        *[TEMPORAL / name for name in (
            "annual_block_bootstrap_intervals.csv",
            "change_from_2011_intervals.csv",
            "metric_coverage_audit.csv",
        )],
        *[PRICE_MARKET / name for name in (
            "quality_market_decomposition.csv", "price_market_summary.csv",
        )],
        *[PRICE_MARKET / f"{year}_lsbg_price_market.gpkg" for year in (2011, 2016, 2021, 2024)],
        JOINT / "joint_access_point_estimates.csv",
        JOINT / "joint_access_dcca_block_intervals.csv",
        JOINT / "lsbg_joint_quality_affordable_access.csv",
        *[JOINT / f"{year}_joint_access_geometry.gpkg" for year in (2016, 2021, 2024)],
        DOWNSTREAM / "scenario_results.csv",
        DOWNSTREAM / "selected_planning_nodes.csv",
        DOWNSTREAM / "priority_population_definition.csv",
        *[DOWNSTREAM / name for name in (
            "joint_subgroup_point_estimates.csv",
            "joint_subgroup_dcca_block_intervals.csv",
            "joint_subgroup_dcca_block_replicates.csv",
        )],
        CONTRASTS / "within_year_contrasts.csv",
        *[AGGREGATION / name for name in (
            "weight_schemes.csv", "variant_agreement.csv", "maup_agreement.csv",
        )],
        *[AGGREGATION / f"{year}_{scale}_sensitivity.gpkg"
          for year in (2011, 2016, 2021, 2024) for scale in ("lsbg", "dcca")],
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    for year in (2011, 2016, 2021, 2024):
        old_geometry = OLD_ATLAS_GEOMETRY / f"{year}_lsbg_components.gpkg"
        if not old_geometry.is_file():
            raise FileNotFoundError(old_geometry)
    points = pd.read_csv(JOINT / "joint_access_point_estimates.csv")
    annual = pd.read_csv(TEMPORAL / "annual_block_bootstrap_intervals.csv")
    scenarios = pd.read_csv(DOWNSTREAM / "scenario_results.csv")
    expected_sdi = {2011: .3769, 2016: .3934, 2021: .4162, 2024: .4257}
    for year, expected in expected_sdi.items():
        row = annual.loc[annual.year.eq(year) & annual.metric.eq("equal_six_strict")]
        if len(row) != 1 or abs(float(row.iloc[0].population_weighted_mean) - expected) > .0001:
            raise AssertionError(f"Fixed SDI series drift: {year}")
    for year, expected_zero in [(2016, .3598), (2021, .2250), (2024, .1964)]:
        row = points.loc[
            points.year.eq(year)
            & points.outcome.eq("joint_access")
            & points.threshold_min.eq(15)
            & points.price_ceiling_hkd.eq(100)
            & points.destination_area_sdi_threshold.eq(.45)
        ]
        if len(row) != 1 or abs(float(row.iloc[0].zero_access_population_share) - expected_zero) > .0001:
            raise AssertionError(f"Fixed joint-access baseline drift: {year}")
    equity = scenarios.loc[
        scenarios.scenario.eq("zero_affordability_gap") & scenarios.budget_sites.eq(10)
    ]
    population = scenarios.loc[
        scenarios.scenario.eq("population_reach") & scenarios.budget_sites.eq(10)
    ]
    if len(equity) != 1 or len(population) != 1:
        raise AssertionError("Missing fixed K=10 planning rows")
    if abs(100 * float(equity.iloc[0].zero_affordability_priority_reached_share) - 49.699852691809804) > 1e-8:
        raise AssertionError("Fixed K=10 equity value drift")
    if abs(100 * float(population.iloc[0].zero_affordability_priority_reached_share) - 4.305363557162996) > 1e-8:
        raise AssertionError("Fixed K=10 population value drift")
    return {str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p) for p in required}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=["12", "13", "15", "16", "17", "18", "19"], required=True)
    args = parser.parse_args()
    input_hashes = assert_inputs()
    OUT.mkdir(parents=True, exist_ok=True)
    renderer_paths = {}

    if args.only in ("12", "16"):
        evidence, evidence_path = load_original("render_evidence_composites")
        renderer_paths[evidence_path.name] = sha256(evidence_path)
        evidence.OUT = OUT
        if args.only == "12":
            original_root = evidence.ROOT

            def fixed_maps():
                frames = {}
                for year in (2011, 2016, 2021, 2024):
                    frame = gpd.read_file(AREA / f"{year}_lsbg_price_market.gpkg").to_crs(2326)
                    if evidence.sdi.base.FIELD not in frame and "sdi_equal_arithmetic" in frame:
                        frame[evidence.sdi.base.FIELD] = frame["sdi_equal_arithmetic"]
                    frame[evidence.sdi.base.FIELD] = pd.to_numeric(
                        frame[evidence.sdi.base.FIELD], errors="coerce"
                    )
                    frame["restaurant_n"] = pd.to_numeric(
                        frame["restaurant_n"], errors="coerce"
                    ).fillna(0)
                    frame["geometry"] = frame.geometry.simplify(10, preserve_topology=True)
                    frames[year] = frame
                return frames

            evidence.sdi.base.load_maps = fixed_maps
            evidence.ROOT = TemporalRoot(original_root)
            evidence.sdi_composite()
        else:
            evidence.joint.DATA = JOINT
            evidence.rain.DATA = JOINT
            evidence.joint_composite()

    if args.only == "13":
        atlas, atlas_path = load_original("fig05_component_atlas")
        renderer_paths[atlas_path.name] = sha256(atlas_path)
        atlas.DATA = AreaSource()
        atlas.OUT = OUT
        # The V134 manuscript-embedded atlas uses a brighter six-row palette
        # than the archived renderer's later/current export. These exact
        # five-class colours were read from the V134 embedded colourbars;
        # class breaks and component values are unchanged.
        manuscript_palettes = {
            "nutrition_score": ["#FBE9EB", "#F1BEC3", "#E28790", "#C84A55", "#8F2632"],
            "carbon_score": ["#FFF0E2", "#F7C99D", "#EBA468", "#D77A2F", "#984713"],
            "diversity_score": ["#FFF8D8", "#F5E7A3", "#E4CD62", "#C9A11C", "#8A6A08"],
            "sustainability_score": ["#EAF6EC", "#B9DEBF", "#7BC386", "#3F9254", "#27663A"],
            "hygiene_score": ["#E8F2FA", "#BBD8ED", "#7EB4DA", "#3B7FB5", "#24577F"],
            "practice_score": ["#F1ECF8", "#D7C9E8", "#B298CF", "#7755A4", "#503678"],
        }
        # Also shorten two verbose row labels at the author's request.
        display_labels = {
            "diversity_score": ("Cuisine\ndiversity", "Diversity"),
            "sustainability_score": ("Environmental\nsustainability", "Environment"),
        }
        shortened = []
        for component in atlas.COMPONENTS:
            key, label, _old_label_colour, _old_colours = component
            if key in display_labels:
                expected, label = display_labels[key]
                if component[1] != expected:
                    raise AssertionError(f"Unexpected historical atlas label: {key}")
            if key not in manuscript_palettes:
                raise AssertionError(f"Unknown atlas component: {key}")
            colours = manuscript_palettes[key]
            shortened.append((key, label, colours[3], colours))
        atlas.COMPONENTS = shortened

        def fixed_atlas_read(path: Path):
            year = int(path.name[:4])
            # The historic display geometry is small and preserves the V134
            # cartographic silhouette; all plotted values come from V135 CSV.
            geometry = gpd.read_file(
                OLD_ATLAS_GEOMETRY / f"{year}_lsbg_components.gpkg"
            )[["lsbg", "geometry"]]
            values = pd.read_csv(AREA / f"{year}_lsbg_components.csv")
            if geometry.lsbg.duplicated().any() or values.lsbg.duplicated().any():
                raise AssertionError(f"Duplicated LSBG key in atlas year {year}")
            if set(geometry.lsbg) != set(values.lsbg):
                raise AssertionError(f"Historical geometry and fixed results differ in LSBG keys: {year}")
            return geometry.merge(values, on="lsbg", validate="one_to_one")

        atlas.gpd = SimpleNamespace(read_file=fixed_atlas_read)
        atlas.main()

    if args.only == "15":
        os.environ["CITIES_FIGURE_DIR"] = str(OUT)
        market, market_path = load_original("fig06_price_market")
        renderer_paths[market_path.name] = sha256(market_path)
        market.DATA_DIR = PRICE_MARKET
        market.main()

    if args.only == "17":
        groups, groups_path = load_original("fig09_social_within_year")
        renderer_paths[groups_path.name] = sha256(groups_path)
        groups.SOURCE = DOWNSTREAM
        groups.OUT = OUT
        points = pd.read_csv(DOWNSTREAM / "joint_subgroup_point_estimates.csv")
        replicates = pd.read_csv(DOWNSTREAM / "joint_subgroup_dcca_block_replicates.csv")
        independently_computed = groups.within_year_contrasts(replicates, points)
        archived = pd.read_csv(CONTRASTS / "within_year_contrasts.csv")
        keys = ["year", "domain", "reference_group", "comparison_group"]
        compared = independently_computed.merge(archived, on=keys, validate="one_to_one", suffixes=("_render", "_fixed"))
        if len(compared) != 51:
            raise AssertionError("Fig17 requires all 51 paired contrasts")
        for column in ("difference_percentage_points", "lower_95_percentage_points",
                       "upper_95_percentage_points", "q_bh_within_domain_year"):
            if not ((compared[f"{column}_render"] - compared[f"{column}_fixed"]).abs() < 1e-8).all():
                raise AssertionError(f"Fig17 paired contrast drift: {column}")
        groups.main()

    if args.only == "18":
        scale, scale_path = load_original("render_scale_weight_composite")
        renderer_paths[scale_path.name] = sha256(scale_path)
        scale.SOURCE = AGGREGATION
        scale.maps.DATA_DIR = VariantMapSource()
        scale.shared.OUT = OUT
        original_export = scale.shared.export

        def historic_numbering_export(fig, stem, audit):
            stems = {
                "Fig9a-h_Scale_Maps_Two_Per_Row": "Fig18a_Spatial_Scale_Sensitivity_Maps",
                "Fig9i-l_Scale_Diagnostics_Continued": "Fig18b_Weighting_And_Cross_Scale_Diagnostics",
            }
            if stem not in stems:
                raise AssertionError(f"Unexpected historical scale figure: {stem}")
            return original_export(fig, stems[stem], audit)

        scale.shared.export = historic_numbering_export
        # V134's four-panel diagnostic is the continuation page of this
        # original two-page renderer (183 × 136 mm), not the later 165-mm
        # single-page variant. Both pages now use fixed V135 sources.
        scale.build(paired_pages=True)

    if args.only == "19":
        planning, planning_path = load_original("render_planning_composite")
        renderer_paths[planning_path.name] = sha256(planning_path)
        planning.source.JOINT_SRC = DOWNSTREAM
        # Keep the historical, fixed public-housing candidate pool and boundaries.
        previous_argv = sys.argv
        try:
            sys.argv = [str(planning_path), "--output-dir", str(OUT)]
            planning.main()
        finally:
            sys.argv = previous_argv

    if input_hashes != assert_inputs():
        raise AssertionError("Fixed empirical inputs changed during rendering")
    names = [
        "Fig12_Restaurant_Quality_SDI_Evolution_Coverage",
        "Fig13_Six_Component_FourYear_Atlas",
        "Fig15_Price_Composition_Socioeconomic_Associations",
        "Fig16_Joint_Quality_Walking_Price_Opportunity",
        "Fig17_SameYear_Socioeconomic_Zero_Joint_Opportunity",
        "Fig18b_Weighting_And_Cross_Scale_Diagnostics",
        "Fig19_Planning_Strategies",
    ]
    outputs = {
        f"{name}.{ext}": sha256(OUT / f"{name}.{ext}")
        for name in names for ext in ("pdf", "svg", "png")
        if (OUT / f"{name}.{ext}").exists()
    }
    for suffix in ("_PREVIEW.png", "_600dpi.png"):
        name = "Fig15_Price_Composition_Socioeconomic_Associations"
        path = OUT / f"{name}{suffix}"
        if path.exists():
            outputs[path.name] = sha256(path)
    report = {
        "status": "RESTRICTED_VISUAL_POLISH_CANDIDATE",
        "claim": "Copied V134 renderers with bounded V138 cosmetic edits and fixed-Nutrition V135 result sources",
        "no_empirical_recomputation": True,
        "public_release_authorized": False,
        "figure_16_panels": 11,
        "figure_17_panels": 7,
        "figure_18b_panels": 4,
        "figure_19_panels": 8,
        "last_rendered_figure": args.only,
        "renderer_sha256_this_run": renderer_paths,
        "historic_atlas_geometry_sha256": {
            str(year): sha256(OLD_ATLAS_GEOMETRY / f"{year}_lsbg_components.gpkg")
            for year in (2011, 2016, 2021, 2024)
        },
        "input_sha256": input_hashes,
        "output_sha256": outputs,
    }
    (OUT / f"V138_FIG{args.only}_VISUAL_POLISH_AUDIT.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(OUT)


if __name__ == "__main__":
    main()
