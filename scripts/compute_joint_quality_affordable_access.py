"""Compute quality-context-qualified affordable walking opportunity.

The estimand is deliberately defined at the measurement level supported by
the audit trail.  A restaurant qualifies when it is reachable through the
pedestrian network, its OpenRice price tier is no higher than the selected
ceiling, and the restaurant is located in an LSBG whose six-component SDI is
at or above the selected threshold.  The script therefore does *not* assign
an area SDI to an individual restaurant or claim a validated restaurant-level
SDI.

Raw restaurant identifiers never leave the restricted input environment.
Released outputs contain LSBG aggregates and opaque opportunity identifiers
only through aggregate counts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


YEARS = (2016, 2021, 2024)
SDI_THRESHOLDS = (0.40, 0.45, 0.50)
PRICE_CEILINGS = (1.0, 2.0, 3.0)  # <=HK$50, <=HK$100, <=HK$200
TIME_THRESHOLDS = (10.0, 15.0)
PRIMARY_SDI_THRESHOLD = 0.45
PRIMARY_PRICE_CEILING = 2.0
PRIMARY_TIME_THRESHOLD = 15.0
SEED = 20260825


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def weighted_metrics(frame: pd.DataFrame, value: str, summarize_inequality) -> dict[str, float]:
    data = frame[[value, "t_pop", "ma_hh"]].replace([np.inf, -np.inf], np.nan).dropna()
    data = data.loc[data["t_pop"].gt(0) & data["ma_hh"].gt(0) & data[value].ge(0)]
    result = summarize_inequality(
        data[value].to_numpy(float),
        data["t_pop"].to_numpy(float),
        data["ma_hh"].to_numpy(float),
    ).set_index("metric")["estimate"]
    return {
        "population_weighted_mean": float(result["population_weighted_mean"]),
        "zero_access_population_share": float(result["zero_access_population_share"]),
        "weighted_gini": float(result["weighted_gini"]),
        "concentration_index": float(result["concentration_index"]),
        "lsbg_n": int(len(data)),
        "population_analyzed": float(data["t_pop"].sum()),
    }


def bootstrap_primary(
    frame: pd.DataFrame,
    repetitions: int,
    rng: np.random.Generator,
    summarize_inequality,
) -> pd.DataFrame:
    groups = [group.copy() for _, group in frame.groupby("dcca", sort=False)]
    if len(groups) < 2:
        raise RuntimeError("DCCA block bootstrap requires at least two non-empty blocks")
    rows: list[dict[str, float | int | str]] = []
    for repetition in range(repetitions):
        sampled = rng.integers(0, len(groups), size=len(groups))
        replicate = pd.concat([groups[index] for index in sampled], ignore_index=True)
        for outcome in ("low_price_access", "joint_access"):
            metrics = weighted_metrics(replicate, outcome, summarize_inequality)
            for metric in (
                "population_weighted_mean",
                "zero_access_population_share",
                "weighted_gini",
                "concentration_index",
            ):
                rows.append(
                    {
                        "replicate": repetition,
                        "outcome": outcome,
                        "metric": metric,
                        "estimate": metrics[metric],
                    }
                )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--master", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=999)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project = args.project_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

    from sus_dining_access.component_pair_diagnostics import normalize_price_order
    from sus_dining_access.count_opportunity import opaque_opportunity_id
    from sus_dining_access.inequality import summarize_inequality

    specs = {
        2016: {
            "namespace": "openrice_2016_count_candidate_v1",
            "eligible": project
            / "outputs/t3_count_candidate_2016/primary_narrow/eligible_restaurant_2016_primary_narrow_count_opportunity.csv",
            "destination_id": "lsbg_id_2016",
            "pairs": project
            / "outputs/restricted/t3_reachable_pairs_2016/lsbg_restaurant_reachable_pairs_2016_15min.csv",
        },
        2021: {
            "namespace": "openrice_2021_count_candidate_v1",
            "eligible": project
            / "outputs/t3_count_candidate/primary_narrow/eligible_restaurant_2021_primary_narrow_count_opportunity.csv",
            "destination_id": "lsbg_id_2021",
            "pairs": project
            / "outputs/restricted/t3_reachable_pairs_2021/lsbg_restaurant_reachable_pairs_2021_15min.csv",
        },
        2024: {
            "namespace": "openrice_2024_count_candidate_v1",
            "eligible": project
            / "outputs/t3_count_candidate_2024/primary_narrow/eligible_restaurant_2024_primary_narrow_count_opportunity.csv",
            "destination_id": "lsbg_id_2021",
            "pairs": project
            / "outputs/restricted/t3_reachable_pairs_2024/lsbg_restaurant_reachable_pairs_2024_15min.csv",
        },
    }
    for year in YEARS:
        specs[year]["area"] = (
            project / f"outputs/restricted/v4_price_market_decomposition/{year}_lsbg_price_market.gpkg"
        )
    network_reference = (
        project / "source_data/fig04_network_price_v4/lsbg_network_price_opportunity.csv"
    )

    input_paths = [args.master, network_reference]
    for spec in specs.values():
        input_paths.extend([spec["eligible"], spec["pairs"], spec["area"]])
    missing = [path for path in input_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing inputs:\n" + "\n".join(map(str, missing)))

    master = pd.read_csv(
        args.master,
        usecols=["restaurant_id", "restaurant_cost_range"],
        low_memory=False,
    )
    master["price_order"] = normalize_price_order(master["restaurant_cost_range"])
    if master["restaurant_id"].duplicated().any():
        raise ValueError("restaurant_id must be unique in the master table")
    network_status = pd.read_csv(network_reference, dtype={"lsbg_id": str})
    network_status = network_status.loc[
        network_status["threshold_min"].eq(PRIMARY_TIME_THRESHOLD),
        ["year", "lsbg_id", "network_available"],
    ].drop_duplicates()

    all_lsbg: list[pd.DataFrame] = []
    point_rows: list[dict[str, float | int]] = []
    coverage_rows: list[dict[str, float | int]] = []
    bootstrap_rows: list[pd.DataFrame] = []
    rng = np.random.default_rng(SEED)

    for year in YEARS:
        spec = specs[year]
        destination_id = str(spec["destination_id"])
        eligible = pd.read_csv(
            spec["eligible"],
            dtype={"opportunity_id": str, destination_id: str},
            usecols=["opportunity_id", destination_id],
        )
        if eligible["opportunity_id"].duplicated().any():
            raise ValueError(f"{year}: duplicate opportunity_id in eligible table")

        price = master[["restaurant_id", "price_order"]].copy()
        price["opportunity_id"] = price["restaurant_id"].map(
            lambda value: opaque_opportunity_id(value, str(spec["namespace"]))
        )
        price = price[["opportunity_id", "price_order"]]

        area = gpd.read_file(spec["area"])
        area["lsbg"] = area["lsbg"].astype(str)
        area["dcca"] = area["dcca"].astype(str)
        area["sdi_equal_arithmetic"] = pd.to_numeric(
            area["sdi_equal_arithmetic"], errors="coerce"
        )
        destination = area[["lsbg", "sdi_equal_arithmetic"]].rename(
            columns={"lsbg": destination_id, "sdi_equal_arithmetic": "destination_area_sdi"}
        )
        destination[destination_id] = destination[destination_id].astype(str)

        attributes = eligible.merge(price, on="opportunity_id", how="left", validate="one_to_one")
        attributes = attributes.merge(destination, on=destination_id, how="left", validate="many_to_one")
        pairs = pd.read_csv(
            spec["pairs"],
            dtype={"lsbg_id": str, "opportunity_id": str},
            usecols=["lsbg_id", "opportunity_id", "travel_time_min"],
        )
        if pairs.duplicated(["lsbg_id", "opportunity_id"]).any():
            raise ValueError(f"{year}: reachable pairs are not unique")
        pairs = pairs.merge(
            attributes[["opportunity_id", "price_order", "destination_area_sdi"]],
            on="opportunity_id",
            how="left",
            validate="many_to_one",
        )

        origin = area[["lsbg", "t_pop", "ma_hh", "dcca", "geometry"]].rename(
            columns={"lsbg": "lsbg_id"}
        )
        origin["year"] = year
        origin = origin.merge(
            network_status.loc[network_status["year"].eq(year), ["lsbg_id", "network_available"]],
            on="lsbg_id",
            how="left",
            validate="one_to_one",
        )
        if origin["network_available"].isna().any():
            raise ValueError(f"{year}: network availability missing for at least one origin")
        origin["network_available"] = origin["network_available"].astype(bool)

        low_mask = pairs["travel_time_min"].le(PRIMARY_TIME_THRESHOLD) & pairs[
            "price_order"
        ].le(PRIMARY_PRICE_CEILING)
        joint_mask = low_mask & pairs["destination_area_sdi"].ge(PRIMARY_SDI_THRESHOLD)
        low_count = pairs.loc[low_mask].groupby("lsbg_id").size().rename("low_price_access")
        joint_count = pairs.loc[joint_mask].groupby("lsbg_id").size().rename("joint_access")
        origin = origin.merge(low_count, on="lsbg_id", how="left")
        origin = origin.merge(joint_count, on="lsbg_id", how="left")
        origin[["low_price_access", "joint_access"]] = origin[
            ["low_price_access", "joint_access"]
        ].fillna(0).astype(int)
        origin["joint_retention_share"] = np.divide(
            origin["joint_access"],
            origin["low_price_access"],
            out=np.full(len(origin), np.nan),
            where=origin["low_price_access"].to_numpy() > 0,
        )

        primary_points: dict[str, dict[str, float]] = {}
        analysis_origin = origin.loc[origin["network_available"]].copy()
        for outcome in ("low_price_access", "joint_access"):
            primary_points[outcome] = weighted_metrics(
                analysis_origin, outcome, summarize_inequality
            )
            point_rows.append(
                {
                    "year": year,
                    "threshold_min": PRIMARY_TIME_THRESHOLD,
                    "price_ceiling_hkd": 100,
                    "destination_area_sdi_threshold": PRIMARY_SDI_THRESHOLD,
                    "outcome": outcome,
                    **primary_points[outcome],
                }
            )
        point_rows.append(
            {
                "year": year,
                "threshold_min": PRIMARY_TIME_THRESHOLD,
                "price_ceiling_hkd": 100,
                "destination_area_sdi_threshold": PRIMARY_SDI_THRESHOLD,
                "outcome": "joint_retention_ratio_of_population_weighted_means",
                "population_weighted_mean": primary_points["joint_access"]["population_weighted_mean"]
                / primary_points["low_price_access"]["population_weighted_mean"],
                "zero_access_population_share": np.nan,
                "weighted_gini": np.nan,
                "concentration_index": np.nan,
                "lsbg_n": primary_points["joint_access"]["lsbg_n"],
                "population_analyzed": primary_points["joint_access"]["population_analyzed"],
            }
        )

        for time_threshold in TIME_THRESHOLDS:
            for price_ceiling in PRICE_CEILINGS:
                for sdi_threshold in SDI_THRESHOLDS:
                    mask = (
                        pairs["travel_time_min"].le(time_threshold)
                        & pairs["price_order"].le(price_ceiling)
                        & pairs["destination_area_sdi"].ge(sdi_threshold)
                    )
                    count = pairs.loc[mask].groupby("lsbg_id").size().rename("value")
                    scenario = analysis_origin[["lsbg_id", "t_pop", "ma_hh"]].merge(
                        count, on="lsbg_id", how="left"
                    )
                    scenario["value"] = scenario["value"].fillna(0)
                    metrics = weighted_metrics(scenario, "value", summarize_inequality)
                    point_rows.append(
                        {
                            "year": year,
                            "threshold_min": time_threshold,
                            "price_ceiling_hkd": {1.0: 50, 2.0: 100, 3.0: 200}[price_ceiling],
                            "destination_area_sdi_threshold": sdi_threshold,
                            "outcome": "joint_access_sensitivity",
                            **metrics,
                        }
                    )

        boot = bootstrap_primary(analysis_origin, args.bootstrap, rng, summarize_inequality)
        boot["year"] = year
        bootstrap_rows.append(boot)

        coverage_rows.append(
            {
                "year": year,
                "eligible_restaurant_n": len(attributes),
                "eligible_with_price_n": int(attributes["price_order"].notna().sum()),
                "eligible_with_price_share": float(attributes["price_order"].notna().mean()),
                "eligible_with_destination_area_sdi_n": int(
                    attributes["destination_area_sdi"].notna().sum()
                ),
                "eligible_with_destination_area_sdi_share": float(
                    attributes["destination_area_sdi"].notna().mean()
                ),
                "reachable_pair_n": len(pairs),
                "reachable_pair_with_price_share": float(pairs["price_order"].notna().mean()),
                "reachable_pair_with_destination_area_sdi_share": float(
                    pairs["destination_area_sdi"].notna().mean()
                ),
                "origin_lsbg_n": len(origin),
                "population_analyzed": float(origin.loc[origin["t_pop"].gt(0), "t_pop"].sum()),
            }
        )

        release = origin.drop(columns="geometry").copy()
        all_lsbg.append(release)
        origin.to_file(output / f"{year}_joint_access_geometry.gpkg", driver="GPKG")

    points = pd.DataFrame(point_rows)
    lsbg = pd.concat(all_lsbg, ignore_index=True)
    coverage = pd.DataFrame(coverage_rows)
    replicates = pd.concat(bootstrap_rows, ignore_index=True)
    primary = points.loc[points["outcome"].isin(["low_price_access", "joint_access"])]
    intervals = (
        replicates.groupby(["year", "outcome", "metric"], observed=True)["estimate"]
        .agg(ci_low=lambda x: x.quantile(0.025), ci_high=lambda x: x.quantile(0.975), valid_replicates="count")
        .reset_index()
    )
    intervals = intervals.merge(
        primary.melt(
            id_vars=["year", "outcome"],
            value_vars=[
                "population_weighted_mean",
                "zero_access_population_share",
                "weighted_gini",
                "concentration_index",
            ],
            var_name="metric",
            value_name="point_estimate",
        ),
        on=["year", "outcome", "metric"],
        how="left",
        validate="one_to_one",
    )

    outputs = {
        "lsbg": output / "lsbg_joint_quality_affordable_access.csv",
        "points": output / "joint_access_point_estimates.csv",
        "intervals": output / "joint_access_dcca_block_intervals.csv",
        "coverage": output / "joint_access_data_coverage_audit.csv",
        "replicates": output / "joint_access_dcca_block_replicates.csv",
    }
    lsbg.to_csv(outputs["lsbg"], index=False)
    points.to_csv(outputs["points"], index=False)
    intervals.to_csv(outputs["intervals"], index=False)
    coverage.to_csv(outputs["coverage"], index=False)
    replicates.to_csv(outputs["replicates"], index=False)

    contract = {
        "analysis_name": "quality-context-qualified affordable walking opportunity",
        "primary_estimand": (
            "Number of restaurants reachable within 15 minutes, with OpenRice price tier "
            "no higher than HK$100, located in an LSBG with strict six-component area SDI >=0.45."
        ),
        "measurement_boundary": (
            "SDI is measured for the destination LSBG restaurant environment, not for an individual "
            "restaurant. The result must not be labelled restaurant-level SDI or household affordability."
        ),
        "years": list(YEARS),
        "excluded_years": {
            "2011": (
                "No audited same-date pedestrian network exists, and the earliest defensible OSM attic "
                "floor is 2012-09-12; no interpolation or relabelling was used."
            )
        },
        "primary_thresholds": {
            "walking_minutes": PRIMARY_TIME_THRESHOLD,
            "price_ceiling_hkd": 100,
            "destination_area_sdi": PRIMARY_SDI_THRESHOLD,
        },
        "sensitivity_thresholds": {
            "walking_minutes": list(TIME_THRESHOLDS),
            "price_ceiling_hkd": [50, 100, 200],
            "destination_area_sdi": list(SDI_THRESHOLDS),
        },
        "population_reference": (
            "Native 2016 census geography and population for 2016; fixed 2021 census geography, "
            "population and income composition for 2021 and 2024."
        ),
        "price_boundary": (
            "OpenRice price tier is an undated platform classification. It is a price-screened opportunity "
            "measure, not household affordability, expenditure, realised use, need or welfare."
        ),
        "uncertainty": f"{args.bootstrap} DCCA spatial-block bootstrap replicates",
        "seed": SEED,
    }
    contract_path = output / "analysis_contract.json"
    contract_path.write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    manifest_rows = []
    for path in input_paths:
        manifest_rows.append({"role": "input", "path": str(path), "sha256": sha256_file(path)})
    for path in [*outputs.values(), contract_path, *[output / f"{y}_joint_access_geometry.gpkg" for y in YEARS]]:
        manifest_rows.append({"role": "output", "path": path.name, "sha256": sha256_file(path)})
    pd.DataFrame(manifest_rows).to_csv(output / "sha256_manifest.csv", index=False)
    run = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "geopandas": gpd.__version__,
        "analysis_contract": contract,
    }
    (output / "run_manifest.json").write_text(
        json.dumps(run, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(primary[["year", "outcome", "population_weighted_mean", "zero_access_population_share", "weighted_gini", "concentration_index"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
