"""Extend the audited joint opportunity estimand to social groups and planning.

Every input is observed or explicitly hypothetical.  The 2016 and 2021
population compositions come from their corresponding census LSBG files;
2024 access is evaluated against the fixed 2021 composition.  Male and female
weights are reconstructed as published census proportions multiplied by the
same LSBG population denominator.

The public-housing intervention is a planning stress test, not a forecast:
one additional outlet per selected estate is assumed to satisfy *all* three
gates (destination-context SDI, price tier and the audited walking network).
No restaurant identifier, census microdata or private origin-destination pair
is exported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix, hstack, vstack


ROOT = Path(__file__).resolve().parents[1]
YEARS = (2016, 2021, 2024)
SEED = 20260826
BUDGETS = (5, 10, 20)

GROUPS = {
    "Ethnicity": {
        "Chinese": "ethn_chi",
        "Filipino": "ethn_phi",
        "Indonesian": "ethn_ind",
        "White": "ethn_wh",
        "Other": "ethn_oth",
    },
    "Occupation": {
        "Managers": "wp_a",
        "Professionals": "wp_b",
        "Clerical": "wp_d",
        "Service/sales": "wp_e",
        "Elementary": "wp_h",
    },
    "Education": {
        "Primary or below": "edu_pri_be",
        "Secondary": "edu_sec",
        "Post-secondary": "edu_psec",
    },
    "Household income": {
        "<HK$10k": "dhi_sb_1",
        "HK$10–20k": "dhi_sb_2",
        "HK$20–40k": "dhi_sb_3",
        "≥HK$40k": "dhi_sb_4",
    },
    "Age": {
        "15–24": "age_2",
        "25–44": "age_3",
        "45–64": "age_4",
        "65+": "age_5",
    },
    "Sex": {
        "Female": "derived_female_count",
        "Male": "derived_male_count",
    },
}


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def census_paths(root: Path) -> dict[int, Path]:
    return {
        2016: root
        / "2016/5. 2016LargeStreetBlockGroups(Large Subunit Group)_1622_SHP/LSBG_16BC_percent.gpkg",
        2021: root
        / "2021/5. 2021LargeSubunitGroups(Large Street Block Group)_1746_SHP/LSUG_21C_percent.gpkg",
    }


def weighted_metrics(frame: pd.DataFrame, count_column: str, multiplier: np.ndarray | None = None):
    weights = pd.to_numeric(frame[count_column], errors="coerce").fillna(0).to_numpy(float)
    if multiplier is not None:
        weights = weights * multiplier
    values = pd.to_numeric(frame["joint_access"], errors="coerce").to_numpy(float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not valid.any():
        return np.nan, np.nan, 0.0
    values = values[valid]
    weights = weights[valid]
    return (
        float(np.average(values, weights=weights)),
        float(weights[values == 0].sum() / weights.sum()),
        float(weights.sum()),
    )


def subgroup_analysis(joint: pd.DataFrame, output: Path, bootstrap_n: int, census_root: Path) -> dict:
    paths = census_paths(census_root)
    raw_fields = sorted(
        {field for domain in GROUPS.values() for field in domain.values() if not field.startswith("derived_")}
    )
    point_rows: list[dict] = []
    replicate_rows: list[dict] = []
    rng = np.random.default_rng(SEED)

    for year in YEARS:
        composition_year = 2016 if year == 2016 else 2021
        census = pyogrio.read_dataframe(
            paths[composition_year],
            columns=["lsbg", "t_pop", "male_p", "female_p", *raw_fields],
            read_geometry=False,
        ).rename(columns={"lsbg": "lsbg_id", "t_pop": "census_t_pop"})
        census["lsbg_id"] = census["lsbg_id"].astype(str)
        for field in ("male_p", "female_p"):
            census[field] = pd.to_numeric(census[field], errors="coerce")
            if ((census[field].dropna() < 0) | (census[field].dropna() > 1)).any():
                raise ValueError(f"{year}: census {field} is not a 0–1 proportion")
        census["derived_male_count"] = census["census_t_pop"] * census["male_p"]
        census["derived_female_count"] = census["census_t_pop"] * census["female_p"]

        current = joint.loc[
            joint["year"].eq(year)
            & joint["network_available"].fillna(False).astype(bool),
            ["lsbg_id", "joint_access", "dcca", "t_pop"],
        ].copy()
        frame = census.merge(current, on="lsbg_id", how="inner", validate="one_to_one")
        if not np.allclose(frame["census_t_pop"], frame["t_pop"], rtol=0, atol=1):
            raise ValueError(f"{year}: joint population differs from the census composition")
        if frame["dcca"].isna().any():
            raise ValueError(f"{year}: DCCA block identifiers are missing")

        block_ids = pd.Index(frame["dcca"].astype(str).unique())
        block_codes = frame["dcca"].astype(str).map(
            pd.Series(np.arange(len(block_ids)), index=block_ids)
        ).to_numpy(int)
        multipliers = []
        for _ in range(bootstrap_n):
            sampled = rng.integers(0, len(block_ids), size=len(block_ids))
            multipliers.append(np.bincount(sampled, minlength=len(block_ids))[block_codes])

        for domain, groups in GROUPS.items():
            for group, field in groups.items():
                mean, zero, denominator = weighted_metrics(frame, field)
                point_rows.append({
                    "year": year,
                    "census_composition_year": composition_year,
                    "temporal_status": (
                        "contemporaneous" if year == composition_year else "2021-composition projection"
                    ),
                    "domain": domain,
                    "group": group,
                    "source_count_field": (
                        "t_pop*female_p" if field == "derived_female_count" else
                        "t_pop*male_p" if field == "derived_male_count" else field
                    ),
                    "mean_joint_options": mean,
                    "zero_joint_access_share": zero,
                    "group_denominator": denominator,
                })
                for replicate, multiplier in enumerate(multipliers):
                    boot_mean, boot_zero, _ = weighted_metrics(frame, field, multiplier)
                    replicate_rows.append({
                        "year": year,
                        "domain": domain,
                        "group": group,
                        "replicate": replicate,
                        "mean_joint_options": boot_mean,
                        "zero_joint_access_share": boot_zero,
                    })

    points = pd.DataFrame(point_rows)
    replicates = pd.DataFrame(replicate_rows)
    intervals = []
    for keys, frame in replicates.groupby(["year", "domain", "group"], sort=False):
        for metric in ("mean_joint_options", "zero_joint_access_share"):
            values = frame[metric].dropna().to_numpy(float)
            intervals.append({
                "year": keys[0],
                "domain": keys[1],
                "group": keys[2],
                "metric": metric,
                "lower_95": float(np.quantile(values, 0.025)),
                "upper_95": float(np.quantile(values, 0.975)),
                "valid_replicate_n": len(values),
            })
    points.to_csv(output / "joint_subgroup_point_estimates.csv", index=False)
    replicates.to_csv(output / "joint_subgroup_dcca_block_replicates.csv", index=False)
    pd.DataFrame(intervals).to_csv(output / "joint_subgroup_dcca_block_intervals.csv", index=False)
    return {"groups": len(points), "replicates": len(replicates), "domains": list(GROUPS)}


def solve_maximum_coverage(
    coverage: np.ndarray, objective_weights: np.ndarray, population: np.ndarray, budget: int
) -> np.ndarray:
    demands, sites = coverage.shape
    primary = objective_weights / max(float(objective_weights.sum()), 1.0)
    secondary = population / max(float(population.sum()), 1.0)
    objective = np.concatenate((
        np.arange(sites, dtype=float) * 1e-10,
        -(primary * 1_000_000.0 + secondary),
    ))
    budget_row = csr_matrix(np.concatenate((np.ones(sites), np.zeros(demands))).reshape(1, -1))
    demand_rows = hstack((-csr_matrix(coverage.astype(float)), csr_matrix(np.eye(demands))), format="csr")
    matrix = vstack((budget_row, demand_rows), format="csr")
    result = milp(
        c=objective,
        integrality=np.ones(sites + demands),
        bounds=Bounds(np.zeros(sites + demands), np.ones(sites + demands)),
        constraints=LinearConstraint(
            matrix,
            np.full(demands + 1, -np.inf),
            np.concatenate(([float(budget)], np.zeros(demands))),
        ),
        options={"time_limit": 120.0, "mip_rel_gap": 0.0},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"joint maximum-coverage MILP failed: {result.message}")
    selected = np.flatnonzero(result.x[:sites] > 0.5)
    if len(selected) != budget:
        raise RuntimeError(f"expected {budget} selected sites, received {len(selected)}")
    return selected


def planning_analysis(project: Path, joint: pd.DataFrame, output: Path) -> dict:
    sys.path.insert(0, str(ROOT / "src"))
    from sus_dining_access.inequality import concentration_index, weighted_gini
    from sus_dining_access.policy_coverage import bottom_population_membership

    site_path = ROOT / "source_data/fig04_equity_siting_v7/candidate_planning_nodes.csv"
    pair_path = (
        project / "outputs/restricted/v7_public_housing_planning_node_coverage_202608/"
        "public_housing_site_lsbg_reachable_pairs_15min.csv"
    )
    base = joint.loc[
        joint["year"].eq(2024)
        & joint["network_available"].fillna(False).astype(bool)
        & joint["t_pop"].gt(0)
        & joint["ma_hh"].gt(0)
    ].copy().sort_values("lsbg_id").reset_index(drop=True)
    sites = pd.read_csv(site_path, dtype={"site_id": str}).sort_values("site_id").reset_index(drop=True)
    pairs = pd.read_csv(pair_path, dtype={"site_id": str, "lsbg_id": str})
    pairs = pairs.loc[pairs["travel_time_min"].le(15.0)]
    demand_index = pd.Series(np.arange(len(base)), index=base["lsbg_id"])
    site_index = pd.Series(np.arange(len(sites)), index=sites["site_id"])
    valid = pairs.loc[pairs["lsbg_id"].isin(demand_index.index) & pairs["site_id"].isin(site_index.index)]
    coverage = np.zeros((len(base), len(sites)), dtype=bool)
    coverage[demand_index.loc[valid["lsbg_id"]].to_numpy(), site_index.loc[valid["site_id"]].to_numpy()] = True

    population = base["t_pop"].to_numpy(float)
    income = base["ma_hh"].to_numpy(float)
    baseline = base["joint_access"].to_numpy(float)
    low_income = bottom_population_membership(income, population, 0.40)
    low_access = bottom_population_membership(baseline, population, 0.40)
    joint_priority = population * low_income * low_access
    zero_priority = population * low_income * (baseline == 0)
    objectives = {
        "zero_affordability_gap": zero_priority,
        "equity_joint": joint_priority,
        "access_deficit": population * low_access,
        "population_reach": population,
    }
    rows: list[dict] = []
    selected_rows: list[dict] = []

    def record(scenario: str, budget: int, chosen: np.ndarray) -> None:
        additions = coverage[:, chosen].sum(axis=1).astype(float) if len(chosen) else np.zeros(len(base))
        simulated = baseline + additions
        covered = coverage[:, chosen].any(axis=1) if len(chosen) else np.zeros(len(base), dtype=bool)
        rows.append({
            "scenario": scenario,
            "budget_sites": budget,
            "weighted_gini": weighted_gini(simulated, population),
            "income_concentration_index": concentration_index(simulated, population, income),
            "zero_access_population_share": float(population[simulated == 0].sum() / population.sum()),
            "population_newly_covered": float(population[(baseline == 0) & (simulated > 0)].sum()),
            "population_within_selected_catchments": float(population[covered].sum()),
            "joint_priority_population_reached": float(joint_priority[covered].sum()),
            "joint_priority_population_denominator": float(joint_priority.sum()),
            "joint_priority_reached_share": float(joint_priority[covered].sum() / max(joint_priority.sum(), 1.0)),
            "zero_affordability_priority_population_reached": float(zero_priority[covered].sum()),
            "zero_affordability_priority_population_denominator": float(zero_priority.sum()),
            "zero_affordability_priority_reached_share": float(zero_priority[covered].sum() / max(zero_priority.sum(), 1.0)),
            "mean_joint_options": float(np.average(simulated, weights=population)),
            "selected_site_ids": "|".join(sites.iloc[chosen]["site_id"].tolist()),
            "baseline_outcome": "joint_access",
        })
        for rank, site in enumerate(sites.iloc[chosen].itertuples(index=False), start=1):
            selected_rows.append({
                "scenario": scenario,
                "budget_sites": budget,
                "selection_rank": rank,
                "site_id": site.site_id,
                "site_name": site.site_name,
                "district": site.district,
                "longitude": site.longitude,
                "latitude": site.latitude,
            })

    record("BAU", 0, np.array([], dtype=int))
    for scenario, weights in objectives.items():
        for budget in BUDGETS:
            record(scenario, budget, solve_maximum_coverage(coverage, weights, population, budget))

    results = pd.DataFrame(rows)
    results.to_csv(output / "scenario_results.csv", index=False)
    pd.DataFrame(selected_rows).to_csv(output / "selected_planning_nodes.csv", index=False)
    demand = base[["lsbg_id", "t_pop", "ma_hh", "low_price_access", "joint_access"]].copy()
    demand["low_income_40_membership"] = low_income
    demand["low_access_40_membership"] = low_access
    demand["joint_priority_population"] = joint_priority
    demand["zero_affordability_priority_population"] = zero_priority
    demand.to_csv(output / "priority_population_definition.csv", index=False)
    return {
        "candidate_sites": len(sites),
        "demand_lsbg": len(base),
        "restricted_pair_sha256": digest(pair_path),
        "strict_priority_denominator": float(zero_priority.sum()),
        "joint_priority_denominator": float(joint_priority.sum()),
        "results": results.to_dict(orient="records"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--census-root", type=Path, required=True,
                        help="Directory containing the authorized 2016 and 2021 census inputs")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=999)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    joint_path = ROOT / "source_data/fig13_joint_quality_affordable_access_v1/lsbg_joint_quality_affordable_access.csv"
    joint = pd.read_csv(joint_path, dtype={"lsbg_id": str, "dcca": str})
    subgroup = subgroup_analysis(joint, output, args.bootstrap, args.census_root.resolve())
    planning = planning_analysis(args.project_root.resolve(), joint, output)
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "estimand": "15-min walking AND platform price <=HK$100 AND destination-LSBG SDI >=0.45",
        "2024_population_composition": 2021,
        "bootstrap_replicates": args.bootstrap,
        "bootstrap_seed": SEED,
        "joint_source_sha256": digest(joint_path),
        "census_source_sha256": {str(year): digest(path) for year, path in census_paths(args.census_root.resolve()).items()},
        "subgroups": subgroup,
        "planning": planning,
        "limitations": [
            "The SDI qualification refers to the destination-area restaurant environment, not an individual restaurant.",
            "Price tier is a platform classification, not demonstrated household affordability.",
            "2024 subgroup estimates use the fixed 2021 census composition.",
            "Planning additions are hypothetical programme-qualified outlets and are not implementation forecasts.",
        ],
    }
    with (output / "joint_analysis_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(json.dumps({"subgroups": subgroup, "planning": {k: v for k, v in planning.items() if k != "results"}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
