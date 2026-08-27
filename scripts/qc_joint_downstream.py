"""Audit released joint-outcome subgroup contrasts and planning scenarios.

The check replays published interval and BH calculations, verifies restricted
input hashes when they are available locally, and checks every scenario against
its released candidate IDs and priority-population denominator. It does not
fabricate observations or publish restricted input rows.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "source_data" / "fig14_joint_subgroup_planning_v1"
PRIMARY = ROOT / "source_data" / "fig13_joint_quality_affordable_access_v1"
PROJECT = Path(os.environ["CITIES_RESTRICTED_PROJECT"]) if os.environ.get("CITIES_RESTRICTED_PROJECT") else None
FIGURES = Path(os.environ.get("CITIES_FIGURE_DIR", str(ROOT / "figures/current")))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted = np.minimum.accumulate(
        (ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1]
    )[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.minimum(adjusted, 1.0)
    return restored


def main() -> int:
    manifest = json.loads((DATA / "joint_analysis_manifest.json").read_text(encoding="utf-8"))
    points = pd.read_csv(DATA / "joint_subgroup_point_estimates.csv")
    intervals = pd.read_csv(DATA / "joint_subgroup_dcca_block_intervals.csv")
    replicates = pd.read_csv(DATA / "joint_subgroup_dcca_block_replicates.csv")
    contrasts = pd.read_csv(
        FIGURES / "Fig9_Joint_Socioeconomic_Sex_Contrasts_v26_NATURE_within_year_contrasts.csv"
    )
    scenarios = pd.read_csv(DATA / "scenario_results.csv")
    selected = pd.read_csv(DATA / "selected_planning_nodes.csv")
    priorities = pd.read_csv(DATA / "priority_population_definition.csv")
    candidates = pd.read_csv(ROOT / "source_data" / "fig04_equity_siting_v7" / "candidate_planning_nodes.csv")

    require(len(points) == 69, f"expected 69 subgroup-year estimates, found {len(points)}")
    require(len(intervals) == 138, f"expected 138 confidence intervals, found {len(intervals)}")
    require(len(replicates) == 69 * 999, f"unexpected paired-bootstrap row count: {len(replicates)}")
    require(len(contrasts) == 51, f"expected 51 contrasts, found {len(contrasts)}")
    require(len(scenarios) == 13, f"expected baseline and 12 scenarios, found {len(scenarios)}")
    require(len(priorities) == 1744, f"expected 1744 demand LSBGs, found {len(priorities)}")
    require(len(candidates) == 237, f"expected 237 candidate nodes, found {len(candidates)}")
    require(
        digest(PRIMARY / "lsbg_joint_quality_affordable_access.csv") == manifest["joint_source_sha256"],
        "primary joint-outcome SHA-256 mismatch",
    )
    require(set(points.year) == {2016, 2021, 2024}, "unapproved network year in subgroup data")
    require(points.loc[points.year.eq(2024), "census_composition_year"].eq(2021).all(),
            "2024 subgroup results do not uniformly use fixed 2021 composition")

    grouped_replicates = replicates.groupby(["year", "domain", "group"], sort=False)
    require(grouped_replicates.size().eq(999).all(), "a subgroup does not have exactly 999 paired replicates")
    for record in intervals.itertuples(index=False):
        draws = grouped_replicates.get_group((record.year, record.domain, record.group))[record.metric]
        expected = np.quantile(draws.to_numpy(float), [0.025, 0.975])
        require(np.allclose(expected, [record.lower_95, record.upper_95], atol=1e-10),
                f"incorrect interval: {record.year} / {record.domain} / {record.group} / {record.metric}")

    for (year, domain), family in contrasts.groupby(["year", "domain"], sort=False):
        expected = bh_adjust(family.p_two_sided.to_numpy(float))
        require(np.allclose(expected, family.q_bh_within_domain_year.to_numpy(float), atol=1e-12),
                f"BH adjustment mismatch: {year} / {domain}")
        require(family.paired_replicates.eq(999).all(), f"missing paired draws: {year} / {domain}")
        for row in family.itertuples(index=False):
            group_draws = grouped_replicates.get_group((year, domain, row.comparison_group))
            reference_draws = grouped_replicates.get_group((year, domain, row.reference_group))
            paired = group_draws[["replicate", "zero_joint_access_share"]].merge(
                reference_draws[["replicate", "zero_joint_access_share"]],
                on="replicate", validate="one_to_one", suffixes=("_group", "_reference"),
            )
            require(len(paired) == 999, f"incomplete contrast pairing: {year} / {domain}")
            differences = (
                paired.zero_joint_access_share_group - paired.zero_joint_access_share_reference
            ).to_numpy(float)
            empirical_p = 2.0 * min(float(np.mean(differences <= 0)), float(np.mean(differences >= 0)))
            empirical_p = min(1.0, max(empirical_p, 2.0 / (len(differences) + 1)))
            require(np.isclose(empirical_p, row.p_two_sided, atol=1e-12),
                    f"empirical probability mismatch: {year} / {domain} / {row.comparison_group}")
            expected_interval = 100 * np.quantile(differences, [0.025, 0.975])
            require(np.allclose(expected_interval,
                                [row.lower_95_percentage_points, row.upper_95_percentage_points], atol=1e-10),
                    f"paired contrast interval mismatch: {year} / {domain} / {row.comparison_group}")
            comparison = points.loc[
                points.year.eq(year) & points.domain.eq(domain) & points.group.eq(row.comparison_group)
            ].iloc[0]
            reference = points.loc[
                points.year.eq(year) & points.domain.eq(domain) & points.group.eq(row.reference_group)
            ].iloc[0]
            expected_difference = 100 * (
                comparison.zero_joint_access_share - reference.zero_joint_access_share
            )
            require(np.isclose(expected_difference, row.difference_percentage_points, atol=1e-10),
                    f"point-estimate contrast mismatch: {year} / {domain} / {row.comparison_group}")

    valid_ids = set(candidates.site_id.astype(str))
    strict_total = float(priorities.zero_affordability_priority_population.sum())
    joint_total = float(priorities.joint_priority_population.sum())
    require(np.isclose(strict_total, manifest["planning"]["strict_priority_denominator"]),
            "strict-priority denominator does not match released LSBG data")
    require(np.isclose(joint_total, manifest["planning"]["joint_priority_denominator"]),
            "joint-priority denominator does not match released LSBG data")
    for row in scenarios.itertuples(index=False):
        require(row.baseline_outcome == "joint_access", f"scenario uses wrong baseline: {row.scenario}")
        if row.budget_sites == 0:
            continue
        site_ids = set(str(row.selected_site_ids).split("|"))
        require(len(site_ids) == row.budget_sites,
                f"wrong selected-site count: {row.scenario} / K={row.budget_sites}")
        require(site_ids.issubset(valid_ids), f"unknown selected candidate: {row.scenario}")
        released = selected.loc[
            selected.scenario.eq(row.scenario) & selected.budget_sites.eq(row.budget_sites), "site_id"
        ].astype(str)
        require(site_ids == set(released), f"selected-site export mismatch: {row.scenario}")
        require(np.isclose(
            row.zero_affordability_priority_population_reached / strict_total,
            row.zero_affordability_priority_reached_share,
        ), f"strict-priority fraction mismatch: {row.scenario} / K={row.budget_sites}")
        require(np.isclose(row.joint_priority_population_reached / joint_total,
                           row.joint_priority_reached_share),
                f"joint-priority fraction mismatch: {row.scenario} / K={row.budget_sites}")

    restricted_pair = (
        PROJECT / "outputs" / "restricted" / "v7_public_housing_planning_node_coverage_202608"
        / "public_housing_site_lsbg_reachable_pairs_15min.csv"
    ) if PROJECT is not None else None
    if restricted_pair is not None and restricted_pair.is_file():
        require(digest(restricted_pair) == manifest["planning"]["restricted_pair_sha256"],
                "restricted origin-destination-pair SHA-256 mismatch")

    else:
        print("Restricted origin-destination input not supplied: its hash check was SKIPPED.")
    significant = int(contrasts.q_bh_within_domain_year.lt(0.05).sum())
    print(
        "JOINT_DOWNSTREAM_QC=PASS "
        f"domains={points.domain.nunique()} subgroup_years={len(points)} "
        f"contrasts={len(contrasts)} significant_bh={significant} "
        f"bootstrap=999 candidates={len(candidates)} demand_lsbg={len(priorities)} "
        f"planning_scenarios={len(scenarios) - 1}"
        " empirical_probabilities_replayed=51 paired_contrast_intervals_replayed=51"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
