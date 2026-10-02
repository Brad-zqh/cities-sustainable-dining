"""Recalculate current manuscript results using separately authorized inputs.

The V180 numerical replay starts from frozen area components and audited walking
pairs. Raw data acquisition, LLM inference and network construction are separate
upstream steps. Reference tables are read only for comparison after computation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"
TASKS = ("aggregation", "temporal", "structural", "variants", "market", "joint",
         "planning", "components")
FOLDERS = {
    "area": "v134_fixed_nutrition_full_chain",
    "aggregation": "v135_fixed_nutrition_aggregation_variants",
    "temporal": "v135_fixed_nutrition_temporal_uncertainty",
    "structural": "v135_fixed_nutrition_structural_inequality_999",
    "variants": "v135_fixed_nutrition_variant_inequality_999",
    "market": "v135_fixed_nutrition_price_market",
    "joint": "v134_fixed_nutrition_joint_access",
    "planning": "v134_fixed_nutrition_joint_downstream",
    "components": "v135_fixed_nutrition_component_social",
}
# All numeric columns and missingness are compared after one-to-one key alignment.
TABLES = {
    "aggregation": {
        "weight_schemes.csv": ["scheme", "component"],
        "maup_agreement.csv": ["year", "variant"],
        "variant_population_summary.csv": ["year", "scale", "variant"],
        "variant_agreement.csv": ["year", "scale", "variant"],
    },
    "temporal": {name: ["year", "metric"] for name in (
        "annual_block_bootstrap_intervals.csv", "change_from_2011_intervals.csv",
        "metric_coverage_audit.csv")},
    "structural": {
        "sdi_inequality_summary.csv": ["year"],
        "sdi_inequality_block_intervals.csv": ["year", "metric"],
        "sdi_income_quintile_means.csv": ["year", "income_quintile"],
        "sdi_income_concentration_curves.csv": ["year", "population_fraction"],
        "component_income_concentration.csv": ["year", "component"],
    },
    "variants": {
        "sdi_variant_income_inequality_intervals.csv": ["year", "variant_field", "metric"],
        "variant_sample_audit.csv": ["year", "variant_field"],
    },
    "market": {
        "quality_market_decomposition.csv": ["year", "model", "predictor"],
        "price_market_summary.csv": ["year"],
    },
    "joint": {
        "joint_access_point_estimates.csv": ["year", "threshold_min", "price_ceiling_hkd",
                                              "destination_area_sdi_threshold", "outcome"],
        "joint_access_dcca_block_intervals.csv": ["year", "outcome", "metric"],
        "lsbg_joint_quality_affordable_access.csv": ["year", "lsbg_id"],
    },
    "planning": {
        "joint_subgroup_point_estimates.csv": ["year", "domain", "group"],
        "joint_subgroup_dcca_block_intervals.csv": ["year", "domain", "group", "metric"],
        "joint_subgroup_dcca_block_replicates.csv": ["year", "domain", "group", "replicate"],
        "scenario_results.csv": ["scenario", "budget_sites"],
        "selected_planning_nodes.csv": ["scenario", "budget_sites", "site_id"],
        "priority_population_definition.csv": ["lsbg_id"],
    },
    "components": {
        "component_domain_group_means_2024.csv": ["domain", "group", "component"],
        "component_domain_reference_contrasts_2024.csv": ["domain", "component"],
        "weighted_component_income_distributions.csv": ["year", "component", "income_quintile"],
    },
}


def sha256(path: Path) -> str:
    """Hash a file using bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(name: str) -> Any:
    """Import archived computation functions without modifying source text."""
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    sys.path.insert(0, str(ROOT / "src"))
    path = SCRIPTS / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compare_csv(actual: Path, reference: Path, keys: list[str]) -> dict[str, Any]:
    """Compare schema, unique keys, values and NaN states of two result tables.

    Numerical tolerance is 1e-10 absolute and relative; selected site identifiers
    and text labels must match exactly. Complexity is O(n log n) for key sorting.
    """
    import numpy as np
    import pandas as pd

    left, right = pd.read_csv(actual), pd.read_csv(reference)
    if set(left.columns) != set(right.columns):
        raise ValueError(f"{actual.name}: result schema differs")
    if left.duplicated(keys).any() or right.duplicated(keys).any():
        raise ValueError(f"{actual.name}: duplicate comparison keys")
    left = left.sort_values(keys).reset_index(drop=True)
    right = right.sort_values(keys).reset_index(drop=True)
    if len(left) != len(right):
        raise ValueError(f"{actual.name}: row counts differ")
    failures, max_errors = [], {}
    for column in right.columns:
        a, b = left[column], right[column]
        if not a.isna().equals(b.isna()):
            failures.append(column + ": missingness")
            continue
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            aa, bb = a.astype(float).to_numpy(), b.astype(float).to_numpy()
            if not np.allclose(aa, bb, atol=1e-10, rtol=1e-10, equal_nan=True):
                failures.append(column + ": values")
            finite = np.isfinite(aa) & np.isfinite(bb)
            max_errors[column] = float(np.max(np.abs(aa[finite] - bb[finite]))) if finite.any() else 0.0
        elif not a.fillna("").astype(str).equals(b.fillna("").astype(str)):
            failures.append(column + ": labels/identifiers")
    return {"table": actual.name, "rows": len(left), "keys": keys,
            "matches_reference": not failures, "differences": failures,
            "maximum_numeric_errors": max_errors,
            "actual_sha256": sha256(actual), "reference_sha256": sha256(reference)}


def fresh_output(output: Path, data_root: Path) -> None:
    """Reject overlapping source/output trees and previous nonempty runs."""
    output, data_root = output.resolve(), data_root.resolve()
    if output == data_root or output.is_relative_to(data_root) or data_root.is_relative_to(output):
        raise ValueError("Output and input directories must be disjoint")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Choose a fresh output directory: {output}")


def run_task(task: str, args: argparse.Namespace) -> None:
    """Run one frozen estimator with explicit, read-only input locations."""
    import pandas as pd

    data = args.data_root / "outputs" / "restricted"
    dest = args.output_dir / task
    dest.mkdir(parents=True, exist_ok=False)
    area = data / FOLDERS["area"]
    variants = args.output_dir / "aggregation"
    if not variants.is_dir():
        variants = data / FOLDERS["aggregation"]
    if task == "aggregation":
        module = load_module("v135_build_fixed_sdi_aggregation_variants")
        sys.argv = [module.__file__, "--input-dir", str(area), "--output-dir", str(dest)]
        module.main()
    elif task == "temporal":
        module = load_module("v4_temporal_uncertainty_coverage")
        module.IN_DIR, module.OUT_DIR = variants, dest
        module.main()
    elif task == "structural":
        module = load_module("v4_sdi_structural_inequality")
        module.RESTRICTED_INPUT, module.OUTPUT = area, dest
        # Fixed primary-only replay; the seven variants have their own frozen RNG.
        module.VARIANTS = {}
        module.main()
    elif task == "variants":
        module = load_module("v135_run_fixed_variant_inequality")
        sys.argv = [module.__file__, "--input-dir", str(variants), "--output-dir", str(dest),
                    "--bootstrap", "999"]
        module.main()
    elif task == "market":
        module = load_module("v4_price_market_decomposition")
        module.RESTAURANTS, module.SENSITIVITY_DIR, module.OUTPUT_DIR = args.master, variants, dest
        module.main()
    elif task == "joint":
        module = load_module("compute_joint_quality_affordable_access")
        sys.argv = [module.__file__, "--project-root", str(args.project_root),
                    "--master", str(args.master), "--area-dir", str(area),
                    "--output-dir", str(dest), "--bootstrap", "999"]
        module.main()
    elif task == "planning":
        module = load_module("compute_joint_subgroups_planning")
        joint = args.output_dir / "joint" / "lsbg_joint_quality_affordable_access.csv"
        if not joint.is_file():
            joint = data / FOLDERS["joint"] / joint.name
        sys.argv = [module.__file__, "--project-root", str(args.project_root),
                    "--census-root", str(args.census_root), "--joint-file", str(joint),
                    "--candidate-file", str(args.data_root / "source_data/fig04_equity_siting_v7/candidate_planning_nodes.csv"),
                    "--output-dir", str(dest), "--bootstrap", "999"]
        module.main()
        renderer = load_module("fig09_social_within_year")
        renderer.within_year_contrasts(
            pd.read_csv(dest / "joint_subgroup_dcca_block_replicates.csv"),
            pd.read_csv(dest / "joint_subgroup_point_estimates.csv")
        ).to_csv(dest / "within_year_contrasts.csv", index=False)
    elif task == "components":
        module = load_module("fig13_component_social_decomposition")
        subgroup = load_module("compute_joint_subgroups_planning")
        concentration = args.output_dir / "structural" / "component_income_concentration.csv"
        if not concentration.is_file():
            concentration = data / FOLDERS["structural"] / concentration.name
        module.COMPONENT_DATA, module.CONCENTRATION_DATA = area, concentration
        module.census_paths = lambda: subgroup.census_paths(args.census_root)
        means = module.build_group_means()
        means.to_csv(dest / "component_domain_group_means_2024.csv", index=False)
        module.build_contrasts(means).to_csv(dest / "component_domain_reference_contrasts_2024.csv", index=False)
        module.build_distributions().to_csv(dest / "weighted_component_income_distributions.csv", index=False)


def input_hashes(args: argparse.Namespace, tasks: list[str]) -> dict[str, str]:
    """Bind prepared inputs and explicitly used upstream files to a run receipt."""
    paths = set((args.data_root / "source_data").rglob("*"))
    paths |= set((args.data_root / "outputs/restricted").rglob("*"))
    if "market" in tasks or "joint" in tasks:
        paths.add(args.master)
    if "planning" in tasks or "components" in tasks:
        module = load_module("compute_joint_subgroups_planning")
        paths.update(module.census_paths(args.census_root).values())
    if "joint" in tasks:
        for year in (2016, 2021, 2024):
            folder = "t3_count_candidate" if year == 2021 else f"t3_count_candidate_{year}"
            paths.add(args.project_root / f"outputs/{folder}/primary_narrow/eligible_restaurant_{year}_primary_narrow_count_opportunity.csv")
            paths.add(args.project_root / f"outputs/restricted/t3_reachable_pairs_{year}/lsbg_restaurant_reachable_pairs_{year}_15min.csv")
        paths.add(args.project_root / "source_data/fig04_network_price_v4/lsbg_network_price_opportunity.csv")
    if "planning" in tasks:
        paths.add(args.project_root / "outputs/restricted/v7_public_housing_planning_node_coverage_202608/public_housing_site_lsbg_reachable_pairs_15min.csv")
    files = {str(path.resolve()): sha256(path) for path in sorted(paths) if path.is_file()}
    if not files:
        raise FileNotFoundError("No prepared empirical inputs found")
    return files


def main() -> int:
    """Dispatch isolated worker processes and save an evidence-backed receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="Authorized prepared-bundle root")
    parser.add_argument("--output-dir", type=Path, required=True, help="Fresh directory outside the input bundle")
    parser.add_argument("--project-root", type=Path, help="Authorized eligible-outlet and walking-pair tree")
    parser.add_argument("--census-root", type=Path, help="Directory containing 2016/ and 2021/ census inputs")
    parser.add_argument("--master", type=Path, help="Authorized restaurant master CSV")
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--authorized", action="store_true", help="Confirm permission to use the supplied private inputs")
    parser.add_argument("--worker", choices=TASKS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.authorized:
        parser.error("Supply --authorized only when you have permission to use these inputs")
    for field in ("data_root", "output_dir", "project_root", "census_root", "master"):
        value = getattr(args, field)
        if value is not None:
            setattr(args, field, value.resolve())
    tasks = args.tasks.split(",")
    if not tasks or len(set(tasks)) != len(tasks) or any(task not in TASKS for task in tasks):
        parser.error("--tasks must contain distinct names from: " + ",".join(TASKS))
    if tasks != sorted(tasks, key=TASKS.index):
        parser.error("Keep task dependency order: " + ",".join(TASKS))
    for required, condition in (
        ("master", bool({"market", "joint"}.intersection(tasks))),
        ("project_root", bool({"joint", "planning"}.intersection(tasks))),
        ("census_root", bool({"planning", "components"}.intersection(tasks))),
    ):
        if condition and getattr(args, required) is None:
            parser.error("Required for requested tasks: --" + required.replace("_", "-"))
    if args.worker:
        run_task(args.worker, args)
        return 0
    fresh_output(args.output_dir, args.data_root)
    # Protect upstream input trees as well as the prepared bundle.
    for source in (args.project_root, args.census_root):
        if source is not None:
            fresh_output(args.output_dir, source)
    before = input_hashes(args, tasks)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "edition": "V180", "python": sys.version, "platform": platform.platform(),
        "tasks": [], "input_sha256": before,
        "source_sha256": {path.relative_to(ROOT).as_posix(): sha256(path)
                          for path in [Path(__file__), *SCRIPTS.glob("*.py"), *(ROOT / "src").rglob("*.py")]},
        "upstream_collection_llm_inference_network_build_reexecuted": False,
        "status": "RUNNING",
    }
    receipt = args.output_dir / "verification.json"
    try:
        for task in tasks:
            command = [sys.executable, "-X", "utf8", str(Path(__file__)),
                       "--worker", task, "--tasks", args.tasks,
                       "--data-root", str(args.data_root), "--output-dir", str(args.output_dir), "--authorized"]
            for field in ("project_root", "census_root", "master"):
                if getattr(args, field) is not None:
                    command.extend(["--" + field.replace("_", "-"), str(getattr(args, field))])
            started = time.monotonic()
            print(f"Recalculating {task}...", flush=True)
            with (args.output_dir / f"{task}.log").open("w", encoding="utf-8") as log:
                process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
            run: dict[str, Any] = {"task": task, "command": command, "returncode": process.returncode,
                                   "elapsed_seconds": time.monotonic() - started, "comparisons": []}
            report["tasks"].append(run)
            if process.returncode:
                raise RuntimeError(f"{task} failed; inspect {task}.log")
            reference = args.data_root / "outputs/restricted" / FOLDERS[task]
            for name, keys in TABLES[task].items():
                run["comparisons"].append(compare_csv(args.output_dir / task / name, reference / name, keys))
            if task == "planning":
                run["comparisons"].append(compare_csv(
                    args.output_dir / task / "within_year_contrasts.csv",
                    args.data_root / "outputs/restricted/v135_fixed_nutrition_subgroup_contrasts/within_year_contrasts.csv",
                    ["year", "domain", "reference_group", "comparison_group"]))
            if not all(check["matches_reference"] for check in run["comparisons"]):
                raise ValueError(f"{task}: differences found; inspect verification.json")
            print(f"{task}: {len(run['comparisons'])} result tables match", flush=True)
            receipt.write_text(json.dumps(report, indent=2), encoding="utf-8")
        report["inputs_unchanged"] = before == input_hashes(args, tasks)
        if not report["inputs_unchanged"]:
            raise RuntimeError("Input files changed during replay")
        report["status"] = "MATCHES_FROZEN_REFERENCE"
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        report["status"], report["error"] = "FAILED", str(exc)
        raise
    finally:
        receipt.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
