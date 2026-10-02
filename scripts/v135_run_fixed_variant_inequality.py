"""DCCA-block uncertainty for all fixed-Nutrition aggregation variants."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
import os

import geopandas as gpd
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "outputs" / "restricted" / "v135_fixed_nutrition_aggregation_variants"
DEFAULT_OUTPUT = ROOT / "outputs" / "restricted" / "v135_fixed_nutrition_variant_inequality_999"
SEED = 20260923


def load_v4_module():
    source = ROOT / "scripts" / "v4_sdi_structural_inequality.py"
    spec = importlib.util.spec_from_file_location("v4_inequality_reused", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load V4 inequality functions: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap", type=int, default=999)
    args = parser.parse_args()
    if args.bootstrap < 1:
        raise ValueError("--bootstrap must be positive")
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    module, source = load_v4_module()
    rows = []
    audit = []
    for year in module.YEARS:
        path = input_dir / f"{year}_lsbg_sensitivity.gpkg"
        raw = gpd.read_file(path).drop(columns="geometry")
        for variant_index, (column, label) in enumerate(module.VARIANTS.items()):
            needed = {column, "t_pop", "ma_hh", "dcca"}
            if not needed.issubset(raw.columns):
                raise ValueError(f"{path.name}: missing {needed.difference(raw.columns)}")
            frame = raw[[column, "t_pop", "ma_hh", "dcca"]].replace(
                [np.inf, -np.inf], np.nan
            ).dropna().copy()
            for field in (column, "t_pop", "ma_hh"):
                frame[field] = pd.to_numeric(frame[field], errors="coerce")
            frame = frame.dropna().loc[
                frame["t_pop"].gt(0) & frame["ma_hh"].gt(0) & frame[column].ge(0)
            ].copy()
            frame["dcca"] = frame["dcca"].astype(str)
            if frame["dcca"].nunique() < 2:
                raise ValueError(f"{year} {column}: fewer than two DCCA blocks")
            point = module.variant_metrics(frame, column)
            groups = [group for _, group in frame.groupby("dcca", sort=True)]
            rng = np.random.default_rng(SEED + year * 10 + variant_index)
            replicated = []
            for repetition in range(args.bootstrap):
                selected = rng.integers(0, len(groups), len(groups))
                sample = pd.concat([groups[int(index)] for index in selected], ignore_index=True)
                replicated.append(module.variant_metrics(sample, column))
            table = pd.DataFrame(replicated)
            for metric in ("concentration_index", "income_quintile_relative_ratio"):
                finite = pd.to_numeric(table[metric], errors="coerce").replace(
                    [np.inf, -np.inf], np.nan
                ).dropna()
                if len(finite) < args.bootstrap * 0.95:
                    raise RuntimeError(f"{year} {column} {metric}: too few valid replicates")
                rows.append({
                    "year": year,
                    "variant": label,
                    "variant_field": column,
                    "metric": metric,
                    "estimate": point[metric],
                    "lower_95": float(finite.quantile(0.025)),
                    "upper_95": float(finite.quantile(0.975)),
                    "lsbg_n": len(frame),
                    "population_analyzed": float(frame["t_pop"].sum()),
                    "dcca_blocks": len(groups),
                    "valid_replicates": len(finite),
                })
            audit.append({
                "year": year,
                "variant_field": column,
                "input_units": len(raw),
                "analyzed_units": len(frame),
                "excluded_units": len(raw) - len(frame),
                "blocks": len(groups),
            })
            print(f"{year} {column}: {len(frame)} units, {len(groups)} DCCA blocks", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_dir / "sdi_variant_income_inequality_intervals.csv", index=False)
    pd.DataFrame(audit).to_csv(output_dir / "variant_sample_audit.csv", index=False)
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "restricted_diagnostic_not_submission_authorized",
        "input_dir": str(input_dir),
        "source_script": str(source),
        "source_script_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "bootstrap_repetitions": args.bootstrap,
        "block_unit": "DCCA",
        "seed_base": SEED,
        "outcomes": list(module.VARIANTS),
        "years": list(module.YEARS),
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
