"""Recompute the seven SDI aggregation sensitivities from a frozen area bundle.

This wrapper reuses the V4 aggregation functions without changing the frozen
V4 outputs. The fixed-Nutrition input is the completed price-market GeoPackage,
which retains the six area component columns and the original area geometry.
"""

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
DEFAULT_INPUT = ROOT / "outputs" / "restricted" / "v134_fixed_nutrition_full_chain"
DEFAULT_OUTPUT = ROOT / "outputs" / "restricted" / "v135_fixed_nutrition_aggregation_variants"


def load_v4_module():
    source = ROOT / "scripts" / "v4_sdi_sensitivity_analysis.py"
    spec = importlib.util.spec_from_file_location("v4_sdi_sensitivity_reused", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load V4 SDI sensitivity functions: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, source


def verify_primary(frame: gpd.GeoDataFrame, module, key: tuple[int, str]) -> None:
    needed = set(module.COMPONENTS) | {
        "sdi_equal_arithmetic", "nutrition_coverage", "carbon_coverage", "t_pop"
    }
    missing = sorted(needed.difference(frame.columns))
    if missing:
        raise ValueError(f"{key} lacks columns: {missing}")
    if frame.crs is None:
        raise ValueError(f"{key} has no CRS")
    for column in module.COMPONENTS:
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        if not values.between(-1e-12, 1 + 1e-12).all():
            raise ValueError(f"{key} {column} is outside [0,1]")
    expected = frame[module.COMPONENTS].mean(axis=1, skipna=False)
    reported = pd.to_numeric(frame["sdi_equal_arithmetic"], errors="coerce")
    if not expected.notna().equals(reported.notna()):
        raise ValueError(f"{key} has a different primary-SDI missingness rule")
    if not np.allclose(expected.dropna(), reported.dropna(), atol=1e-10, rtol=0):
        raise ValueError(f"{key} primary SDI differs from the six-component mean")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(input_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing sensitivity bundle: {output_dir}")

    module, source = load_v4_module()
    raw_frames = {}
    for year in module.YEARS:
        for scale in module.SCALES:
            path = input_dir / f"{year}_{scale}_price_market.gpkg"
            frame = gpd.read_file(path)
            verify_primary(frame, module, (year, scale))
            raw_frames[(year, scale)] = frame

    pooled = pd.concat(
        [raw_frames[(year, "lsbg")][module.COMPONENTS] for year in module.YEARS],
        ignore_index=True,
    )
    weights, weight_metadata = module.fit_weights(pooled)
    frames = {key: module.add_variants(frame, weights) for key, frame in raw_frames.items()}
    for key, frame in frames.items():
        if not np.allclose(
            frame["equal_six_strict"].dropna(),
            frame["sdi_equal_arithmetic"].dropna(),
            atol=1e-10,
            rtol=0,
        ):
            raise ValueError(f"{key}: aggregation reference drift")
        for variant in module.VARIANTS:
            values = pd.to_numeric(frame[variant], errors="coerce").dropna()
            if not values.between(-1e-10, 1 + 1e-10).all():
                raise ValueError(f"{key} {variant} is outside [0,1]")

    summary, agreement = module.summarize_variants(frames)
    maup = module.maup_agreement(frames)
    output_dir.mkdir(parents=True, exist_ok=True)
    weights.assign(component_label=weights["component"].map(module.COMPONENT_LABELS)).to_csv(
        output_dir / "weight_schemes.csv", index=False
    )
    summary.to_csv(output_dir / "variant_population_summary.csv", index=False)
    agreement.to_csv(output_dir / "variant_agreement.csv", index=False)
    maup.to_csv(output_dir / "maup_agreement.csv", index=False)
    for (year, scale), frame in frames.items():
        frame.to_file(output_dir / f"{year}_{scale}_sensitivity.gpkg", driver="GPKG")

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "restricted_diagnostic_not_submission_authorized",
        "nutrition_definition": "fixed bounded nutrient-density score",
        "years": module.YEARS,
        "scales": [scale.upper() for scale in module.SCALES],
        "variants": module.VARIANTS,
        "primary_equality_check": "all eight equal_six_strict fields equal the source six-component SDI",
        "input_dir": str(input_dir),
        "source_script": str(source),
        "source_script_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "pooled_reference": "four-year complete LSBG observations",
        **weight_metadata,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(summary.loc[summary["scale"].eq("LSBG")].pivot(
        index="variant", columns="year", values="population_weighted_mean"
    ).round(6).to_string())


if __name__ == "__main__":
    main()
