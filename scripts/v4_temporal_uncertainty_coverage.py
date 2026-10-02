"""DCCA-block uncertainty and early-year coverage sensitivity for v4 LSBG results."""

from __future__ import annotations

import json
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
IN_DIR = Path(os.environ.get(
    "CITIES_SDI_VARIANT_DIR",
    str(ROOT / "outputs" / "restricted" / "v4_sdi_sensitivity"),
))
OUT_DIR = Path(os.environ.get(
    "CITIES_TEMPORAL_OUTPUT_DIR",
    str(ROOT / "outputs" / "restricted" / "v4_temporal_uncertainty_coverage"),
))
YEARS = [2011, 2016, 2021, 2024]
METRICS = {
    "nutrition_score": "Nutrition",
    "carbon_score": "Carbon",
    "sustainability_score": "Sustainability practice text",
    "hygiene_score": "Hygiene",
    "practice_score": "Current practice tag (backcast)",
    "diversity_score": "Cuisine diversity",
    "equal_six_strict": "Equal six (strict)",
    "equal_five_no_practice": "Equal five (no practice)",
    "equal_six_highcoverage": "Equal six (coverage >=50%)",
}
N_BOOT = 999
SEED = 20260820


def block_bootstrap(df: pd.DataFrame, metric: str, rng: np.random.Generator) -> tuple[float, np.ndarray, int, float, int]:
    work = df[["dcca", "t_pop", metric]].copy()
    work = work[work.dcca.notna() & work.t_pop.gt(0) & np.isfinite(work[metric])]
    if work.empty:
        return np.nan, np.full(N_BOOT, np.nan), 0, 0.0, 0
    work["wx"] = work.t_pop * work[metric]
    grouped = work.groupby("dcca", observed=True).agg(wx=("wx", "sum"), w=("t_pop", "sum"))
    wx, w = grouped.wx.to_numpy(float), grouped.w.to_numpy(float)
    point = wx.sum() / w.sum()
    pick = rng.integers(0, len(grouped), size=(N_BOOT, len(grouped)))
    boot = wx[pick].sum(axis=1) / w[pick].sum(axis=1)
    return float(point), boot, int(len(work)), float(w.sum()), int(len(grouped))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    annual_rows: list[dict] = []
    boot_store: dict[tuple[int, str], np.ndarray] = {}
    frames: dict[int, pd.DataFrame] = {}
    crosswalk_rows: list[dict] = []
    for year in YEARS:
        frame = gpd.read_file(IN_DIR / f"{year}_lsbg_sensitivity.gpkg")
        if frame["dcca"].notna().sum() == 0:
            dcca = gpd.read_file(IN_DIR / f"{year}_dcca_sensitivity.gpkg")[["dcca", "geometry"]]
            points = frame[["geometry"]].copy()
            points["geometry"] = points.geometry.representative_point()
            joined = gpd.sjoin(points, dcca, how="left", predicate="within")
            if joined.index.duplicated().any():
                raise ValueError(f"{year}: non-unique LSBG representative-point to DCCA crosswalk")
            missing_idx = joined.index[joined["dcca"].isna()]
            nearest_max = 0.0
            if len(missing_idx):
                nearest = gpd.sjoin_nearest(points.loc[missing_idx], dcca, how="left", distance_col="distance_m")
                if nearest.index.duplicated().any() or nearest.dcca.isna().any():
                    raise ValueError(f"{year}: nearest-DCCA fallback is not one-to-one")
                nearest_max = float(nearest.distance_m.max())
                if nearest_max > 500:
                    raise ValueError(f"{year}: nearest-DCCA fallback exceeds 500 m ({nearest_max:.1f} m)")
                joined.loc[missing_idx, "dcca"] = nearest.loc[missing_idx, "dcca"]
            crosswalk_rows.append(
                {"year": year, "lsbg_n": len(frame), "within_n": len(frame)-len(missing_idx),
                 "nearest_fallback_n": len(missing_idx), "nearest_fallback_max_m": nearest_max}
            )
            frame["dcca"] = joined.loc[frame.index, "dcca"].to_numpy()
        frames[year] = frame
        for metric, label in METRICS.items():
            point, boot, unit_n, pop_n, block_n = block_bootstrap(frame, metric, rng)
            boot_store[(year, metric)] = boot
            lo, hi = np.nanquantile(boot, [.025, .975])
            annual_rows.append(
                {
                    "year": year,
                    "metric": metric,
                    "metric_label": label,
                    "population_weighted_mean": point,
                    "ci_low": lo,
                    "ci_high": hi,
                    "valid_lsbg_n": unit_n,
                    "valid_population": pop_n,
                    "dcca_block_n": block_n,
                    "bootstrap_replicates": N_BOOT,
                }
            )
    annual = pd.DataFrame(annual_rows)
    annual.to_csv(OUT_DIR / "annual_block_bootstrap_intervals.csv", index=False)
    pd.DataFrame(crosswalk_rows).to_csv(OUT_DIR / "lsbg_dcca_crosswalk_qa.csv", index=False)

    change_rows: list[dict] = []
    for metric, label in METRICS.items():
        base_point = annual[(annual.year == 2011) & (annual.metric == metric)].population_weighted_mean.iloc[0]
        for year in YEARS[1:]:
            point = annual[(annual.year == year) & (annual.metric == metric)].population_weighted_mean.iloc[0]
            delta_boot = boot_store[(year, metric)] - boot_store[(2011, metric)]
            lo, hi = np.nanquantile(delta_boot, [.025, .975])
            change_rows.append(
                {
                    "reference_year": 2011,
                    "year": year,
                    "metric": metric,
                    "metric_label": label,
                    "difference": point - base_point,
                    "ci_low": lo,
                    "ci_high": hi,
                    "bootstrap_replicates": N_BOOT,
                }
            )
    pd.DataFrame(change_rows).to_csv(OUT_DIR / "change_from_2011_intervals.csv", index=False)

    coverage_rows: list[dict] = []
    for year, frame in frames.items():
        total_pop = frame.loc[frame.t_pop.gt(0), "t_pop"].sum()
        total_units = frame.t_pop.gt(0).sum()
        for metric, label in METRICS.items():
            valid = frame.t_pop.gt(0) & np.isfinite(frame[metric])
            coverage_rows.append(
                {
                    "year": year,
                    "metric": metric,
                    "metric_label": label,
                    "valid_lsbg_n": int(valid.sum()),
                    "all_lsbg_n": int(total_units),
                    "lsbg_coverage_share": float(valid.sum() / total_units),
                    "valid_population": float(frame.loc[valid, "t_pop"].sum()),
                    "all_population": float(total_pop),
                    "population_coverage_share": float(frame.loc[valid, "t_pop"].sum() / total_pop),
                }
            )
    coverage = pd.DataFrame(coverage_rows)
    strict_pop = coverage[coverage.metric == "equal_six_strict"][["year", "valid_population"]].rename(
        columns={"valid_population": "strict_valid_population"}
    )
    coverage = coverage.merge(strict_pop, on="year", how="left", validate="many_to_one")
    coverage["population_share_of_strict_denominator"] = coverage.valid_population / coverage.strict_valid_population
    coverage.to_csv(OUT_DIR / "metric_coverage_audit.csv", index=False)

    strict = annual[annual.metric == "equal_six_strict"].set_index("year")
    high = annual[annual.metric == "equal_six_highcoverage"].set_index("year")
    high_cov = coverage[coverage.metric == "equal_six_highcoverage"].set_index("year")
    summary = {
        "seed": SEED,
        "bootstrap_replicates": N_BOOT,
        "block": "DCCA",
        "strict_means": strict.population_weighted_mean.to_dict(),
        "highcoverage_means": high.population_weighted_mean.to_dict(),
        "highcoverage_population_share_full_geography": high_cov.population_coverage_share.to_dict(),
        "highcoverage_population_share_of_strict_denominator": high_cov.population_share_of_strict_denominator.to_dict(),
        "2011_high_minus_strict": float(high.loc[2011, "population_weighted_mean"] - strict.loc[2011, "population_weighted_mean"]),
        "highcoverage_monotonic_non_decreasing": bool(np.all(np.diff(high.loc[YEARS, "population_weighted_mean"]) >= 0)),
        "interpretation_boundary": "Spatial-block uncertainty for repeated ecological cross-sections; not a causal trend or individual-level effect.",
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
