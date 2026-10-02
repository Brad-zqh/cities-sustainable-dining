from __future__ import annotations

import json
import sys
from pathlib import Path
import os

import numpy as np
import pandas as pd
import geopandas as gpd


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("CITIES_DATA_ROOT", str(ROOT)))
sys.path.insert(0, str(ROOT / "src"))

from sus_dining_access.inequality import (  # noqa: E402
    concentration_index,
    population_quintile_means,
    summarize_inequality,
)


YEARS = (2011, 2016, 2021, 2024)
COMPONENTS = {
    "nutrition_score": "Nutrition",
    "carbon_score": "Carbon",
    "diversity_score": "Cuisine diversity",
    "sustainability_score": "Sustainability text",
    "hygiene_score": "Hygiene",
    "practice_score": "Practice tag*",
}
VARIANTS = {
    "equal_six_strict": "Equal six (strict)",
    "equal_five_no_practice": "Equal five (no practice)",
    "equal_six_highcoverage": "Equal six (≥50% coverage)",
    "equal_six_available4": "Available components (≥4/6)",
    "geometric_six": "Geometric six",
    "pca_six": "Absolute-PC1 weights",
    "entropy_six": "Entropy weights",
}
REPETITIONS = 999
SEED = 20260821
INPUT = DATA_ROOT / "source_data" / "fig_v4_four_year"
RESTRICTED_INPUT = ROOT / "outputs" / "restricted" / "v4_price_market_decomposition"
OUTPUT = DATA_ROOT / "source_data" / "figS_sdi_structural_inequality_v4"


def clean_frame(year: int, outcome: str) -> pd.DataFrame:
    frame = gpd.read_file(RESTRICTED_INPUT / f"{year}_lsbg_price_market.gpkg").drop(columns="geometry")
    frame["lsbg"] = frame["lsbg"].astype(str)
    frame["dcca"] = frame["dcca"].astype(str)
    keep = ["lsbg", "dcca", "t_pop", "ma_hh", outcome]
    frame = frame[keep].replace([np.inf, -np.inf], np.nan).dropna().copy()
    for column in ["t_pop", "ma_hh", outcome]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna().loc[
        frame["t_pop"].gt(0) & frame["ma_hh"].gt(0) & frame[outcome].ge(0)
    ]
    return frame


def dashboard(frame: pd.DataFrame, outcome: str) -> dict[str, float]:
    table = summarize_inequality(
        frame[outcome].to_numpy(float),
        frame["t_pop"].to_numpy(float),
        frame["ma_hh"].to_numpy(float),
    ).set_index("metric")["estimate"]
    return {str(key): float(value) for key, value in table.items()}


def bootstrap(frame: pd.DataFrame, outcome: str, rng: np.random.Generator) -> pd.DataFrame:
    groups = [group.copy() for _, group in frame.groupby("dcca", sort=False)]
    if len(groups) < 2:
        raise RuntimeError("DCCA block bootstrap requires at least two non-empty clusters")
    metrics = ["weighted_gini", "theil_t", "concentration_index", "income_quintile_relative_ratio"]
    rows: list[dict[str, float | int | str]] = []
    for repetition in range(REPETITIONS):
        sampled = rng.integers(0, len(groups), size=len(groups))
        replicate = pd.concat([groups[index] for index in sampled], ignore_index=True)
        estimates = dashboard(replicate, outcome)
        for metric in metrics:
            rows.append(
                {"replicate": repetition, "metric": metric, "estimate": estimates[metric]}
            )
    return pd.DataFrame(rows)


def concentration_curve(frame: pd.DataFrame, outcome: str) -> pd.DataFrame:
    data = frame.sort_values(["ma_hh", "lsbg"], kind="mergesort").copy()
    weighted_value = data[outcome].to_numpy(float) * data["t_pop"].to_numpy(float)
    cumulative_population = np.r_[0.0, np.cumsum(data["t_pop"].to_numpy(float))]
    cumulative_value = np.r_[0.0, np.cumsum(weighted_value)]
    cumulative_population /= cumulative_population[-1]
    cumulative_value /= cumulative_value[-1]
    grid = np.linspace(0.0, 1.0, 101)
    return pd.DataFrame(
        {
            "population_fraction": grid,
            "cumulative_sdi_fraction": np.interp(grid, cumulative_population, cumulative_value),
        }
    )


def variant_metrics(frame: pd.DataFrame, outcome: str) -> dict[str, float]:
    quintiles = population_quintile_means(frame[outcome], frame["t_pop"], frame["ma_hh"])
    return {
        "concentration_index": concentration_index(frame[outcome], frame["t_pop"], frame["ma_hh"]),
        "income_quintile_relative_ratio": float(quintiles[-1] / quintiles[0]),
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    summary_rows = []
    interval_rows = []
    quintile_rows = []
    curve_rows = []
    component_rows = []
    variant_interval_rows = []

    for year in YEARS:
        frame = clean_frame(year, "sdi_equal_arithmetic")
        estimates = dashboard(frame, "sdi_equal_arithmetic")
        summary_rows.append(
            {
                "year": year,
                "lsbg_n": len(frame),
                "population_analyzed": frame["t_pop"].sum(),
                **estimates,
            }
        )
        replicates = bootstrap(frame, "sdi_equal_arithmetic", rng)
        for metric, values in replicates.groupby("metric", sort=False)["estimate"]:
            finite = values[np.isfinite(values)]
            interval_rows.append(
                {
                    "year": year,
                    "metric": metric,
                    "estimate": estimates[metric],
                    "lower_95": finite.quantile(0.025),
                    "upper_95": finite.quantile(0.975),
                    "valid_replicates": len(finite),
                }
            )
        quintiles = population_quintile_means(
            frame["sdi_equal_arithmetic"], frame["t_pop"], frame["ma_hh"]
        )
        for quintile, value in enumerate(quintiles, start=1):
            quintile_rows.append({"year": year, "income_quintile": quintile, "mean_sdi": value})
        curve = concentration_curve(frame, "sdi_equal_arithmetic")
        curve["year"] = year
        curve_rows.append(curve)

        raw = gpd.read_file(RESTRICTED_INPUT / f"{year}_lsbg_price_market.gpkg").drop(columns="geometry")
        for column, label in COMPONENTS.items():
            subset = raw[[column, "t_pop", "ma_hh"]].replace([np.inf, -np.inf], np.nan).dropna()
            subset = subset.loc[subset["t_pop"].gt(0) & subset["ma_hh"].gt(0) & subset[column].ge(0)]
            component_rows.append(
                {
                    "year": year,
                    "component": label,
                    "concentration_index": concentration_index(subset[column], subset["t_pop"], subset["ma_hh"]),
                    "lsbg_n": len(subset),
                    "population_analyzed": subset["t_pop"].sum(),
                }
            )

        for column, label in VARIANTS.items():
            subset = raw[[column, "t_pop", "ma_hh", "dcca"]].replace([np.inf, -np.inf], np.nan).dropna()
            subset = subset.loc[subset["t_pop"].gt(0) & subset["ma_hh"].gt(0) & subset[column].ge(0)].copy()
            point = variant_metrics(subset, column)
            groups = [group.copy() for _, group in subset.groupby("dcca", sort=False)]
            replicate_rows = []
            for _ in range(REPETITIONS):
                sampled = rng.integers(0, len(groups), size=len(groups))
                replicate = pd.concat([groups[index] for index in sampled], ignore_index=True)
                replicate_rows.append(variant_metrics(replicate, column))
            replicate_table = pd.DataFrame(replicate_rows)
            for metric in ["concentration_index", "income_quintile_relative_ratio"]:
                values = replicate_table[metric].replace([np.inf, -np.inf], np.nan).dropna()
                variant_interval_rows.append(
                    {
                        "year": year,
                        "variant": label,
                        "metric": metric,
                        "estimate": point[metric],
                        "lower_95": values.quantile(0.025),
                        "upper_95": values.quantile(0.975),
                        "lsbg_n": len(subset),
                        "population_analyzed": subset["t_pop"].sum(),
                        "valid_replicates": len(values),
                    }
                )

    summary = pd.DataFrame(summary_rows)
    intervals = pd.DataFrame(interval_rows)
    quintiles = pd.DataFrame(quintile_rows)
    curves = pd.concat(curve_rows, ignore_index=True)
    components = pd.DataFrame(component_rows)
    variant_intervals = pd.DataFrame(variant_interval_rows)
    summary.to_csv(OUTPUT / "sdi_inequality_summary.csv", index=False)
    intervals.to_csv(OUTPUT / "sdi_inequality_block_intervals.csv", index=False)
    quintiles.to_csv(OUTPUT / "sdi_income_quintile_means.csv", index=False)
    curves.to_csv(OUTPUT / "sdi_income_concentration_curves.csv", index=False)
    components.to_csv(OUTPUT / "component_income_concentration.csv", index=False)
    variant_intervals.to_csv(OUTPUT / "sdi_variant_income_inequality_intervals.csv", index=False)
    contract = {
        "scope": "population-weighted structural inequality in LSBG area SDI; not walking accessibility or household affordability",
        "years": list(YEARS),
        "population_reference": (
            "native 2011/2016/2021 census population and income; "
            "2024 restaurant evidence uses fixed 2021 population-income geography"
        ),
        "primary_outcome": "strict six-component equal-arithmetic SDI",
        "uncertainty": f"{REPETITIONS} DCCA spatial-block bootstrap replicates",
        "practice_note": "practice tag is a historical-backcast sensitivity because participation dates are unavailable",
        "seed": SEED,
    }
    (OUTPUT / "analysis_contract.json").write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    print(summary[["year", "weighted_gini", "theil_t", "concentration_index", "income_quintile_relative_ratio"]].to_string(index=False))


if __name__ == "__main__":
    main()
