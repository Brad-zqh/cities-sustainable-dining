from __future__ import annotations

import json
from pathlib import Path
import os

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


YEARS = [2011, 2016, 2021, 2024]
SCALES = ["lsbg", "dcca"]
COMPONENTS = [
    "nutrition_score",
    "carbon_score",
    "diversity_score",
    "sustainability_score",
    "hygiene_score",
    "practice_score",
]
COMPONENT_LABELS = {
    "nutrition_score": "Nutrition",
    "carbon_score": "Carbon",
    "diversity_score": "Cuisine diversity",
    "sustainability_score": "Sustainability signal",
    "hygiene_score": "Hygiene",
    "practice_score": "Practice tag",
}

ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "outputs" / "restricted" / "v4_four_year_dual_scale_components"
OUTPUT_DIR = ROOT / "outputs" / "restricted" / "v4_sdi_sensitivity"


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce")
    weights = pd.to_numeric(weights, errors="coerce")
    valid = values.notna() & weights.notna() & weights.gt(0)
    if not valid.any():
        return np.nan
    return float(np.average(values[valid], weights=weights[valid]))


def safe_corr(a: pd.Series, b: pd.Series, method: str) -> float:
    pair = pd.concat([a, b], axis=1).dropna()
    if len(pair) < 3 or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return np.nan
    if method == "pearson":
        return float(pearsonr(pair.iloc[:, 0], pair.iloc[:, 1]).statistic)
    return float(spearmanr(pair.iloc[:, 0], pair.iloc[:, 1]).statistic)


def tail_jaccard(reference: pd.Series, candidate: pd.Series, tail: str) -> float:
    pair = pd.concat([reference, candidate], axis=1).dropna()
    if len(pair) < 10:
        return np.nan
    ref_q = pair.iloc[:, 0].quantile(0.2 if tail == "low" else 0.8)
    can_q = pair.iloc[:, 1].quantile(0.2 if tail == "low" else 0.8)
    if tail == "low":
        ref_set = set(pair.index[pair.iloc[:, 0] <= ref_q])
        can_set = set(pair.index[pair.iloc[:, 1] <= can_q])
    else:
        ref_set = set(pair.index[pair.iloc[:, 0] >= ref_q])
        can_set = set(pair.index[pair.iloc[:, 1] >= can_q])
    union = ref_set | can_set
    return float(len(ref_set & can_set) / len(union)) if union else np.nan


def fit_weights(reference: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    complete = reference[COMPONENTS].dropna()
    if len(complete) < 20:
        raise RuntimeError("Insufficient complete LSBG observations for pooled weights")

    matrix = complete.to_numpy(dtype=float)
    means = matrix.mean(axis=0)
    standard_deviations = matrix.std(axis=0, ddof=0)
    if np.any(standard_deviations == 0):
        raise RuntimeError("PCA weighting cannot use a constant component")
    standardized = (matrix - means) / standard_deviations
    _, singular_values, vt = np.linalg.svd(standardized, full_matrices=False)
    pca_weights = np.abs(vt[0])
    pca_weights = pca_weights / pca_weights.sum()
    explained_variance_ratio = float(
        singular_values[0] ** 2 / np.square(singular_values).sum()
    )

    x = complete.to_numpy(dtype=float)
    x = np.clip(x, 0.0, 1.0)
    column_sums = x.sum(axis=0)
    p = np.divide(x, column_sums, out=np.zeros_like(x), where=column_sums > 0)
    logp = np.zeros_like(p)
    positive = p > 0
    logp[positive] = np.log(p[positive])
    entropy = -(p * logp).sum(axis=0) / np.log(len(complete))
    divergence = 1.0 - entropy
    entropy_weights = divergence / divergence.sum()

    rows = []
    for component, pca_weight, entropy_weight in zip(
        COMPONENTS, pca_weights, entropy_weights, strict=True
    ):
        rows.extend(
            [
                {
                    "scheme": "Equal six",
                    "component": component,
                    "weight": 1.0 / len(COMPONENTS),
                },
                {
                    "scheme": "Absolute PC1 loading",
                    "component": component,
                    "weight": float(pca_weight),
                },
                {
                    "scheme": "Entropy",
                    "component": component,
                    "weight": float(entropy_weight),
                },
            ]
        )
    metadata = {
        "pooled_complete_lsbg_n": int(len(complete)),
        "pc1_explained_variance_ratio": explained_variance_ratio,
    }
    return pd.DataFrame(rows), metadata


def add_variants(frame: gpd.GeoDataFrame, weights: pd.DataFrame) -> gpd.GeoDataFrame:
    output = frame.copy()
    pca_weights = (
        weights.loc[weights["scheme"] == "Absolute PC1 loading"]
        .set_index("component")["weight"]
        .reindex(COMPONENTS)
    )
    entropy_weights = (
        weights.loc[weights["scheme"] == "Entropy"]
        .set_index("component")["weight"]
        .reindex(COMPONENTS)
    )
    complete = output[COMPONENTS].notna().all(axis=1)
    at_least_four = output[COMPONENTS].notna().sum(axis=1).ge(4)
    no_practice_components = [c for c in COMPONENTS if c != "practice_score"]
    complete_no_practice = output[no_practice_components].notna().all(axis=1)

    output["equal_six_strict"] = output[COMPONENTS].mean(axis=1, skipna=False)
    output["equal_six_available4"] = output[COMPONENTS].mean(axis=1, skipna=True).where(
        at_least_four
    )
    output["equal_five_no_practice"] = output[no_practice_components].mean(
        axis=1, skipna=False
    )
    geometric_input = output[COMPONENTS]
    geometric_complete = geometric_input.notna().all(axis=1)
    geometric_zero = geometric_input.eq(0.0).any(axis=1)
    output["geometric_six"] = np.nan
    positive_complete = geometric_complete & ~geometric_zero
    output.loc[positive_complete, "geometric_six"] = np.exp(
        np.log(geometric_input.loc[positive_complete]).mean(axis=1)
    )
    output.loc[geometric_complete & geometric_zero, "geometric_six"] = 0.0
    output["pca_six"] = output[COMPONENTS].mul(pca_weights, axis=1).sum(
        axis=1, min_count=len(COMPONENTS)
    )
    output["entropy_six"] = output[COMPONENTS].mul(entropy_weights, axis=1).sum(
        axis=1, min_count=len(COMPONENTS)
    )
    output["equal_six_highcoverage"] = output["equal_six_strict"].where(
        complete
        & pd.to_numeric(output["nutrition_coverage"], errors="coerce").ge(0.5)
        & pd.to_numeric(output["carbon_coverage"], errors="coerce").ge(0.5)
    )
    output.loc[~complete, ["pca_six", "entropy_six", "geometric_six"]] = np.nan
    output.loc[~complete_no_practice, "equal_five_no_practice"] = np.nan
    return output


VARIANTS = {
    "equal_six_strict": "Equal six (strict)",
    "equal_six_available4": "Available components (≥4/6)",
    "equal_five_no_practice": "Equal five (no practice)",
    "geometric_six": "Geometric six",
    "pca_six": "Absolute-PC1 weights",
    "entropy_six": "Entropy weights",
    "equal_six_highcoverage": "Equal six (coverage ≥50%)",
}


def summarize_variants(frames: dict[tuple[int, str], gpd.GeoDataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    agreement_rows = []
    for (year, scale), frame in frames.items():
        population = pd.to_numeric(frame.get("t_pop"), errors="coerce")
        reference = frame["equal_six_strict"]
        for column, label in VARIANTS.items():
            values = frame[column]
            valid = values.notna()
            summary_rows.append(
                {
                    "year": year,
                    "scale": scale.upper(),
                    "variant": label,
                    "unit_n": int(valid.sum()),
                    "population_n": float(population[valid].sum()),
                    "population_weighted_mean": weighted_mean(values, population),
                    "median": float(values.median()) if valid.any() else np.nan,
                    "iqr": float(values.quantile(0.75) - values.quantile(0.25)) if valid.any() else np.nan,
                }
            )
            agreement_rows.append(
                {
                    "year": year,
                    "scale": scale.upper(),
                    "variant": label,
                    "spearman_vs_equal": safe_corr(reference, values, "spearman"),
                    "pearson_vs_equal": safe_corr(reference, values, "pearson"),
                    "low_quintile_jaccard": tail_jaccard(reference, values, "low"),
                    "high_quintile_jaccard": tail_jaccard(reference, values, "high"),
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(agreement_rows)


def maup_agreement(frames: dict[tuple[int, str], gpd.GeoDataFrame]) -> pd.DataFrame:
    rows = []
    compare_variants = [
        "equal_six_strict",
        "equal_five_no_practice",
        "geometric_six",
        "pca_six",
        "entropy_six",
    ]
    for year in YEARS:
        lsbg = frames[(year, "lsbg")].copy()
        dcca = frames[(year, "dcca")].copy()
        if "dcca" in lsbg.columns:
            lsbg = lsbg.drop(columns="dcca")
        origins = lsbg.copy()
        origins.geometry = origins.geometry.representative_point()
        crosswalk = gpd.sjoin(
            origins,
            dcca[["dcca", "geometry"]],
            how="left",
            predicate="within",
        )
        for variant in compare_variants:
            def aggregate(group: pd.DataFrame) -> float:
                return weighted_mean(group[variant], group["t_pop"])

            aggregated = (
                crosswalk.dropna(subset=["dcca"])
                .groupby("dcca", sort=False)
                .apply(aggregate, include_groups=False)
                .rename("lsbg_aggregated")
                .reset_index()
            )
            direct = dcca[["dcca", variant, "t_pop"]].rename(columns={variant: "dcca_direct"})
            pair = direct.merge(aggregated, on="dcca", how="inner").dropna(
                subset=["dcca_direct", "lsbg_aggregated"]
            )
            rows.append(
                {
                    "year": year,
                    "variant": VARIANTS[variant],
                    "dcca_n": int(len(pair)),
                    "pearson": safe_corr(pair["dcca_direct"], pair["lsbg_aggregated"], "pearson"),
                    "spearman": safe_corr(pair["dcca_direct"], pair["lsbg_aggregated"], "spearman"),
                    "mae": float((pair["dcca_direct"] - pair["lsbg_aggregated"]).abs().mean()),
                    "direct_population_weighted_mean": weighted_mean(pair["dcca_direct"], pair["t_pop"]),
                    "aggregated_population_weighted_mean": weighted_mean(pair["lsbg_aggregated"], pair["t_pop"]),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_frames: dict[tuple[int, str], gpd.GeoDataFrame] = {}
    for year in YEARS:
        for scale in SCALES:
            path = INPUT_DIR / f"{year}_{scale}_components.gpkg"
            raw_frames[(year, scale)] = gpd.read_file(path)

    pooled_lsbg = pd.concat(
        [raw_frames[(year, "lsbg")][COMPONENTS] for year in YEARS],
        ignore_index=True,
    )
    weights, weight_metadata = fit_weights(pooled_lsbg)
    frames = {
        key: add_variants(frame, weights) for key, frame in raw_frames.items()
    }
    summary, agreement = summarize_variants(frames)
    maup = maup_agreement(frames)

    weights.assign(component_label=weights["component"].map(COMPONENT_LABELS)).to_csv(
        OUTPUT_DIR / "weight_schemes.csv", index=False
    )
    summary.to_csv(OUTPUT_DIR / "variant_population_summary.csv", index=False)
    agreement.to_csv(OUTPUT_DIR / "variant_agreement.csv", index=False)
    maup.to_csv(OUTPUT_DIR / "maup_agreement.csv", index=False)
    for (year, scale), frame in frames.items():
        frame.to_file(
            OUTPUT_DIR / f"{year}_{scale}_sensitivity.gpkg",
            driver="GPKG",
        )

    manifest = {
        "status": "candidate sensitivity results; not yet authorized for manuscript",
        "years": YEARS,
        "scales": [scale.upper() for scale in SCALES],
        "components": COMPONENTS,
        "reference": "pooled four-year LSBG complete observations",
        **weight_metadata,
        "missing_rule": "missing values are never coded as zero",
        "high_coverage_rule": "nutrition_coverage >= 0.5 and carbon_coverage >= 0.5",
        "maup_crosswalk": "LSBG representative point within DCCA; population-weighted aggregation",
    }
    (OUTPUT_DIR / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print("\nPopulation-weighted LSBG means:")
    print(
        summary.loc[summary["scale"].eq("LSBG")]
        .pivot(index="variant", columns="year", values="population_weighted_mean")
        .round(4)
        .to_string()
    )
    print("\nMAUP Spearman agreement:")
    print(
        maup.pivot(index="variant", columns="year", values="spearman")
        .round(3)
        .to_string()
    )


if __name__ == "__main__":
    main()
