from __future__ import annotations

import json
import re
from pathlib import Path
import os

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.stats import t as student_t


YEARS = [2011, 2016, 2021, 2024]
ROOT = Path(__file__).resolve().parents[1]
RESTAURANTS = Path(os.environ.get("CITIES_RESTAURANT_MASTER", "authorized_inputs/restaurant_master.csv"))
SENSITIVITY_DIR = ROOT / "outputs" / "restricted" / "v4_sdi_sensitivity"
OUTPUT_DIR = ROOT / "outputs" / "restricted" / "v4_price_market_decomposition"


def parse_price_ceiling(value: object) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).replace(",", "")
    numbers = [int(number) for number in re.findall(r"\d+", text)]
    if not numbers:
        return np.nan
    if len(numbers) >= 2:
        return float(max(numbers))
    number = numbers[0]
    if number >= 801:
        return np.inf
    return float(number)


def load_restaurants() -> pd.DataFrame:
    columns = [
        "restaurant_id",
        "Latitude_new",
        "Longitude_new",
        "restaurant_cost_range",
        "operation_early_year",
        "operation_latest_year",
    ]
    data = pd.read_csv(RESTAURANTS, usecols=columns, low_memory=False)
    data["price_ceiling_hkd"] = data["restaurant_cost_range"].map(parse_price_ceiling)
    data["low_price_le100"] = data["price_ceiling_hkd"].le(100)
    return data


def price_aggregate(restaurants: pd.DataFrame, polygons: gpd.GeoDataFrame, year: int, id_field: str) -> pd.DataFrame:
    active = restaurants.loc[
        pd.to_numeric(restaurants["operation_early_year"], errors="coerce").le(year)
        & pd.to_numeric(restaurants["operation_latest_year"], errors="coerce").ge(year)
        & pd.to_numeric(restaurants["Latitude_new"], errors="coerce").between(22.10, 22.60)
        & pd.to_numeric(restaurants["Longitude_new"], errors="coerce").between(113.80, 114.50)
    ].copy()
    points = gpd.GeoDataFrame(
        active,
        geometry=gpd.points_from_xy(active["Longitude_new"], active["Latitude_new"]),
        crs="EPSG:4326",
    ).to_crs(polygons.crs)
    joined = gpd.sjoin(
        points,
        polygons[[id_field, "geometry"]],
        how="left",
        predicate="within",
    )
    missing = joined[id_field].isna()
    if missing.any():
        nearest = gpd.sjoin_nearest(
            joined.loc[missing, points.columns],
            polygons[[id_field, "geometry"]],
            how="left",
            max_distance=100.0,
        )
        joined.loc[missing, id_field] = nearest[id_field].to_numpy()

    joined["price_valid"] = joined["price_ceiling_hkd"].notna()
    aggregate = (
        joined.dropna(subset=[id_field])
        .groupby(id_field, sort=False)
        .agg(
            active_restaurant_n=("restaurant_id", "size"),
            valid_price_n=("price_valid", "sum"),
            low_price_n=("low_price_le100", "sum"),
            median_price_ceiling_hkd=("price_ceiling_hkd", "median"),
        )
        .reset_index()
    )
    return aggregate


def weighted_standardize(frame: pd.DataFrame, columns: list[str], weight: pd.Series) -> pd.DataFrame:
    output = pd.DataFrame(index=frame.index)
    valid_weight = pd.to_numeric(weight, errors="coerce").fillna(0.0).clip(lower=0.0)
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        valid = values.notna() & valid_weight.gt(0)
        if not valid.any():
            output[column] = np.nan
            continue
        mean = np.average(values[valid], weights=valid_weight[valid])
        variance = np.average((values[valid] - mean) ** 2, weights=valid_weight[valid])
        output[column] = (values - mean) / np.sqrt(variance) if variance > 0 else np.nan
    return output


def fit_model(frame: pd.DataFrame, year: int, model: str, predictors: list[str]) -> list[dict[str, object]]:
    columns = ["equal_five_no_practice", "t_pop", "dcca", *predictors]
    data = frame[columns].replace([np.inf, -np.inf], np.nan).dropna().copy()
    data = data.loc[pd.to_numeric(data["t_pop"], errors="coerce").gt(0)]
    if len(data) < 40:
        return []
    standardized = weighted_standardize(
        data,
        ["equal_five_no_practice", *predictors],
        data["t_pop"],
    ).dropna()
    data = data.loc[standardized.index]
    y = standardized["equal_five_no_practice"].to_numpy(dtype=float)
    x_names = ["constant", *predictors]
    x = np.column_stack(
        [np.ones(len(data), dtype=float), standardized[predictors].to_numpy(dtype=float)]
    )
    weights = pd.to_numeric(data["t_pop"], errors="coerce").to_numpy(dtype=float)
    sqrt_weights = np.sqrt(weights)
    xw = x * sqrt_weights[:, None]
    yw = y * sqrt_weights
    bread = np.linalg.pinv(xw.T @ xw)
    beta = bread @ (xw.T @ yw)
    residual = y - x @ beta
    transformed_residual = residual * sqrt_weights

    group_codes, unique_groups = pd.factorize(data["dcca"], sort=False)
    meat = np.zeros((x.shape[1], x.shape[1]), dtype=float)
    for group_index in range(len(unique_groups)):
        mask = group_codes == group_index
        score = xw[mask].T @ transformed_residual[mask]
        meat += np.outer(score, score)
    n = len(data)
    k = x.shape[1]
    cluster_n = len(unique_groups)
    correction = (cluster_n / (cluster_n - 1)) * ((n - 1) / (n - k))
    covariance = correction * bread @ meat @ bread
    standard_error = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    degrees_freedom = max(cluster_n - 1, 1)
    critical = float(student_t.ppf(0.975, df=degrees_freedom))
    t_statistic = np.divide(
        beta,
        standard_error,
        out=np.full_like(beta, np.nan),
        where=standard_error > 0,
    )
    p_values = 2.0 * student_t.sf(np.abs(t_statistic), df=degrees_freedom)
    weighted_mean_y = np.average(y, weights=weights)
    weighted_sse = float(np.sum(weights * residual**2))
    weighted_sst = float(np.sum(weights * (y - weighted_mean_y) ** 2))
    weighted_r2 = 1.0 - weighted_sse / weighted_sst if weighted_sst > 0 else np.nan
    parameter_index = {name: index for index, name in enumerate(x_names)}
    rows = []
    for predictor in predictors:
        index = parameter_index[predictor]
        rows.append(
            {
                "year": year,
                "model": model,
                "predictor": predictor,
                "standardized_beta": float(beta[index]),
                "se_cluster_dcca": float(standard_error[index]),
                "ci_low": float(beta[index] - critical * standard_error[index]),
                "ci_high": float(beta[index] + critical * standard_error[index]),
                "p_value": float(p_values[index]),
                "lsbg_n": int(n),
                "dcca_cluster_n": int(cluster_n),
                "weighted_r2": float(weighted_r2),
            }
        )
    return rows


def weighted_summary(frame: pd.DataFrame, year: int) -> dict[str, object]:
    population = pd.to_numeric(frame["t_pop"], errors="coerce")

    def wmean(column: str) -> float:
        values = pd.to_numeric(frame[column], errors="coerce")
        valid = values.notna() & population.notna() & population.gt(0)
        return float(np.average(values[valid], weights=population[valid])) if valid.any() else np.nan

    return {
        "year": year,
        "population_reference_year": 2021 if year == 2024 else year,
        "active_restaurant_n": int(frame["active_restaurant_n"].sum()),
        "valid_price_n": int(frame["valid_price_n"].sum()),
        "valid_price_share": float(frame["valid_price_n"].sum() / frame["active_restaurant_n"].sum()),
        "low_price_le100_share_of_valid": float(frame["low_price_n"].sum() / frame["valid_price_n"].sum()),
        "population_weighted_low_price_share": wmean("low_price_share"),
        "population_weighted_low_price_per1000": wmean("low_price_per1000"),
        "population_weighted_total_supply_per1000": wmean("total_supply_per1000"),
        "population_weighted_quality_no_practice": wmean("equal_five_no_practice"),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    restaurants = load_restaurants()
    frames: dict[int, gpd.GeoDataFrame] = {}
    regression_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for year in YEARS:
        frame = gpd.read_file(SENSITIVITY_DIR / f"{year}_lsbg_sensitivity.gpkg")
        if "dcca" not in frame.columns or frame["dcca"].isna().all():
            dcca_geometry = gpd.read_file(
                SENSITIVITY_DIR / f"{year}_dcca_sensitivity.gpkg"
            )[["dcca", "geometry"]]
            origins = frame[["lsbg", "geometry"]].copy()
            origins.geometry = origins.geometry.representative_point()
            crosswalk = gpd.sjoin(
                origins,
                dcca_geometry,
                how="left",
                predicate="within",
            )[["lsbg", "dcca"]]
            frame = frame.drop(columns="dcca", errors="ignore").merge(
                crosswalk,
                on="lsbg",
                how="left",
            )
        price = price_aggregate(restaurants, frame, year, "lsbg")
        frame = frame.merge(price, on="lsbg", how="left")
        count_columns = ["active_restaurant_n", "valid_price_n", "low_price_n"]
        frame[count_columns] = frame[count_columns].fillna(0).astype(int)
        frame["low_price_share"] = frame["low_price_n"].div(frame["valid_price_n"].replace(0, np.nan))
        frame["low_price_per1000"] = frame["low_price_n"].div(
            pd.to_numeric(frame["t_pop"], errors="coerce").replace(0, np.nan)
        ) * 1000.0
        frame["total_supply_per1000"] = frame["active_restaurant_n"].div(
            pd.to_numeric(frame["t_pop"], errors="coerce").replace(0, np.nan)
        ) * 1000.0
        frame["log_income"] = np.log1p(pd.to_numeric(frame["ma_hh"], errors="coerce"))
        frame["log_supply"] = np.log1p(frame["total_supply_per1000"])
        frame["ageing_share"] = pd.to_numeric(frame["age_65o_p"], errors="coerce")

        regression_rows.extend(fit_model(frame, year, "Income only", ["log_income"]))
        regression_rows.extend(
            fit_model(
                frame,
                year,
                "Income + market composition",
                ["log_income", "log_supply", "low_price_share", "ageing_share"],
            )
        )
        summary_rows.append(weighted_summary(frame, year))
        frames[year] = frame
        frame.to_file(OUTPUT_DIR / f"{year}_lsbg_price_market.gpkg", driver="GPKG")

    regression = pd.DataFrame(regression_rows)
    summary = pd.DataFrame(summary_rows)
    regression.to_csv(OUTPUT_DIR / "quality_market_decomposition.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "price_market_summary.csv", index=False)
    manifest = {
        "status": "temporal-mismatch market decomposition sensitivity; not household affordability",
        "years": YEARS,
        "outcome": "equal-five no-practice area quality score",
        "price_measure": "undated current OpenRice price tier reused within each active cohort",
        "low_price_definition": "platform price ceiling <= HKD 100",
        "model": "population-weighted least squares with DCCA-clustered standard errors",
        "full_predictors": ["log income", "log supply per 1000", "low-price share", "ageing share"],
        "interpretation_boundary": "ecological market-composition sensitivity; not causal, household affordability, need, realised use, or historical price change",
        "population_reference": "native 2011/2016/2021 census; 2024 supply uses fixed 2021 population geography",
        "formal_manuscript_result_authorized": False,
    }
    (OUTPUT_DIR / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print("\nSummary")
    print(summary.round(4).to_string(index=False))
    print("\nRegression")
    print(regression.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
