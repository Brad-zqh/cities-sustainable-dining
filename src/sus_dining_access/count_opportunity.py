"""Build auditable count-only restaurant opportunity candidates.

The builder separates candidate construction from Gate A authorization. It
never labels an output READY and emits no restaurant names or platform IDs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from .outlet_boundary import classify_outlet_tokens, valid_hong_kong_coordinate


class CountOpportunityError(ValueError):
    """Raised when a count-candidate input violates the frozen contract."""


def opaque_opportunity_id(value: object, namespace: str) -> str:
    if pd.isna(value) or not str(value).strip():
        raise CountOpportunityError("restaurant ID must be nonmissing")
    payload = f"{namespace}:{str(value).strip()}".encode("utf-8")
    return "opp_" + hashlib.sha256(payload).hexdigest()[:24]


def _tokens(value: object) -> list[str]:
    if not isinstance(value, str):
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    return [str(item) for item in parsed] if isinstance(parsed, list) else []


def _validate_unique(frame: pd.DataFrame, column: str, label: str) -> None:
    if column not in frame:
        raise CountOpportunityError(f"{label} missing required column: {column}")
    if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
        raise CountOpportunityError(f"{label} {column} must be complete")
    if frame[column].duplicated().any():
        raise CountOpportunityError(f"{label} {column} must be unique")


def build_count_candidates(
    master: pd.DataFrame,
    taxonomy: pd.DataFrame,
    lsbg: gpd.GeoDataFrame,
    land_mask: gpd.GeoDataFrame,
    config: dict[str, Any],
    *,
    boundary_scope: str,
    source_snapshot_id: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return a restricted candidate table and aggregate construction QA."""
    if boundary_scope not in config["boundary_scopes"]:
        raise CountOpportunityError(f"unsupported boundary_scope: {boundary_scope}")
    fields = config["fields"]
    master_required = [
        fields["id"],
        fields["early_year"],
        fields["latest_year"],
        fields["latitude"],
        fields["longitude"],
    ]
    missing = sorted(set(master_required).difference(master.columns))
    if missing:
        raise CountOpportunityError(f"master missing columns: {missing}")
    taxonomy_required = {"restaurant_id", "taxonomy_parse_status", "restaurant_tokens_json"}
    if missing_taxonomy := sorted(taxonomy_required.difference(taxonomy.columns)):
        raise CountOpportunityError(f"taxonomy missing columns: {missing_taxonomy}")
    _validate_unique(master, fields["id"], "master")
    _validate_unique(taxonomy, "restaurant_id", "taxonomy")

    frame = master[master_required].merge(
        taxonomy[list(taxonomy_required)],
        left_on=fields["id"],
        right_on="restaurant_id",
        how="left",
        validate="one_to_one",
    )
    year = int(config["analysis_year"])
    early = pd.to_numeric(frame[fields["early_year"]], errors="coerce")
    latest = pd.to_numeric(frame[fields["latest_year"]], errors="coerce")
    operation = early.le(year) & latest.ge(year)

    decisions = [
        classify_outlet_tokens(
            _tokens(row.restaurant_tokens_json),
            parse_status=str(row.taxonomy_parse_status),
            definite_exclusions=config["definite_exclusions"],
            ambiguous_types=config["ambiguous_types"],
        )
        for row in frame[["taxonomy_parse_status", "restaurant_tokens_json"]].itertuples(index=False)
    ]
    if boundary_scope == "primary_narrow":
        boundary = pd.Series(
            [item.included_primary_narrow for item in decisions], index=frame.index
        )
    elif boundary_scope == "broad_sensitivity":
        boundary = pd.Series(
            [item.included_broad_sensitivity for item in decisions], index=frame.index
        )
    else:
        boundary = pd.Series(True, index=frame.index)

    coordinate_valid = pd.Series(
        [
            valid_hong_kong_coordinate(lat, lon, bounds=config["coordinate_bounds"])
            for lat, lon in zip(frame[fields["latitude"]], frame[fields["longitude"]])
        ],
        index=frame.index,
    )
    base_selected = operation & boundary & coordinate_valid
    selected = frame.loc[base_selected].copy()
    selected[fields["latitude"]] = pd.to_numeric(
        selected[fields["latitude"]], errors="coerce"
    )
    selected[fields["longitude"]] = pd.to_numeric(
        selected[fields["longitude"]], errors="coerce"
    )
    points = gpd.GeoDataFrame(
        selected,
        geometry=gpd.points_from_xy(
            selected[fields["longitude"]], selected[fields["latitude"]]
        ),
        crs="EPSG:4326",
    )

    if land_mask.empty or land_mask.crs is None:
        raise CountOpportunityError("land_mask must be nonempty with a CRS")
    land = land_mask.to_crs("EPSG:4326").copy()
    land_geometry = shapely.union_all(land.geometry.make_valid().to_numpy())
    inside_land = points.geometry.apply(land_geometry.covers)
    points = points.loc[inside_land].copy()

    lsbg_id = fields["lsbg_id"]
    if lsbg.empty or lsbg.crs is None or lsbg_id not in lsbg:
        raise CountOpportunityError("LSBG geometry must be nonempty, projected and keyed")
    polygons = lsbg[[lsbg_id, "geometry"]].copy()
    _validate_unique(polygons, lsbg_id, "LSBG")
    polygons.geometry = polygons.geometry.make_valid()
    projected_points = points.to_crs(polygons.crs)
    joined = gpd.sjoin(projected_points, polygons, how="left", predicate="within")
    unmatched = joined[lsbg_id].isna()
    if unmatched.any():
        boundary_join = gpd.sjoin(
            projected_points.loc[unmatched, projected_points.columns],
            polygons,
            how="left",
            predicate="intersects",
        )
        if boundary_join.index.duplicated().any():
            duplicated = boundary_join.index[boundary_join.index.duplicated()].unique()
            boundary_join.loc[duplicated, lsbg_id] = np.nan
            boundary_join = boundary_join.loc[~boundary_join.index.duplicated(keep="first")]
        joined.loc[boundary_join.index, lsbg_id] = boundary_join[lsbg_id]
    matched = joined[lsbg_id].notna()
    eligible = joined.loc[matched].copy()

    output = pd.DataFrame(
        {
            "opportunity_id": [
                opaque_opportunity_id(value, config["opportunity_id_namespace"])
                for value in eligible[fields["id"]]
            ],
            "analysis_mode": config["analysis_mode"],
            "analysis_year": year,
            "reference_date": config["reference_date"],
            "comparison_role": config["comparison_role"],
            "operational_window_year": year,
            "temporal_alignment_status": config["temporal_alignment_status"],
            "restaurant_source": config["restaurant_source"],
            "source_snapshot_id": source_snapshot_id,
            "longitude_wgs84": eligible[fields["longitude"]].to_numpy(float),
            "latitude_wgs84": eligible[fields["latitude"]].to_numpy(float),
            "crs": "EPSG:4326",
            "location_status": "validated_inside_D016_land_mask",
            config.get("output_lsbg_id_field", f"lsbg_id_{year}"): eligible[lsbg_id]
            .astype(str)
            .to_numpy(),
            "lsbg_geography_year": year,
            "land_mask_id": config["land_mask_id"],
            "eligible_primary": True,
            "eligibility_status": "eligible_primary",
            "exclusion_reason": None,
            "model_validation_status": "not_model_derived",
            "gate_a_evidence_id": config["gate_a_evidence_id"],
            "analysis_version": config["analysis_version"],
            "release_class": config["release_class"],
            "opportunity_count_value": 1.0,
            "opportunity_count_status": "observed",
            "opportunity_count_source_year": year,
            "opportunity_count_match_type": "exact_year",
            "opportunity_count_unit": "restaurant",
        }
    ).sort_values("opportunity_id", ignore_index=True)
    _validate_unique(output, "opportunity_id", "candidate")

    qa = {
        "status": "candidate_built_not_gate_a_authorized",
        "analysis_year": year,
        "analysis_mode": "count_only",
        "boundary_scope": boundary_scope,
        "master_row_n": int(len(frame)),
        "operation_window_n": int(operation.sum()),
        "boundary_included_n": int((operation & boundary).sum()),
        "valid_coordinate_n": int(base_selected.sum()),
        "inside_land_n": int(inside_land.sum()),
        "lsbg_matched_n": int(matched.sum()),
        "lsbg_unmatched_n": int((~matched).sum()),
        "candidate_row_n": int(len(output)),
        "candidate_unique_lsbg_n": int(
            output[config.get("output_lsbg_id_field", f"lsbg_id_{year}")].nunique()
        ),
        "formal_run_authorized": False,
        "representation_boundary": "OpenRice-derived candidate; FEHD coverage calibration remains a separate Gate A requirement.",
    }
    return output, qa
