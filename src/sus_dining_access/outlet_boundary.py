"""Fail-closed outlet-universe boundary rules for walking access."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class OutletBoundaryDecision:
    status: str
    included_primary_narrow: bool
    included_broad_sensitivity: bool
    matched_definite_exclusions: tuple[str, ...]
    matched_ambiguous_types: tuple[str, ...]


def classify_outlet_tokens(
    tokens: Iterable[str],
    *,
    parse_status: str,
    definite_exclusions: Iterable[str],
    ambiguous_types: Iterable[str],
) -> OutletBoundaryDecision:
    """Classify one venue under narrow and broad prespecified universes."""

    token_set = {str(token) for token in tokens}
    definite = tuple(sorted(token_set.intersection(definite_exclusions)))
    ambiguous = tuple(sorted(token_set.intersection(ambiguous_types)))
    if parse_status != "parsed_unique":
        return OutletBoundaryDecision(
            status="unresolved_missing_or_unparsed_taxonomy",
            included_primary_narrow=False,
            included_broad_sensitivity=False,
            matched_definite_exclusions=definite,
            matched_ambiguous_types=ambiguous,
        )
    if not token_set:
        return OutletBoundaryDecision(
            status="included_parsed_without_venue_type_label",
            included_primary_narrow=True,
            included_broad_sensitivity=True,
            matched_definite_exclusions=definite,
            matched_ambiguous_types=ambiguous,
        )
    if definite:
        return OutletBoundaryDecision(
            status="excluded_definite_non_walkin_or_restricted_type",
            included_primary_narrow=False,
            included_broad_sensitivity=False,
            matched_definite_exclusions=definite,
            matched_ambiguous_types=ambiguous,
        )
    if ambiguous:
        return OutletBoundaryDecision(
            status="ambiguous_in_broad_only",
            included_primary_narrow=False,
            included_broad_sensitivity=True,
            matched_definite_exclusions=definite,
            matched_ambiguous_types=ambiguous,
        )
    return OutletBoundaryDecision(
        status="included_food_service_candidate",
        included_primary_narrow=True,
        included_broad_sensitivity=True,
        matched_definite_exclusions=definite,
        matched_ambiguous_types=ambiguous,
    )


def valid_hong_kong_coordinate(
    latitude: float,
    longitude: float,
    *,
    bounds: dict[str, float],
) -> bool:
    """Return whether a finite coordinate lies in the frozen broad HK envelope."""

    try:
        lat = float(latitude)
        lon = float(longitude)
    except (TypeError, ValueError):
        return False
    return (
        bounds["latitude_min"] <= lat <= bounds["latitude_max"]
        and bounds["longitude_min"] <= lon <= bounds["longitude_max"]
    )
