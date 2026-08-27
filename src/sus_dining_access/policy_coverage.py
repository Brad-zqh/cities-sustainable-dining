"""Transparent maximum-coverage helpers for real-site planning scenarios."""

from __future__ import annotations

from itertools import combinations
from typing import Sequence

import numpy as np


def bottom_population_membership(
    values: Sequence[float], weights: Sequence[float], share: float
) -> np.ndarray:
    """Fractional membership in the lowest outcome-ranked population share.

    Geographic units tied at the cutoff receive the same fractional membership,
    so the definition is invariant to row or identifier ordering.
    """
    x = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    if x.ndim != 1 or w.shape != x.shape or len(x) == 0:
        raise ValueError("values and weights must be aligned non-empty vectors")
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(w)) or np.any(w <= 0):
        raise ValueError("values must be finite and weights finite and positive")
    if not 0 < share <= 1:
        raise ValueError("share must be in (0, 1]")
    target = float(w.sum() * share)
    membership = np.zeros(len(x), dtype=float)
    assigned = 0.0
    for value in np.unique(np.sort(x)):
        tied = x == value
        tied_weight = float(w[tied].sum())
        remaining = target - assigned
        if remaining <= 1e-10:
            break
        fraction = min(1.0, remaining / tied_weight)
        membership[tied] = fraction
        assigned += tied_weight * fraction
    if not np.isclose(float(np.sum(w * membership)), target, rtol=0, atol=1e-6):
        raise ValueError("bottom-population allocation failed to reach target")
    return membership


def exact_maximum_coverage(
    coverage: np.ndarray,
    target_weights: Sequence[float],
    total_weights: Sequence[float],
    candidate_ids: Sequence[str],
    budget_k: int,
) -> dict[str, object]:
    """Enumerate a small real candidate set with deterministic tie breaking."""
    matrix = np.asarray(coverage, dtype=bool)
    target = np.asarray(target_weights, dtype=float)
    total = np.asarray(total_weights, dtype=float)
    ids = tuple(map(str, candidate_ids))
    if matrix.ndim != 2 or matrix.shape != (len(target), len(ids)):
        raise ValueError("coverage must be demand-by-candidate")
    if total.shape != target.shape or np.any(target < 0) or np.any(total <= 0):
        raise ValueError("invalid weights")
    if np.any(target > total + 1e-12):
        raise ValueError("target weights cannot exceed total weights")
    if not 1 <= budget_k <= len(ids):
        raise ValueError("budget must be between one and candidate count")
    best = None
    for selected in combinations(range(len(ids)), budget_k):
        covered = matrix[:, selected].any(axis=1)
        target_covered = float(target[covered].sum())
        total_covered = float(total[covered].sum())
        selected_ids = tuple(sorted(ids[index] for index in selected))
        key = (target_covered, total_covered, tuple(reversed(selected_ids)))
        if best is None or key[:2] > best[0][:2] or (
            key[:2] == best[0][:2] and selected_ids < best[1]
        ):
            best = (key, selected_ids, covered)
    assert best is not None
    return {
        "selected_site_ids": best[1],
        "target_population_reached": float(target[best[2]].sum()),
        "total_population_reached": float(total[best[2]].sum()),
        "covered_demand_mask": best[2],
    }


__all__ = ["bottom_population_membership", "exact_maximum_coverage"]
