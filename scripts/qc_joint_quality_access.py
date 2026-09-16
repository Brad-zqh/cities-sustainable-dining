"""Integrity checks for the joint quality-context/price/network analysis."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "source_data" / "fig13_joint_quality_affordable_access_v1"
FIG = Path(os.environ.get("CITIES_FIGURE_DIR", str(ROOT / "figures/current"))) / "Fig16_Joint_Quality_Walking_Price_Opportunity"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    points = pd.read_csv(DATA / "joint_access_point_estimates.csv")
    intervals = pd.read_csv(DATA / "joint_access_dcca_block_intervals.csv")
    coverage = pd.read_csv(DATA / "joint_access_data_coverage_audit.csv")
    lsbg = pd.read_csv(DATA / "lsbg_joint_quality_affordable_access.csv")
    contract = json.loads((DATA / "analysis_contract.json").read_text(encoding="utf-8"))

    require(set(points["year"].unique()) == {2016, 2021, 2024}, "unexpected analysis year")
    require(2011 not in points["year"].unique(), "2011 must not enter the network analysis")
    require("restaurant_id" not in lsbg.columns, "raw restaurant identifiers leaked")
    require((lsbg["joint_access"] <= lsbg["low_price_access"]).all(), "joint count exceeds low-price count")
    require(coverage["eligible_with_price_share"].min() > 0.99, "price linkage below 99%")
    require(
        coverage["eligible_with_destination_area_sdi_share"].min() > 0.96,
        "destination-area SDI linkage below 96%",
    )
    require(intervals["valid_replicates"].min() == 999, "bootstrap count is not 999")
    require(
        "not for an individual restaurant" in contract["measurement_boundary"],
        "measurement-level warning absent",
    )

    primary = points.loc[points["outcome"].eq("low_price_access")].set_index("year")
    reference = pd.read_csv(
        ROOT / "source_data/fig04_network_price_v4/network_price_inequality_point_estimates.csv"
    )
    reference = reference.loc[
        reference["threshold_min"].eq(15.0)
        & reference["outcome"].eq("opportunity_n_le_100")
    ].set_index("year")
    for metric in (
        "population_weighted_mean",
        "zero_access_population_share",
        "weighted_gini",
        "concentration_index",
    ):
        delta = (primary[metric] - reference[metric]).abs().max()
        require(delta < 1e-10, f"low-price baseline mismatch for {metric}: {delta}")

    sensitivity = points.loc[points["outcome"].eq("joint_access_sensitivity")].copy()
    for _, frame in sensitivity.groupby(["year", "threshold_min", "price_ceiling_hkd"]):
        frame = frame.sort_values("destination_area_sdi_threshold")
        require(frame["population_weighted_mean"].diff().dropna().le(1e-12).all(), "mean not monotone in SDI")
        require(frame["zero_access_population_share"].diff().dropna().ge(-1e-12).all(), "zero share not monotone in SDI")
    for _, frame in sensitivity.groupby(["year", "threshold_min", "destination_area_sdi_threshold"]):
        frame = frame.sort_values("price_ceiling_hkd")
        require(frame["population_weighted_mean"].diff().dropna().ge(-1e-12).all(), "mean not monotone in price ceiling")
        require(frame["zero_access_population_share"].diff().dropna().le(1e-12).all(), "zero share not monotone in price ceiling")
    for _, frame in sensitivity.groupby(["year", "price_ceiling_hkd", "destination_area_sdi_threshold"]):
        frame = frame.sort_values("threshold_min")
        require(frame["population_weighted_mean"].diff().dropna().ge(-1e-12).all(), "mean not monotone in walk time")
        require(frame["zero_access_population_share"].diff().dropna().le(1e-12).all(), "zero share not monotone in walk time")

    manifest = pd.read_csv(DATA / "sha256_manifest.csv")
    for row in manifest.loc[manifest["role"].eq("output")].itertuples(index=False):
        path = DATA / row.path
        require(path.exists(), f"manifest output missing: {row.path}")
        require(digest(path) == row.sha256, f"hash mismatch: {row.path}")

    for suffix in (".png", ".pdf", ".svg"):
        path = FIG.with_suffix(suffix)
        require(path.exists() and path.stat().st_size > 20_000, f"figure output invalid: {suffix}")
    with Image.open(FIG.with_suffix(".png")) as image:
        require(image.width >= 4000 and image.height >= 2500, "PNG below publication pixel dimensions")
        dpi = image.info.get("dpi", (0, 0))
        require(min(dpi) >= 590, f"PNG DPI below 600 target: {dpi}")

    print(
        "JOINT_ACCESS_QC=PASS "
        f"years={sorted(points.year.unique())} "
        f"bootstrap_min={int(intervals.valid_replicates.min())} "
        f"price_link_min={coverage.eligible_with_price_share.min():.3f} "
        f"sdi_link_min={coverage.eligible_with_destination_area_sdi_share.min():.3f}"
    )


if __name__ == "__main__":
    main()
