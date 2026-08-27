"""Draw a privacy-preserving, evidence-locked Hong Kong study-area atlas.

The manuscript map describes real restaurant locations, aggregated into fixed
200-m cells before public release.  Every displayed count is derived from the
restricted platform master; no restaurant identifiers, addresses or individual
coordinates are exported.  A sparse 2025-01-01 major-road layer is retained
only as a vector reference beneath the restaurant cells; no raster/tile
basemap is drawn and the road layer is never represented as a historic network.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
from matplotlib.gridspec import GridSpec

from v5_cities_visual_system import configure, north_arrow, outer_boundary, segmented_scale_bar


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source_data" / "fig03_study_area_annual_distribution_v1"
GEOMETRY = ROOT / "source_data" / "fig_v4_four_year" / "2024_lsbg_components.gpkg"
OUT = Path(os.environ.get("CITIES_FIGURE_DIR", str(ROOT / "figures")))
YEARS = (2011, 2016, 2021, 2024)
COLORS = {2011: "#B62B45", 2016: "#24659A", 2021: "#33854F", 2024: "#CE752D"}
CELL_METRES = 200
COUNT_BREAKS = np.array([1, 2, 5, 10, 20, 40, 80, 160, 320], dtype=float)
PROJECTED_CRS = "EPSG:2326"


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-source", action="store_true")
    parser.add_argument("--restricted-master", type=Path)
    parser.add_argument("--osm-context", type=Path)
    parser.add_argument("--coverage-audit", type=Path)
    return parser.parse_args()


def build_public_source(args: argparse.Namespace) -> None:
    required = {
        "restricted restaurant master": args.restricted_master,
        "dated OSM context": args.osm_context,
        "fixed-geography coverage audit": args.coverage_audit,
    }
    for label, path in required.items():
        if path is None or not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")
    if not GEOMETRY.is_file():
        raise FileNotFoundError(GEOMETRY)

    columns = [
        "restaurant_id",
        "Longitude_new",
        "Latitude_new",
        "operation_early_year",
        "operation_latest_year",
    ]
    frame = pd.read_csv(args.restricted_master, usecols=columns, low_memory=False)
    if frame["restaurant_id"].duplicated().any():
        raise ValueError("Restricted restaurant identifiers are not unique")
    for column in columns[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    coordinates_valid = frame["Longitude_new"].between(113.80, 114.50) & frame[
        "Latitude_new"
    ].between(22.10, 22.60)
    coverage = pd.read_csv(args.coverage_audit).set_index("year")

    yearly_cells: list[pd.DataFrame] = []
    summary_rows: list[dict[str, int | float]] = []
    for year in YEARS:
        operation = frame["operation_early_year"].le(year) & frame[
            "operation_latest_year"
        ].ge(year)
        valid = frame.loc[operation & coordinates_valid, ["Longitude_new", "Latitude_new"]]
        projected = gpd.GeoDataFrame(
            valid,
            geometry=gpd.points_from_xy(valid["Longitude_new"], valid["Latitude_new"]),
            crs="EPSG:4326",
        ).to_crs(PROJECTED_CRS)
        cells = pd.DataFrame(
            {
                "grid_easting": np.floor(projected.geometry.x / CELL_METRES).astype(int),
                "grid_northing": np.floor(projected.geometry.y / CELL_METRES).astype(int),
            }
        )
        cells = (
            cells.groupby(["grid_easting", "grid_northing"], sort=True)
            .size()
            .rename("restaurant_n")
            .reset_index()
        )
        cells.insert(0, "year", year)
        cells["easting_m"] = cells["grid_easting"] * CELL_METRES + CELL_METRES / 2
        cells["northing_m"] = cells["grid_northing"] * CELL_METRES + CELL_METRES / 2
        if int(cells["restaurant_n"].sum()) != len(valid):
            raise AssertionError(f"{year}: gridded count does not match valid restaurants")
        audited_n = int(coverage.at[year, "valid_coordinate_n"])
        if len(valid) != audited_n:
            raise AssertionError(f"{year}: coordinate count {len(valid)} != audited {audited_n}")
        yearly_cells.append(cells)
        summary_rows.append(
            {
                "year": year,
                "interval_proxy_restaurant_n": int(operation.sum()),
                "valid_coordinate_restaurant_n": len(valid),
                "coordinate_coverage_pct": 100 * len(valid) / int(operation.sum()),
                "occupied_200m_cells_n": len(cells),
                "occupied_fixed_2021_lsbg_n": int(coverage.at[year, "lsbg_with_any_outlet_n"]),
                "fixed_2021_lsbg_n": int(coverage.at[year, "lsbg_n"]),
                "occupied_fixed_2021_lsbg_share_pct": 100
                * float(coverage.at[year, "lsbg_with_any_outlet_share"]),
            }
        )

    # Publish only a lightly simplified major-road contextual subset.  OSM is
    # ODbL data; its vintage and attribution remain visible on every output.
    roads = gpd.read_file(args.osm_context, columns=["highway", "geometry"])
    if roads.crs is None:
        raise ValueError("The OSM context layer has no declared CRS")
    roads = roads.loc[roads["highway"].isin(["primary", "secondary"])].copy()
    roads = roads.to_crs(PROJECTED_CRS)
    roads["geometry"] = roads.geometry.simplify(35, preserve_topology=True)
    roads = roads.loc[roads.geometry.notna() & ~roads.geometry.is_empty].reset_index(drop=True)

    SOURCE.mkdir(parents=True, exist_ok=True)
    cells_path = SOURCE / "annual_restaurant_density_200m.csv"
    summary_path = SOURCE / "annual_restaurant_coordinate_coverage.csv"
    roads_path = SOURCE / "osm_major_walkable_context_2025_01_01.gpkg"
    pd.concat(yearly_cells, ignore_index=True).to_csv(cells_path, index=False)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    roads[["highway", "geometry"]].to_file(roads_path, driver="GPKG")

    contract = {
        "analysis_years": list(YEARS),
        "cell_size_m": CELL_METRES,
        "projection": PROJECTED_CRS,
        "membership_rule": "operation_early_year <= year <= operation_latest_year",
        "coordinate_validity": "113.80 <= longitude <= 114.50; 22.10 <= latitude <= 22.60",
        "source_restaurant_master": {
            "filename": args.restricted_master.name,
            "sha256": digest(args.restricted_master),
            "redistribution": "restricted; no IDs, individual coordinates, addresses or names released",
        },
        "osm_context": {
            "snapshot_date": "2025-01-01",
            "source_sha256": digest(args.osm_context),
            "selected_highway_values": ["primary", "secondary"],
            "role": "single fixed cartographic background only; not a historical network estimate",
            "license": "OpenStreetMap contributors; Open Database License (ODbL)",
            "feature_n": len(roads),
        },
        "coverage_audit_sha256": digest(args.coverage_audit),
        "annual_restaurant_counts": {
            str(row["year"]): row["valid_coordinate_restaurant_n"] for row in summary_rows
        },
        "privacy": "Only aggregated fixed-grid centroids and cell counts are released.",
    }
    (SOURCE / "analysis_contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    print(f"SOURCE PASS: {len(yearly_cells)} years; {len(roads)} contextual OSM features")


def single_hue_map(color: str) -> LinearSegmentedColormap:
    rgb = np.array(mpl.colors.to_rgb(color))
    white = np.ones(3)
    stops = [white * (1 - alpha) + rgb * alpha for alpha in (0.18, 0.38, 0.63, 0.84, 1.00)]
    stops[-1] = np.clip(rgb * 0.72, 0, 1)
    return LinearSegmentedColormap.from_list(f"study_{color[1:]}", stops)


def line_collection(ax: plt.Axes, roads: gpd.GeoDataFrame) -> None:
    secondary = roads.loc[roads["highway"].eq("secondary")]
    primary = roads.loc[roads["highway"].eq("primary")]
    secondary.plot(ax=ax, color="#D7DBDD", linewidth=0.18, alpha=0.90, zorder=2)
    primary.plot(ax=ax, color="#AEB6BA", linewidth=0.30, alpha=0.95, zorder=3)


def decorate_map(
    ax: plt.Axes,
    polygons: gpd.GeoDataFrame,
    roads: gpd.GeoDataFrame,
) -> None:
    xmin, ymin, xmax, ymax = polygons.total_bounds
    dx, dy = xmax - xmin, ymax - ymin
    ax.set_xlim(xmin - .01 * dx, xmax + .035 * dx)
    ax.set_ylim(ymin - .12 * dy, ymax + .025 * dy)
    ax.set_aspect("equal")
    ax.set_axis_off()
    # A clean analytical field replaces the former Voyager tile layer.  The
    # fully opaque land polygons and sparse vector roads retain orientation
    # without introducing contemporary basemap colour or labels.
    polygons.plot(ax=ax, color="#F7F8F7", edgecolor="#C9CECC", linewidth=0.14, alpha=1.0, zorder=1)
    line_collection(ax, roads)
    outer_boundary(ax, polygons, linewidth=.38)


def panel_colorbar(fig: plt.Figure, ax: plt.Axes, scatter, label: str) -> None:
    box = ax.get_position()
    # Match the visible map-frame height, while keeping the bar narrow enough
    # to remain subordinate to the restaurant pattern.
    cax = fig.add_axes([box.x1 + 0.003, box.y0, 0.0105, box.height])
    cb = fig.colorbar(scatter, cax=cax, ticks=[1, 5, 20, 80, 320], spacing="uniform")
    cb.set_ticklabels(["1", "5", "20", "80", "320+"])
    cb.ax.tick_params(length=1.8, width=0.45, pad=1.4, labelsize=6.0, colors="black")
    cb.outline.set_linewidth(0.5)
    cax.text(0.5, 1.055, label, transform=cax.transAxes, ha="center", va="bottom", fontsize=6.2)


def summary_panel(ax: plt.Axes, summary: pd.DataFrame) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.plot([0.012, 0.988], [0.94, 0.94], color="#D9DCDC", lw=0.8)
    ax.text(0.015, 0.70, "Evidence audit", fontweight="bold", fontsize=8.6, va="center")
    ax.text(0.015, 0.36, "200-m cells  |  fixed 2021 census geography", fontsize=7.2, va="center")
    for index, year in enumerate(YEARS):
        item = summary.loc[summary["year"].eq(year)].iloc[0]
        left = 0.39 + index * 0.148
        ax.add_patch(Rectangle((left, 0.21), 0.012, 0.55, color=COLORS[year], lw=0))
        ax.text(left + 0.021, 0.71, str(year), fontsize=7.5, fontweight="bold", va="center")
        ax.text(
            left + 0.021,
            0.45,
            f"{int(item['valid_coordinate_restaurant_n']):,} outlets",
            fontsize=6.9,
            va="center",
        )
        ax.text(
            left + 0.021,
            0.22,
            f"{item['occupied_fixed_2021_lsbg_share_pct']:.1f}% LSBGs",
            fontsize=6.5,
            va="center",
        )


def draw() -> None:
    cells = pd.read_csv(SOURCE / "annual_restaurant_density_200m.csv")
    summary = pd.read_csv(SOURCE / "annual_restaurant_coordinate_coverage.csv")
    contract = json.loads((SOURCE / "analysis_contract.json").read_text(encoding="utf-8"))
    polygons = gpd.read_file(GEOMETRY)
    roads = gpd.read_file(SOURCE / "osm_major_walkable_context_2025_01_01.gpkg")
    if polygons.crs.to_epsg() != 2326 or roads.crs.to_epsg() != 2326:
        raise ValueError("All analytical map layers must use EPSG:2326")
    for year in YEARS:
        counted = int(cells.loc[cells["year"].eq(year), "restaurant_n"].sum())
        audited = int(summary.loc[summary["year"].eq(year), "valid_coordinate_restaurant_n"].iat[0])
        if counted != audited or counted != contract["annual_restaurant_counts"][str(year)]:
            raise ValueError(f"{year}: public cell counts and audited annual counts disagree")

    configure()
    mpl.rcParams.update({"text.color": "black", "axes.labelcolor": "black", "font.size": 8.0})
    fig = plt.figure(figsize=(185 / 25.4, 158 / 25.4), facecolor="white")
    # Explanatory prose belongs in the manuscript caption, not inside the
    # analytical plate.  The released figure therefore retains only the
    # identifiers and quantitative decoding elements needed to read the map.
    grid = GridSpec(2, 2, figure=fig, left=0.052, right=0.930, top=0.965, bottom=0.032, hspace=0.115, wspace=0.19)
    norm = BoundaryNorm(COUNT_BREAKS, 256, clip=True)
    for index, year in enumerate(YEARS):
        ax = fig.add_subplot(grid[index // 2, index % 2])
        decorate_map(ax, polygons, roads)
        subset = cells.loc[cells["year"].eq(year)].sort_values("restaurant_n")
        sizes = np.clip(0.65 + np.log2(subset["restaurant_n"].to_numpy(float) + 1) * 0.70, 1.05, 5.2)
        plot = ax.scatter(
            subset["easting_m"],
            subset["northing_m"],
            c=subset["restaurant_n"],
            cmap=single_hue_map(COLORS[year]),
            norm=norm,
            s=sizes,
            linewidths=0,
            marker="o",
            alpha=1.0,
            rasterized=True,
            zorder=10,
        )
        # Put both the panel letter and year outside the map frame.  This keeps
        # the data field free of headings and matches the manuscript's other
        # atlas-style figures.
        box = ax.get_position()
        fig.text(box.x0, box.y1 + 0.006, chr(ord("a") + index), fontsize=9.0,
                 fontweight="bold", ha="left", va="bottom")
        fig.text((box.x0 + box.x1) / 2, box.y1 + 0.006, str(year), fontsize=9.5,
                 ha="center", va="bottom")
        ax.text(
            0.50,
            -0.012,
            f"{int(summary.loc[summary['year'].eq(year), 'valid_coordinate_restaurant_n'].iat[0]):,} observed restaurants",
            transform=ax.transAxes,
            fontsize=7.0,
            ha="center",
            va="top",
        )
        # Keep cartographic furniture above/below the data field and clear of
        # the right-side colour scale.
        north_arrow(ax, x=.065, y=.940, height=.052)
        segmented_scale_bar(ax, length_km=10, x=.055, y=.022)
        panel_colorbar(fig, ax, plot, "n")

    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / "Fig03_Study_Area_Annual_Restaurant_Distribution_NO_BASEMAP_NATURE"
    fig.savefig(stem.with_suffix(".png"), dpi=360, bbox_inches="tight", pad_inches=0.07, facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.07, facecolor="white")
    plt.close(fig)
    print(f"FIGURE PASS: {stem.with_suffix('.png')}")


def main() -> None:
    args = parse_args()
    if args.build_source:
        build_public_source(args)
    draw()


if __name__ == "__main__":
    main()
