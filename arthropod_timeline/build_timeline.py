"""
Arthropod-dataset loader and driver.

Everything dataset-agnostic (NLCD enrichment, the timeline, the map, CSV
export, the page shell) lives in the shared ../pipeline.py. This file only
knows how to turn the raw arthropod study files in data/ into the standard
site table pipeline.py expects, plus the small amount of config (title,
captions) specific to this dataset.
"""

import json
import logging
import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pipeline

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*default value for compat will change.*",
)

# The dashboard is built straight from the raw study files in data/:
#   * sampling CSVs, one row per organism record, carrying site_code and
#     sample_date (e.g. 41_core_arthropods.csv)
#   * location GeoJSONs describing where each site sits
#     (e.g. 41_core_arthropods_locations.geojson)
# Nothing below names a specific file, so dropping updated or additional study
# files into data/ is all that is needed to refresh the dashboard. Files that
# do not carry these columns are ignored.
SAMPLING_CSV_COLUMNS = {"site_code", "sample_date"}

# A sample_date counts as a sampling event when a trap was actually collected.
# An empty trap is a real event with a real result and so stays in; these flags
# mean no sample was obtained, or that the record is unusable.
EXCLUDED_FLAGS = {"trap_not_collected", "miscoded", "empty_sampling_event"}

# A site counts as having run to the end of the study when its last sampling
# event falls within this window of the most recent event anywhere in the data.
STUDY_END_WINDOW = pd.Timedelta(days=365)
STATUS_TO_STUDY_END = "Sampled to study end"
STATUS_RETIRED_EARLY = "Retired early"

# The McDowell location file carries one polygon per *pair* of sites, keyed by a
# composite name, and each polygon is only the bounding box of its two sites, so
# per-site positions cannot be recovered from it. These ten were carried forward
# from the project's earlier hand-distilled site table (arthros_temporal.csv),
# which is the only surviving record of them; _check_paired_polygons() confirms
# each still falls inside the polygon that covers it on every build.
MCDOWELL_SITE_COORDS = {
    "Bell": (33.64092797, -111.85715019),
    "Gateway": (33.644507375, -111.84884062),
    "Mine": (33.652722605, -111.788845595),
    "Prospector": (33.65168216, -111.797182615),
    "Rincon": (33.597271495, -111.810251345),
    "Sunrise": (33.60752663, -111.804051625),
    "Dixileta": (33.75610739, -111.844353495),
    "LoneMtn": (33.76264058, -111.842862815),
    "Paraiso": (33.693904995, -111.81341669),
    "TomThumb": (33.691912325, -111.80063395),
}

# Which sites each composite polygon covers. Note that "DixieMine" in the
# polygon name is the site coded "Mine" in the sampling CSV.
PAIRED_POLYGON_MEMBERS = {
    "Bell_Gateway": ("Bell", "Gateway"),
    "DixieMine_Prospector": ("Mine", "Prospector"),
    "Dixileta_LoneMtn": ("Dixileta", "LoneMtn"),
    "Rincon_Sunrise": ("Rincon", "Sunrise"),
    "TomThumb_Paraiso": ("TomThumb", "Paraiso"),
}


def _polygon_centroid(ring: list[list[float]]) -> tuple[float, float]:
    """Area-weighted centroid of a closed GeoJSON linear ring, as (lat, lon)."""
    points = ring[:-1] if ring[0] == ring[-1] else ring
    area = cx = cy = 0.0
    for i in range(len(points)):
        x0, y0 = points[i]
        x1, y1 = points[(i + 1) % len(points)]
        cross = x0 * y1 - x1 * y0
        area += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    area *= 0.5

    if area == 0:  # degenerate ring; fall back to the mean vertex
        return (
            sum(p[1] for p in points) / len(points),
            sum(p[0] for p in points) / len(points),
        )
    return cy / (6 * area), cx / (6 * area)


def _feature_centroid(geometry: dict) -> tuple[float, float] | None:
    """Representative (lat, lon) for a GeoJSON geometry, or None if unsupported."""
    coords = geometry.get("coordinates")
    if not coords:
        return None

    geom_type = geometry.get("type")
    if geom_type == "Point":
        return coords[1], coords[0]
    if geom_type == "Polygon":
        return _polygon_centroid(coords[0])
    if geom_type == "MultiPolygon":
        return _polygon_centroid(coords[0][0])

    logging.warning("Unsupported geometry type %r; skipping feature", geom_type)
    return None


def _check_paired_polygons(
    paired_rings: dict[str, list[list[float]]],
    locations: dict[str, tuple[float, float]],
) -> None:
    """
    Guard MCDOWELL_SITE_COORDS against drift.

    Those coordinates are maintained by hand, so confirm each one still falls
    inside the polygon that is supposed to cover it, and complain about any
    paired location this script does not know how to split into sites.
    """
    for name, ring in paired_rings.items():
        members = PAIRED_POLYGON_MEMBERS.get(name)
        if members is None:
            logging.warning(
                "Paired location %r has no entry in PAIRED_POLYGON_MEMBERS; its "
                "sites will have no coordinates unless listed in MCDOWELL_SITE_COORDS",
                name,
            )
            continue

        lons = [p[0] for p in ring]
        lats = [p[1] for p in ring]
        for site in members:
            point = locations.get(site)
            if point is None:
                logging.warning(
                    "Polygon %r covers site %r, which has no coordinate", name, site
                )
            elif not (
                min(lats) <= point[0] <= max(lats)
                and min(lons) <= point[1] <= max(lons)
            ):
                logging.warning(
                    "Coordinate for %r lies outside polygon %r; check "
                    "MCDOWELL_SITE_COORDS against the location file",
                    site,
                    name,
                )


def load_site_locations(data_dir: Path) -> dict[str, tuple[float, float]]:
    """
    Read every *.geojson in data_dir and return {site_code: (lat, lon)}.

    Features keyed by `site_code` describe a single site and place it at the
    polygon centroid. Features keyed by `sampling_location` cover a pair of
    sites and are used only to validate the hand-kept McDowell coordinates.
    """
    locations: dict[str, tuple[float, float]] = {}
    paired_rings: dict[str, list[list[float]]] = {}

    for path in sorted(data_dir.glob("*.geojson")):
        features = json.loads(path.read_text(encoding="utf-8")).get("features", [])
        for feature in features:
            props = feature.get("properties") or {}
            geometry = feature.get("geometry") or {}

            if "site_code" in props:
                centre = _feature_centroid(geometry)
                if centre is not None:
                    locations[props["site_code"]] = centre
            elif "sampling_location" in props and geometry.get("type") == "Polygon":
                paired_rings[props["sampling_location"]] = geometry["coordinates"][0]

        logging.info("Read %d location features from %s", len(features), path.name)

    locations.update(MCDOWELL_SITE_COORDS)
    _check_paired_polygons(paired_rings, locations)
    return locations


def load_sampling_spans(data_dir: Path) -> pd.DataFrame:
    """
    Read every sampling CSV in data_dir and return one row per site giving the
    first and last date on which that site was sampled.
    """
    frames = []

    for path in sorted(data_dir.glob("*.csv")):
        columns = set(pd.read_csv(path, nrows=0).columns)
        if not SAMPLING_CSV_COLUMNS.issubset(columns):
            logging.info("Skipping %s: not a sampling table", path.name)
            continue

        frame = pd.read_csv(
            path,
            usecols=sorted(SAMPLING_CSV_COLUMNS | (columns & {"flags"})),
            parse_dates=["sample_date"],
        )
        kept = frame[~frame.get("flags", pd.Series(dtype=object)).isin(EXCLUDED_FLAGS)]
        logging.info(
            "Read %d sampling records from %s (%d excluded by flag)",
            len(kept),
            path.name,
            len(frame) - len(kept),
        )
        frames.append(kept[["site_code", "sample_date"]])

    if not frames:
        raise FileNotFoundError(
            f"No sampling CSVs found in {data_dir}; expected files with "
            f"{sorted(SAMPLING_CSV_COLUMNS)} columns"
        )

    records = pd.concat(frames, ignore_index=True).dropna(subset=["sample_date"])
    return (
        records.groupby("site_code")["sample_date"]
        .agg(start_date="min", end_date="max")
        .reset_index()
    )


def build_site_table(data_dir: Path) -> pd.DataFrame:
    """
    Assemble one row per site: its sampling span from the sampling CSVs and its
    position from the location GeoJSONs.
    """
    df = load_sampling_spans(data_dir)
    locations = load_site_locations(data_dir)

    df["lat"] = df["site_code"].map(lambda s: locations.get(s, (None, None))[0])
    df["long"] = df["site_code"].map(lambda s: locations.get(s, (None, None))[1])

    unplaced = sorted(df.loc[df["lat"].isna(), "site_code"])
    if unplaced:
        logging.warning(
            "No coordinates for %d sampled site(s): %s",
            len(unplaced),
            ", ".join(unplaced),
        )

    unsampled = sorted(set(locations) - set(df["site_code"]))
    if unsampled:
        logging.info(
            "%d located site(s) have no sampling records: %s",
            len(unsampled),
            ", ".join(unsampled),
        )

    df = pipeline.assign_study_end_status(
        df, STUDY_END_WINDOW, STATUS_TO_STUDY_END, STATUS_RETIRED_EARLY
    )
    pipeline.warn_duplicate_coordinates(df)

    return df.sort_values("site_code").reset_index(drop=True)


def main():
    logging.basicConfig(level=logging.INFO)

    root = Path(__file__).parent
    data_dir = root / "data"
    output_dir = root / "output"
    build_dir = root.parent / "build"
    build_dir.mkdir(exist_ok=True)

    # One row per site, derived entirely from the raw study files in data/
    df = build_site_table(data_dir)
    logging.info(
        "Site table: %d sites sampled %s to %s",
        len(df),
        df["start_date"].min().date(),
        df["end_date"].max().date(),
    )

    # Add NLCD land-use info + codes_by_index
    df, codes_by_index = pipeline.enrich_with_land_use(df)

    # Export enriched data to CSV files
    pipeline.export_enriched_data(df, codes_by_index, output_dir)

    # Build HTML snippets
    timeline_html, timeline_height = pipeline.build_segmented_timeline_html(
        df, codes_by_index
    )
    map_html, map_bounds = pipeline.build_map_html(df)

    # Build full page
    full_page = pipeline.build_full_page(
        timeline_html,
        timeline_height,
        map_html,
        map_bounds,
        nav_html=pipeline.build_nav_html(active="index.html"),
    )

    out_path = build_dir / "index.html"
    out_path.write_text(full_page, encoding="utf-8")
    print(f"Timeline & map page built -> {out_path}")

if __name__ == "__main__":
    main()
