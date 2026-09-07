import json
import logging
import math
import warnings
from pathlib import Path

import pandas as pd
import plotly.express as px
from plotly.offline import get_plotlyjs_version
from pygeohydro import nlcd as nlcd_mod

# Pin the CDN bundle to the plotly.js that generated the figure JSON, so the
# page can never drift from the installed plotly version
PLOTLYJS_VERSION = get_plotlyjs_version()

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*default value for compat will change.*",
)

# NLCD snapshot years we will use
NLCD_YEARS = [2001, 2006, 2011, 2016, 2019, 2021]

# A committed lookup of NLCD codes keyed by coordinate, shared by every dataset
# built from this repo. The MRLC service this queries is unreliable (observed
# outright ServiceUnavailableError and incomplete responses during development),
# so once a coordinate has been looked up it is never queried again: a build
# with nothing new to look up has no network dependency at all. Keyed by
# coordinate rather than site_code because the same site_code can denote a
# different physical location in a different dataset (confirmed true of the
# arthropod and bird site tables), so a site-keyed cache could silently mix
# them up.
NLCD_CACHE_PATH = Path(__file__).resolve().parent.parent / "nlcd_lookup.json"

# Map NLCD numeric codes -> human-readable labels
NLCD_LABELS = {
    11: "Open water",
    12: "Perennial ice/snow",
    21: "Developed, open space",
    22: "Developed, low intensity",
    23: "Developed, medium intensity",
    24: "Developed, high intensity",
    31: "Barren land",
    41: "Deciduous forest",
    42: "Evergreen forest",
    43: "Mixed forest",
    52: "Shrub/scrub",
    71: "Grassland/herbaceous",
    81: "Pasture/hay",
    82: "Cultivated crops",
    90: "Woody wetlands",
    95: "Emergent herbaceous wetlands",
}

# Nice colors for land-use
LANDUSE_COLORS = {
    "Cultivated crops": "#15803d",
    "Developed, high intensity": "#b91c1c",
    "Developed, medium intensity": "#f97373",
    "Developed, low intensity": "#fecaca",
    "Developed, open space": "#facc15",
    "Shrub/scrub": "#fbbf77",
    "Grassland/herbaceous": "#22c55e",
    "Deciduous forest": "#16a34a",
    "Evergreen forest": "#166534",
    "Mixed forest": "#4ade80",
    "Open water": "#0ea5e9",
    "Barren land": "#a3a3a3",
    "Woody wetlands": "#4b5563",
    "Emergent herbaceous wetlands": "#22c55e",
    "Other": "#6b7280",
    "Unknown": "#9ca3af",
}

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


# Geometry of the drawn map. The height is fixed by the figure, but the width
# follows the browser window, so the view is fitted to a deliberately narrow
# viewport and wider windows simply show more surrounding context.
MAP_HEIGHT = 460
MAP_MARGIN = {"l": 0, "r": 0, "t": 40, "b": 0}
MAP_TOP_MARGIN = MAP_MARGIN["t"]
MAP_MIN_WIDTH_PX = 600
MAP_TILE_SIZE = 512  # MapLibre's Web Mercator tile size
MAP_PADDING = 1.15  # keep the outermost sites clear of the edges

# Below this drawn width the legend is moved off the map and laid out beneath
# it, where it needs this much room.
LEGEND_REFLOW_PX = 560
LEGEND_REFLOW_HEIGHT = 132


def _mercator_y(lat: float) -> float:
    """Web Mercator northing for a latitude, normalised to 0..1."""
    radians = math.radians(lat)
    return (1 - math.log(math.tan(radians) + 1 / math.cos(radians)) / math.pi) / 2


def _mercator_lat(y: float) -> float:
    """Inverse of _mercator_y()."""
    return math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y))))


def _fit_map_view(
    lats: "pd.Series", lons: "pd.Series"
) -> tuple[dict[str, float], float]:
    """
    Centre and zoom that keep every site inside the map.

    Plotly has no fit-to-bounds for map subplots, so the view is derived from
    the extent of the data. A fixed zoom cannot do this: it crops whichever
    sites fall outside whatever the chosen level happens to cover.
    """
    lat_min, lat_max = float(lats.min()), float(lats.max())
    lon_min, lon_max = float(lons.min()), float(lons.max())

    # y grows southward, so the northern edge gives the smaller value
    y_north, y_south = _mercator_y(lat_max), _mercator_y(lat_min)
    span_x = max((lon_max - lon_min) / 360, 1e-9) * MAP_PADDING
    span_y = max(y_south - y_north, 1e-9) * MAP_PADDING

    zoom = min(
        math.log2(MAP_MIN_WIDTH_PX / (MAP_TILE_SIZE * span_x)),
        math.log2((MAP_HEIGHT - MAP_TOP_MARGIN) / (MAP_TILE_SIZE * span_y)),
    )
    center = {
        "lat": _mercator_lat((y_north + y_south) / 2),
        "lon": (lon_min + lon_max) / 2,
    }
    return center, zoom


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

    # Sites sampled right up to the end of the study, versus retired before it
    study_end = df["end_date"].max()
    df["status"] = (df["end_date"] >= study_end - STUDY_END_WINDOW).map(
        {True: STATUS_TO_STUDY_END, False: STATUS_RETIRED_EARLY}
    )

    return df.sort_values("site_code").reset_index(drop=True)


def _choose_nlcd_year(sample_year: int | None, available_years: list[int]) -> int:
    """
    Given a calendar year and list of NLCD years, choose the closest NLCD
    year at or before sample_year. If sample_year is before all NLCD years,
    use the earliest; if sample_year is None, use the latest.
    """
    if sample_year is None:
        return max(available_years)

    past_years = [y for y in available_years if y <= sample_year]
    if past_years:
        return max(past_years)
    return min(available_years)


def _code_to_label(code: int | float | None) -> str | None:
    """Convert an NLCD numeric code into a nice label."""
    if pd.isna(code):
        return None
    try:
        return NLCD_LABELS.get(int(code), "Other")
    except Exception:
        return "Other"

def _coord_key(lat: float, lon: float) -> str:
    # 6 decimal places is ~0.1 m, far finer than coordinates need to be to
    # select the right 30 m NLCD cell, and stable across runs since the
    # upstream centroid math is itself deterministic.
    return f"{lat:.6f},{lon:.6f}"


def _load_nlcd_cache() -> dict[str, dict[str, int]]:
    if NLCD_CACHE_PATH.exists():
        return json.loads(NLCD_CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def _save_nlcd_cache(cache: dict[str, dict[str, int]]) -> None:
    NLCD_CACHE_PATH.write_text(
        json.dumps(cache, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def get_nlcd_codes(
    coords: list[tuple[float, float]], years: list[int]
) -> dict[tuple[float, float], dict[int, int]]:
    """
    NLCD cover code per (lat, lon) for each requested year, backed by the
    committed cache at NLCD_CACHE_PATH.

    Coordinates already in the cache for every requested year are returned
    without any network access. Anything else is queried from the MRLC
    service; on success the cache is updated and saved so the same
    coordinates never need to be queried again. If the service is unavailable
    or returns incomplete data for a coordinate not already cached, this
    raises rather than returning partial results — a build should fail
    outright rather than publish a page with silently missing land-use.
    """
    cache = _load_nlcd_cache()
    result: dict[tuple[float, float], dict[int, int]] = {}
    missing: list[tuple[float, float]] = []

    for lat, lon in coords:
        entry = cache.get(_coord_key(lat, lon), {})
        codes = {y: entry[str(y)] for y in years if str(y) in entry}
        if len(codes) == len(years):
            result[(lat, lon)] = codes
        else:
            missing.append((lat, lon))

    if not missing:
        return result

    logging.info(
        "Querying NLCD for %d coordinate(s) not yet in %s",
        len(missing),
        NLCD_CACHE_PATH.name,
    )
    try:
        nlcd_gdf = nlcd_mod.nlcd_bycoords(
            [(lon, lat) for lat, lon in missing],
            years={"cover": years},
            region="L48",
        )
    except Exception as exc:
        raise RuntimeError(
            f"NLCD service unavailable and {len(missing)} coordinate(s) are not "
            f"yet in {NLCD_CACHE_PATH.name}; cannot build without land-use data "
            f"for: {missing[:5]}{' ...' if len(missing) > 5 else ''}. Once cached, "
            f"a coordinate never depends on this service again."
        ) from exc

    for (lat, lon), row in zip(missing, nlcd_gdf.itertuples()):
        codes = {y: getattr(row, f"cover_{y}") for y in years if hasattr(row, f"cover_{y}")}
        if len(codes) < len(years):
            raise RuntimeError(
                f"NLCD service returned incomplete data for ({lat}, {lon}): "
                f"got years {sorted(codes)}, need {years}"
            )
        result[(lat, lon)] = codes
        cache.setdefault(_coord_key(lat, lon), {}).update(
            {str(y): int(c) for y, c in codes.items()}
        )

    _save_nlcd_cache(cache)
    logging.info(
        "Saved %d new coordinate(s) to %s", len(missing), NLCD_CACHE_PATH.name
    )
    return result


def enrich_with_land_use(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, dict[int, int]]]:

    df = df.copy()

    # Clean coordinates
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["long"] = pd.to_numeric(df["long"], errors="coerce")
    valid_mask = df["lat"].notna() & df["long"].notna()
    df_valid = df[valid_mask]

    # Default empty columns
    df["nlcd_code_start"] = pd.NA
    df["nlcd_code_end"] = pd.NA
    df["nlcd_code_latest"] = pd.NA
    df["land_use_start"] = pd.NA
    df["land_use_end"] = pd.NA
    df["land_use"] = pd.NA  # latest land-use, used for the map
    df["land_use_changed"] = False

    codes_by_index: dict[int, dict[int, int]] = {}

    if df_valid.empty:
        logging.warning("No valid lat/long values – skipping NLCD enrichment.")
        return df, codes_by_index

    # get_nlcd_codes() raises if it cannot obtain real data for a coordinate
    # (cached or freshly queried); a build must not silently continue with
    # missing land-use, so no try/except sits around this call.
    codes_by_coord = get_nlcd_codes(
        list(zip(df_valid["lat"], df_valid["long"])), NLCD_YEARS
    )

    # Map from df index -> {year: code}
    for idx, lat, lon in zip(df_valid.index, df_valid["lat"], df_valid["long"]):
        year_to_code = codes_by_coord[(lat, lon)]
        codes_by_index[idx] = year_to_code

        years_avail = sorted(year_to_code.keys())

        start_date = df.at[idx, "start_date"]
        end_date = df.at[idx, "end_date"]

        start_year = int(start_date.year) if pd.notna(start_date) else None
        end_year = int(end_date.year) if pd.notna(end_date) else None

        start_snap = _choose_nlcd_year(start_year, years_avail)
        end_snap = _choose_nlcd_year(end_year, years_avail)
        latest_snap = max(years_avail)

        code_start = year_to_code.get(start_snap, pd.NA)
        code_end = year_to_code.get(end_snap, pd.NA)
        code_latest = year_to_code.get(latest_snap, pd.NA)

        df.at[idx, "nlcd_code_start"] = code_start
        df.at[idx, "nlcd_code_end"] = code_end
        df.at[idx, "nlcd_code_latest"] = code_latest

        label_start = _code_to_label(code_start)
        label_end = _code_to_label(code_end)
        label_latest = _code_to_label(code_latest)

        if label_start is not None:
            df.at[idx, "land_use_start"] = label_start
        if label_end is not None:
            df.at[idx, "land_use_end"] = label_end
        if label_latest is not None:
            df.at[idx, "land_use"] = label_latest

    # Check if land-use changed across any NLCD snapshot years
    # This is more comprehensive than just start vs end
    for idx in df.index:
        codes = codes_by_index.get(idx, {})
        if codes:
            land_use_values = []
            for year in sorted(codes.keys()):
                code = codes[year]
                label = _code_to_label(code)
                if label:
                    land_use_values.append(label)
            # Check if there are multiple unique land-use values
            unique_land_uses = set(land_use_values)
            df.at[idx, "land_use_changed"] = len(unique_land_uses) > 1

    return df, codes_by_index

def build_segmented_timeline_html(
    df: pd.DataFrame, codes_by_index: dict[int, dict[int, int]]
) -> str:
    """
    Build a timeline where each site is one row, segmented by NLCD snapshot years.
    Each segment shows the land-use for that NLCD period (e.g., 2001-2006 uses 2001 data).
    For ended sites, segments stop at the actual end_date (creating a blank/gap after).
    """
    rows = []

    for idx, row in df.iterrows():
        site = row["site_code"]
        status = row["status"]
        codes = codes_by_index.get(idx)

        start_date = row["start_date"]
        end_date = row["end_date"]  # last date this site was sampled
 #validation check (skips sites with no NLCD data)
        if codes is None or not codes:
            continue
        if pd.isna(start_date) or pd.isna(end_date):
            continue
        years_avail = sorted(codes.keys())
        if not years_avail:
            continue

        relevant_snapshots = sorted([y for y in NLCD_YEARS if y in years_avail])
        if not relevant_snapshots:
            continue

        # Show ALL land-use changes during site's active period
        boundaries = [pd.Timestamp(year=y, month=1, day=1) for y in relevant_snapshots]
        site_end_date = end_date

        def get_land_use(year):
            return _code_to_label(codes.get(year)) or "Unknown"
        
        land_use_by_year = {y: get_land_use(y) for y in relevant_snapshots}
        
        segment_start = start_date
        current_land_use = None
        
        # Find the snapshot that applies at start_date (latest snapshot <= start_date)
        snap_for_start_idx = 0
        for i, boundary in enumerate(boundaries):
            if boundary <= start_date:
                snap_for_start_idx = i
            else:
                break
        
        # Process each NLCD snapshot boundary
        for i, snap_year in enumerate(relevant_snapshots):
            boundary = boundaries[i]
            if boundary < start_date:
                continue
            
            # Stop if boundary is after site ended
            if boundary > site_end_date:
                break
            
            # Determine land-use for period ending at this boundary
            if i == snap_for_start_idx or (i > 0 and boundaries[i-1] < start_date):
                # First period uses the snapshot that applies at start_date
                period_land_use = land_use_by_year[relevant_snapshots[snap_for_start_idx]]
            else:
                prev_snap = relevant_snapshots[i - 1]
                period_land_use = land_use_by_year[prev_snap]
            
            segment_end = min(boundary, site_end_date)
#checks if land use will change at next boundary
            if segment_end > segment_start:
                if current_land_use is None:
                    # First segment
                    rows.append({
                        "site_code": site,
                        "status": status,
                        "segment_start": segment_start,
                        "segment_end": segment_end,
                        "land_use_segment": period_land_use,
                    })
                    current_land_use = period_land_use
                elif period_land_use != current_land_use:
                    # Land-use changed - create new segment
                    rows.append({
                        "site_code": site,
                        "status": status,
                        "segment_start": segment_start,
                        "segment_end": segment_end,
                        "land_use_segment": period_land_use,
                    })
                    current_land_use = period_land_use
                else:
                    # Same land-use - extend current segment
                    if rows and rows[-1]["site_code"] == site:
                        rows[-1]["segment_end"] = segment_end
            
            segment_start = segment_end
            
            # Check if land-use changes at next boundary
            if i < len(relevant_snapshots) - 1:
                next_snap = relevant_snapshots[i + 1]
                next_land_use = land_use_by_year[next_snap]
                if next_land_use != land_use_by_year[snap_year]:
                    # Will change at next boundary - reset for next iteration
                    current_land_use = None
        
# Final segment after last snapshot 
        if segment_start < site_end_date:
            final_land_use = land_use_by_year[relevant_snapshots[-1]]
            if current_land_use is None or final_land_use != current_land_use:
                rows.append({
                    "site_code": site,
                    "status": status,
                    "segment_start": segment_start,
                    "segment_end": site_end_date,
                    "land_use_segment": final_land_use,
                })
            elif rows and rows[-1]["site_code"] == site:
                rows[-1]["segment_end"] = site_end_date

    if not rows:
        logging.warning("No rows built for segmented land-use timeline.")
        return "<!-- empty timeline -->", 600

    # Clean up segments: merge adjacent segments with same land-use
    seg_df = pd.DataFrame(rows).sort_values(["site_code", "segment_start"])
    cleaned_rows = []
    
    for site_code in seg_df["site_code"].unique():
        site_segs = seg_df[seg_df["site_code"] == site_code].copy()
        site_segs = site_segs[site_segs["segment_end"] > site_segs["segment_start"]]
        
        if len(site_segs) == 0:
            continue
        
        for idx, row in site_segs.iterrows():
            if not cleaned_rows or cleaned_rows[-1]["site_code"] != site_code:
                cleaned_rows.append(row.to_dict())
            else:
                prev = cleaned_rows[-1]
                # Merge if same land-use and adjacent/overlapping
                if (row["land_use_segment"] == prev["land_use_segment"] and
                    row["segment_start"] <= prev["segment_end"]):
                    prev["segment_end"] = max(prev["segment_end"], row["segment_end"])
                else:
                    # Ensure no gap
                    if row["segment_start"] < prev["segment_end"]:
                        row["segment_start"] = prev["segment_end"]
                    if row["segment_end"] > row["segment_start"]:
                        cleaned_rows.append(row.to_dict())
    
    seg_df = pd.DataFrame(cleaned_rows).sort_values(["site_code", "segment_start"]) if cleaned_rows else seg_df
    
    # Format dates for better tooltip display (for hovering over segments)
    seg_df["segment_start_str"] = seg_df["segment_start"].dt.strftime("%Y-%m-%d")
    seg_df["segment_end_str"] = seg_df["segment_end"].dt.strftime("%Y-%m-%d")

    # Get all unique sites and calculate dynamic height
    unique_sites = sorted(seg_df["site_code"].unique())
    num_sites = len(unique_sites)
    logging.info(f"Timeline includes {num_sites} unique sites")
    logging.info(f"All sites in timeline: {unique_sites}")
    # Calculate height: use ~14px per site to keep bars thin while ensuring ALL labels are visible
    # Plotly automatically hides overlapping labels if there's not enough space
    # With 67 sites * 14px = 938px, this keeps bars thin while ensuring all 67 site names display
    # This is the minimum height needed to prevent Plotly from hiding labels
    calculated_height = max(600, num_sites * 14)

#create timeline plot
    fig = px.timeline(
        seg_df,
        x_start="segment_start",
        x_end="segment_end",
        y="site_code",
        color="land_use_segment",
        title="",
        color_discrete_map=LANDUSE_COLORS,
        custom_data=["land_use_segment", "status", "segment_start_str", "segment_end_str"],
    )

    fig.update_yaxes(autorange="reversed")

    fig.update_traces(
        hovertemplate=(
            "<b>Site:</b> %{y}<br>"
            "<b>Land-use:</b> %{customdata[0]}<br>"
            "<b>Site status:</b> %{customdata[1]}<br>"
            "<b>Period:</b> %{customdata[2]} to %{customdata[3]}<extra></extra>"
        ),
        marker=dict(line=dict(width=0.5, color="rgba(255,255,255,0.4)")),
    )

# add dotted lines for each NLCD snapshot year with individual labels
    nlcd_annotations = []
    for snap_year in NLCD_YEARS:
        snap_date = pd.Timestamp(year=snap_year, month=1, day=1)
        fig.add_shape(
            type="line",
            x0=snap_date,
            x1=snap_date,
            y0=0,
            y1=1,
            yref="paper",
            line=dict(
                width=1,
                dash="dot",
                color="rgba(100, 100, 100, 0.25)",
            ),
        )
        # Collect annotations to add to layout
        nlcd_annotations.append(
            dict(
                x=snap_date,
                y=1.02,
                xref="x",
                yref="paper",
                text=f"NLCD {snap_year}",
                showarrow=False,
                font=dict(size=9, color="rgba(50, 50, 50, 0.9)"),
                xanchor="center",
                yanchor="bottom",
                bgcolor="rgba(255, 255, 255, 0.85)",
                bordercolor="rgba(100, 100, 100, 0.4)",
                borderwidth=1,
                borderpad=3,
            )
        )

    # Add horizontal grey lines every 5 bars to improve readability
    # Calculate positions for lines after every 5th site
    for i in range(5, num_sites, 5):
        # Position the line between sites (after the 5th, 10th, 15th, etc.)
        # Since y-axis is reversed, calculate position from top
        # Each site takes 1/num_sites of the space
        y_position = 1 - (i / num_sites)
        fig.add_shape(
            type="line",
            x0=0,
            x1=1,
            xref="paper",
            y0=y_position,
            y1=y_position,
            yref="paper",
            line=dict(
                width=1,
                color="rgba(200, 200, 200, 0.4)",  # Light grey
                dash="solid",
            ),
            layer="below",  # Draw behind the bars
        )

    fig.update_xaxes(
        title_text="Year",
        showgrid=True,
        gridcolor="rgba(148,163,184,0.2)",
        gridwidth=1,
    )
    # Configure y-axis to show ALL site names with consistent spacing
    # For categorical y-axis in timeline plots, ensure all categories are shown
    # by setting categoryorder and categoryarray, and increasing height if needed
    # With sufficient height (22px per site), Plotly should show all labels
    # Note: For categorical axes, we use categoryarray, not tickmode/tickvals
    fig.update_yaxes(
        title_text="Site",
        tickfont=dict(size=8),  # Normal font, not bold
        showgrid=False,
        categoryorder="array",  # Use explicit array order
        categoryarray=unique_sites,  # All site names in order - this ensures all 67 sites are categories
        showticklabels=True,  # Explicitly enable tick labels
        automargin=True,  # Allow margin adjustment for labels
        type="category",  # Explicitly set as category type
    )

    fig.update_layout(
        legend_title="NLCD land-cover class",
        template="plotly_white",
        margin=dict(l=140, r=40, t=100, b=50),  # Increased top margin for NLCD labels, left margin for site labels
        height=calculated_height,  # Use dynamic height
        bargap=0,  # Keep bars touching for continuous timeline appearance
        font=dict(
            family="system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
            size=11,
        ),
        hovermode="closest",
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            font=dict(size=10),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(200,200,200,0.3)",
            borderwidth=1,
        ),
        annotations=nlcd_annotations + [
            dict(
                text="Segmented by NLCD snapshot years (2001, 2006, 2011, 2016, 2019). Dotted lines mark snapshot boundaries. Ended sites stop at their end date.",
                xref="paper",
                yref="paper",
                x=0.5,
                y=-0.1,
                xanchor="center",
                yanchor="top",
                showarrow=False,
                font=dict(size=9, color="rgba(100,100,100,0.7)"),
            )
        ],
    )

    html = fig.to_html(
        full_html=False,
        include_plotlyjs=False,
        div_id="timeline-plot",
        config={"responsive": True},
    )
    # The height is driven by the number of sites, so the container is sized to
    # match it: a responsive plot is resized to its container, and a shorter one
    # would start dropping site labels.
    return html, calculated_height

def build_map_html(df: pd.DataFrame) -> str:
    df = df.copy()
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["long"] = pd.to_numeric(df["long"], errors="coerce")
    df = df.dropna(subset=["lat", "long"])

    if df.empty:
        return "<!-- empty map -->", None

    # Initial framing for first paint; refit_map() in the page then refits this
    # to whatever size the map is actually drawn at on the reader's screen.
    center, zoom = _fit_map_view(df["lat"], df["long"])
    bounds = {
        "latMin": float(df["lat"].min()),
        "latMax": float(df["lat"].max()),
        "lonMin": float(df["long"].min()),
        "lonMax": float(df["long"].max()),
    }

    df["land_use"] = df["land_use"].fillna("Unknown")

    df["start_date_formatted"] = df["start_date"].dt.strftime("%Y-%m-%d")
    df["end_date_formatted"] = df["end_date"].dt.strftime("%Y-%m-%d")

    fig = px.scatter_map(
        df,
        lat="lat",
        lon="long",
        hover_name="site_code",
        color="land_use",
        color_discrete_map=LANDUSE_COLORS,
        zoom=zoom,
        center=center,
        # No explicit height: Plotly only lets a responsive plot follow its
        # container's height when the layout does not pin one.
        hover_data={
            "land_use": True,
            "status": True,
            "start_date_formatted": True,
            "end_date_formatted": True,
            "lat": False,
            "long": False,
        },
        title="",
    )

    fig.update_traces(
        # Some sites are only a few hundred metres apart (Rincon/Sunrise, the
        # NDV cluster, MVLH2/VALLUT) and their markers overlap until the reader
        # zooms in; the slight transparency keeps a stack of them legible.
        marker=dict(size=8, opacity=0.85),
        hovertemplate=(
            "<b>%{hovertext}</b><br><br>"
            "land_use=%{customdata[0]}<br>"
            "status=%{customdata[1]}<br>"
            "start_date=%{customdata[2]}<br>"
            "end_date=%{customdata[3]}<extra></extra>"
        ),
    )

    fig.update_layout(
        # MapLibre basemap; resolves to CARTO's vector style, which needs no
        # API key (the legacy raster tiles are now watermarked without one)
        map_style="carto-positron",
        margin=MAP_MARGIN,
        legend_title_text="Land-use class (latest snapshot)",
        font=dict(
            family="system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
            size=11,
        ),
        # Sit the legend on top of the map rather than beside it. Beside it, the
        # legend claimed a fixed ~200px that the map could not use, which ate
        # most of the width on a small screen.
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="right",
            x=0.995,
            font=dict(size=10),
            bgcolor="rgba(255,255,255,0.82)",
            bordercolor="rgba(200,200,200,0.35)",
            borderwidth=1,
        ),
    )

    html = fig.to_html(
        full_html=False,
        include_plotlyjs=False,
        div_id="map-plot",
        config={"responsive": True},
    )
    return html, bounds

def export_enriched_data(
    df: pd.DataFrame, codes_by_index: dict[int, dict[int, int]], output_dir: Path
):
    """
    Export enriched data to CSV files:
    1. Summary CSV: One row per site with start/end land-use
    2. Detailed CSV: One row per land-use segment showing all changes
    """
    # Prepare summary CSV (one row per site)
    summary_cols = [
        "site_code",
        "start_date",
        "end_date",
        "status",
        "lat",
        "long",
        "land_use_start",
        "land_use_end",
        "land_use",  # latest
        "land_use_changed",
        "nlcd_code_start",
        "nlcd_code_end",
        "nlcd_code_latest",
    ]
    
    # Add NLCD year columns for each snapshot
    for year in NLCD_YEARS:
        summary_cols.append(f"nlcd_code_{year}")
        summary_cols.append(f"land_use_{year}")
    
    summary_df = df.copy()
    
    # Add NLCD data for each snapshot year
    for year in NLCD_YEARS:
        summary_df[f"nlcd_code_{year}"] = pd.NA
        summary_df[f"land_use_{year}"] = pd.NA
        
        for idx in summary_df.index:
            codes = codes_by_index.get(idx, {})
            if year in codes:
                code = codes[year]
                summary_df.at[idx, f"nlcd_code_{year}"] = code
                label = _code_to_label(code)
                if label:
                    summary_df.at[idx, f"land_use_{year}"] = label
    
    # Select and reorder columns
    available_cols = [col for col in summary_cols if col in summary_df.columns]
    summary_df = summary_df[available_cols].copy()
    
    # Format dates for CSV export
    if "start_date" in summary_df.columns:
        summary_df["start_date"] = summary_df["start_date"].dt.strftime("%Y-%m-%d")
    if "end_date" in summary_df.columns:
        summary_df["end_date"] = summary_df["end_date"].dt.strftime("%Y-%m-%d")
    
    # Prepare detailed CSV (one row per land-use segment)
    detailed_rows = []
    
    for idx, row in df.iterrows():
        site = row["site_code"]
        status = row["status"]
        codes = codes_by_index.get(idx)
        
        start_date = row["start_date"]
        end_date = row["end_date"]
        lat = row.get("lat", pd.NA)
        long = row.get("long", pd.NA)

        if codes is None or not codes:
            continue
        if pd.isna(start_date) or pd.isna(end_date):
            continue
        
        years_avail = sorted(codes.keys())
        if not years_avail:
            continue
        
        actual_end = end_date
        relevant_snapshots = sorted([y for y in NLCD_YEARS if y in years_avail])
        
        if not relevant_snapshots:
            continue
        
        current_start = start_date
        boundaries = [pd.Timestamp(year=y, month=1, day=1) for y in relevant_snapshots]
        
        for i in range(len(relevant_snapshots) + 1):
            if i == 0:
                snap_year = relevant_snapshots[0]
                period_start = current_start
                period_end = boundaries[0] if boundaries else actual_end
            elif i < len(relevant_snapshots):
                snap_year = relevant_snapshots[i - 1]
                period_start = boundaries[i - 1]
                period_end = boundaries[i]
            else:
                snap_year = relevant_snapshots[-1]
                period_start = boundaries[-1]
                period_end = actual_end
            
            land_use_code = codes.get(snap_year)
            land_use_label = _code_to_label(land_use_code) or "Unknown"
            
            segment_end = min(period_end, actual_end)
            
            if segment_end > period_start and period_start < actual_end:
                segment_start = max(period_start, current_start)
                if segment_end > segment_start:
                    detailed_rows.append({
                        "site_code": site,
                        "status": status,
                        "lat": lat,
                        "long": long,
                        "segment_start": segment_start.strftime("%Y-%m-%d"),
                        "segment_end": segment_end.strftime("%Y-%m-%d"),
                        "land_use_class": land_use_label,
                        "nlcd_code": land_use_code,
                        "nlcd_snapshot_year": snap_year,
                    })
                    current_start = segment_end
            
            if segment_end >= actual_end:
                break
    
    detailed_df = pd.DataFrame(detailed_rows)
    
    # Export both CSVs
    summary_path = output_dir / "summary.csv"
    detailed_path = output_dir / "detailed.csv"
    
    try:
        summary_df.to_csv(summary_path, index=False, encoding="utf-8")
        detailed_df.to_csv(detailed_path, index=False, encoding="utf-8")
        print(f"Summary CSV exported -> {summary_path}")
        print(f"  ({len(summary_df)} sites)")
        print(f"Detailed CSV exported -> {detailed_path}")
        print(f"  ({len(detailed_df)} land-use segments)")
    except PermissionError:
        print(f"Warning: Could not write CSV files (they may be open in another program)")
        print(f"  Summary: {len(summary_df)} sites")
        print(f"  Detailed: {len(detailed_df)} land-use segments")
    
    # Print summary of sites with land-use changes
    changed_sites = summary_df[summary_df["land_use_changed"] == True]
    if len(changed_sites) > 0:
        print(f"\nSites with land-use changes: {len(changed_sites)}")
        print(f"  Sites: {', '.join(changed_sites['site_code'].tolist())}")
    else:
        print(f"\nNo sites with land-use changes detected")


def build_full_page(
    timeline_html: str,
    timeline_height: int,
    map_html: str,
    map_bounds: dict[str, float] | None,
) -> str:
    """HTML wrapper with timeline above map."""
    map_bounds_js = json.dumps(map_bounds) if map_bounds else "null"
    map_padding = MAP_PADDING
    map_tile_size = MAP_TILE_SIZE
    map_margin_js = json.dumps(MAP_MARGIN)
    legend_reflow_px = LEGEND_REFLOW_PX
    legend_reflow_height = LEGEND_REFLOW_HEIGHT

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>CAP LTER – Arthropod Timeline & Map</title>
  <script src="https://cdn.plot.ly/plotly-{PLOTLYJS_VERSION}.min.js"></script>
  <style>
    :root {{
      --accent: #0f766e;
      --accent-soft: #ccfbf1;
      --bg: #f5f5f5;
      --card-bg: #ffffff;
      --border-subtle: #e5e7eb;
      --text-main: #111827;
      --text-muted: #6b7280;
    }}
    * {{
      box-sizing: border-box;
    }}
    body {{
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 0;
      padding: 24px 32px 48px;
      background:
        radial-gradient(circle at top left, #e0f2fe 0, #f5f5f5 45%, #f5f5f5 100%);
      color: var(--text-main);
    }}
    .page {{
      max-width: min(1600px, 100%);
      margin: 0 auto;
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      padding: 4px 11px;
      border-radius: 999px;
      font-size: 0.75rem;
      font-weight: 600;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      background: var(--accent-soft);
      color: var(--accent);
      margin-bottom: 8px;
    }}
    h1 {{
      margin: 4px 0 6px;
      font-size: 1.9rem;
    }}
    .subtitle {{
      margin-top: 0;
      margin-bottom: 24px;
      color: var(--text-muted);
      font-size: 0.95rem;
      max-width: 900px;
    }}
    .section-label {{
      text-transform: uppercase;
      font-size: 0.78rem;
      letter-spacing: 0.12em;
      color: var(--text-muted);
      margin: 22px 4px 6px;
    }}
    .card {{
      background: var(--card-bg);
      border-radius: 18px;
      padding: 18px 20px 18px;
      box-shadow: 0 18px 40px rgba(15, 23, 42, 0.08);
      margin-bottom: 26px;
      border: 1px solid rgba(148, 163, 184, 0.16);
    }}
    .card h2 {{
      margin: 0 0 4px;
      font-size: 1.15rem;
    }}
    .card-sub {{
      margin: 0 0 10px;
      color: var(--text-muted);
      font-size: 0.85rem;
    }}
    .chip-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-bottom: 10px;
    }}
    .chip {{
      display: inline-flex;
      align-items: center;
      padding: 3px 9px;
      border-radius: 999px;
      font-size: 0.78rem;
      border: 1px dashed var(--border-subtle);
      color: var(--text-muted);
      background: #f9fafb;
      white-space: nowrap;
    }}
    .chip strong {{
      font-weight: 600;
      color: var(--accent);
    }}
    /* Both plots are responsive, so Plotly sizes them to these containers. */
    #timeline-container {{
      width: 100%;
      /* one row per site, so the height is set by the data, not the viewport */
      height: {timeline_height}px;
    }}
    #map-container {{
      width: 100%;
      height: clamp(340px, 58vh, 760px);
    }}
    #timeline-plot {{
      width: 100%;
    }}
    #map-plot {{
      width: 100%;
      height: 100%;
    }}
    @media (max-width: 900px) {{
      body {{
        padding: 16px;
      }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <h1>CAP LTER Arthropod Sites (Sampling and Land Use)</h1>
    <p class="subtitle">
    </p>

    <div class="section-label">Timeline</div>
    <div class="card">
      <h2>Sampling history by land-use change</h2>
      <p class="card-sub" style="margin: 0 0 10px 0;">Arthropod Sampling Sites (Timeline by land-use change)</p>
      <div id="timeline-container">
        {timeline_html}
      </div>
    </div>

    <div class="section-label">Map</div>
    <div class="card">
      <h2>Site locations by land-use context</h2>
      <p class="card-sub" style="margin: 0 0 10px 0;">Arthropod Sampling Sites (Land-Use context with latest NLCD)</p>
      <p class="card-sub">
        Points are colored by the most recent NLCD land-use class at each site location.
        Hover to see site name and sampling dates. Some sites lie within a few hundred
        metres of each other and their markers merge until you zoom in.
      </p>
      <div id="map-container">
        {map_html}
      </div>
    </div>
  </div>
  <script>
    // Refit the map to the size it is actually drawn at. A zoom level baked in
    // at build time only suits the viewport it was computed for and crops sites
    // on any smaller one, so the fit is redone here and on every resize.
    (function () {{
      var BOUNDS = {map_bounds_js};
      var PADDING = {map_padding};
      var TILE = {map_tile_size};
      var MARGIN = {map_margin_js};
      if (!BOUNDS) return;

      function mercatorY(lat) {{
        var r = lat * Math.PI / 180;
        return (1 - Math.log(Math.tan(r) + 1 / Math.cos(r)) / Math.PI) / 2;
      }}
      function mercatorLat(y) {{
        return Math.atan(Math.sinh(Math.PI * (1 - 2 * y))) * 180 / Math.PI;
      }}

      var userInteracted = false;

      function applyFit() {{
        var gd = document.getElementById('map-plot');
        if (userInteracted || !gd || !gd.data || typeof Plotly === 'undefined') return;

        // On a narrow screen a legend sitting on the map covers most of it, so
        // lay it out below the map instead and give it room in the margin.
        var narrow = gd.clientWidth < {legend_reflow_px};
        var marginB = narrow ? {legend_reflow_height} : MARGIN.b;

        var width = gd.clientWidth - MARGIN.l - MARGIN.r;
        var height = gd.clientHeight - MARGIN.t - marginB;
        if (width <= 0 || height <= 0) return;

        var yNorth = mercatorY(BOUNDS.latMax);
        var ySouth = mercatorY(BOUNDS.latMin);
        var spanX = Math.max((BOUNDS.lonMax - BOUNDS.lonMin) / 360, 1e-9) * PADDING;
        var spanY = Math.max(ySouth - yNorth, 1e-9) * PADDING;

        Plotly.relayout(gd, {{
          'map.center': {{
            lat: mercatorLat((yNorth + ySouth) / 2),
            lon: (BOUNDS.lonMin + BOUNDS.lonMax) / 2
          }},
          'map.zoom': Math.min(
            Math.log2(width / (TILE * spanX)),
            Math.log2(height / (TILE * spanY))
          ),
          'margin.b': marginB,
          'legend.orientation': narrow ? 'h' : 'v',
          'legend.x': narrow ? 0 : 0.995,
          'legend.xanchor': narrow ? 'left' : 'right',
          'legend.y': narrow ? -0.02 : 0.98,
          'legend.bgcolor': narrow ? 'rgba(255,255,255,0)' : 'rgba(255,255,255,0.82)',
          'legend.bordercolor': narrow ? 'rgba(0,0,0,0)' : 'rgba(200,200,200,0.35)',
          'legend.font.size': narrow ? 9 : 10,
          'legend.entrywidth': narrow ? 0 : null
        }});
      }}

      // MapLibre refuses camera changes until its style has finished loading,
      // and there is no public event for that, so the fit is reapplied a few
      // times on a decaying schedule. Each call is idempotent.
      function refitRepeatedly() {{
        [400, 1200, 2500, 5000].forEach(function (delay) {{
          setTimeout(applyFit, delay);
        }});
      }}

      var pending;
      function scheduleRefit() {{
        clearTimeout(pending);
        pending = setTimeout(applyFit, 200);
      }}

      refitRepeatedly();
      window.addEventListener('resize', scheduleRefit);

      // Once the reader has panned or zoomed, leave their view alone
      var container = document.getElementById('map-container');
      ['mousedown', 'wheel', 'touchstart'].forEach(function (evt) {{
        container.addEventListener(evt, function () {{
          userInteracted = true;
          window.removeEventListener('resize', scheduleRefit);
          clearTimeout(pending);
        }}, {{ once: true, passive: true }});
      }});
    }})();
  </script>
</body>
</html>
"""
    return page


def main():
    logging.basicConfig(level=logging.INFO)

    root = Path(__file__).parent
    data_dir = root / "data"
    build_dir = root / "build"
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
    df, codes_by_index = enrich_with_land_use(df)

    # Export enriched data to CSV files
    export_enriched_data(df, codes_by_index, build_dir)

    # Build HTML snippets
    timeline_html, timeline_height = build_segmented_timeline_html(df, codes_by_index)
    map_html, map_bounds = build_map_html(df)

    # Build full page
    full_page = build_full_page(timeline_html, timeline_height, map_html, map_bounds)

    out_path = build_dir / "index.html"
    out_path.write_text(full_page, encoding="utf-8")
    print(f"Timeline & map page built -> {out_path}")

if __name__ == "__main__":
    main()