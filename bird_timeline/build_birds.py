"""
Bird-dataset loader and driver.

Everything dataset-agnostic (NLCD enrichment, the timeline, the map, CSV
export, the page shell) lives in the shared ../pipeline.py. This file only
knows how to turn the raw bird study files in data/ into the standard site
table pipeline.py expects, plus the small amount of config (title, captions)
specific to this dataset.

Unlike the arthropod sampling CSVs, there is no `flags` column here to filter
by -- every row in the surveys file is a real, conducted survey, so a site's
span is simply its first to last survey_date.
"""

import datetime as dt
import logging
import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
import pipeline

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*default value for compat will change.*",
)

# Files are matched by column shape, not by name, so an additional bird study
# file can be dropped into data/ the same way an additional arthropod study
# can: a "survey" table (one row per visit) needs site_code + survey_date; a
# "location" table (one row per site, or per site per period it occupied a
# given point) needs site_code + lat + long.
SURVEY_REQUIRED_COLUMNS = {"site_code", "survey_date"}
LOCATION_REQUIRED_COLUMNS = {"site_code", "lat", "long"}

STUDY_END_WINDOW = pd.Timedelta(days=365)
STATUS_TO_STUDY_END = "Sampled to study end"
STATUS_RETIRED_EARLY = "Retired early"


def _composite_date(row: pd.Series, kind: str) -> str | None:
    """
    Effective begin/end date for one location row.

    Most rows carry a full `<kind>_date`. Where that is blank, the location
    file instead carries `<kind>_date_year`/`<kind>_date_month` (populated
    for exactly the rows the full date is missing for), so this falls back to
    the first of the month for a begin date, or the last day of the month for
    an end date. Returns None if neither is available.
    """
    full = row.get(f"{kind}_date")
    if pd.notna(full):
        return str(full)

    year = row.get(f"{kind}_date_year")
    if pd.isna(year):
        return None
    year = int(year)

    month_val = row.get(f"{kind}_date_month")
    month = int(month_val) if pd.notna(month_val) else (1 if kind == "begin" else 12)

    if kind == "begin":
        return f"{year:04d}-{month:02d}-01"

    next_month = dt.date(year + (month == 12), (month % 12) + 1, 1)
    return str(next_month - dt.timedelta(days=1))


def load_bird_locations(data_dir: Path) -> dict[str, tuple[float, float]]:
    """
    Read every location CSV in data_dir and return {site_code: (lat, lon)}.

    A site can have more than one location row when it was physically moved
    over the study; the most recently established row is used (an
    open-ended row -- no end_date -- outranks any closed one, then the row
    with the latest begin date, then the latest end date).
    """
    resolved: dict[str, tuple[float, float]] = {}

    for path in sorted(data_dir.glob("*.csv")):
        columns = set(pd.read_csv(path, nrows=0).columns)
        if not LOCATION_REQUIRED_COLUMNS.issubset(columns):
            logging.info("Skipping %s: not a location table", path.name)
            continue

        frame = pd.read_csv(path)
        logging.info("Read %d location row(s) from %s", len(frame), path.name)

        for site, rows in frame.groupby("site_code"):
            candidates = []
            for _, r in rows.iterrows():
                begin = _composite_date(r, "begin")
                end = _composite_date(r, "end")
                candidates.append((begin, end, float(r["lat"]), float(r["long"])))
            candidates.sort(
                key=lambda c: (c[1] is None, c[0] or "", c[1] or ""), reverse=True
            )
            _, _, lat, lon = candidates[0]
            resolved[site] = (lat, lon)

    return resolved


def load_bird_surveys(data_dir: Path) -> pd.DataFrame:
    """
    Read every survey CSV in data_dir and return one row per site giving the
    first and last date it was surveyed, plus its location_type if the survey
    table carries one.
    """
    frames = []

    for path in sorted(data_dir.glob("*.csv")):
        columns = set(pd.read_csv(path, nrows=0).columns)
        if not SURVEY_REQUIRED_COLUMNS.issubset(columns):
            logging.info("Skipping %s: not a survey table", path.name)
            continue

        usecols = sorted(SURVEY_REQUIRED_COLUMNS | (columns & {"location_type"}))
        frame = pd.read_csv(path, usecols=usecols, parse_dates=["survey_date"])
        logging.info("Read %d survey record(s) from %s", len(frame), path.name)
        frames.append(frame)

    if not frames:
        raise FileNotFoundError(
            f"No survey CSVs found in {data_dir}; expected files with "
            f"{sorted(SURVEY_REQUIRED_COLUMNS)} columns"
        )

    records = pd.concat(frames, ignore_index=True).dropna(subset=["survey_date"])
    spans = (
        records.groupby("site_code")["survey_date"]
        .agg(start_date="min", end_date="max")
    )

    if "location_type" in records.columns:
        distinct_per_site = records.groupby("site_code")["location_type"].nunique()
        inconsistent = sorted(distinct_per_site[distinct_per_site > 1].index)
        if inconsistent:
            logging.warning(
                "Site(s) recorded under more than one location_type: %s",
                ", ".join(inconsistent),
            )
        spans["location_type"] = records.groupby("site_code")["location_type"].first()

    return spans.reset_index()


def build_site_table(data_dir: Path) -> pd.DataFrame:
    """
    Assemble one row per site: its survey span (+ location_type) from the
    survey CSVs, and its position from the location CSVs.
    """
    df = load_bird_surveys(data_dir)
    locations = load_bird_locations(data_dir)

    df["lat"] = df["site_code"].map(lambda s: locations.get(s, (None, None))[0])
    df["long"] = df["site_code"].map(lambda s: locations.get(s, (None, None))[1])

    unplaced = sorted(df.loc[df["lat"].isna(), "site_code"])
    if unplaced:
        logging.warning(
            "No coordinates for %d surveyed site(s): %s",
            len(unplaced),
            ", ".join(unplaced),
        )

    unsurveyed = sorted(set(locations) - set(df["site_code"]))
    if unsurveyed:
        logging.info(
            "%d located site(s) have no survey records: %s",
            len(unsurveyed),
            ", ".join(unsurveyed),
        )

    df = pipeline.assign_study_end_status(
        df, STUDY_END_WINDOW, STATUS_TO_STUDY_END, STATUS_RETIRED_EARLY
    )
    pipeline.warn_duplicate_coordinates(df)

    sort_cols = ["location_type", "site_code"] if "location_type" in df.columns else ["site_code"]
    return df.sort_values(sort_cols).reset_index(drop=True)


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
        "Site table: %d sites surveyed %s to %s",
        len(df),
        df["start_date"].min().date(),
        df["end_date"].max().date(),
    )

    # Add NLCD land-use info + codes_by_index
    df, codes_by_index = pipeline.enrich_with_land_use(df)

    # Export enriched data to CSV files
    pipeline.export_enriched_data(df, codes_by_index, output_dir)

    # Build HTML snippets. Grouped by location_type: the six survey protocols
    # (ESCA, PASS, SRBP, riparian, desert_fertilization, NDV) are different
    # kinds of site, and reading the timeline by protocol first, site second,
    # makes that visible instead of interleaving them alphabetically.
    timeline_html, timeline_height = pipeline.build_segmented_timeline_html(
        df, codes_by_index, group_by="location_type"
    )
    map_html, map_bounds = pipeline.build_map_html(df)

    # Build full page
    full_page = pipeline.build_full_page(
        timeline_html,
        timeline_height,
        map_html,
        map_bounds,
        page_title="CAP LTER – Bird Timeline & Map",
        heading="CAP LTER Bird Survey Sites (Sampling and Land Use)",
        timeline_card_heading="Survey history by land-use change",
        timeline_card_caption=(
            "Bird Survey Sites (Timeline by land-use change, grouped by location type)"
        ),
        map_card_heading="Site locations by land-use context",
        map_card_caption="Bird Survey Sites (Land-Use context with latest NLCD)",
        map_note=(
            "Points are colored by the most recent NLCD land-use class at each site location. "
            "Hover to see site name and survey dates. Some sites lie close to each other and "
            "their markers merge until you zoom in."
        ),
        nav_html=pipeline.build_nav_html(active="birds.html"),
    )

    out_path = build_dir / "birds.html"
    out_path.write_text(full_page, encoding="utf-8")
    print(f"Bird timeline & map page built -> {out_path}")


if __name__ == "__main__":
    main()
