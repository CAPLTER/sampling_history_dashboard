# CAP LTER Arthropod Timeline Visualization

This project creates an interactive timeline and map visualization for CAP LTER arthropod sampling sites, showing their sampling history and land-use changes over time.

# Overview

The visualization consists of two main components:
1. **Timeline Chart**: A horizontal bar chart showing the land-use history for
   each sampling site, segmented by NLCD (National Land Cover Database)
snapshot years
2. **Interactive Map**: A map showing the geographic locations of all sites,
   colored by their latest NLCD land-use classification

## Features

**Dynamic Timeline**: Automatically adjusts to show all sites in the dataset
**Land-Use Segmentation**: Timeline bars are segmented by NLCD snapshot years
(2001, 2006, 2011, 2016, 2019)
**Interactive Elements**: Hover tooltips show detailed information about each
site and time period
 **Visual Indicators**: 
  - Vertical dotted lines mark NLCD snapshot boundaries
  - Horizontal grey lines every 5 sites for improved readability
  - Color-coded bars representing different land-use classes
**Responsive Design**: Works well on different screen sizes

## Project Structure

```
arthropod_timeline/
├── build_timeline.py      # Main script that generates the visualization
├── data/                  # Raw study files (see "Input Data")
│   ├── 41_core_arthropods.csv
│   ├── 41_core_arthropods_locations.geojson
│   ├── 643_mcdowell_pitfall_arthropods.csv
│   └── 643_mcdowell_arthropod_locations.geojson
├── build/
│   ├── index.html         # Generated HTML visualization page
│   ├── summary.csv        # One row per site with summary information
│   └── detailed.csv       # One row per land-use segment showing all changes
├── cache/                 # Cached NLCD data (created automatically)
├── requirements.txt       # Python dependencies
└── README.md             
```

## Installation

### Prerequisites

- Python 3.8 or higher
- Internet connection (for downloading NLCD data)

### Setup

1. Install the required Python packages:
```bash
pip install -r requirements.txt
```
Or install individually:
```bash
pip install pandas plotly pygeohydro
```

## Usage

### Running the Script

1. Ensure the study files are in `data/` (see [Input Data](#input-data))
2. Run the script:
```bash
python build_timeline.py
```

3. The generated visualization will be saved to `build/index.html`
4. Open `build/index.html` in a web browser to view the visualization

### Input Data

The dashboard is built directly from the raw study files in `data/`. Nothing in
`build_timeline.py` names a specific file, so the directory is scanned by shape
rather than by name and additional studies can simply be dropped in.

**Sampling CSVs** — any `.csv` carrying both a `site_code` and a `sample_date`
column. One row per organism record; the other columns of the CAP LTER exports
(`count`, `flags`, `trap_name`, and so on) may be present but only `flags` is
used. Files without those two columns are ignored, and the script logs which
ones it skipped.

**Location GeoJSONs** — any `.geojson` whose features carry a `site_code`
property. Each site is placed at the centroid of its polygon; `Point` and
`MultiPolygon` geometries are also accepted.

A site's sampling span runs from its first to its last sampling event. A date
counts as a sampling event when a trap was actually collected: empty traps are
kept, because an empty trap is a real event with a real result, while records
flagged `trap_not_collected`, `miscoded`, or `empty_sampling_event` are
excluded. See `EXCLUDED_FLAGS` in `build_timeline.py`.

Because every boundary comes from the data, rebuilding unchanged inputs
reproduces byte-identical output — the build does not depend on the date it is
run.

#### Paired locations

`643_mcdowell_arthropod_locations.geojson` is the one exception to the rule
above. It carries a single polygon per *pair* of sites, keyed by a composite
`sampling_location` name (`Bell_Gateway`, `DixieMine_Prospector`, ...), and each
polygon is only the bounding box of its two sites, so per-site positions cannot
be recovered from it. Those ten coordinates are therefore kept in
`MCDOWELL_SITE_COORDS`, and the build re-checks each one against the polygon
that covers it, warning if either drifts. Note that `DixieMine` in a polygon
name is the site coded `Mine` in the sampling CSV; that pairing lives in
`PAIRED_POLYGON_MEMBERS`.

### Output Files

The script generates three output files in the `build/` directory. All three are
**independent products of the same in-memory table** — none is an input to
another. In particular, the CSVs are *not* read when the visualization is built:
`index.html` is rendered straight from the enriched site table, so deleting both
CSVs would leave the page byte-identical. They exist to publish the underlying
numbers, and the deploy workflow uploads the whole `build/` directory, so both
are downloadable from the live site alongside the page.

1. **index.html**: Interactive visualization page with timeline and map
2. **summary.csv**: Summary data with one row per site, including:
   - Site information (code, dates, status, coordinates)
   - Land-use at start, end, and latest snapshot
   - Whether land-use changed over time
   - NLCD codes and labels for each snapshot year
3. **detailed.csv**: Detailed segment data with one row per land-use segment, including:
   - Site code and status
   - Segment start and end dates
   - Land-use class for that segment
   - NLCD code and snapshot year used

## Data Processing Pipeline

The build is a short linear stage that assembles one enriched table, followed by
three outputs generated independently from it:

```
data/  (sampling CSVs + location GeoJSONs)
   |
   +-> build_site_table()       one row per site: span + coordinates
   +-> enrich_with_land_use()   adds NLCD codes and land-use labels
                |
                |  (the enriched table, held in memory)
                |
    +-----------+-----------------------+
    |           |                       |
    v           v                       v
export_       build_segmented_    build_map_html()
enriched_     timeline_html()
data()              |                   |
    |               +--------+----------+
    v                        v
summary.csv             index.html
detailed.csv
```

1. **Data Loading**: Scans `data/` for sampling CSVs and location GeoJSONs, and
   reduces the sampling records to a first and last sampling date per site
2. **Status Determination**: Sites whose last sampling event falls within a year
   of the most recent event anywhere in the data are labelled "Sampled to study
   end"; the rest are "Retired early"
3. **NLCD Enrichment**: For each site, retrieves NLCD land-use data for all snapshot years
4. **Outputs**: The timeline, the map, and the two CSVs are each derived from
   the enriched table. `export_enriched_data()` happens to run first, but nothing
   after it reads what it wrote

### Two segmentations, on purpose

Because the timeline and `detailed.csv` are derived separately, the script
contains two implementations of the "cut a site's span at NLCD boundaries" idea,
and they deliberately differ:

- `build_segmented_timeline_html()` **merges** adjacent periods that share a
  land-use class, so a bar is drawn per visible change
- the `detailed_rows` loop in `export_enriched_data()` emits **one row per NLCD
  snapshot period**, whether or not the class changed

That is why the current data yields 211 rows in `detailed.csv` but only 75 bars
in the timeline: 67 sites, most of whose land-use never changed, collapse to one
bar each. The extra granularity in the CSV is intended, but note that a change to
how spans are cut has to be made in both places — nothing cross-checks them.

# NLCD Land-Use Data

The script uses the `pygeohydro` library to retrieve NLCD (National Land Cover Database) land-use classifications for each site location. NLCD provides land cover data at multiple snapshot years: 2001, 2006, 2011, 2016, 2019 (Latest)

For each site, the script:
- Retrieves land-use data for all available snapshot years
- Determines which snapshot applies to each time period
- Segments the timeline when land-use changes between snapshots

# Updating Data
# Adding New Sites or Updating Existing Data

1. **Add or replace the study files**: Drop the updated sampling CSV and its
   location GeoJSON into `data/`. A new study needs no code change, provided its
   CSV has `site_code` and `sample_date` columns and its GeoJSON features carry
   a `site_code` property; the build reports any sampled site it could not
   place, and any location with no sampling records
2. **Run the script**: Execute `python build_timeline.py`
3. **View results**: Open `build/index.html` to see the updated visualization

The script automatically:
- Detects all sites in the data
- Calculates appropriate height for the timeline
- Updates the map with all site locations
- Generates new summary and detailed CSV files

# Adding New NLCD Snapshot Years

If new NLCD snapshots are released (e.g., 2024, 2027), update the `NLCD_YEARS` constant in `build_timeline.py`:

```python
NLCD_YEARS = [2001, 2006, 2011, 2016, 2019, 2024]  # Add new years here
```
# Configuration
# Adjusting Timeline Appearance

Key parameters in `build_timeline.py` that can be adjusted:

- **Bar thickness**: Modify `calculated_height = max(600, num_sites * 14)` - change the multiplier (14) to adjust bar thickness
- **Font sizes**: Adjust `tickfont=dict(size=8)` for y-axis labels
- **Margins**: Modify `margin=dict(l=140, r=40, t=100, b=50)` for spacing
- **Colors**: Update `LANDUSE_COLORS` dictionary to change color scheme

# Technical Details
# Dependencies

- **pandas** (>=2.0.0): Data manipulation and CSV handling
- **plotly** (>=5.0.0): Interactive visualization library
- **pygeohydro** (>=0.14.0): NLCD data retrieval from USGS

# Performance Notes

- **First Run**: May take longer as NLCD data is downloaded and cached
- **Subsequent Runs**: Faster due to caching (data stored in `cache/` directory)
- **Internet Required**: Script needs internet access to download NLCD data
- **Site Filtering**: Sites without valid coordinates are automatically skipped
