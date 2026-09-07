# CAP LTER Bird Timeline Visualization

This project creates an interactive timeline and map visualization for CAP
LTER bird survey sites, showing their survey history and land-use changes
over time. It is the bird counterpart to
[../arthropod_timeline/](../arthropod_timeline/); both are built from the
same shared code in [../common/pipeline.py](../common/pipeline.py) -- see the
top-level [README.md](../README.md) for how the pieces fit together.

# Overview

The visualization consists of two main components:
1. **Timeline Chart**: A horizontal bar chart showing the land-use history for
   each survey site, segmented by NLCD (National Land Cover Database) snapshot
   years, and grouped by `location_type` (the survey protocol a site was run
   under)
2. **Interactive Map**: A map showing the geographic locations of all sites,
   colored by their latest NLCD land-use classification

## Project Structure

```
sampling_history_dashboard/
├── common/
│   ├── pipeline.py          # Shared code (dataset-agnostic)
│   └── nlcd_lookup.json     # Committed NLCD cache, shared across datasets
├── build/
│   ├── birds.html           # This dataset's page (deployed to Pages)
│   └── index.html           # The arthropod dataset's page
└── bird_timeline/
    ├── build_birds.py        # This dataset's loader + driver
    ├── data/                 # Raw study files (see "Input Data")
    │   ├── 46_bird_surveys.csv
    │   └── 46_bird_survey_locations.csv
    ├── output/               # Generated CSVs -- not served on the website
    │   ├── summary.csv       # One row per site with summary information
    │   └── detailed.csv      # One row per land-use segment showing all changes
    ├── requirements.txt      # Python dependencies
    └── README.md
```

## Installation

### Prerequisites

- Python 3.8 or higher
- Internet connection (only needed the first time a site's coordinate is
  looked up -- see [Performance Notes](#performance-notes))

### Setup

```bash
pip install -r requirements.txt
```

## Usage

### Running the Script

1. Ensure the study files are in `data/` (see [Input Data](#input-data))
2. Run the script from this directory:
```bash
python build_birds.py
```
3. The generated visualization is saved to `../build/birds.html` (a directory
   shared with the arthropod dataset); the CSVs are saved to `output/`
4. Open `../build/birds.html` in a web browser to view the visualization

### Input Data

The dashboard is built directly from the raw study files in `data/`. Nothing
in `build_birds.py` names a specific file, so the directory is scanned by
shape rather than by name and additional bird studies can simply be dropped
in.

**Survey CSVs** — any `.csv` carrying both a `site_code` and a `survey_date`
column. One row per visit to a site; if a `location_type` column is present,
one value is taken per site (and any site recorded under more than one
`location_type` is logged as a warning). Unlike the arthropod dataset there is
no `flags` column to filter by -- every row is a real, conducted survey, so a
site's span is simply its first to last `survey_date`.

**Location CSVs** — any `.csv` carrying `site_code`, `lat`, and `long`
columns. A site can have more than one location row if it was physically
moved during the study; see [Moved locations](#moved-locations) below for how
that is resolved.

Because every boundary comes from the data, rebuilding unchanged inputs
reproduces byte-identical output — the build does not depend on the date it
is run.

#### Moved locations

15 of the 106 sites in `46_bird_survey_locations.csv` carry more than one
location row: the site was physically relocated during the study, and each
row's `begin_date`/`end_date` (or, where those are blank, the paired
`begin_date_year`/`begin_date_month` and `end_date_year`/`end_date_month`
columns) records when that particular point was occupied. The **most
recently established** row is used for each site -- the one with no
`end_date` if one exists, else the row with the latest `begin_date`, then the
latest `end_date`. This was chosen over "the row valid at the site's last
survey" because it agrees with that alternative on 105 of 106 sites and, for
the one site where the two disagree, is the rule that avoids two different
sites resolving to the same coordinate.

### Output Files

The script generates three output files, split across two directories, on the
same terms as the arthropod dataset: `../build/birds.html` is the only one
served on the website; `output/summary.csv` and `output/detailed.csv` publish
the underlying numbers for anyone working with the repository directly, and
are not read when the page is built.

1. **birds.html**: Interactive visualization page with timeline and map
2. **summary.csv**: Summary data with one row per site, including
   `location_type`, survey dates, status, coordinates, and NLCD codes/labels
   for each snapshot year
3. **detailed.csv**: Detailed segment data with one row per land-use segment

See the arthropod README's [Data Processing
Pipeline](../arthropod_timeline/README.md#data-processing-pipeline) section
for how these are assembled -- the process is identical, since both datasets
share the same code.

### Grouping by location_type

The survey sites were run under six different protocols (`ESCA`, `PASS`,
`SRBP`, `riparian`, `desert_fertilization`, `NDV`), recorded in the
`location_type` column. The timeline orders sites by protocol first, then
alphabetically by `site_code` within a protocol, with a solid divider line and
a rotated label marking each group -- see `build_segmented_timeline_html()`'s
`group_by` parameter in `../common/pipeline.py`. The arthropod dataset has no
such grouping, so it passes no `group_by` and keeps its original layout.

# NLCD Land-Use Data

Land-use enrichment works identically to the arthropod dataset: see
[Updating the NLCD Cache](../arthropod_timeline/README.md#updating-the-nlcd-cache)
in the arthropod README. The cache is shared, keyed by coordinate rather than
by site, precisely so that both datasets benefit from each other's lookups
without any risk of one dataset's `site_code` being confused for the other's.

# Technical Details
# Dependencies

- **pandas** (>=2.0.0): Data manipulation and CSV handling
- **plotly** (>=5.24.0): Interactive visualization library (needs `px.scatter_map`)
- **pygeohydro** (>=0.14.0): NLCD data retrieval from USGS

# Performance Notes

- **First run for a new site**: Needs internet access, to query the MRLC
  service once for that coordinate
- **Every run after that**: No internet dependency at all -- every coordinate
  already looked up is served from the committed `../common/nlcd_lookup.json`
- **Site Filtering**: Sites without valid coordinates are automatically skipped
