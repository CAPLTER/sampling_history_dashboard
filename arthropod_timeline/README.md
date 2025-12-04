# CAP LTER Arthropod Timeline Visualization

This project creates an interactive timeline and map visualization for CAP LTER arthropod sampling sites, showing their sampling history and land-use changes over time.

# Overview

The visualization consists of two main components:
1. **Timeline Chart**: A horizontal bar chart showing the land-use history for each sampling site, segmented by NLCD (National Land Cover Database) snapshot years
2. **Interactive Map**: A map showing the geographic locations of all sites, colored by their latest NLCD land-use classification

## Features

**Dynamic Timeline**: Automatically adjusts to show all sites in the dataset
**Land-Use Segmentation**: Timeline bars are segmented by NLCD snapshot years (2001, 2006, 2011, 2016, 2019)
**Interactive Elements**: Hover tooltips show detailed information about each site and time period
 **Visual Indicators**: 
  - Vertical dotted lines mark NLCD snapshot boundaries
  - Horizontal grey lines every 5 sites for improved readability
  - Color-coded bars representing different land-use classes
**Responsive Design**: Works well on different screen sizes

## Project Structure

```
arthropod_timeline/
├── build_timeline.py      # Main script that generates the visualization
├── data/
│   └── arthros_temporal.csv  # Input data file (site codes, dates, coordinates)
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

1. Ensure your data file is in the correct location: `data/arthros_temporal.csv`
2. Run the script:
```bash
python build_timeline.py
```

3. The generated visualization will be saved to `build/index.html`
4. Open `build/index.html` in a web browser to view the visualization

### Input Data Format

The input CSV file (`data/arthros_temporal.csv`) must contain the following columns:

- `site_code`: Unique identifier for each site (e.g., "AA-17", "AB-19")
- `start_date`: Start date of sampling (format: YYYY-MM-DD)
- `end_date`: End date of sampling (format: YYYY-MM-DD, can be empty for active sites)
- `lat`: Latitude coordinate (decimal degrees)
- `long`: Longitude coordinate (decimal degrees)

### Output Files

The script generates three output files in the `build/` directory:

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

#Data Processing Pipeline

1. **Data Loading**: Reads the temporal CSV file and parses dates
2. **Status Determination**: Sites with missing `end_date` are marked as "Active"
3. **NLCD Enrichment**: For each site, retrieves NLCD land-use data for all snapshot years
4. **Timeline Segmentation**: Creates segments for each period between NLCD snapshots
5. **Visualization Generation**: Creates interactive Plotly charts
6. **Export**: Generates HTML page and CSV exports

# NLCD Land-Use Data

The script uses the `pygeohydro` library to retrieve NLCD (National Land Cover Database) land-use classifications for each site location. NLCD provides land cover data at multiple snapshot years: 2001, 2006, 2011, 2016, 2019 (Latest)

For each site, the script:
- Retrieves land-use data for all available snapshot years
- Determines which snapshot applies to each time period
- Segments the timeline when land-use changes between snapshots

# Updating Data
# Adding New Sites or Updating Existing Data

1. **Update the CSV file**: Edit `data/arthros_temporal.csv` with new or updated site information
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