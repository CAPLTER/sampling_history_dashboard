import logging
import warnings
from pathlib import Path

import pandas as pd
import plotly.express as px
from pygeohydro import nlcd as nlcd_mod

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*default value for compat will change.*",
)

# NLCD snapshot years we will use
NLCD_YEARS = [2001, 2006, 2011, 2016, 2019]

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

    coords = list(zip(df_valid["long"], df_valid["lat"]))  # (lon, lat)

    try:
        nlcd_gdf = nlcd_mod.nlcd_bycoords(
            coords,
            years={"cover": NLCD_YEARS},
            region="L48",
        )

        today_year = pd.Timestamp.today().year

        # Map from df index -> {year: code}
        for idx, row in zip(df_valid.index, nlcd_gdf.itertuples()):
            year_to_code = {}
            for year in NLCD_YEARS:
                colname = f"cover_{year}"
                if hasattr(row, colname):
                    year_to_code[year] = getattr(row, colname)
            codes_by_index[idx] = year_to_code

            if not year_to_code:
                continue

            years_avail = sorted(year_to_code.keys())

            start_date = df.at[idx, "start_date"]
            end_date = df.at[idx, "end_date"]

            start_year = int(start_date.year) if pd.notna(start_date) else None
            end_year = int(end_date.year) if pd.notna(end_date) else today_year

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

    except Exception as e:
        logging.error(f"Failed to retrieve NLCD land-use data: {e}")

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
        end_date = row["end_date"]  # Actual end date (may be NaN for active sites)
        end_for_plot = row["end_for_plot"]  # end_date or today for active sites
 #validation check (skips sites with no NLCD data)
        if codes is None or not codes:
            continue
        if pd.isna(start_date) or pd.isna(end_for_plot):
            continue
        years_avail = sorted(codes.keys())
        if not years_avail:
            continue

        relevant_snapshots = sorted([y for y in NLCD_YEARS if y in years_avail])
        if not relevant_snapshots:
            continue

        # Show ALL land-use changes during site's active period
        boundaries = [pd.Timestamp(year=y, month=1, day=1) for y in relevant_snapshots]
        site_end_date = end_date if pd.notna(end_date) else end_for_plot

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
        return "<!-- empty timeline -->"

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

    return fig.to_html(full_html=False, include_plotlyjs=False, div_id="timeline-plot")

def build_map_html(df: pd.DataFrame) -> str:
    df = df.copy()
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["long"] = pd.to_numeric(df["long"], errors="coerce")
    df = df.dropna(subset=["lat", "long"])

    if df.empty:
        return "<!-- empty map -->"

 # calculate center of map
    center_lat = df["lat"].mean()
    center_lon = df["long"].mean()

    df["land_use"] = df["land_use"].fillna("Unknown")

    df["start_date_formatted"] = df["start_date"].dt.strftime("%Y-%m-%d")
    df["end_date_formatted"] = df["end_date"].dt.strftime("%Y-%m-%d").fillna("Active")

    fig = px.scatter_mapbox(
        df,
        lat="lat",
        lon="long",
        hover_name="site_code",
        color="land_use",
        color_discrete_map=LANDUSE_COLORS,
        zoom=9,
        center={"lat": center_lat, "lon": center_lon},
        height=460,
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
        marker=dict(size=8, opacity=0.9),
        hovertemplate=(
            "<b>%{hovertext}</b><br><br>"
            "land_use=%{customdata[0]}<br>"
            "status=%{customdata[1]}<br>"
            "start_date=%{customdata[2]}<br>"
            "end_date=%{customdata[3]}<extra></extra>"
        ),
    )

    fig.update_layout(
        mapbox_style="carto-positron",
        margin=dict(l=0, r=0, t=40, b=0),
        legend_title_text="Land-use class (latest snapshot)",
        font=dict(
            family="system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
            size=11,
        ),
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
    )

    return fig.to_html(full_html=False, include_plotlyjs=False, div_id="map-plot")

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
        end_for_plot = row["end_for_plot"]
        lat = row.get("lat", pd.NA)
        long = row.get("long", pd.NA)
        
        if codes is None or not codes:
            continue
        if pd.isna(start_date) or pd.isna(end_for_plot):
            continue
        
        years_avail = sorted(codes.keys())
        if not years_avail:
            continue
        
        actual_end = end_date if pd.notna(end_date) else end_for_plot
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


def build_full_page(timeline_html: str, map_html: str) -> str:
    """HTML wrapper with timeline above map."""
    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>CAP LTER – Arthropod Timeline & Map</title>
  <script src="https://cdn.plot.ly/plotly-3.0.0.min.js"></script>
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
      max-width: 1100px;
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
    #timeline-container {{
      width: 100%;
      min-height: 600px;
      /* Allow container to grow with plot height, scrolling if needed */
    }}
    #map-container {{
      width: 100%;
      height: 460px;
    }}
    @media (max-width: 900px) {{
      body {{
        padding: 16px;
      }}
      #timeline-container {{
        height: 620px;
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
        Hover to see site name, sampling dates, and whether sampling is still active
      </p>
      <div id="map-container">
        {map_html}
      </div>
    </div>
  </div>
</body>
</html>
"""
    return page


def main():
    logging.basicConfig(level=logging.INFO)

    root = Path(__file__).parent
    data_path = root / "data" / "arthros_temporal.csv"
    build_dir = root / "build"
    build_dir.mkdir(exist_ok=True)

    df = pd.read_csv(
        data_path,
        parse_dates=["start_date", "end_date"],
    )

    # Status: Active if end_date is missing
    df["status"] = df["end_date"].isna().map({True: "Active", False: "Ended"})

    # For plotting, replace missing end_date with today
    today = pd.Timestamp.today().normalize()
    df["end_for_plot"] = df["end_date"].fillna(today)

    # Add NLCD land-use info + codes_by_index
    df, codes_by_index = enrich_with_land_use(df)

    # Export enriched data to CSV files
    export_enriched_data(df, codes_by_index, build_dir)

    # Build HTML snippets
    timeline_html = build_segmented_timeline_html(df, codes_by_index)
    map_html = build_map_html(df)

    # Build full page
    full_page = build_full_page(timeline_html, map_html)

    out_path = build_dir / "index.html"
    out_path.write_text(full_page, encoding="utf-8")
    print(f"Timeline & map page built -> {out_path}")

if __name__ == "__main__":
    main()