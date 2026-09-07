# sampling_history_dashboard

Repository to house code and resources for a dashboard of the sampling
histories of CAP LTER long-term monitoring programs.

The site publishes one page per monitoring program, sharing a nav bar:

- **Arthropods** — `build/index.html`, built from [arthropod_timeline/](arthropod_timeline/)
- **Birds** — `build/birds.html`, built from [bird_timeline/](bird_timeline/)

## Structure

```
sampling_history_dashboard/
├── common/                  # Shared tools -- code used by every dataset
│   ├── pipeline.py           # Dataset-agnostic build code
│   └── nlcd_lookup.json      # Committed NLCD cache, shared across datasets
├── build/                   # Deploy artifact -- HTML pages only, no data
│   ├── index.html
│   └── birds.html
├── arthropod_timeline/       # Arthropod-specific: loader + raw data + CSV output
└── bird_timeline/            # Bird-specific: loader + raw data + CSV output
```

Each dataset directory has its own README with details specific to that
dataset (input file formats, status labels, configuration). This file covers
what's shared.

### Why a shared `common/pipeline.py`

Every dataset needs the same things done to it once it's in a standard shape
(one row per site: `site_code`, `lat`, `long`, `start_date`, `end_date`,
`status`): NLCD enrichment, a segmented timeline, a map, CSV export, and the
page shell. `common/pipeline.py` holds all of that. Each dataset directory holds only
the small, genuinely different part -- how to turn *that* dataset's raw files
into the standard site table -- plus a short driver that calls into
`common/pipeline.py`.

Adding a third monitoring program means writing a new loader next to
`arthropod_timeline/` and `bird_timeline/`, not touching either of the
existing ones.

### The NLCD cache

NLCD land-use lookups are backed by `nlcd_lookup.json`, keyed by coordinate
(rounded to ~0.1 m) rather than by site, since the same `site_code` can denote
a different physical location in a different dataset -- confirmed true of the
arthropod and bird site tables, which are otherwise entirely unrelated
programs and must never be joined or deduplicated by `site_code`.

A coordinate already cached for every year in `common.pipeline.NLCD_YEARS` is never
re-queried, so an ordinary rebuild has no dependency on the NLCD service (the
MRLC WMS endpoint this queries has been observed to fail outright or return
incomplete data). Anything not yet cached is queried once; if the service is
unavailable for a coordinate not already cached, the build raises rather than
publishing a page with missing land-use. Commit the updated
`nlcd_lookup.json` whenever a rebuild adds new sites or a new NLCD year.

### Outputs and what gets deployed

`build/` is exactly what the GitHub Actions workflow uploads to Pages, so it
holds only the HTML pages -- no CSVs. Each dataset's `output/summary.csv` and
`output/detailed.csv` publish the same underlying numbers for anyone working
with the repository directly, but are not served on the website.

### Adding a new dataset

1. Create `<name>_timeline/data/` with the raw study files.
2. Write `<name>_timeline/build_<name>.py`: a loader that returns a standard
   site table (see any existing dataset's `build_site_table()`), plus a
   `main()` that calls `pipeline.enrich_with_land_use()`,
   `pipeline.export_enriched_data()`, `pipeline.build_segmented_timeline_html()`,
   `pipeline.build_map_html()`, and `pipeline.build_full_page()`, writing CSVs
   to `<name>_timeline/output/` and the page to `build/<name>.html`.
3. Add `("<name>.html", "<Label>")` to `pipeline.NAV_LINKS` so every page's
   nav bar picks it up.
4. Add the new build script to `.github/workflows/pages.yml`.
