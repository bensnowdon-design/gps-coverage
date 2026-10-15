# GPS Coverage Project

Python tools for converting GPS files and plotting recording coverage.

## Project layout

- `scripts/` — conversion and plotting programs.
- `data/recordings/` — original GPX recordings.
- `outputs/` — saved coverage grid and plot images.
- `requirements.txt` — Python dependencies.

## Setup

From the project root, install dependencies with:

```sh
python3 -m pip install -r requirements.txt
```

## Common commands

Generate a recording plot and coverage grid:

```sh
python3 scripts/plot_recordings.py data/recordings \
  --plot outputs/recordings_plot.png \
  --grid outputs/coverage_grid.csv
```

Render the grid CSV as an image:

```sh
python3 scripts/plot_coverage_grid.py outputs/coverage_grid.csv \
  --output outputs/coverage_grid.png
```

Convert recordings between GPX and CSV while keeping generated files under `outputs/`:

```sh
python3 scripts/gpx_to_csv.py data/recordings --output-dir outputs/csv
python3 scripts/csv_to_gpx.py outputs/csv --output-dir outputs/gpx
```

The conversion scripts accept an optional `--output-dir`; if omitted, they write beside their input files.
