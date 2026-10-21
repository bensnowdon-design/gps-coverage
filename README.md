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
python3 scripts/plot_recordings.py data/recordings
```

By default, outputs go to `outputs/<input-folder-name>/`. For example, the command above writes `outputs/recordings/recordings_plot.png` and `outputs/recordings/coverage_grid.csv`. Use `--plot` or `--grid` to override either output path.

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

Prototype a prize-collecting route through enclosed uncovered regions:

```sh
python3 scripts/plan_coverage_route.py \
  outputs/13_10_2026_evening/coverage_grid.csv \
  data/13_10_2026_evening
```

The planner writes `planned_route.csv` and `planned_route.png` beside the grid. Route CSV coordinates use the same local east/north metre frame as the grid, and the `action` column identifies exploration points, connectors, and points that follow a recorded track. The main tuning options are:

| Option | Default | Meaning |
| --- | ---: | --- |
| `--coverage` | `balanced` | `light`, `balanced`, or `thorough`: how strongly new coverage is favored over distance |
| `--route-length` | `2000` | Maximum total route length, in metres |
| `--leg-length` | `20` | Maximum length of a direct exploration leg, in metres |
| `--target-spacing` | `5` | Spacing between sampled exploration targets, in metres |
| `--track-use` | `frequent` | `occasional`, `regular`, or `frequent`: trades track section length against how many sections are allowed |
| `--output`, `--plot` | beside grid | Override route CSV and diagnostic PNG paths |

`--track-use occasional` allows up to 5 track sections of 20 m each; `regular` allows up to 15 sections of 15 m; `frequent` allows up to 30 sections of 10 m. The planner still includes the connector when scoring a track section. Enclosed regions are uncovered grid components inside the convex hull of covered cells. The route starts at the recorded point nearest any target. This prototype does not write GPX.

Detailed options such as grid spacing, sensor radius, random seed, scoring weights, and stop limits remain available as advanced CLI overrides, but are hidden from the short `--help` list. `--turn-cost` defaults to `0.75`: a 180-degree turn subtracts 75% of that candidate's coverage and track reward and adds 7.5 times the normal distance penalty, strongly favoring a short step after a sharp turn. The added distance penalty falls with the square of the turn angle. `--crossing-cost` defaults to `0.2`, subtracting 20% of a candidate's reward for each interior crossing of the earlier route. Increase either value to discourage the behavior more strongly; set it to `0` to disable that penalty. For example, `--sensor-radius 8` overrides the default sensor model. Grid spacing must match the input CSV.

The conversion scripts accept an optional `--output-dir`; if omitted, they write beside their input files.
