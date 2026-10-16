"""Plot GPS recordings and export a 50 cm coverage grid."""

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import gpxpy
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from geographiclib.geodesic import Geodesic

################
STEP = 0.5      
RADIUS = 10.0   
################

# Opens a CSV and yields its rows as (timestamp, latitude, longitude).
def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            if not row["timestamp"]:
                raise ValueError(f"Missing timestamp in {path}")
            yield datetime.fromisoformat(row["timestamp"].strip().replace("Z", "+00:00")), float(row["latitude"]), float(row["longitude"])

# Parses a GPX file and yields its track points as (timestamp, latitude, longitude).
def read_gpx(path):
    with path.open(encoding="utf-8") as file:
        gpx = gpxpy.parse(file)
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                if point.time is None:
                    raise ValueError(f"Missing timestamp in {path}")
                yield point.time, point.latitude, point.longitude

# Load gpx and csv files in folder, remove exact duplicates, sort by timestamp, return (recording_name, timestamp, latitude, longitude)
def load_points(folder):
    files = sorted((*folder.glob("*.csv"), *folder.glob("*.gpx")))
    if not files:
        raise ValueError(f"No CSV or GPX recordings found in {folder}")

    points, seen = [], set()
    for path in files:
        try:
            rows = read_csv(path) if path.suffix.lower() == ".csv" else read_gpx(path)
            for timestamp, lat, lon in rows:
                key_time = timestamp.astimezone(timezone.utc) if timestamp.tzinfo else timestamp
                key = (path.stem, key_time, lat, lon)
                if key not in seen:
                    points.append((path.stem, timestamp, lat, lon))
                    seen.add(key)
        except Exception as exc:
            raise ValueError(f"Could not read {path.name}: {exc}") from exc

    if not points:
        raise ValueError("No recorded points found")
    aware = {timestamp.tzinfo is not None for _, timestamp, _, _ in points}
    if len(aware) > 1:
        raise ValueError("Mixing timezone-aware and timezone-naive timestamps!")
    points.sort(key=lambda point: point[1].astimezone(timezone.utc) if point[1].tzinfo else point[1])
    return points


def local_xy(lat_lon_points):
    _, _, lat0, lon0 = lat_lon_points[0]
    xy_points = []
    for source, timestamp, lat, lon in lat_lon_points:
        result = Geodesic.WGS84.Inverse(lat0, lon0, lat, lon)
        azimuth = math.radians(result["azi1"])
        distance = result["s12"]
        xy_points.append((source, timestamp, distance * math.sin(azimuth), distance * math.cos(azimuth)))
    return xy_points


def save_origin_metadata(path, points):
    source, timestamp, latitude, longitude = points[0]
    metadata = {
        "origin": {
            "latitude": latitude,
            "longitude": longitude,
            "recording": source,
            "timestamp": timestamp.isoformat(),
        },
        "coordinate_frame": "Local east/north metres from the origin, using WGS84 geodesic distance and azimuth",
    }
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def iter_segments(xy):
    """Yield nonzero adjacent path segments, ordered by time within each recording."""
    tracks = {}
    for source, timestamp, x, y in xy:
        tracks.setdefault(source, []).append((timestamp, x, y))

    for source, points in tracks.items():
        points.sort(key=lambda point: point[0])
        for first, second in zip(points, points[1:]):
            _, x1, y1 = first
            _, x2, y2 = second
            if x1 != x2 or y1 != y2:
                yield source, (x1, y1), (x2, y2)


def segment_covers_cell(start, end, cell_x, cell_y):
    """Return whether a cell center is in a 90-degree cone swept along a segment."""
    x1, y1 = start
    x2, y2 = end
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    if length == 0:
        return False

    unit_x, unit_y = dx / length, dy / length
    offset_x, offset_y = cell_x - x1, cell_y - y1
    along = offset_x * unit_x + offset_y * unit_y
    across = abs(-offset_x * unit_y + offset_y * unit_x)

    if across > RADIUS:
        return False

    # Along the path, the cone apex can range from start to end. At any
    # apex, the cell must be ahead, within 45 degrees, and within RADIUS.
    minimum_along = max(0.0, along - length, across)
    maximum_along = min(along, math.sqrt(max(0.0, RADIUS**2 - across**2)))
    return minimum_along <= maximum_along


def save_grid(path, xy):
    xmin = math.floor((min(x for _, _, x, _ in xy) - RADIUS) / STEP)
    xmax = math.ceil((max(x for _, _, x, _ in xy) + RADIUS) / STEP)
    ymin = math.floor((min(y for _, _, _, y in xy) - RADIUS) / STEP)
    ymax = math.ceil((max(y for _, _, _, y in xy) + RADIUS) / STEP)

    covered = set()
    for _, start, end in iter_segments(xy):
        x1, y1 = start
        x2, y2 = end
        gx_min = math.floor((min(x1, x2) - RADIUS) / STEP)
        gx_max = math.ceil((max(x1, x2) + RADIUS) / STEP)
        gy_min = math.floor((min(y1, y2) - RADIUS) / STEP)
        gy_max = math.ceil((max(y1, y2) + RADIUS) / STEP)

        for gx in range(gx_min, gx_max + 1):
            for gy in range(gy_min, gy_max + 1):
                if segment_covers_cell(start, end, gx * STEP, gy * STEP):
                    covered.add((gx, gy))

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("east_m", "north_m", "covered"))
        for gx in range(xmin, xmax + 1):
            for gy in range(ymin, ymax + 1):
                writer.writerow((f"{gx * STEP:.2f}", f"{gy * STEP:.2f}", int((gx, gy) in covered)))


def save_plot(path, xy):
    fig, ax = plt.subplots()
    tracks = {}
    for source, timestamp, x, y in xy:
        tracks.setdefault(source, []).append((timestamp, x, y))
    cmap = plt.get_cmap("Dark2")
    count = len(tracks)
    for index, (name, track_points) in enumerate(sorted(tracks.items())):
        color = cmap(index / max(count - 1, 1))
        track_points.sort(key=lambda point: point[0])
        points = [(x, y) for _, x, y in track_points]
        xs, ys = zip(*points)
        ax.plot(xs, ys, color=color, linewidth=1, label=name, zorder=1)
        ax.scatter(xs, ys, s=12, color=color, zorder=2)

        for _, start, end in iter_segments(
            [(name, timestamp, x, y) for timestamp, x, y in track_points]
        ):
            x1, y1 = start
            x2, y2 = end
            dx, dy = x2 - x1, y2 - y1
            length = math.hypot(dx, dy)
            unit_x, unit_y = dx / length, dy / length
            normal_x, normal_y = -unit_y, unit_x
            half_width = RADIUS / math.sqrt(2)

            # Draw the rear V, the circular front cap, and both flanks.
            local_boundary = [(half_width, half_width)]
            for angle in range(45, -46, -5):
                radians = math.radians(angle)
                local_boundary.append((
                    length + RADIUS * math.cos(radians),
                    RADIUS * math.sin(radians),
                ))
            local_boundary.extend(((half_width, -half_width), (0, 0)))
            boundary = [
                (
                    x1 + along * unit_x + across * normal_x,
                    y1 + along * unit_y + across * normal_y,
                )
                for along, across in local_boundary
            ]
            ax.add_patch(Polygon(boundary, closed=True, facecolor=color, edgecolor="none", alpha=0.2))
    ax.set(xlabel="East (m)", ylabel="North (m)", title="Recording coverage")
    ax.legend(title="Recording", loc="best")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--plot", type=Path)
    parser.add_argument("--grid", type=Path)
    args = parser.parse_args()
    folder = args.folder
    output_folder = Path(__file__).resolve().parent.parent / "outputs" / folder.name
    plot_path = args.plot or output_folder / "recordings_plot.png"
    grid_path = args.grid or output_folder / "coverage_grid.csv"
    origin_path = output_folder / "coordinate_origin.json"
    sources = (*folder.glob("*.csv"), *folder.glob("*.gpx"))
    if plot_path.resolve() == grid_path.resolve():
        parser.error("Plot and grid outputs must use different paths")
    if any(path.resolve() == plot_path.resolve() for path in sources):
        parser.error("Plot output would overwrite an input recording")
    try:
        points = load_points(folder)
        xy = local_xy(points)
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        grid_path.parent.mkdir(parents=True, exist_ok=True)
        save_plot(plot_path, xy)
        save_grid(grid_path, xy)
        save_origin_metadata(origin_path, points)
    except (OSError, ValueError, csv.Error) as exc:
        parser.error(str(exc))
    print(f"Plotted {len(xy)} points: {plot_path}; {grid_path}; {origin_path}")


if __name__ == "__main__":
    main()
