"""Plot GPS recordings and export a 50 cm coverage grid."""

import argparse
import csv
from datetime import datetime, timezone
import math
from pathlib import Path

import gpxpy
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from geographiclib.geodesic import Geodesic

################
STEP = 0.5      
RADIUS = 10.0   
################

def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            yield datetime.fromisoformat(row["timestamp"].strip().replace("Z", "+00:00")), float(row["latitude"]), float(row["longitude"])


def read_gpx(path):
    with path.open(encoding="utf-8") as file:
        gpx = gpxpy.parse(file)
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                if point.time is None:
                    raise ValueError(f"Missing timestamp in {path}")
                yield point.time, point.latitude, point.longitude


def load_points(folder, grid_path):
    files = sorted((*folder.glob("*.csv"), *folder.glob("*.gpx")))
    same_path = [path for path in files if path.resolve() == grid_path.resolve()]
    if same_path:
        with same_path[0].open(newline="", encoding="utf-8-sig") as file:
            if next(csv.reader(file), []) != ["east_m", "north_m", "covered"]:
                raise ValueError(f"Grid output would overwrite input recording: {same_path[0]}")
    files = [path for path in files if path.resolve() != grid_path.resolve()]
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
        raise ValueError("Cannot mix timezone-aware and timezone-naive timestamps")
    points.sort(key=lambda point: point[1].astimezone(timezone.utc) if point[1].tzinfo else point[1])
    return points


def local_xy(points):
    _, _, lat0, lon0 = points[0]
    xy = []
    for source, _, lat, lon in points:
        result = Geodesic.WGS84.Inverse(lat0, lon0, lat, lon)
        azimuth = math.radians(result["azi1"])
        distance = result["s12"]
        xy.append((source, distance * math.sin(azimuth), distance * math.cos(azimuth)))
    return xy


def save_grid(path, xy):
    xmin = math.floor((min(x for _, x, _ in xy) - RADIUS) / STEP)
    xmax = math.ceil((max(x for _, x, _ in xy) + RADIUS) / STEP)
    ymin = math.floor((min(y for _, _, y in xy) - RADIUS) / STEP)
    ymax = math.ceil((max(y for _, _, y in xy) + RADIUS) / STEP)
    covered = set()
    reach = math.ceil(RADIUS / STEP)
    for _, x, y in xy:
        ix, iy = round(x / STEP), round(y / STEP)
        for gx in range(ix - reach, ix + reach + 1):
            for gy in range(iy - reach, iy + reach + 1):
                cx, cy = gx * STEP, gy * STEP
                if (cx - x) ** 2 + (cy - y) ** 2 <= RADIUS ** 2:
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
    for source, x, y in xy:
        tracks.setdefault(source, []).append((x, y))
    cmap = plt.get_cmap("turbo")
    count = len(tracks)
    for index, (name, points) in enumerate(sorted(tracks.items())):
        color = cmap(index / max(count - 1, 1))
        xs, ys = zip(*points)
        ax.plot(xs, ys, color=color, linewidth=1, label=name, zorder=1)
        for x, y in points:
            ax.add_patch(Circle((x, y), RADIUS, fill=False, edgecolor=color, alpha=0.3))
        ax.scatter(xs, ys, s=12, color=color, zorder=2)
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
    sources = (*folder.glob("*.csv"), *folder.glob("*.gpx"))
    if plot_path.resolve() == grid_path.resolve():
        parser.error("Plot and grid outputs must use different paths")
    if any(path.resolve() == plot_path.resolve() for path in sources):
        parser.error("Plot output would overwrite an input recording")
    try:
        points = load_points(folder, grid_path)
        xy = local_xy(points)
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        grid_path.parent.mkdir(parents=True, exist_ok=True)
        save_plot(plot_path, xy)
        save_grid(grid_path, xy)
    except (OSError, ValueError, csv.Error) as exc:
        parser.error(str(exc))
    print(f"Plotted {len(xy)} points: {plot_path}; {grid_path}")


if __name__ == "__main__":
    main()
