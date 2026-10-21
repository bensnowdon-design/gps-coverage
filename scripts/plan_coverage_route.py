"""Plan a prize-collecting route through enclosed uncovered grid regions."""

import argparse
import csv
import math
from pathlib import Path

import gpxpy
import gpxpy.gpx
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.colors import ListedColormap
import numpy as np
from scipy import ndimage
from geographiclib.geodesic import Geodesic
from shapely.geometry import LineString, Point, box
from shapely.ops import unary_union


def read_grid(path, step):
    with path.open(newline="", encoding="utf-8-sig") as file:
        if next(csv.reader(file), []) != ["east_m", "north_m", "covered"]:
            raise ValueError("Expected columns east_m,north_m,covered")
    data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    if data.shape[1] != 3 or not np.isfinite(data).all() or not np.isin(data[:, 2], (0, 1)).all():
        raise ValueError("Grid contains invalid values")
    east, north = np.unique(data[:, 0]), np.unique(data[:, 1])
    if len(data) != len(east) * len(north):
        raise ValueError("Grid must contain exactly one value per cell")
    if (len(east) > 1 and not np.allclose(np.diff(east), step)) or (len(north) > 1 and not np.allclose(np.diff(north), step)):
        raise ValueError(f"Grid cells must be spaced {step:g} m apart")
    cols, rows = np.searchsorted(east, data[:, 0]), np.searchsorted(north, data[:, 1])
    indexes = rows * len(east) + cols
    if np.unique(indexes).size != len(data):
        raise ValueError("Grid contains duplicate cells")
    grid = np.empty((len(north), len(east)), dtype=bool)
    grid.ravel()[indexes] = data[:, 2].astype(bool)
    return east, north, grid


def track_lines(folder, origin, max_points=10000):
    lines = []
    for path in sorted(Path(folder).glob("*.gpx")):
        with path.open(encoding="utf-8") as file:
            gpx = gpxpy.parse(file)
        for track in gpx.tracks:
            for segment in track.segments:
                points = []
                source = [(p.time, p.latitude, p.longitude) for p in segment.points if p.time]
                source.sort(key=lambda item: item[0])
                for _, lat, lon in source:
                    result = Geodesic.WGS84.Inverse(*origin, lat, lon)
                    azimuth, distance = math.radians(result["azi1"]), result["s12"]
                    points.append((distance * math.sin(azimuth), distance * math.cos(azimuth)))
                if len(points) > max_points:
                    indexes = np.linspace(0, len(points) - 1, max_points).astype(int)
                    points = [points[i] for i in indexes]
                if len(points) >= 2:
                    lines.append(points)
    return lines


def track_chunks(tracks, max_length):
    """Return overlapping short track sections, preserving each recording's order."""
    chunks = []
    stride = max(max_length / 2, 1.0)
    for track_id, track in enumerate(tracks):
        cumulative = [0.0]
        for first, second in zip(track, track[1:]):
            cumulative.append(cumulative[-1] + math.dist(first, second))
        total = cumulative[-1]
        if total <= 0:
            continue

        def point_at(distance):
            if distance >= total:
                return track[-1]
            index = max(0, int(np.searchsorted(cumulative, distance, side="right") - 1))
            segment_length = cumulative[index + 1] - cumulative[index]
            fraction = (distance - cumulative[index]) / segment_length if segment_length else 0
            first, second = track[index], track[index + 1]
            return (first[0] + fraction * (second[0] - first[0]),
                    first[1] + fraction * (second[1] - first[1]))

        start = 0.0
        while start < total:
            end = min(start + max_length, total)
            if end - start >= min(1.0, max_length * 0.25):
                points = [point_at(start)]
                points.extend(track[i] for i in range(1, len(track) - 1)
                              if start < cumulative[i] < end)
                points.append(point_at(end))
                chunks.append((track_id, start, end, points))
            start += stride
    return chunks


def enclosed_regions(east, north, grid, step, min_area):
    covered = [box(x - step / 2, y - step / 2, x + step / 2, y + step / 2)
               for y, x in zip(*np.where(grid))]
    covered_centers = [Point(east[col], north[row]) for row, col in zip(*np.where(grid))]
    if not covered:
        raise ValueError("Coverage grid has no covered cells")
    hull = unary_union(covered_centers).convex_hull
    labels, count = ndimage.label(~grid, structure=ndimage.generate_binary_structure(2, 1))
    regions = []
    for label in range(1, count + 1):
        rows, cols = np.where(labels == label)
        coords = [(east[col], north[row]) for row, col in zip(rows, cols)]
        if not coords or not all(hull.covers(Point(x, y)) for x, y in coords):
            continue
        area = len(coords) * step * step
        if area < min_area:
            continue
        cells = [box(x - step / 2, y - step / 2, x + step / 2, y + step / 2)
                 for x, y in coords]
        regions.append(unary_union(cells))
    return hull, regions


def poisson_points(regions, spacing, seed, attempts):
    rng = np.random.default_rng(seed)
    points = []
    for region in regions:
        minx, miny, maxx, maxy = region.bounds
        accepted = []
        for _ in range(attempts):
            candidate = (rng.uniform(minx, maxx), rng.uniform(miny, maxy))
            if region.contains(Point(candidate)) and all(math.dist(candidate, p) >= spacing for p in accepted):
                accepted.append(candidate)
        points.extend(accepted)
    return points


def segment_covered_cells(start, end, xs, ys, step, radius):
    x1, y1 = start
    x2, y2 = end
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    if length < 1e-9:
        return set()
    ux, uy = dx / length, dy / length
    cells = set()
    c0 = max(0, int(np.searchsorted(xs, min(x1, x2) - radius - step)))
    c1 = min(len(xs), int(np.searchsorted(xs, max(x1, x2) + radius + step, side="right")))
    r0 = max(0, int(np.searchsorted(ys, min(y1, y2) - radius - step)))
    r1 = min(len(ys), int(np.searchsorted(ys, max(y1, y2) + radius + step, side="right")))
    for row in range(r0, r1):
        y = ys[row]
        for col in range(c0, c1):
            x = xs[col]
            ox, oy = x - x1, y - y1
            along = ox * ux + oy * uy
            across = abs(-ox * uy + oy * ux)
            if across > radius:
                continue
            low = max(0.0, along - length, across)
            high = min(along, math.sqrt(max(0, radius * radius - across * across)))
            if low <= high:
                cells.add((row, col))
    return cells


def path_length(points):
    return sum(math.dist(first, second) for first, second in zip(points, points[1:]))


def path_covered_cells(points, xs, ys, step, radius):
    cells = set()
    for first, second in zip(points, points[1:]):
        cells.update(segment_covered_cells(first, second, xs, ys, step, radius))
    return cells


def turn_penalty(route, points, reward, travel, distance_cost, turn_cost):
    """Penalize sharp turns and make long legs after them increasingly costly."""
    if len(route) < 2 or turn_cost <= 0:
        return 0.0
    previous = route[-2]
    current = route[-1]
    incoming = (current[0] - previous[0], current[1] - previous[1])
    outgoing = next(((point[0] - current[0], point[1] - current[1])
                     for point in points[1:]
                     if math.dist(current, point) > 1e-9), None)
    if outgoing is None:
        return 0.0
    dot = incoming[0] * outgoing[0] + incoming[1] * outgoing[1]
    cross = incoming[0] * outgoing[1] - incoming[1] * outgoing[0]
    severity = abs(math.atan2(cross, dot)) / math.pi
    reward_penalty = reward * turn_cost * severity
    long_leg_penalty = distance_cost * travel * 10 * turn_cost * severity**2
    return reward_penalty + long_leg_penalty


def crossing_count(route, points):
    """Count interior intersections with earlier, non-adjacent route segments."""
    if len(route) < 3:
        return 0
    previous_segments = [LineString((route[i], route[i + 1]))
                         for i in range(len(route) - 2)]
    crossings = set()
    for start, end in zip(points, points[1:]):
        candidate = LineString((start, end))
        for previous in previous_segments:
            intersection = candidate.intersection(previous)
            if intersection.is_empty:
                continue
            intersections = ([intersection] if intersection.geom_type == "Point" else
                             list(intersection.geoms) if hasattr(intersection, "geoms") else [])
            for point in intersections:
                if point.geom_type != "Point":
                    continue
                if point.distance(Point(start)) <= 0.05 or point.distance(Point(end)) <= 0.05:
                    continue
                crossings.add((round(point.x, 2), round(point.y, 2)))
    return len(crossings)


def plan_route(targets, initial, xs, ys, original, step, radius, coverage_reward,
               track_reward, chunks, track_join_distance, distance_cost,
               max_distance, max_stops, max_track_segments, max_leg_length,
               turn_cost=0.75, crossing_cost=0.2):
    route = [initial]
    actions = []
    remaining = list(targets)
    covered = set()
    used_chunks = set()
    used_track_intervals = []
    distance = 0.0
    track_count = 0
    while len(actions) < max_stops:
        current = route[-1]
        choices = []
        for target in remaining:
            points = [current, target]
            travel = path_length(points)
            if travel > max_leg_length or distance + travel > max_distance:
                continue
            gained = path_covered_cells(points, xs, ys, step, radius) - original - covered
            reward = coverage_reward * len(gained)
            score = (reward - distance_cost * travel
                     - turn_penalty(route, points, reward, travel, distance_cost, turn_cost)
                     - crossing_cost * reward * crossing_count(route, points))
            if score > 0:
                choices.append((score, "target", points, gained, travel, target, None))

        if track_count < max_track_segments and track_reward > 0:
            for chunk_id, (track_id, start, end, chunk) in enumerate(chunks):
                if chunk_id in used_chunks or math.dist(current, chunk[0]) > track_join_distance:
                    continue
                if any(track_id == used_track and max(start, used_start) < min(end, used_end)
                       for used_track, used_start, used_end in used_track_intervals):
                    continue
                points = [current, *chunk]
                travel = path_length(points)
                if distance + travel > max_distance:
                    continue
                gained = path_covered_cells(points, xs, ys, step, radius) - original - covered
                bonus = track_reward
                reward = coverage_reward * len(gained) + bonus
                score = (reward - distance_cost * travel
                         - turn_penalty(route, points, reward, travel, distance_cost, turn_cost)
                         - crossing_cost * reward * crossing_count(route, points))
                if score > 0:
                    choices.append((score, "track", points, gained, travel,
                                    (track_id, start, end), chunk_id))
        if not choices:
            break
        _, kind, points, gained, travel, destination, chunk_id = max(choices, key=lambda item: item[0])
        if kind == "target":
            remaining.remove(destination)
        else:
            track_id, start, end = destination
            used_chunks.add(chunk_id)
            used_track_intervals.append((track_id, start, end))
            track_count += 1
        actions.append((kind, points))
        route.extend(point for point in points[1:] if math.dist(route[-1], point) > 0.05)
        covered.update(gained)
        distance += travel
    return route, actions, covered, distance


def write_route_gpx(path, route, origin):
    """Export ordered route points from local east/north metres to WGS84."""
    gpx = gpxpy.gpx.GPX()
    gpx.creator = "GPS Coverage route planner"
    planned_route = gpxpy.gpx.GPXRoute(name="Coverage route")
    gpx.routes.append(planned_route)
    for index, (east, north) in enumerate(route):
        position = Geodesic.WGS84.Direct(
            *origin, math.degrees(math.atan2(east, north)), math.hypot(east, north)
        )
        name = "Start" if index == 0 else "End" if index == len(route) - 1 else f"Point {index}"
        planned_route.points.append(gpxpy.gpx.GPXRoutePoint(
            latitude=position["lat2"], longitude=position["lon2"], name=name,
        ))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(gpx.to_xml(), encoding="utf-8")


def plot(path, east, north, grid, regions, targets, route, route_actions, tracks):
    background, text = "#FAFAF7", "#303B40"
    teal, plum = "#087F8C", "#754668"
    route_halo = [path_effects.Stroke(linewidth=4, foreground="white"),
                  path_effects.Normal()]
    fig, ax = plt.subplots(figsize=(10, 8), facecolor=background)
    ax.set_facecolor(background)
    ax.imshow(grid, origin="lower", interpolation="nearest",
              extent=(east[0] - .25, east[-1] + .25, north[0] - .25, north[-1] + .25),
              cmap=ListedColormap([background, "#DFE3DF"]), vmin=0, vmax=1,
              zorder=0)
    for region in regions:
        x, y = region.exterior.xy
        ax.fill(x, y, color="#F3E7C7", linewidth=0, zorder=1)
    for track in tracks:
        ax.plot(*zip(*track), color="#8C9EAD", alpha=.65, linewidth=.8, zorder=2)
    if targets:
        ax.scatter(*zip(*targets), s=6, color="#BC7865", alpha=.65,
                   edgecolors="none", zorder=3, label="Sampled targets")
    track_labelled = exploration_labelled = join_labelled = False
    for kind, points in route_actions:
        if kind == "target":
            ax.plot(*zip(*points), "o-", color=teal, linewidth=2,
                    markersize=4, markeredgewidth=0, path_effects=route_halo, zorder=4,
                    label="Exploration leg" if not exploration_labelled else None)
            exploration_labelled = True
        else:
            if len(points) > 1 and math.dist(points[0], points[1]) > 0.05:
                ax.plot(*zip(*points[:2]), "--", color=teal, linewidth=1.5,
                        path_effects=route_halo, zorder=4,
                        label="Join to original path" if not join_labelled else None)
                join_labelled = True
            if len(points) > 1:
                ax.plot(*zip(*points[1:]), "-", color=plum, linewidth=2.5,
                        path_effects=route_halo, zorder=4,
                        label="Forward track segment" if not track_labelled else None)
                track_labelled = True
    if route:
        ax.scatter(*route[0], s=90, marker="o", color=teal, edgecolor="white",
                   linewidth=1.5, zorder=5, label="Start")
        ax.annotate("Start", route[0], xytext=(6, 7), textcoords="offset points",
                    color=teal, weight="bold", zorder=7)
        ax.scatter(*route[-1], s=110, marker="X", color=plum, edgecolor="white",
                   linewidth=1.2, zorder=6, label="End")
        ax.annotate("End", route[-1], xytext=(6, 7), textcoords="offset points",
                    color=plum, weight="bold", zorder=7)
    ax.set(xlabel="East (m)", ylabel="North (m)", title="Prize-collecting coverage route")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_axisbelow(True)
    ax.grid(color="#303B40", alpha=.07, linewidth=.7)
    ax.tick_params(colors=text)
    for label in (ax.xaxis.label, ax.yaxis.label, ax.title):
        label.set_color(text)
    for spine in ax.spines.values():
        spine.set_color("#C8CECA")
        spine.set_linewidth(.8)
    ax.legend(loc="best", facecolor=background, edgecolor="#DFE3DF",
              labelcolor=text, framealpha=.95)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("grid", type=Path)
    parser.add_argument("tracks", type=Path, help="Folder of original GPX recordings")
    parser.add_argument("--output", type=Path, help="Route GPX (default: beside grid)")
    parser.add_argument("--plot", type=Path, help="Diagnostic PNG (default: beside route GPX)")

    parser.add_argument("--coverage", choices=("light", "balanced", "thorough"),
                        default="balanced", help="How strongly to favor new coverage")
    parser.add_argument("--route-length", type=float, default=2000.0,
                        help="Maximum planned route distance in metres")
    parser.add_argument("--leg-length", type=float, default=30.0,
                        help="Maximum direct exploration leg in metres")
    parser.add_argument("--target-spacing", type=float, default=2.0,
                        help="Spacing between sampled exploration targets in metres")
    parser.add_argument("--track-use", choices=("occasional", "regular", "frequent"),
                        default="frequent", help="How often and how long to follow recorded tracks")

    # Advanced overrides keep detailed tuning available without cluttering --help.
    advanced = {
        "grid-step": (float, 0.5), "sensor-radius": (float, 10.0),
        "seed": (int, 42), "sampling-attempts": (int, 5000),
        "min-region-area": (float, 5.0), "coverage-reward": (float, None),
        "track-reward": (float, 50.0), "track-max-length": (float, None),
        "track-join-distance": (float, 10.0), "max-track-segments": (int, None),
        "distance-cost": (float, None), "max-distance": (float, None),
        "max-stops": (int, 50), "max-leg-length": (float, None),
        "turn-cost": (float, 0.75),
        "crossing-cost": (float, 0.8),
    }
    for option, (value_type, default) in advanced.items():
        parser.add_argument(f"--{option}", type=value_type, default=default,
                            help=argparse.SUPPRESS)
    args = parser.parse_args()

    coverage_profiles = {
        "light": (0.25, 0.5),
        "balanced": (0.5, 0.5),
        "thorough": (1.0, 0.5),
    }
    track_profiles = {
        "occasional": (20.0, 5),
        "regular": (15.0, 15),
        "frequent": (10.0, 30),
    }
    defaults = {
        "coverage_reward": coverage_profiles[args.coverage][0],
        "distance_cost": coverage_profiles[args.coverage][1],
        "track_max_length": track_profiles[args.track_use][0],
        "max_track_segments": track_profiles[args.track_use][1],
        "max_distance": args.route_length,
        "max_leg_length": args.leg_length,
    }
    for name, default in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, default)
    if any(getattr(args, name) <= 0 for name in ("grid_step", "sensor_radius", "target_spacing", "sampling_attempts", "coverage_reward", "track_max_length", "track_join_distance", "max_distance", "max_stops", "max_leg_length")):
        parser.error("Distances, rewards, attempts and stop limit must be positive")
    if args.track_reward < 0 or args.max_track_segments < 0:
        parser.error("Track reward and track segment limit cannot be negative")
    if args.turn_cost < 0:
        parser.error("Turn cost cannot be negative")
    if args.crossing_cost < 0:
        parser.error("Crossing cost cannot be negative")
    output = args.output or args.grid.with_name("planned_route.gpx")
    plot_path = args.plot or output.with_suffix(".png")
    try:
        east, north, grid = read_grid(args.grid, args.grid_step)
        meta_path = args.grid.with_name("coordinate_origin.json")
        import json
        with meta_path.open(encoding="utf-8") as file:
            meta = json.load(file)["origin"]
        origin = meta["latitude"], meta["longitude"]
        tracks = track_lines(args.tracks, origin)
        hull, regions = enclosed_regions(east, north, grid, args.grid_step, args.min_region_area)
        targets = poisson_points(regions, args.target_spacing, args.seed, args.sampling_attempts)
        if not targets:
            raise ValueError("No enclosed uncovered regions or sample points found")
        covered_cells = set(zip(*np.where(grid)))
        if not tracks:
            raise ValueError("No ordered GPX track points found")
        start = min((point for track in tracks for point in track),
                    key=lambda point: min(math.dist(point, target) for target in targets))
        chunks = track_chunks(tracks, args.track_max_length)
        route, route_actions, newly_covered, length = plan_route(
            targets, start, east, north, covered_cells, args.grid_step, args.sensor_radius,
            args.coverage_reward, args.track_reward, chunks, args.track_join_distance,
            args.distance_cost, args.max_distance, args.max_stops, args.max_track_segments,
            args.max_leg_length, args.turn_cost, args.crossing_cost,
        )
        write_route_gpx(output, route, origin)
        plot(plot_path, east, north, grid, regions, targets, route, route_actions, tracks)
    except (OSError, ValueError, KeyError, IndexError) as exc:
        parser.error(str(exc))
    followed = sum(kind == "track" for kind, _ in route_actions)
    print(f"Saved {len(route)} route points ({length:.1f} m, {len(newly_covered)} new cells, {followed} track segments): {output}; {plot_path}")


if __name__ == "__main__":
    main()
