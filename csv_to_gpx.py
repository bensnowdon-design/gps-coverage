"""Convert a GPS CSV (timestamp, latitude, longitude, optional altitude) to GPX."""

import argparse
import csv
from datetime import datetime
from pathlib import Path

import gpxpy.gpx


def convert_folder(folder):
    input_files = sorted(Path(folder).glob("*.csv"))
    if not input_files:
        print(f"No CSV files found in {folder}")
        return

    print(f"Converting {len(input_files)} CSV file(s) to GPX...")
    for input_csv in input_files:
        output_gpx = input_csv.with_suffix(".gpx")
        gpx = gpxpy.gpx.GPX()
        track = gpxpy.gpx.GPXTrack()
        segment = gpxpy.gpx.GPXTrackSegment()
        track.segments.append(segment)
        gpx.tracks.append(track)

        with input_csv.open(newline="", encoding="utf-8-sig") as file:
            for row in csv.DictReader(file):
                segment.points.append(gpxpy.gpx.GPXTrackPoint(
                    latitude=float(row["latitude"]),
                    longitude=float(row["longitude"]),
                    elevation=float(row["altitude"]) if row.get("altitude") else None,
                    time=datetime.fromisoformat(row["timestamp"]),
                ))

        output_gpx.write_text(gpx.to_xml(), encoding="utf-8")
        print(f"Created {output_gpx}")

    print("Conversion complete.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder")
    args = parser.parse_args()
    convert_folder(args.folder)


if __name__ == "__main__":
    main()