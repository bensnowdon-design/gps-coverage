"""Convert GPX tracks to CSV files with timestamp, latitude, longitude and altitude."""

import argparse
import csv
from pathlib import Path

import gpxpy


def convert_folder(folder, output_dir=None):
    input_files = sorted(Path(folder).glob("*.gpx"))
    if not input_files:
        print(f"No GPX files found in {folder}")
        return

    print(f"Converting {len(input_files)} GPX file(s) to CSV...")
    output_dir = Path(output_dir) if output_dir else Path(folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    for input_gpx in input_files:
        output_csv = output_dir / input_gpx.with_suffix(".csv").name
        with input_gpx.open(encoding="utf-8") as file:
            gpx = gpxpy.parse(file)

        with output_csv.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(("timestamp", "latitude", "longitude", "altitude"))
            for track in gpx.tracks:
                for segment in track.segments:
                    for point in segment.points:
                        writer.writerow((
                            point.time.isoformat() if point.time else "",
                            point.latitude,
                            point.longitude,
                            point.elevation if point.elevation is not None else "",
                        ))
        print(f"Created {output_csv}")

    print("Conversion complete.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder")
    parser.add_argument("--output-dir", help="directory for converted CSV files (default: input folder)")
    args = parser.parse_args()
    convert_folder(args.folder, args.output_dir)


if __name__ == "__main__":
    main()
