"""Render a coverage-grid CSV as a PNG image."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


#################
STEP = 0.5
################


def read_grid(path):
    with path.open(newline="", encoding="utf-8-sig") as file:
        if next(csv.reader(file), []) != ["east_m", "north_m", "covered"]:
            raise ValueError("Expected columns east_m,north_m,covered")

    data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    if data.shape[1] != 3 or not np.isfinite(data).all():
        raise ValueError("Grid contains invalid values")
    if not np.isin(data[:, 2], (0, 1)).all():
        raise ValueError("covered values must be 0 or 1")

    east = np.unique(data[:, 0])
    north = np.unique(data[:, 1])
    if len(data) != len(east) * len(north):
        raise ValueError("Grid must contain exactly one value per cell")
    if (len(east) > 1 and not np.allclose(np.diff(east), STEP)) or (
        len(north) > 1 and not np.allclose(np.diff(north), STEP)
    ):
        raise ValueError("Grid cells must be spaced 0.5 m apart")

    cols = np.searchsorted(east, data[:, 0])
    rows = np.searchsorted(north, data[:, 1])
    indexes = rows * len(east) + cols
    if np.unique(indexes).size != len(data):
        raise ValueError("Grid contains duplicate cells")
    grid = np.empty((len(north), len(east)), dtype=np.uint8)
    grid.ravel()[indexes] = data[:, 2].astype(np.uint8)
    return east, north, grid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("grid", type=Path, help="coverage_grid.csv")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.grid.with_name("coverage_grid.png")
    if args.grid.resolve() == output.resolve():
        parser.error("Output must not overwrite the grid CSV")

    try:
        east, north, grid = read_grid(args.grid)
        fig, ax = plt.subplots()
        image = ax.imshow(
            grid,
            origin="lower",
            interpolation="nearest",
            extent=(east[0] - STEP / 2, east[-1] + STEP / 2,
                    north[0] - STEP / 2, north[-1] + STEP / 2),
            cmap=ListedColormap(("white", "#2563eb")),
            vmin=0,
            vmax=1,
            aspect="equal",
        )
        ax.set(xlabel="East (m)", ylabel="North (m)", title="Coverage grid")
        colorbar = fig.colorbar(image, ax=ax, ticks=(0, 1))
        colorbar.ax.set_yticklabels(("Uncovered", "Covered"))
        fig.tight_layout()
        fig.savefig(output, dpi=200)
        plt.close(fig)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
