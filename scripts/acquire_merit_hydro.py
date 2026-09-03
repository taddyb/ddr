#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "earthengine-api>=1.5",
#   "geedim>=2,<3",
# ]
# ///
"""Acquire MERIT Hydro rasters from Google Earth Engine.

Downloads ``MERIT/Hydro/v1_0_1`` bands (default: elv, dir, upa, wth) as
native-grid GeoTIFF tiles — 3 arc-second resolution, 5°x5° tiles named by
their lower-left corner (MERIT convention, e.g. ``n40w080``) — clipped to a
bbox (default: CONUS). Resumable: existing tiles are skipped.

This bypasses the registration-gated Dropbox distribution of MERIT Hydro;
the GEE asset carries the same bands. D8 codes here are ESRI-style
(1=E, 2=SE, 4=S, ... 128=NE; 0=river mouth, -1=inland sink).

One-time setup:
    uv run --with earthengine-api earthengine authenticate

Usage:
    uv run scripts/acquire_merit_hydro.py --project <gee-cloud-project>
"""

from __future__ import annotations

import argparse
import logging
import math
import os
from pathlib import Path

ASSET = "MERIT/Hydro/v1_0_1"
DEFAULT_BANDS = ("elv", "dir", "upa", "wth")
CONUS_BBOX = (-125.0, 24.0, -66.0, 53.0)
TILE_DEG = 5
RES = 1 / 1200  # 3 arc-seconds
TILE_SHAPE = (6000, 6000)
DEFAULT_OUT = Path("/mnt/ssd1/data/merit_hydro/rasters")

log = logging.getLogger("acquire_merit_hydro")


def tile_name(lon0: int, lat0: int) -> str:
    """MERIT-style tile name from the lower-left corner, e.g. (-80, 40) -> n40w080."""
    ns = "s" if lat0 < 0 else "n"
    ew = "w" if lon0 < 0 else "e"
    return f"{ns}{abs(lat0):02d}{ew}{abs(lon0):03d}"


def tiles_for_bbox(
    bbox: tuple[float, float, float, float], tile_deg: int = TILE_DEG
) -> list[tuple[int, int]]:
    """5° tile lower-left corners covering bbox (xmin, ymin, xmax, ymax); exact edges don't spill."""
    xmin, ymin, xmax, ymax = bbox
    lon0 = math.floor(xmin / tile_deg) * tile_deg
    lat0 = math.floor(ymin / tile_deg) * tile_deg
    lon1 = math.ceil(xmax / tile_deg) * tile_deg
    lat1 = math.ceil(ymax / tile_deg) * tile_deg
    return [(lon, lat) for lon in range(lon0, lon1, tile_deg) for lat in range(lat0, lat1, tile_deg)]


def tile_crs_transform(lon0: int, lat0: int, tile_deg: int = TILE_DEG) -> list[float]:
    """Affine transform anchored at the tile's top-left corner (north-up 3" grid)."""
    return [RES, 0, lon0, 0, -RES, lat0 + tile_deg]


def out_path(out_dir: Path, band: str, tile: tuple[int, int]) -> Path:
    """Tile file path: <out_dir>/<band>/merit_hydro_<band>_<tile>.tif."""
    return out_dir / band / f"merit_hydro_{band}_{tile_name(*tile)}.tif"


def download(
    bands: tuple[str, ...],
    bbox: tuple[float, float, float, float],
    out_dir: Path,
    project: str,
) -> None:
    """Download each band x tile as a GeoTIFF, skipping tiles that already exist."""
    # GEE deps are lazy so tests can import the pure helpers without them
    import ee
    import geedim as gd

    gd.Initialize(project=project)
    tiles = tiles_for_bbox(bbox)
    log.info("%d tiles x %d bands from %s -> %s", len(tiles), len(bands), ASSET, out_dir)
    failed: list[str] = []
    for band in bands:
        image = gd.MaskedImage(ee.Image(ASSET).select(band))
        for tile in tiles:
            path = out_path(out_dir, band, tile)
            if path.exists():
                log.info("skip (exists): %s", path.name)
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                image.download(
                    path,
                    crs="EPSG:4326",
                    crs_transform=tile_crs_transform(*tile),
                    shape=TILE_SHAPE,
                )
            except Exception:
                log.exception("failed: %s (rerun to retry)", path.name)
                path.unlink(missing_ok=True)  # no partial tiles — keeps the skip-if-exists resume sound
                failed.append(path.name)
    if failed:
        raise SystemExit(f"{len(failed)} tile(s) failed: {failed}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bands", nargs="+", default=list(DEFAULT_BANDS))
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        default=list(CONUS_BBOX),
        metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--project",
        default=os.environ.get("EARTHENGINE_PROJECT"),
        help="GEE-registered Google Cloud project (or set EARTHENGINE_PROJECT)",
    )
    args = parser.parse_args()
    if not args.project:
        parser.error("--project (or EARTHENGINE_PROJECT) is required")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    download(tuple(args.bands), tuple(args.bbox), args.out, args.project)


if __name__ == "__main__":
    main()
