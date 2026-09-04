#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "earthengine-api>=1.5",
#   "geedim>=2,<3",
# ]
# ///
"""Acquire gridded-KAN attribute rasters from Google Earth Engine.

Downloads each registry layer as one CONUS GeoTIFF (EPSG:4326 at the source's
native nominal scale) into /mnt/ssd1/data/gridded_attrs/gee/, ready for
extractrs zonal stats onto DDM30 0.5° cells. Resumable: existing files are
skipped. See wiki/gridded-attribute-sources.md for the full source catalog.

Choices baked in (flag if they should change):
- Soils (HiHydroSoil) use the 0-5 cm surface depth; deeper layers exist in
  the same collections. HiHydroSoil values are int x10,000 -> multiply 1e-4.
- NDVI = MOD13A2 mean over 2000-2019 (x0.0001); snow_fraction = MOD10A1
  NDSI_Snow_Cover mean over 2000-2020 (codes >100 masked).
- GLWD: `glwd_v2_openwater_pct` (classes 1-6, the paper's FW) is the one to
  use; `glwd_v2_area_pct` (all 33 classes) is kept for reference.
- SoilGrids clay/sand/silt are NOT pulled from GEE (Mollweide, needs
  resampling) - use ISRIC's pre-aggregated 1 km GeoTIFFs directly.

Usage:
    uv run scripts/acquire_gee_attributes.py --project <gee-cloud-project> [--only name ...]
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path

CONUS_BBOX = (-125.0, 24.0, -66.0, 53.0)
DEFAULT_OUT = Path("/mnt/ssd1/data/gridded_attrs/gee")

log = logging.getLogger("acquire_gee_attributes")


@dataclass(frozen=True)
class Layer:
    """One GEE layer: how to build the ee.Image and at what scale to export."""

    name: str
    asset: str
    kind: str  # "image" | "ic_filter" (one image by index) | "ic_sum" (sum of indices) | "ic_mean"
    scale_m: float
    band: str | None = None  # band to select ("image"/"ic_mean")
    index: str | None = None  # system:index to filter ("ic_filter")
    indices: tuple[str, ...] = ()  # system:index values to sum ("ic_sum")
    date_range: tuple[str, str] | None = None  # for "ic_mean"
    valid_max: float | None = None  # mask values above this before reducing
    max_requests: int = 32  # lower for compute-heavy reductions (EE concurrency limit)


_HHS = "projects/sat-io/open-datasets/HiHydroSoilv2_0"

REGISTRY: list[Layer] = [
    Layer("gmted2010_mea", "USGS/GMTED2010_FULL", "image", 231.92, band="mea"),
    *[
        Layer(f"hihydrosoil_{var}", f"{_HHS}/{var}", "ic_filter", 250, index=f"{prefix}_0-5cm_M_250m")
        for var, prefix in [
            ("ksat", "Ksat"),
            ("alpha", "ALFA"),
            ("N", "N"),
            ("ormc", "ORMC"),
            ("wcpf2", "WCpF2"),
            ("wcsat", "WCsat"),
        ]
    ],
    Layer(
        "glwd_v2_area_pct",
        "projects/sat-io/open-datasets/GLWD/GLWD_V2_DELTA_AREA_PCT",
        "image",
        463,
    ),
    Layer(  # paper's FW = "fraction of open water": lakes, reservoirs, rivers, permanent waterbodies
        "glwd_v2_openwater_pct",
        "projects/sat-io/open-datasets/GLWD/DELTA_AREA_CLASS_PCT",
        "ic_sum",
        463,
        indices=tuple(f"GLWD_v2_delta_class_{i:02d}_pct" for i in range(1, 7)),
        max_requests=8,
    ),
    Layer(
        "ndvi_mod13_mean",
        "MODIS/061/MOD13A2",
        "ic_mean",
        1000,
        band="NDVI",
        date_range=("2000-02-18", "2020-01-01"),
    ),
    Layer(
        "snow_mod10_mean",
        "MODIS/061/MOD10A1",
        "ic_mean",
        1000,  # native 500 m, but the 21-yr daily mean is compute-heavy; 1 km is ample for 0.5deg cells
        band="NDSI_Snow_Cover",
        date_range=("2000-02-24", "2021-01-01"),
        valid_max=100,
        max_requests=4,
    ),
]


def out_path(out_dir: Path, name: str) -> Path:
    """Layer file path: <out_dir>/<name>.tif."""
    return out_dir / f"{name}.tif"


def select_entries(only: list[str] | None) -> list[Layer]:
    """Registry subset for --only names (all when None); unknown names abort."""
    if only is None:
        return list(REGISTRY)
    by_name = {e.name: e for e in REGISTRY}
    unknown = [n for n in only if n not in by_name]
    if unknown:
        raise SystemExit(f"unknown layer(s): {unknown}; available: {sorted(by_name)}")
    return [by_name[n] for n in only]


def build_ee_image(layer: Layer):  # type: ignore[no-untyped-def] # returns ee.Image; ee is a lazy import
    """Materialize the ee.Image for a registry layer."""
    import ee

    if layer.kind == "image":
        img = ee.Image(layer.asset)
        return img.select(layer.band) if layer.band else img
    ic = ee.ImageCollection(layer.asset)
    if layer.kind == "ic_filter":
        return ee.Image(ic.filter(ee.Filter.eq("system:index", layer.index)).first())
    if layer.kind == "ic_sum":
        return ic.filter(ee.Filter.inList("system:index", list(layer.indices))).sum()
    assert layer.date_range is not None
    ic = ic.filterDate(*layer.date_range).select(layer.band)
    if layer.valid_max is not None:
        vmax = layer.valid_max
        ic = ic.map(lambda im: im.updateMask(im.lte(vmax)))
    return ic.mean()


def download(entries: list[Layer], out_dir: Path, project: str) -> None:
    """Download each layer as a CONUS GeoTIFF, skipping existing files."""
    import ee
    import geedim as gd

    gd.Initialize(project=project)
    region = ee.Geometry.Rectangle(list(CONUS_BBOX), proj="EPSG:4326", geodesic=False)
    failed: list[str] = []
    for layer in entries:
        path = out_path(out_dir, layer.name)
        if path.exists():
            log.info("skip (exists): %s", path.name)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        log.info("downloading %s (%s @ %.0f m)", layer.name, layer.asset, layer.scale_m)
        try:
            gd.MaskedImage(build_ee_image(layer)).download(
                path,
                crs="EPSG:4326",
                scale=layer.scale_m,
                region=region,
                max_requests=layer.max_requests,
            )
        except Exception:
            log.exception("failed: %s (rerun to retry)", layer.name)
            path.unlink(missing_ok=True)
            failed.append(layer.name)
    if failed:
        raise SystemExit(f"{len(failed)} layer(s) failed: {failed}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", default=None, help="subset of layer names")
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
    download(select_entries(args.only), args.out, args.project)


if __name__ == "__main__":
    main()
