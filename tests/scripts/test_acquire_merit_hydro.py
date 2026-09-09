"""Tests for scripts/acquire_merit_hydro.py pure helpers (tile math, naming).

The script is a standalone PEP 723 uv script, so it is loaded from its file path;
its GEE imports (ee, geedim) are lazy and not required here.
"""

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).parents[2]
SCRIPT = REPO_ROOT / "scripts" / "acquire_merit_hydro.py"

spec = importlib.util.spec_from_file_location("acquire_merit_hydro", SCRIPT)
assert spec is not None and spec.loader is not None
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class TestTileName:
    def test_northwest_quadrant(self) -> None:
        assert mod.tile_name(-80, 40) == "n40w080"

    def test_conus_southwest_corner(self) -> None:
        assert mod.tile_name(-125, 25) == "n25w125"

    def test_southern_eastern_hemisphere(self) -> None:
        assert mod.tile_name(5, -10) == "s10e005"

    def test_equator_and_meridian_are_n_and_e(self) -> None:
        assert mod.tile_name(0, 0) == "n00e000"


class TestTilesForBbox:
    def test_point_bbox_single_tile(self) -> None:
        assert mod.tiles_for_bbox((-77.5, 40.2, -76.9, 40.3)) == [(-80, 40)]

    def test_exact_tile_bbox_does_not_spill(self) -> None:
        assert mod.tiles_for_bbox((-80.0, 40.0, -75.0, 45.0)) == [(-80, 40)]

    def test_conus_bbox_tile_count(self) -> None:
        tiles = mod.tiles_for_bbox(mod.CONUS_BBOX)
        # lon -125..-70 (12 tiles) x lat 20..50 (7 tiles)
        assert len(tiles) == 84
        assert (-125, 20) in tiles
        assert (-70, 50) in tiles

    def test_tiles_are_sorted_and_unique(self) -> None:
        tiles = mod.tiles_for_bbox(mod.CONUS_BBOX)
        assert tiles == sorted(set(tiles))


class TestCrsTransform:
    def test_transform_anchors_top_left(self) -> None:
        res = 1 / 1200  # 3 arc-seconds
        assert mod.tile_crs_transform(-80, 40) == [res, 0, -80, 0, -res, 45]

    def test_shape_matches_3arcsec_tile(self) -> None:
        assert mod.TILE_SHAPE == (6000, 6000)


class TestOutPath:
    def test_layout(self, tmp_path: Path) -> None:
        p = mod.out_path(tmp_path, "elv", (-80, 40))
        assert p == tmp_path / "elv" / "merit_hydro_elv_n40w080.tif"
