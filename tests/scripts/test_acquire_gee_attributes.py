"""Tests for scripts/acquire_gee_attributes.py declarative registry (no GEE deps)."""

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parents[2]
SCRIPT = REPO_ROOT / "scripts" / "acquire_gee_attributes.py"

spec = importlib.util.spec_from_file_location("acquire_gee_attributes", SCRIPT)
assert spec is not None and spec.loader is not None
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod  # dataclasses resolve annotations via sys.modules
spec.loader.exec_module(mod)


class TestRegistry:
    def test_names_unique(self) -> None:
        names = [e.name for e in mod.REGISTRY]
        assert names == sorted(set(names), key=names.index)
        assert len(names) == len(set(names))

    def test_scales_positive(self) -> None:
        assert all(e.scale_m > 0 for e in mod.REGISTRY)

    def test_kinds_valid(self) -> None:
        assert all(e.kind in ("image", "ic_filter", "ic_mean", "ic_sum") for e in mod.REGISTRY)

    def test_ic_entries_have_selector(self) -> None:
        for e in mod.REGISTRY:
            if e.kind == "ic_filter":
                assert e.index, e.name
            if e.kind == "ic_mean":
                assert e.band, e.name
            if e.kind == "ic_sum":
                assert e.indices, e.name

    def test_expected_layers_present(self) -> None:
        names = {e.name for e in mod.REGISTRY}
        assert {
            "gmted2010_mea",
            "glwd_v2_area_pct",
            "glwd_v2_openwater_pct",
            "ndvi_mod13_mean",
            "snow_mod10_mean",
        } <= names
        assert {f"hihydrosoil_{v}" for v in ("ksat", "alpha", "N", "ormc", "wcpf2", "wcsat")} <= names


class TestOutPath:
    def test_layout(self, tmp_path: Path) -> None:
        assert mod.out_path(tmp_path, "gmted2010_mea") == tmp_path / "gmted2010_mea.tif"


class TestSelect:
    def test_only_filters_registry(self) -> None:
        sel = mod.select_entries(["gmted2010_mea"])
        assert [e.name for e in sel] == ["gmted2010_mea"]

    def test_unknown_name_raises(self) -> None:
        import pytest

        with pytest.raises(SystemExit):
            mod.select_entries(["nope"])

    def test_none_returns_all(self) -> None:
        assert mod.select_entries(None) == list(mod.REGISTRY)


class TestGlwdOpenWater:
    def test_open_water_is_classes_1_to_6(self) -> None:
        layer = next(e for e in mod.REGISTRY if e.name == "glwd_v2_openwater_pct")
        assert layer.kind == "ic_sum"
        assert layer.indices == tuple(f"GLWD_v2_delta_class_{i:02d}_pct" for i in range(1, 7))


class TestGlobalMode:
    def test_global_bbox_covers_ddm30_grid(self) -> None:
        xmin, ymin, xmax, ymax = mod.GLOBAL_BBOX
        assert (xmin, xmax) == (-180.0, 180.0)
        assert ymin <= -56.0 and ymax >= 84.0  # DDM30 rows span -55.75..83.75

    def test_every_layer_has_a_global_scale(self) -> None:
        for e in mod.REGISTRY:
            assert mod.scale_for(e, global_mode=True) > 0

    def test_global_scale_is_coarser_or_equal(self) -> None:
        for e in mod.REGISTRY:
            assert mod.scale_for(e, global_mode=True) >= e.scale_m

    def test_conus_scale_unchanged(self) -> None:
        e = next(x for x in mod.REGISTRY if x.name == "gmted2010_mea")
        assert mod.scale_for(e, global_mode=False) == e.scale_m
