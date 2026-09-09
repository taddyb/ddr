"""Tests for the gridded CLI entrypoint."""

import zarr
from ddr_engine.gridded.__main__ import main


class TestCli:
    def test_builds_adjacency_zarr(self, sandbox_nc_dir, tmp_path):
        main([str(sandbox_nc_dir), "--path", str(tmp_path)])
        root = zarr.open_group(store=tmp_path / "ddm30_adjacency.zarr", mode="r")
        assert root.attrs["geodataset"] == "ddm30"
        assert len(root["order"][:]) == 10

    def test_builds_gauge_subsets(self, sandbox_nc_dir, tmp_path):
        gages_csv = tmp_path / "gages.csv"
        gages_csv.write_text("STAID,LAT,LON\n01567000,41.25,-76.75\n")
        main([str(sandbox_nc_dir), "--path", str(tmp_path), "--gages", str(gages_csv)])
        root = zarr.open_group(store=tmp_path / "ddm30_gages_adjacency.zarr", mode="r")
        assert set(root["01567000"]["order"][:].tolist()) == {8, 9, 13, 10}
