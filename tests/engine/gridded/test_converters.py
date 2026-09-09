"""Tests for the ddm30 grid-cell order converter in the core registry."""

import numpy as np
from ddr_engine.core.converters import get_converter, list_geodatasets


class TestDdm30Converter:
    """The ddm30 geodataset must be registered so coo_from_zarr auto-detects it."""

    def test_registered(self):
        assert "ddm30" in list_geodatasets()

    def test_to_zarr_int32_identity(self):
        converter = get_converter("ddm30")
        result = converter.to_zarr([138445, 0, 201599])
        assert result.dtype == np.int32
        assert result.tolist() == [138445, 0, 201599]

    def test_roundtrip(self):
        converter = get_converter("ddm30")
        ids = [138443, 138444, 139163, 138445]
        assert converter.from_zarr(converter.to_zarr(ids)) == ids
