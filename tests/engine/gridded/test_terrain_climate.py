"""Tests for ddr_engine.gridded.terrain (slope) and .climate (Hargreaves PET)."""

import numpy as np
import pytest
import xarray as xr
from ddr_engine.gridded.attributes import extract_from_dataarray
from ddr_engine.gridded.climate import extraterrestrial_radiation_mm, hargreaves_pet_mm_yr
from ddr_engine.gridded.terrain import slope_degrees

JUNIATA_OUTLET = 138445


def _da(values: np.ndarray, bounds: tuple[float, float, float, float]) -> xr.DataArray:
    xmin, ymin, xmax, ymax = bounds
    ny, nx = values.shape
    dx, dy = (xmax - xmin) / nx, (ymax - ymin) / ny
    da = xr.DataArray(
        values,
        dims=("y", "x"),
        coords={"y": ymax - dy * (np.arange(ny) + 0.5), "x": xmin + dx * (np.arange(nx) + 0.5)},
    )
    return da.rio.write_crs("EPSG:4326")


class TestExtractFromDataArray:
    def test_mean_and_max(self) -> None:
        v = np.full((10, 10), 2.0, dtype="float32")
        v[0, 0] = 10.0
        da = _da(v, (-77.5, 40.0, -77.0, 40.5))
        mean = extract_from_dataarray(da, [JUNIATA_OUTLET], stat="mean")
        mx = extract_from_dataarray(da, [JUNIATA_OUTLET], stat="max")
        assert mean.loc[JUNIATA_OUTLET] == pytest.approx(2.08)
        assert mx.loc[JUNIATA_OUTLET] == pytest.approx(10.0)


class TestSlopeDegrees:
    # 3" pixel at the equator is 6371 km * (1/1200 deg in rad) = 92.67 m; a rise of 0.9267 m/px is a 1% grade
    PX_M = 6371000 * np.radians(1 / 1200)

    def test_planar_dem_east_west(self) -> None:
        res_deg = 1 / 1200
        elv = np.tile(np.arange(50) * 0.01 * self.PX_M, (20, 1))
        s = slope_degrees(elv, np.full(20, 0.0), res_deg)
        assert s[5:-5, 5:-5] == pytest.approx(np.degrees(np.arctan(0.01)), rel=1e-3)

    def test_latitude_shrinks_dx(self) -> None:
        # same field at 60N: east-west pixel is half as wide -> grade doubles to 2%
        res_deg = 1 / 1200
        elv = np.tile(np.arange(50) * 0.01 * self.PX_M, (20, 1))
        s = slope_degrees(elv, np.full(20, 60.0), res_deg)
        assert s[10, 25] == pytest.approx(np.degrees(np.arctan(0.02)), rel=1e-2)

    def test_north_south_uses_constant_dy(self) -> None:
        res_deg = 1 / 1200
        elv = np.tile((np.arange(20) * 0.01 * self.PX_M)[:, None], (1, 50))
        s = slope_degrees(elv, np.full(20, 60.0), res_deg)
        assert s[10, 25] == pytest.approx(np.degrees(np.arctan(0.01)), rel=1e-2)


class TestHargreaves:
    def test_ra_matches_fao56_example(self) -> None:
        # FAO-56 Example 8: 20S, 3 September -> Ra = 32.2 MJ m-2 d-1 = 13.1 mm/d (x0.408)
        ra = extraterrestrial_radiation_mm(np.array([-20.0]), day_of_year=246)
        assert ra[0] == pytest.approx(32.2 * 0.408, rel=0.01)

    def test_annual_pet_reasonable_for_temperate_site(self) -> None:
        # ~monthly climatology for a 40N humid-temperate site
        tmin = np.array([-6, -5, -1, 4, 9, 14, 17, 16, 12, 6, 1, -4], dtype=float)
        tmax = np.array([3, 5, 10, 17, 23, 27, 30, 29, 25, 18, 11, 5], dtype=float)
        pet = hargreaves_pet_mm_yr(tmin[:, None, None], tmax[:, None, None], np.array([40.0]))
        assert 700 < float(pet[0, 0]) < 1100

    def test_pet_increases_with_temperature(self) -> None:
        tmin = np.full((12, 1, 1), 5.0)
        tmax = np.full((12, 1, 1), 20.0)
        lat = np.array([40.0])
        cool = hargreaves_pet_mm_yr(tmin, tmax, lat)
        warm = hargreaves_pet_mm_yr(tmin + 5, tmax + 5, lat)
        assert float(warm[0, 0]) > float(cool[0, 0])


class TestSeasonalityIndex:
    def test_uniform_is_zero(self) -> None:
        from ddr_engine.gridded.climate import seasonality_index

        monthly = np.full((12, 2, 2), 50.0)
        assert seasonality_index(monthly) == pytest.approx(0.0)

    def test_single_month_is_walsh_lawler_max(self) -> None:
        from ddr_engine.gridded.climate import seasonality_index

        monthly = np.zeros((12, 1, 1))
        monthly[6] = 600.0
        assert float(seasonality_index(monthly)[0, 0]) == pytest.approx(1.833, abs=1e-3)


class TestMonthlyPet:
    def test_monthly_sums_to_annual(self) -> None:
        from ddr_engine.gridded.climate import hargreaves_pet_mm_yr, hargreaves_pet_monthly_mm

        tmin = np.linspace(-5, 15, 12)[:, None, None] * np.ones((12, 2, 3))
        tmax = tmin + 10
        lat = np.array([40.0, 45.0])
        monthly = hargreaves_pet_monthly_mm(tmin, tmax, lat)
        assert monthly.shape == (12, 2, 3)
        assert np.allclose(monthly.sum(axis=0), hargreaves_pet_mm_yr(tmin, tmax, lat))


class TestSnowfallFraction:
    def test_all_cold_is_one_all_warm_is_zero(self) -> None:
        from ddr_engine.gridded.climate import snowfall_fraction

        prec = np.full((12, 1, 1), 50.0)
        assert float(snowfall_fraction(prec, np.full((12, 1, 1), -10.0))[0, 0]) == pytest.approx(1.0)
        assert float(snowfall_fraction(prec, np.full((12, 1, 1), 20.0))[0, 0]) == pytest.approx(0.0)

    def test_precip_weighted_ramp(self) -> None:
        from ddr_engine.gridded.climate import snowfall_fraction

        # 6 cold months (-10C) with 100 mm each, 6 warm months (+20C) with 300 mm each -> 600/2400
        prec = np.array([100] * 6 + [300] * 6, dtype=float)[:, None, None]
        temp = np.array([-10] * 6 + [20] * 6, dtype=float)[:, None, None]
        assert float(snowfall_fraction(prec, temp)[0, 0]) == pytest.approx(0.25)

    def test_ramp_midpoint(self) -> None:
        from ddr_engine.gridded.climate import RAIN_C, SNOW_C, snowfall_fraction

        mid = 0.5 * (RAIN_C + SNOW_C)
        assert float(
            snowfall_fraction(np.full((12, 1, 1), 1.0), np.full((12, 1, 1), mid))[0, 0]
        ) == pytest.approx(0.5)


class TestSlopeFromDataArray:
    def test_resolution_read_from_coords(self) -> None:
        """slope_from_dataarray must use the raster's own pixel size, not an assumed one."""
        from ddr_engine.gridded.terrain import slope_from_dataarray

        res_deg = 1 / 480  # 7.5" (GMTED native), not the 30" global export
        px_m = 6371000 * np.radians(res_deg)
        ny, nx = 20, 50
        elv = np.tile(np.arange(nx) * 0.01 * px_m, (ny, 1))
        da = xr.DataArray(
            elv,
            dims=("y", "x"),
            coords={"y": np.arange(ny) * -res_deg, "x": np.arange(nx) * res_deg},
        ).rio.write_crs("EPSG:4326")
        out = slope_from_dataarray(da)
        assert float(out.values[10, 25]) == pytest.approx(np.degrees(np.arctan(0.01)), rel=1e-2)


class TestAridity:
    def test_uses_precipitation_floor(self) -> None:
        """Aridity must stay finite where WorldClim rounds annual precipitation to ~0."""
        from ddr_engine.gridded.climate import MIN_P_MM_YR, aridity_index

        a = aridity_index(np.array([1000.0, 1000.0, 1000.0]), np.array([500.0, 0.0, 0.0004]))
        assert a[0] == pytest.approx(2.0)
        assert np.isfinite(a).all()
        assert a[1] == pytest.approx(1000.0 / MIN_P_MM_YR)
        assert a[2] == pytest.approx(1000.0 / MIN_P_MM_YR)

    def test_monotone_in_precipitation(self) -> None:
        from ddr_engine.gridded.climate import aridity_index

        a = aridity_index(np.full(3, 1000.0), np.array([100.0, 500.0, 2000.0]))
        assert a[0] > a[1] > a[2]

    def test_nan_precipitation_propagates(self) -> None:
        from ddr_engine.gridded.climate import aridity_index

        assert np.isnan(aridity_index(np.array([1000.0]), np.array([np.nan])))[0]
