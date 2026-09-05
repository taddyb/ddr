"""Validation checks for the gridded (DDM30) network and its attribute store.

Three families, all returning ``Check`` records so a CLI can print one table:

* :func:`check_network` — flow-direction/topology invariants of the built adjacency
* :func:`check_units` — every attribute inside its documented physical range
* :func:`check_attribute_consistency` — cross-variable identities (texture sums,
  field capacity below saturation, aridity = PET/P, accumulation >= own area)

Muskingum-Cunge stability is validated separately (it needs ddr's geometry code);
see ``scripts/validate_gridded.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Check:
    """One validation result."""

    name: str
    passed: bool
    detail: str = ""


@dataclass(frozen=True)
class Spec:
    """Documented unit and physical range of an attribute."""

    unit: str
    lo: float
    hi: float


# Units follow merit_global_attributes_v2.nc; ranges are physical bounds, not observed extremes.
UNITS: dict[str, Spec] = {
    "meanelevation": Spec("m", -450, 9000),  # Dead Sea shore to Everest
    "meanslope": Spec("degree", 0, 60),
    "SoilGrids1km_clay": Spec("%", 0, 100),
    "SoilGrids1km_sand": Spec("%", 0, 100),
    "SoilGrids1km_silt": Spec("%", 0, 100),
    "NDVI": Spec("-", -0.3, 1.0),
    "snow_fraction": Spec("1", 0, 1),
    "snowfall_fraction": Spec("1", 0, 1),
    "FW": Spec("1", 0, 1),
    "Ksat": Spec("cm/day", 0, 2000),
    "ALPHA": Spec("1/cm", 0, 1),
    "N": Spec("-", 1, 3),
    "ORMC": Spec("%", 0, 100),
    "WCpF2": Spec("m3/m3", 0, 1),
    "WCsat": Spec("m3/m3", 0, 1),
    "catchsize": Spec("km2", 0, 3100),  # 0.5 deg cell at the equator is 3092 km2
    "log10_uparea": Spec("log10 km2", 0, 7.2),  # Amazon ~6.1e6 km2
    "meanP": Spec("mm/year", 0, 13000),
    "meanTa": Spec("degC", -60, 45),
    "ETPOT_Hargr": Spec("mm/year", 0, 4000),
    "aridity": Spec("-", 0, 2000),
    "seasonality_P": Spec("-", 0, 1.834),
    "seasonality_PET": Spec("-", 0, 1.834),
}


def check_units(df: pd.DataFrame) -> list[Check]:
    """Every column inside its documented range, and every column documented."""
    out = []
    for col in df.columns:
        spec = UNITS.get(col)
        if spec is None:
            out.append(Check(f"{col}: documented unit", False, "no entry in UNITS"))
            continue
        v = df[col].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        bad = int(((v < spec.lo) | (v > spec.hi)).sum())
        detail = f"[{spec.lo}, {spec.hi}] {spec.unit}"
        if bad:
            detail += f" — {bad} values outside (min {v.min():.4g}, max {v.max():.4g})"
        out.append(Check(f"{col} in range", bad == 0, detail))
    return out


def check_attribute_consistency(df: pd.DataFrame, tol: float = 1e-3) -> list[Check]:
    """Cross-variable identities that must hold regardless of source."""
    out = []
    texture = ["SoilGrids1km_clay", "SoilGrids1km_sand", "SoilGrids1km_silt"]
    if all(c in df for c in texture):
        # min_count keeps all-missing rows missing (water cells) instead of summing to 0
        s = df[texture].sum(axis=1, min_count=1).dropna()
        bad = int((np.abs(s - 100) > 0.5).sum())
        out.append(Check("texture sums to 100%", bad == 0, f"{bad} cells off by >0.5%"))
    if {"WCpF2", "WCsat"} <= set(df):
        bad = int((df["WCpF2"] > df["WCsat"] + tol).sum())
        out.append(Check("WCpF2 <= WCsat", bad == 0, f"{bad} cells with field capacity above saturation"))
    if {"aridity", "ETPOT_Hargr", "meanP"} <= set(df):
        expect = df["ETPOT_Hargr"] / df["meanP"]
        bad = int((np.abs(expect - df["aridity"]) > 1e-6 * np.maximum(1.0, expect)).sum())
        out.append(Check("aridity = PET/P", bad == 0, f"{bad} cells violate the identity"))
    if {"log10_uparea", "catchsize"} <= set(df):
        bad = int((df["log10_uparea"] < np.log10(df["catchsize"]) - tol).sum())
        out.append(
            Check("log10_uparea >= cell area", bad == 0, f"{bad} cells accumulate less than their own area")
        )
    return out


def check_network(
    rows: np.ndarray,
    cols: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    basin: np.ndarray,
    ncols: int,
    cell_deg: float = 0.5,
) -> list[Check]:
    """Flow-direction invariants of the built adjacency (``rows`` = downstream, ``cols`` = upstream)."""
    out = []
    _, counts = np.unique(cols, return_counts=True)
    out.append(
        Check("single downstream", bool((counts <= 1).all()), f"max downstreams per cell {counts.max()}")
    )
    n_upper = int((rows <= cols).sum())
    out.append(Check("lower triangular", n_upper == 0, f"{n_upper} edges not strictly lower-triangular"))
    n_self = int((rows == cols).sum())
    out.append(Check("no self loops", n_self == 0, f"{n_self} self edges"))

    dlat = np.abs(lat[rows] - lat[cols])
    dlon = np.abs(lon[rows] - lon[cols])
    dlon = np.minimum(dlon, ncols * cell_deg - dlon)  # longitude wrap
    n_far = int(((dlat > cell_deg * 1.01) | (dlon > cell_deg * 1.01)).sum())
    out.append(Check("downstream is a neighbour", n_far == 0, f"{n_far} edges span more than one cell"))

    n_cross = int((basin[rows] != basin[cols]).sum())
    out.append(Check("edges within basin", n_cross == 0, f"{n_cross} edges cross basin boundaries"))
    return out


def check_mass_balance(
    area: np.ndarray, accumulated: np.ndarray, dn: np.ndarray, tol: float = 1e-6
) -> list[Check]:
    """Accumulated area conserves total area and increases downstream."""
    terminals = dn < 0
    total, routed = float(area.sum()), float(accumulated[terminals].sum())
    rel = abs(routed - total) / total
    out = [
        Check("accumulation conserves area", rel < tol, f"terminals {routed:.6g} vs total {total:.6g} km2")
    ]
    inner = ~terminals
    n_dec = int((accumulated[dn[inner]] < accumulated[inner] - tol).sum())
    out.append(Check("accumulation increases downstream", n_dec == 0, f"{n_dec} edges decrease"))
    return out


def check_coverage(df: pd.DataFrame, max_missing: float = 0.05) -> list[Check]:
    """Missing-data fraction per attribute; small gaps pass but are always reported."""
    out = []
    for col in df.columns:
        n_missing = int(df[col].isna().sum())
        frac = n_missing / max(1, len(df))
        detail = f"{n_missing} cell{'' if n_missing == 1 else 's'} missing ({100 * frac:.2f}%)"
        out.append(Check(f"{col} coverage", frac <= max_missing, detail))
    return out
