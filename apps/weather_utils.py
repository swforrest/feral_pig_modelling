"""Psychrometric helpers for building physically consistent weather scenarios.

Air temperature, relative humidity, dew point and wet bulb temperature are
linked, so a scenario that changes one of them has to recompute the others.
These are the relations used by the inference notebook
(deepssf_piggies_inference_landcover.ipynb) and the habitat-selection app:

* dew point from temperature and RH by the Magnus formula (a = 17.62,
  b = 243.12 °C), and its inverse, RH from temperature and dew point;
* wet bulb temperature from temperature and RH by Stull (2011), optionally
  with a linear correction fitted to the station's observed wet bulb (Stull
  assumes sea-level pressure; at the Oenpelli station it reads ~0.2 °C low).

All functions take and return NumPy arrays (or scalars) in °C and %.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

MAGNUS_A = 17.62
MAGNUS_B = 243.12  # °C


def dew_point_c(air_temp_c, rel_humidity_pct):
    """Dew point (°C) from air temperature (°C) and RH (%) — Magnus formula."""
    t = np.asarray(air_temp_c, dtype=float)
    rh = np.asarray(rel_humidity_pct, dtype=float)
    gamma = np.log(rh / 100.0) + MAGNUS_A * t / (MAGNUS_B + t)
    return MAGNUS_B * gamma / (MAGNUS_A - gamma)


def rel_humidity_pct(air_temp_c, dew_point_c_):
    """RH (%) from air temperature and dew point (°C) — inverse of Magnus."""
    t = np.asarray(air_temp_c, dtype=float)
    td = np.asarray(dew_point_c_, dtype=float)
    return 100.0 * np.exp(
        MAGNUS_A * td / (MAGNUS_B + td) - MAGNUS_A * t / (MAGNUS_B + t)
    )


def wet_bulb_stull_c(air_temp_c, rel_humidity_pct_):
    """Wet bulb temperature (°C) from air temperature (°C) and RH (%).

    Stull (2011), *J. Appl. Meteor. Climatol.* 50, 2267–2269; valid for RH
    5–99 % and -20–50 °C at sea-level pressure.
    """
    t = np.asarray(air_temp_c, dtype=float)
    rh = np.asarray(rel_humidity_pct_, dtype=float)
    return (
        t * np.arctan(0.151977 * np.sqrt(rh + 8.313659))
        + np.arctan(t + rh)
        - np.arctan(rh - 1.676331)
        + 0.00391838 * rh**1.5 * np.arctan(0.023101 * rh)
        - 4.686035
    )


def wet_bulb_c(air_temp_c, rel_humidity_pct_, correction=None):
    """Wet bulb (°C): Stull, then ``intercept + slope * stull`` if *correction*.

    *correction* is a dict with ``intercept`` and ``slope`` (see
    :func:`fit_wet_bulb_correction`), or None for plain Stull.  The result is
    capped at the air temperature (the wet bulb can't exceed it; near
    saturation the linear correction would otherwise overshoot by ~0.3 °C).
    """
    stull = wet_bulb_stull_c(air_temp_c, rel_humidity_pct_)
    if correction is None:
        return stull
    corrected = correction["intercept"] + correction["slope"] * stull
    return np.minimum(corrected, np.asarray(air_temp_c, dtype=float))


def fit_wet_bulb_correction(air_temp_c, rel_humidity_pct_, observed_wet_bulb_c):
    """Least-squares ``observed ≈ intercept + slope * stull``; returns a dict."""
    stull = wet_bulb_stull_c(air_temp_c, rel_humidity_pct_)
    slope, intercept = np.polyfit(stull, np.asarray(observed_wet_bulb_c, float), 1)
    return {"intercept": float(intercept), "slope": float(slope)}


def save_correction(correction, path):
    Path(path).write_text(json.dumps(correction, indent=2) + "\n")


def load_correction(path):
    return json.loads(Path(path).read_text())


# ---------------------------------------------------------------------------
# Consistent scenario updates (natural-unit DataFrames with onpli_* columns)
# ---------------------------------------------------------------------------

TEMP = "onpli_air_temp_c"
RH = "onpli_rel_humidity_pct"
DEW = "onpli_dew_point_temp_c"
WET = "onpli_wet_bulb_temp_c"


def set_temp_fixed_dew_point(rows: pd.DataFrame, air_temp_c, correction=None):
    """Set air temperature, keeping each row's dew point (absolute humidity).

    RH and wet bulb are recomputed.  Where the new temperature is below the
    dew point the air would be supersaturated, so the dew point is lowered to
    the temperature (RH = 100 %).
    """
    out = rows.copy()
    t = np.broadcast_to(np.asarray(air_temp_c, dtype=float), len(out))
    td = np.minimum(out[DEW].to_numpy(float), t)
    out[TEMP] = t
    out[DEW] = td
    out[RH] = rel_humidity_pct(t, td)
    out[WET] = wet_bulb_c(t, out[RH].to_numpy(float), correction)
    return out


def set_rh_fixed_temp(rows: pd.DataFrame, rel_humidity, correction=None):
    """Set RH, keeping each row's air temperature; dew point and wet bulb follow."""
    out = rows.copy()
    rh = np.broadcast_to(np.asarray(rel_humidity, dtype=float), len(out))
    t = out[TEMP].to_numpy(float)
    out[RH] = rh
    out[DEW] = dew_point_c(t, rh)
    out[WET] = wet_bulb_c(t, rh, correction)
    return out


def set_temp_rh(rows: pd.DataFrame, air_temp_c, rel_humidity, correction=None):
    """Set both air temperature and RH; dew point and wet bulb follow."""
    out = rows.copy()
    t = np.broadcast_to(np.asarray(air_temp_c, dtype=float), len(out))
    rh = np.broadcast_to(np.asarray(rel_humidity, dtype=float), len(out))
    out[TEMP] = t
    out[RH] = rh
    out[DEW] = dew_point_c(t, rh)
    out[WET] = wet_bulb_c(t, rh, correction)
    return out


# ---------------------------------------------------------------------------
# Heat-stress axis: temperature and humidity moving together
# ---------------------------------------------------------------------------


def rh_for_wet_bulb(air_temp_c, wet_bulb_target_c, correction=None, tol=1e-6):
    """RH (%) giving the target wet bulb at the given temperature (bisection).

    Wet bulb increases with RH at fixed temperature, so the root is unique.
    Targets above the temperature (impossible) return 100 %.
    """
    t = np.asarray(air_temp_c, dtype=float)
    target = np.asarray(wet_bulb_target_c, dtype=float)
    t, target = np.broadcast_arrays(t, target)
    lo = np.full(t.shape, 1.0)
    hi = np.full(t.shape, 100.0)
    for _ in range(60):
        mid = (lo + hi) / 2
        too_low = wet_bulb_c(t, mid, correction) < target
        lo = np.where(too_low, mid, lo)
        hi = np.where(too_low, hi, mid)
        if np.all(hi - lo < tol):
            break
    return (lo + hi) / 2


def heat_stress_path(observed: pd.DataFrame, n_bins: int = 20) -> pd.DataFrame:
    """The typical air temperature at each wet bulb, from observed rows.

    Rows are cut into wet bulb quantile bins; the path is the median wet bulb
    and median air temperature of each bin.  :func:`set_heat_stress` moves
    along it.
    """
    wb = observed[WET]
    bins = pd.qcut(wb, n_bins, duplicates="drop")
    path = observed.groupby(bins, observed=True).agg(
        wet_bulb=(WET, "median"), air_temp=(TEMP, "median")
    )
    return path.reset_index(drop=True)


def set_heat_stress(rows: pd.DataFrame, wet_bulb_c_, path: pd.DataFrame,
                    correction=None):
    """Move rows along the heat-stress axis to the given wet bulb.

    Air temperature is the path's typical temperature at that wet bulb
    (linear interpolation, flat beyond the ends), RH is solved so the wet bulb
    is exactly the target, and dew point follows.  All four thermal variables
    of every row are replaced.
    """
    wb = np.broadcast_to(np.asarray(wet_bulb_c_, dtype=float), len(rows))
    t = np.interp(wb, path["wet_bulb"], path["air_temp"])
    t = np.maximum(t, wb)  # wet bulb cannot exceed air temperature
    rh = rh_for_wet_bulb(t, wb, correction)
    return set_temp_rh(rows, t, rh, correction)
