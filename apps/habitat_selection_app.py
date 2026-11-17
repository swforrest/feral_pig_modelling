"""Feral pig habitat selection under a chosen time and weather (Streamlit app).

Run from the repository root:

    /opt/miniconda3/envs/deepssf/bin/streamlit run apps/habitat_selection_app.py

The app is deliberately thin: every model calculation is a deepssf function
(ScalarEncoder, HabitatLandscapePredictor, class_selection, kernel_summary,
add_heatmap_overlay).  What it needs from a training run is written by the
"Files for the app" section of deepssf_piggies_inference_landcover.ipynb:
``inference/app_config.json`` (paths, window size, class names), the scalar
spec, the wet bulb correction and the availability mask.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import folium
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
import torch  # noqa: E402
from rasterio.warp import transform as warp_transform  # noqa: E402
from streamlit_folium import st_folium  # noqa: E402

import deepssf  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from apps import weather_utils as wx  # noqa: E402

DEFAULT_RUN = "outputs/landcover/2026-11-17_1"
VEG_COLOURS = {
    "Dry Rainforest": "#1B5E3A",
    "Eucalypt Open Forest": "#B5772B",
    "Eucalypt Woodland": "#E0B65C",
    "Floodplain Sedgeland": "#D6E2A3",
    "Melaleuca Woodland": "#A9CCE3",
    "Melaleuca Open Forest": "#5B97BF",
    "Melaleuca Closed Forest": "#2B5F8A",
    "Mangrove": "#2A9D8F",
    "Mudflat": "#8A7968",
    "Water": "#264653",
}
WIND_DIR = "onpli_wind_dir_deg"
WEATHER = ["onpli_rain_30min_mm", "onpli_air_temp_c", "onpli_wet_bulb_temp_c",
           "onpli_dew_point_temp_c", "onpli_rel_humidity_pct", "onpli_wind_speed_kmh"]

st.set_page_config(page_title="Feral pig habitat selection", layout="wide")


def label_of(cls: str) -> str:
    return cls.removeprefix("veg_").replace("_", " ").title()


# ---------------------------------------------------------------------------
# Loading (cached across reruns)
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading the model and rasters…")
def load_run(run_dir: str):
    run = REPO / run_dir
    cfg = json.loads((run / "inference" / "app_config.json").read_text())
    device = deepssf.get_device()
    layer_paths = {k: str(REPO / v) for k, v in cfg["layer_paths"].items()}
    env, transform = deepssf.load_environmental_layers(layer_paths)
    layers = []
    for arr in env.values():
        t = torch.from_numpy(np.asarray(arr).astype(np.float32))
        layers.extend([t] if t.ndim == 2 else list(t))

    encoder = deepssf.ScalarEncoder.from_json(run / cfg["scalar_spec"])
    ckpt = run / cfg["checkpoint"]
    params = deepssf.params_from_checkpoint(
        str(ckpt), image_dim=cfg["window_size"], pixel_size=cfg["pixel_size"],
        n_scalar_covariates=len(encoder.scalar_cols), device=device)
    model = deepssf.ConvJointModel(params).to(device)
    deepssf.load_checkpoint(str(ckpt), model)
    model.eval()

    mask = np.load(run / cfg["availability_mask"])
    predictors = {
        True: deepssf.HabitatLandscapePredictor(model, layers, len(encoder.scalar_cols),
                                                mask=mask),
        False: deepssf.HabitatLandscapePredictor(model, layers, len(encoder.scalar_cols)),
    }
    class_rasters = np.asarray(env[cfg["class_layer"]], dtype=np.float32)
    class_crops = {
        k: torch.from_numpy(class_rasters[:, p.row_slice, p.col_slice]).to(device)
        for k, p in predictors.items()
    }
    correction = json.loads((run / cfg["wet_bulb_correction"]).read_text())
    return dict(cfg=cfg, device=device, transform=transform, layers=layers,
                landscape_dev=torch.stack(layers).to(device), encoder=encoder,
                model=model, predictors=predictors, class_crops=class_crops,
                class_rasters=class_rasters, correction=correction)


@st.cache_data(show_spinner="Reading the GPS fixes…")
def load_fixes(csv_path: str, every: int = 60):
    cols = ["name", "t1_", "x1_", "y1_", WIND_DIR, *WEATHER]
    df = pd.read_csv(csv_path, usecols=cols)
    df["t1_"] = (pd.to_datetime(df["t1_"], utc=True).dt.tz_convert("Australia/Darwin")
                 .dt.tz_localize(None))
    # Weather is per half-hour at the station: one row per half-hour
    weather = (df.dropna(subset=[WIND_DIR, *WEATHER])
               .assign(slot=lambda d: d["t1_"].dt.floor("30min"))
               .drop_duplicates("slot").set_index("slot").sort_index()[[WIND_DIR, *WEATHER]])
    fixes = df.iloc[::every][["name", "t1_", "x1_", "y1_"]].reset_index(drop=True)
    lon, lat = warp_transform("EPSG:3112", "EPSG:4326", fixes["x1_"].tolist(),
                              fixes["y1_"].tolist())
    fixes["lon"], fixes["lat"] = lon, lat
    # One MultiPoint feature: ~25x smaller and faster to draw than a marker per fix
    geojson = {"type": "FeatureCollection", "features": [{
        "type": "Feature", "properties": {},
        "geometry": {"type": "MultiPoint",
                     "coordinates": [[round(x, 5), round(y, 5)] for x, y in zip(lon, lat,
                                                                                strict=True)]},
    }]}
    return fixes, weather, geojson


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

st.sidebar.title("Scenario")
run_dir = st.sidebar.text_input("Training run directory", DEFAULT_RUN)
try:
    R = load_run(run_dir)
except FileNotFoundError as exc:
    st.error(f"Could not load the run in {run_dir!r}: {exc}.  Run the 'Files for the app' "
             "section of the inference notebook first.")
    st.stop()
fixes, weather_table, fixes_geojson = load_fixes(str(REPO / R["cfg"]["csv_path"]))
encoder = R["encoder"]

mode = st.sidebar.radio("Weather", ["Set the weather", "Observed weather at a date-time"])
if mode == "Set the weather":
    date = st.sidebar.date_input("Date (sets the day of year)", pd.Timestamp("2025-07-19"))
    hour = st.sidebar.slider("Hour of day (Darwin time)", 0.0, 23.5, 9.0, 0.5)
    rain = st.sidebar.slider("Rain in the previous 30 min (mm)", 0.0, 35.0, 0.0, 0.2)
    calm = st.sidebar.checkbox("Calm (no wind)", value=False)
    wind_speed = 0.0 if calm else st.sidebar.slider("Wind speed (km/h)", 0.5, 37.0, 7.0, 0.5)
    wind_dir = st.sidebar.slider("Wind direction (° from north, blowing from)", 0, 350, 110,
                                 10, disabled=calm)
    air_temp = st.sidebar.slider("Air temperature (°C)", 11.0, 40.0, 27.0, 0.5)
    rh = st.sidebar.slider("Relative humidity (%)", 13.0, 100.0, 75.0, 1.0)
    dew = float(wx.dew_point_c(air_temp, rh))
    wet = float(wx.wet_bulb_c(air_temp, rh, R["correction"]))
    st.sidebar.caption(f"Derived: dew point **{dew:.1f} °C**, wet bulb **{wet:.1f} °C**")
    yday = pd.Timestamp(date).day_of_year
    natural = {"hour_t1": hour, "yday_t1": yday, "dt_hour": 0.5,
               "onpli_rain_30min_mm": rain, "onpli_air_temp_c": air_temp,
               "onpli_wet_bulb_temp_c": wet, "onpli_dew_point_temp_c": dew,
               "onpli_rel_humidity_pct": rh, "onpli_wind_speed_kmh": wind_speed,
               WIND_DIR: float(wind_dir)}
else:
    days = weather_table.index.normalize().unique()
    date = st.sidebar.date_input("Date", days[len(days) // 2], min_value=days.min(),
                                 max_value=days.max())
    slots = weather_table.loc[str(pd.Timestamp(date).date())].index
    if len(slots) == 0:
        st.sidebar.warning("No weather records on that date.")
        st.stop()
    slot = st.sidebar.select_slider("Time (Darwin)", options=list(slots),
                                    value=slots[len(slots) // 2],
                                    format_func=lambda t: t.strftime("%H:%M"))
    obs = weather_table.loc[slot]
    natural = {"hour_t1": slot.hour + slot.minute / 60, "yday_t1": slot.day_of_year,
               "dt_hour": 0.5, **{c: float(obs[c]) for c in [*WEATHER, WIND_DIR]}}
    st.sidebar.dataframe(pd.Series(natural).round(2).rename("value"), width="stretch")

st.sidebar.divider()
use_mask = st.sidebar.toggle("Only the availability mask (within 500 m of fixes)", value=True)
fixed_limits = st.sidebar.toggle("Fixed colour limits (compare scenarios)", value=True)
show_fixes = st.sidebar.toggle("Show observed fixes", value=True)

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

t0 = time.perf_counter()
for msg in encoder.check_ranges(natural):
    st.warning(f"Outside the training data: {msg}")
encoded = encoder.encode(natural)
predictor = R["predictors"][use_mask]
surface = predictor.predict(encoded, as_numpy=False)
selection = deepssf.class_selection(surface, R["class_crops"][use_mask], R["cfg"]["class_names"])
log_p = surface[0].cpu().numpy()
t_model = time.perf_counter() - t0


@st.cache_data
def reference_limits(run_dir: str, use_mask: bool):
    """Fixed colour limits: percentiles over a spread of observed-like scenarios."""
    rng = np.random.default_rng(0)
    rows = weather_table.sample(24, random_state=0)
    nat = pd.DataFrame({"hour_t1": rng.uniform(0, 24, 24), "yday_t1": rng.integers(1, 366, 24),
                        "dt_hour": 0.5, **{c: rows[c].to_numpy() for c in [*WEATHER, WIND_DIR]}})
    s = R["predictors"][use_mask].predict(encoder.encode(nat))
    return tuple(np.nanpercentile(s, [2, 99.8]))


if fixed_limits:
    lo, hi = reference_limits(run_dir, use_mask)
else:
    lo, hi = np.nanpercentile(log_p, [2, 99.8])

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

st.title("Feral pig habitat selection: East Alligator")
st.caption("Habitat-selection surface from the fitted deepSSF model for the chosen time and "
           "weather (habitat head only, so every cell is treated as reachable).  Brighter = "
           "more strongly selected.")
col_map, col_bar = st.columns([3, 2])

with col_map:
    t1 = time.perf_counter()
    valid = np.isfinite(log_p)
    shown = np.where(valid, np.clip(log_p, lo, hi) - lo + 1.0, 0.0)
    centre_lat, centre_lon = fixes["lat"].median(), fixes["lon"].median()
    fmap = folium.Map(location=[centre_lat, centre_lon], zoom_start=11, tiles=None,
                      control_scale=True)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(fmap)
    folium.TileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery", name="Satellite").add_to(fmap)
    deepssf.add_heatmap_overlay(
        fmap, shown, predictor.crop_transform(R["transform"]), name="Habitat selection",
        cmap="viridis", log_scale=False, vmin=1.0, vmax=hi - lo + 1.0, opacity=0.75)
    if show_fixes:
        folium.GeoJson(
            fixes_geojson, name="Observed fixes (every 60th)",
            marker=folium.CircleMarker(radius=1.2, weight=0, fill=True, fill_opacity=0.6,
                                       fill_color="#ff4b4b"),
        ).add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    clicked = st_folium(fmap, height=620, width="stretch", key="map",
                        returned_objects=["last_clicked"])
    t_map = time.perf_counter() - t1
    st.caption(f"Colour: log-probability per cell, limits {lo:.1f} to {hi:.1f} "
               f"({'fixed across scenarios' if fixed_limits else 'this scenario'}).  "
               f"Model {t_model * 1000:.0f} ms; map {t_map * 1000:.0f} ms.")

with col_bar:
    sel = selection.assign(label=selection["class"].map(label_of))
    sel = sel.set_index("label").loc[[k for k in VEG_COLOURS if k in set(sel["label"])]]
    plot = sel.drop(index="Water", errors="ignore")
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.barh(plot.index[::-1], plot["log_sr"][::-1],
            color=[VEG_COLOURS[k] for k in plot.index[::-1]], edgecolor="white")
    ax.axvline(0, color="#444444", lw=0.8)
    ax.set_xlabel("log selection ratio (0 = used as available)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#e3e3e3", lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title("Selection by landcover class" + (" within the mask" if use_mask else ""),
                 fontsize=10)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.dataframe(sel[["use", "availability", "log_sr"]].round(3), width="stretch")
    with st.expander("Encoded model inputs"):
        st.dataframe(pd.Series(encoded[0], index=encoder.scalar_cols).round(4).rename("value"),
                     width="stretch")

# ---------------------------------------------------------------------------
# Optional: next-step kernels at a clicked point
# ---------------------------------------------------------------------------

st.subheader("Next-step kernels at a clicked point")
last = (clicked or {}).get("last_clicked")
if not last:
    st.info("Click the map to see the habitat, movement and joint next-step surfaces for a "
            "pig at that point under this scenario.")
else:
    heading = st.slider("Previous heading (° anticlockwise from east)", -180, 180, 0, 15)
    xs, ys = warp_transform("EPSG:4326", "EPSG:3112", [last["lng"]], [last["lat"]])
    ws = R["cfg"]["window_size"]
    spatial = deepssf.crop_windows(R["landscape_dev"], xs, ys, R["transform"], ws)
    scal = torch.from_numpy(encoded).to(R["device"])
    bear = torch.tensor([[np.radians(heading)]], dtype=torch.float32, device=R["device"])
    # Habitat and joint surfaces exclude the outer edge_buffer cells, where the habitat
    # CNN's zero-padding leaves artefacts; the movement kernel is parametric
    edge = deepssf.habitat_edge_buffer(R["model"])
    surfaces = deepssf.next_step_surfaces(R["model"], spatial, scal, bear, edge_buffer=edge)
    block = R["model"].movement_grid_output
    summ = pd.concat([
        deepssf.kernel_summary(surfaces[name], block.distance_layer, block.bearing_layer,
                               bear[:, 0], edge=k_edge).assign(kernel=name)
        for name, k_edge in (("movement", 0), ("joint", edge))
    ]).set_index("kernel")
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
    for ax, name in zip(axes, ("habitat", "movement", "joint"), strict=True):
        arr = surfaces[name][0].cpu().numpy()
        arr = np.where(np.isfinite(arr), arr, np.nan)  # excluded edge cells: blank
        ax.imshow(arr, cmap="viridis", vmin=np.nanpercentile(arr, 5), vmax=np.nanmax(arr))
        ax.plot(ws // 2, ws // 2, "w+", ms=8, mew=1.5)
        ax.set(title=name.capitalize(), xticks=[], yticks=[])
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption(f"Window of {ws} × {ws} cells centred on ({last['lat']:.4f}, {last['lng']:.4f}); "
               f"log-probability, north up.  The outer {edge} cells of the habitat and joint "
               "surfaces are left out (blank): the habitat network's zero-padding distorts them.")
    st.dataframe(summ[["expected_distance", "q0.5", "p_stay", "turn_resultant",
                       "outer_ring_mass"]].round(3), width="stretch")
