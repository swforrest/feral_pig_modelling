# Apps

## Habitat selection explorer (`habitat_selection_app.py`)

A local Streamlit app that maps the fitted deepSSF model's habitat-selection
surface for any time and weather. It shows:

- a map of the habitat surface over the observed fixes;
- a bar chart of the log selection ratio for each landcover class;
- optionally, the habitat, movement and joint next-step kernels at a point you
  click on the map.

### Running it

From the repository root:

```sh
/opt/miniconda3/envs/deepssf/bin/streamlit run apps/habitat_selection_app.py
```

The app needs `streamlit` and `streamlit-folium` in the `deepssf` environment.
These were installed with `pip install streamlit streamlit-folium` (streamlit 1.67.0,
streamlit-folium 0.27.4).

It reads its inputs from the training run in the sidebar's **Training run
directory** (default `outputs/landcover/2026-11-17_1`):

- `best_model.pt` and `scalar_spec.json`;
- `inference/app_config.json`, `inference/availability_mask.npy` and
  `inference/wet_bulb_correction.json`.

The "Files for the app" section of `deepssf_piggies_inference_landcover.ipynb`
writes the last three. Run the notebook first for a new training run.

### Inputs

There are two ways to set the weather:

- **Set the weather:** choose the date (which sets the day of year), the hour,
  rain, wind and air temperature + relative humidity. Dew point and wet bulb are
  worked out from temperature and humidity with the helpers in `weather_utils.py`:
  Magnus for dew point, and Stull for wet bulb with the station-fitted correction.
  Those are the same relations the notebook uses.
- **Observed weather at a date-time:** use the weather recorded at the station at
  a chosen half-hour.

The app warns you when an input is outside the range seen in training.

The sidebar also has three toggles:

- **Only the availability mask:** restricts the map and the selection ratios to
  cells within 500 m of an observed fix (on by default), or uses the whole
  landscape.
- **Fixed colour limits:** keeps the colour scale the same across scenarios (on
  by default), so maps can be compared.
- **Show observed fixes:** shows or hides every 60th fix.

### Files

- `habitat_selection_app.py`: the app. All model calculations are `deepssf`
  functions; the app only gathers inputs and draws the outputs.
- `weather_utils.py`: the psychrometric helpers (dew point, wet bulb and the
  "heat-stress" axis), shared with the inference notebook.
