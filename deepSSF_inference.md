---
title: "deepSSF inference: working plan and stage prompts"
draft: true   # keeps Quarto from publishing this page (it renders an empty stub only)
---

# deepSSF inference: landcover × weather × time

Plan and copy-paste prompts for using the fitted deepSSF model to understand:

- how **selection of each landcover class** changes with time of day, time of year and weather (partial dependence, ICE, ALE, 2-D interactions, feature importance);
- how the **movement kernel** (step length, turn angle) changes with the same variables, optionally depending on the landcover the pig starts in ("on the floodplain, how does movement change at a high wet bulb temperature?");
- an **interactive habitat-selection map**: first as a local app, then as a static page on the website.

The empirical, binned versions of these questions are in `weather_analysis_acute_east_alligator.qmd` (R/ggplot). The model-based results should be directly comparable to them.

---

## 1. How to use this file

- Run **one stage per session**. Paste the stage's prompt (Section 6) into Claude Code from the `feral_pig_modelling` directory, then go through its **Done when** list.
- Every prompt begins by asking Claude to read Sections 2–5 of this file. Those sections hold the facts and decisions, so they don't have to be re-derived or re-argued.
- When a stage is done, add a line to the **Progress log** (Section 7): the date, commit hashes and anything that changed from the plan. If a decision changes, edit Sections 4–5 here too, so later stages build on the current version.
- Nothing is committed without asking.

---

## 2. Current state (verified 2026-11-17)

### Repos, environment and files

| What | Where |
|---|---|
| Analysis repo (this one) | `/Users/scottforrest/Library/Mobile Documents/com~apple~CloudDocs/github_repos/feral_pig_modelling` |
| deepSSF package source | `/Users/scottforrest/Library/Mobile Documents/com~apple~CloudDocs/github_repos/deepSSF_package` (`src/deepssf/`, `tests/`) |
| Python / kernel | conda env `deepssf`, `/opt/miniconda3/envs/deepssf/bin/python` (Jupyter kernel `deepSSF_kernel`) |
| Installed package | deepSSF 0.3.2. Currently a **regular, non-editable** install, so edits to `src/` don't reach the kernel until it's reinstalled. Stage 0 switches it to editable. |
| Fitted model | `outputs/landcover/2026-11-17_1/` holds `best_model.pt` (joint; use this by default), `best_model_habitat.pt`, `best_model_movement.pt` and `scalar_scaling.csv`. `outputs/` is gitignored. |
| Training notebook | `deepssf_piggies_train_landcover.ipynb`. Note: it now says `WINDOW_SIZE = 75` and `N_CONV_LAYERS_HAB = 4`, but the saved model was trained with **51** and **8**. |
| Prediction notebook | `deepssf_piggies_predict_landcover.ipynb`. Its setup cells (model load, weather encoding, `SCALING`, `make_scalar_row`, landscape prediction) are the starting point for inference. |
| Data | `data/pigs_east_alligator_steps_covs_20261015.csv`: 254,397 fixes, 19 pigs, 30-min fixes, Darwin time. 247,274 steps remain after weather NAs are dropped. |
| Rasters | `mapping/DEM_east_alligator_3112.tif`, `mapping/slope_east_alligator_3112.tif`, `mapping/landcover_dummy_east_alligator.tif` (10 one-hot bands). All 1335 × 1715 cells at 20 m, EPSG:3112. |
| Website | Quarto website, output to `docs/`, published by GitHub Pages from `main:/docs`. Notebooks are rendered from their **saved outputs**, not re-executed. `_quarto.yml` copies `outputs/**/*.html` as resources. |
| App environments | No streamlit/shiny/plotly/scipy in `deepssf`. `folium 0.20` is there. There's an old `deepSSF-app` env with streamlit 1.12 and torch 2.4; don't use it. |

### Model inputs (the order matters)

- **Raster channels (12)**, in `LAYER_PATHS` order:
  - `dem`, `slope`;
  - then the landcover bands: `veg_mudflat`, `veg_dry_rainforest`, `veg_eucalypt_open_forest`, `veg_eucalypt_woodland`, `veg_floodplain_sedgeland`, `veg_mangrove`, `veg_melaleuca_closed_forest`, `veg_melaleuca_open_forest`, `veg_melaleuca_woodland`, `veg_water`.
  - Each raster is min–max scaled on load by `load_environmental_layers`. Band names are in the GeoTIFF descriptions.
- **Scalar channels (13)**, the `SCALAR_COLS` order:
  - `hour_t1_sin1, hour_t1_cos1, yday_t1_sin1, yday_t1_cos1, dt_hour`. The encodings are sin/cos(2π·hour/24) and sin/cos(2π·yday/365.25); hour is Darwin local clock time. `dt_hour` is raw hours (median 0.5).
  - `onpli_rain_30min_mm_scaled, onpli_air_temp_c_scaled, onpli_wet_bulb_temp_c_scaled, onpli_dew_point_temp_c_scaled, onpli_rel_humidity_pct_scaled, onpli_wind_speed_kmh_scaled`. These are z-scores using `scalar_scaling.csv` (transform, mean, sd); rain is log1p'd first.
  - `onpli_wind_dir_sin1, onpli_wind_dir_cos1`. Wind direction is BoM: the direction the wind blows *from*, 0 = north, clockwise. The data uses 360 for north and **0 for calm**. Calm (speed 0) is encoded as (0, 0).
- **Architecture** (read with `params_from_checkpoint`): window `image_dim = 51`, pixel 20 m, `input_channels = 25`, `output_channels = 8`, `n_conv_layers_hab = 8`, `n_conv_layers_move = 4`, `dense_dim_in_all = 72`, `dense_dim_hidden = 128`, `num_movement_params = 12`.

### How the model works (line refs: deepSSF 0.3.2 plus the `scalar_cols` change, commit 2649889)

- **Scalars enter only the first conv layer of each head.** `ConvJointModel.forward` (`model.py:461-477`) tiles the scalars into constant channels (`Scalar_to_Grid_Block`), concatenates them **after** the rasters, and feeds the same tensor to `conv_habitat` and `conv_movement`.
  - Only the first conv layer of each head sees the scalar channels (`model.py:73-75`, `121`).
  - The movement fully connected layers (`fcn_movement_all`) get only the flattened conv output; there's no direct scalar input.
- **⇒ Habitat shortcut.** For any interior pixel, the scalars' contribution to the first habitat layer's pre-activation is a constant per output channel: `shift_c(s) = Σ_j s_j · Σ_kernel W[c, n_raster + j, :, :]`. That's a vector of length `output_channels` (8 here).
  - So the whole habitat surface depends on the scalars only through this short vector.
  - Compute the raster part of layer 1 once, then for each scenario add the shift and run the remaining layers. This is exact away from the zero-padded edge, which is masked anyway (`habitat_edge_buffer`).
- **Output.** `forward((spatial [B,C,D,D], scalars [B,S], bearing [B,1]))` returns `[B,D,D,2]`:
  - `[...,0]` is habitat log-probability and `[...,1]` is movement log-probability, each normalised over the window (`model.py:83`, `381`).
  - The joint next-step surface is `hab + move`, renormalised by the caller (as in the loss, the simulator and validation).
  - The spatial size must equal `image_dim`, because `Scalar_to_Grid_Block` expands to a fixed size.
- **Movement kernel** (`Params_to_Grid_Block_ChV`, `model.py:345-400`). The 12 raw outputs are:

  | Index | Parameter |
  |---|---|
  | 0 | gamma 1 shape |
  | 1 | gamma 1 scale (m) |
  | 2 | gamma 1 weight logit |
  | 3 | gamma 2 shape |
  | 4 | gamma 2 scale (×500, m) |
  | 5 | gamma 2 weight logit |
  | 6 | von Mises 1 mean offset (rad) |
  | 7 | von Mises 1 κ |
  | 8 | von Mises 1 weight logit |
  | 9 | von Mises 2 mean offset |
  | 10 | von Mises 2 κ |
  | 11 | von Mises 2 weight logit |

  - Positive parameters use `_positive(x) = exp(clamp(x, -10, 10))`. Weights are `log_softmax` over the pairs (2, 5) and (8, 11).
  - The von Mises means are offset by `bearing_tm1`. That is the **only** use of the previous bearing; it's not a network input.
  - The grid (`distance_layer`, `bearing_layer` buffers) is in metres. Bearing is 0 = east, counter-clockwise. The centre cell distance is set to 0.3826 × pixel.
  - The gamma log-density includes the polar→Cartesian 1/r Jacobian.
  - The kernel is **truncated to the window**: 500 m along the axes, about 707 m at the corners for 51 px × 20 m.
- **No helper returns the raw movement parameters.** The pieces can be called separately: `model.scalar_grid_output`, `model.conv_movement`, `model.fcn_movement_all`, `model.movement_grid_output(params, bearing)`, and its methods `_positive`, `_gamma_log`, `_vonmises_log`.
- **Package limits to know:**
  - `make_simulation_inputs` is fixed at 5 scalar columns (`simulate.py:72`), so `simulate_trajectory` and `simulate_trajectories_batched` **cannot** run this 13-scalar model. That's Stage 8.
  - `validate_next_step_probs` has stale default `scalar_cols`; it works when they're passed explicitly. It also rounds the target pixel where training floors it.
- **Reusable helpers:**
  - `predict_habitat_landscape(model, landscape_rasters, scalars, normalise, edge_buffer, chunk_rows)` and `habitat_edge_buffer(model)` (`predict.py`);
  - `_pad_landscape` / `_crop_windows` / `_pixel_from_coords` / `_coords_from_pixel` (`simulate.py:368-422`);
  - `subset_layer_vectorized` (`utils.py:226`), which is what `MovementDataset` uses (floor of the pixel, -1 padding);
  - `recover_hour` / `recover_yday` (`utils.py:296-309`);
  - `params_from_checkpoint`, `load_checkpoint`, `load_head_weights` (`train.py`);
  - `MovementDataset` (`data.py`);
  - `add_heatmap_overlay(fmap, counts, transform, src_crs, cmap, log_scale, vmin, vmax, opacity)` (`plot.py:306`), which reprojects a raster to Web Mercator for folium.

### Landcover: landscape share vs observed use

| Class (band) | % of landscape | % of fixes |
|---|---|---|
| Mudflat (1) | 7.3 | 1.7 |
| Dry rainforest (2) | 1.5 | 17.9 |
| Eucalypt open forest (3) | 27.9 | 23.8 |
| Eucalypt woodland (4) | 19.2 | 8.7 |
| Floodplain sedgeland (5) | 26.8 | 33.9 |
| Mangrove (6) | 1.7 | 0.7 |
| Melaleuca closed forest (7) | 1.7 | 4.9 |
| Melaleuca open forest (8) | 2.2 | 3.7 |
| Melaleuca woodland (9) | 2.7 | 4.6 |
| Water (10) | 9.0 | 0.0 (1 fix) |

**Colours, labels and binning to reuse** (from `weather_analysis_acute_east_alligator.qmd`):

- `veg_colours`: Dry Rainforest `#1B5E3A`, Eucalypt Open Forest `#B5772B`, Eucalypt Woodland `#E0B65C`, Floodplain Sedgeland `#D6E2A3`, Melaleuca Woodland `#A9CCE3`, Melaleuca Open Forest `#5B97BF`, Melaleuca Closed Forest `#2B5F8A`, Mangrove `#2A9D8F`, Mudflat `#8A7968`, Water `#264653`.
- The empirical plots drop `veg_water`.
- Weather labels come from `data/weather_variable_labels.csv`.
- Half-hourly variables use `bin_weather(x, n_bins = 10)`: quantile bins, a separate `"0"` bin when more than 10% of values are zero, and bins with n ≤ 50 dropped.
- Time of day is shown as `day_fraction` (0 = sunrise, 1 = sunset). The model uses clock hour, so mark the median sunrise and sunset for the season (from the CSV's `sunrise` / `sunset`) on hour plots.

---

## 3. Generality requirements (apply to every package function)

The package code must work for **any** deepSSF model and input set, not just this one. Nothing about the current model may be hard-coded.

- **Inputs.**
  - No fixed counts or names for raster bands or scalars.
  - Raster channel count = `conv_habitat.conv2d[0].in_channels − n_scalars`, checked against the rasters supplied. The scalar channels are the slice *after* the rasters.
  - Class or band names come from an argument or from the GeoTIFF descriptions.
  - Which bands form a one-hot categorical set is an argument (a list of channel indices or names).
- **Scalar encoding.** `ScalarEncoder` is driven by a spec, not by built-in variable names. Each entry maps natural input(s) to model column(s), with one of these types:
  - `cyclic`: period, e.g. hour/24 or yday/365.25 → sin and cos columns;
  - `linear`: transform `none`/`log1p`, mean, sd, read from a scaling table;
  - `circular_degrees`: → sin and cos, with an optional calm rule (e.g. calm when another input == 0, encoded as (0, 0));
  - `raw`: passthrough (e.g. `dt_hour`).
  - Column order comes from the `SCALAR_COLS` the model was trained with, and the spec must cover them exactly.
  - The pig-specific spec (onpli weather, the BoM calm rule) is built in the notebook and saved as JSON next to the checkpoint.
- **Window size.** Read from the model (`movement_grid_output.distance_layer.shape`, or `params.image_dim`); never assume 51. Pixel size comes from the raster transform or the grid buffers. The availability-mask radius is an argument whose default the caller derives from the window.
- **Model size.**
  - Conv layer counts, channel counts, kernel size and padding are read from the module structure.
  - The fast habitat path loops over whatever is in `conv_habitat.conv2d`, uses `habitat_edge_buffer(model)`, and computes the shift for any number of output channels.
  - Movement decoding reuses `model.movement_grid_output`'s own methods and buffers rather than copies of them. It checks `num_movement_params` against the layout it decodes, and raises a clear error if the kernel definition is different.
- **Analysis choices are arguments.** That includes PDP variables, grids, ICE sample size, feature-importance groups (`dict[str, list[str]]` of scalar columns or raster channel names), quantile bins and the device.
- **Tests** (`tests/test_inference.py`):
  - Parametrise over toy `ConvJointModel`s of at least two shapes, e.g. windows 21 and 33, 1 vs 3 habitat conv layers, 4 vs 16 channels, 1 vs 5 raster bands, 3 vs 7 scalars.
  - Check each function against brute-force `model.forward` / `predict_habitat_landscape` results.
  - Look at how `tests/test_deepssf.py` builds toy models and reuse that.

---

## 4. Method decisions

1. **Habitat selection metric (per class, per scenario `s`).**
   - Predict the habitat surface over the landscape and keep only cells inside an **availability mask**: cells within the window half-width (500 m) of any observed fix, made by dilating a raster of fixes with a torch max-pool. Also drop the masked edge buffer.
   - Then, for each class k:
     - `use_k(s)` is the share of predicted probability in class k;
     - `avail_k` is class k's share of the mask;
     - the **selection ratio** is `SR_k(s) = use_k / avail_k`, plotted as `log SR`, where 0 means the class is used in proportion to its availability.
   - This averages over the real spatial contexts each class occurs in, which is the PDP marginalisation over the landscape.
   - Keep water in tables but leave it out of most plots (1 fix; the empirical analyses drop it).
2. **Fast habitat path.** Use the Section 2 shortcut on a crop of the mask's bounding box plus the edge buffer. Before it's used anywhere, it must match `predict_habitat_landscape` on interior cells (max abs difference < 1e-4 in log-probability).
3. **Correlated weather.** Air temperature, wet bulb, dew point and humidity are physically linked, so a classic PDP that varies one of them alone feeds the model impossible weather. Use three complementary views:
   - **PDP + ICE with physically consistent perturbations.** Vary *air temperature at fixed dew point*, or *humidity at fixed temperature*, and recompute the linked variables. Dew point uses Magnus: `γ = ln(RH/100) + a·T/(b+T)`, `Td = b·γ/(a−γ)`, a = 17.62, b = 243.12. Wet bulb uses Stull (2011): `Tw = T·atan(0.151977·(RH+8.313659)^0.5) + atan(T+RH) − atan(RH−1.676331) + 0.00391838·RH^1.5·atan(0.023101·RH) − 4.686035`. Validate both against the observed columns first (Stage 2). Also offer a "heat-stress" axis that moves temperature and humidity together along the observed wet-bulb relationship.
   - **ALE** (accumulated local effects) for each thermal variable. It uses only observed rows within each bin, so it avoids extrapolation.
   - **M-plot:** model predictions averaged over *observed* rows in `bin_weather`-style quantile bins. This is directly comparable to the empirical binned plots, but it mixes in correlated variables.
4. **Time.**
   - Vary hour, day of year and wind direction as their sin/cos pairs. Fix `dt_hour` at 0.5.
   - Time and weather are correlated (nights are cooler), so show **both**:
     - the PDP of hour at observed weather, the "pure clock" effect;
     - the M-plot by hour, the "typical diel cycle" including typical weather.
   - Say which is which on every figure.
5. **ICE base rows.** A fixed-seed sample of observed steps, stratified by month × 3-hour block. Start at M = 50 rows and size it from the Stage 1 timing. Natural-unit rows are encoded with `ScalarEncoder`.
6. **Feature importance.**
   - **Grouped permutation importance:** the increase in negative log-likelihood (NLL) when a group is shuffled across steps, reported **per head** (habitat, movement, joint).
   - It's computed on a fixed-seed sample of steps (e.g. 20k), with a few repeats.
   - Groups:
     - time of day (hour sin/cos)
     - day of year
     - `dt_hour`
     - rain
     - thermal (air temp, wet bulb, dew point, RH)
     - wind (speed + direction sin/cos)
     - DEM
     - slope
     - landcover (all 10 bands)
   - Raster groups are shuffled by swapping those channels' windows between steps, keeping each step's target.
   - Report individual variables too, noting that the thermal variables' individual values are unreliable because they're correlated.
   - The sample includes training steps (the training split was random), so this measures how much the *model relies* on each input, not out-of-sample generalisation.
   - A secondary, cheap measure: the variance of each class's PDP.
7. **Movement.**
   - Extract the movement parameters for batches of **real observed windows** (spatial context + observed scalars).
   - Summarise each kernel **exactly from the grid**, instead of sampling n steps:
     - expected step length, SD and quantiles;
     - P(stay), i.e. r < 1 pixel;
     - the turning-angle mean resultant length relative to `bearing_tm1`;
     - the probability mass in the outer ring (a truncation warning).
   - Also report the untruncated gamma-mixture moments (`mean = Σ w·k·θ`) and the analytic von Mises resultant (`A(κ) = i1e(κ)/i0e(κ)`), and explain how they differ from the grid versions.
   - Repeat for the **joint** next-step surface: the realised step length, and the probability that the next step lands in each class.
   - "On the floodplain" means windows centred on fixes whose class is floodplain sedgeland.
   - Sampling n steps is only for illustrative histograms.
8. **Calibration checks.**
   - The model's expected step length by hour, against the observed mean and median `sl_` by hour.
   - The probability integral transform (PIT) of observed step lengths under the joint kernel: a randomised PIT on the discrete grid should look uniform.
   - Report both before interpreting any movement PDPs.
9. **Outputs.**
   - CSV + PNG in `outputs/landcover/<run>/inference/<stage>/` (gitignored).
   - Figures are also kept in the notebook outputs, so the website renders them.
   - Plots use matplotlib and the `veg_colours` palette.

---

## 5. Conventions

- **Package code.** Goes in `src/deepssf/inference.py`, with public names exported in `src/deepssf/__init__.py`.
  - Docstrings follow the existing NumPy style.
  - Tests go in `tests/test_inference.py`.
  - Add a CHANGELOG `## [Unreleased]` entry for each stage.
  - Run `PYTHONPATH=src /opt/miniconda3/envs/deepssf/bin/python -m pytest tests -q` (or plain pytest once the editable install is in place).
- **Analysis notebook.** `deepssf_piggies_inference_landcover.ipynb` in this repo, one `#` section per stage.
  - If it grows past about 20 MB, split the movement work into `deepssf_piggies_inference_movement.ipynb`, with the same setup cells.
  - Setup cells come from `deepssf_piggies_predict_landcover.ipynb`.
- **Notebook editing.** Notebooks may be open in VS Code. Edit cells **by content match** through the JSON (`json.load` → modify → `json.dumps(nb, indent=1, ensure_ascii=False) + "\n"`), which round-trips byte-for-byte. Never `Read` a large `.ipynb` directly; print cell sources with Python instead.
- **Testing notebook code headlessly.** Concatenate the code cells into a script, use the matplotlib `Agg` backend, and replace `plt.savefig` with a no-op unless outputs are wanted. If the model gets retrained, update `OUTPUT_DIR` and `WINDOW_SIZE`.
- **App.** `apps/` in this repo.
- **Package version.** Stays 0.3.2 until a release; the version bump happens then.
- **End of each stage:** run pytest and the notebook headlessly, report results and figures, add a Progress log line, and ask before committing.

---

## 6. Stage prompts

### Stage 0: Setup

```text
Read deepSSF_inference.md Sections 1–5 and the Progress log first.

Stage 0 (setup):
1. Switch the deepssf env to an editable install of the package:
   /opt/miniconda3/envs/deepssf/bin/pip install --no-deps -e "<deepSSF_package path>"
   Confirm deepssf.__file__ points into src/, and that the full pytest suite passes.
2. Create src/deepssf/inference.py (module docstring that summarises the Section 2
   "How the model works" facts it relies on) and tests/test_inference.py with a
   parametrised toy-model fixture covering at least two shapes (Section 3).
   Export nothing yet.
3. Create deepssf_piggies_inference_landcover.ipynb in feral_pig_modelling.
   - Front matter: title "deepSSF Inference".
   - Copy the setup cells from deepssf_piggies_predict_landcover.ipynb by content
     match: imports, seed/device, paths (OUTPUT_DIR = outputs/landcover/2026-11-17_1),
     data, layers, SCALAR_COLS, prepare_movement_df, the NA drop, wind encoding,
     SCALING load, WINDOW_SIZE = 51, PIXEL_SIZE, window filter,
     params_from_checkpoint, model and checkpoint load, get_landscape.
   - Add INFERENCE_DIR = OUTPUT_DIR / "inference".
4. Time predict_habitat_landscape for one scenario (3 warm-up runs, then 5 timed
   runs on the device) and print the per-map time. Record it in the Progress log.

Done when: editable install confirmed; the full pytest suite passes; the notebook
runs headlessly end to end; the timing baseline is logged.
Stop, report and ask before committing.
```

### Stage 1: Scalar encoding and the fast habitat path

```text
Read deepSSF_inference.md Sections 1–5 and the Progress log first.
Follow the Section 3 generality requirements for all package code.

Stage 1, package (src/deepssf/inference.py):
1. ScalarEncoder(spec, scalar_cols):
   - Spec entries of type cyclic / linear / circular_degrees / raw, as in Section 3.
   - encode(natural: DataFrame | dict) -> float32 array [N, len(scalar_cols)] in
     scalar_cols order.
   - decode(encoded) -> DataFrame of natural units (cyclic -> hour/yday via
     atan2; circular_degrees -> 0–360 with NaN for calm; linear -> unscaled with
     the inverse transform).
   - Optional per-input training ranges, with check_ranges() returning
     out-of-range warnings.
   - to_json() / from_json(); a classmethod that builds the linear entries from a
     scaling table (columns: column, transform, mean, sd; model column = column
     + suffix).
   - Validation: the spec must cover scalar_cols exactly (clear error listing
     missing/extra columns).
2. crop_windows(rasters, xs, ys, transform, window_size, fill=-1.0) -> Tensor
   [N, C, ws, ws]:
   - Public, batched; rasters as a list of 2-D tensors or a [C, H, W] tensor.
   - Must reproduce MovementDataset's crop exactly (floor of pixel, -1 padding).
     Test against MovementDataset items.
3. HabitatLandscapePredictor(model, landscape_rasters, n_scalars, mask=None,
   crop_to_mask=True, device=None):
   - Precompute the raster part of conv_habitat.conv2d[0] once (cropped to the
     mask's bounding box plus the edge buffer).
   - predict(encoded_scalars [S] or [B, S], normalise=True) -> [B, H, W]
     log-probability, NaN outside the valid area (edge buffer, mask).
   - Normalised over the valid cells, as predict_habitat_landscape does.
   - Batch over scenarios with a memory-aware chunk size.
   - Derive the layer count, channels, kernel/padding and edge buffer from the model.
4. Tests (several toy shapes):
   - encode/decode round trip (including calm and the 360/0 wrap);
   - spec validation errors;
   - crop_windows vs MovementDataset;
   - HabitatLandscapePredictor vs predict_habitat_landscape (interior max abs
     diff < 1e-5 on CPU), with and without a mask;
   - batch == loop.

Stage 1, notebook:
5. Build the pig spec:
   - TIME_COLS cyclic: hour period 24, yday period 365.25; dt_hour raw.
   - The 6 weather columns linear from scalar_scaling.csv (suffix "_scaled").
   - Wind direction circular_degrees with calm when onpli_wind_speed_kmh == 0.
   - Training ranges from step_df.
   Save it to OUTPUT_DIR / "scalar_spec.json".
6. Check that the encoder rebuilds the model inputs of 200 random training rows
   exactly (atol 1e-5), replacing make_scalar_row.
7. Check HabitatLandscapePredictor against predict_habitat_landscape on the real
   model for 3 scenarios (max abs diff on interior cells; must be < 1e-4 on the
   device). Report the speed-up per scenario and for a batch of 16.

Done when: tests pass for at least two toy model shapes; the full suite passes;
the real-model identity check passes; the timings are logged.
Stop, report and ask before committing.
```

### Stage 2: Landcover selection metric

```text
Read deepSSF_inference.md Sections 1–5 and the Progress log first.
Follow the Section 3 generality requirements for all package code.

Stage 2, package:
1. availability_mask(raster_shape, transform, xs, ys, radius_px) -> bool [H, W].
   Rasterise the points and dilate with a torch max-pool (square, or a disc via
   distance). The caller derives radius_px, by default from the window
   half-width.
2. class_selection(log_surfaces [B, H, W] or [H, W], class_rasters [K, H, W]
   (one-hot), class_names, mask=None, scenario_ids=None) -> long DataFrame with
   columns scenario, class, use, availability, selection_ratio, log_sr.
   - Use NaN-aware sums over the valid cells.
   - Classes with zero availability get NaN.
3. Tests: hand-built surfaces with known answers (a uniform surface gives SR = 1
   everywhere; all mass in one class), plus mask handling.

Stage 2, notebook:
4. Psychrometric helpers (notebook-level; they're domain-specific): Magnus dew
   point and Stull wet bulb from air temperature + RH (Section 4.3).
   - Validate against onpli_dew_point_temp_c and onpli_wet_bulb_temp_c: RMSE, bias,
     and a scatter plot.
   - If wet bulb is biased, fit a small linear correction and say so.
5. Build the availability mask from the observed fixes with a radius of the
   window half-width. Plot it. Report class availability inside the mask vs
   across the whole landscape.
6. Reference scenarios:
   - weather at the median of observed rows (keep the thermal variables
     consistent by taking a real observed row near the medians);
   - yday 200; hours 03:00, 09:00, 15:00, 21:00.
   Show a bar chart of log SR per class (veg_colours; water in the table only).
7. Sanity check against empirical use/availability: the share of fixes per class
   in the mask vs availability. Show model log SR (averaged over a sample of
   observed scenarios) vs empirical log SR as a scatter plot with a 1:1 line.
   Expect dry rainforest high and water/mudflat low.

Done when: tests pass for at least two toy model shapes; the psychrometric RMSE
is reported; the reference and empirical comparison figures exist; CSVs are
saved to INFERENCE_DIR / "stage2".
Stop, report and ask before committing.
```

### Stage 3: Habitat partial dependence

```text
Read deepSSF_inference.md Sections 1–5 and the Progress log first (use the
Stage 1 timings to size grids). Follow the Section 3 generality requirements for
all package code.

Stage 3, package:
1. partial_dependence(metric_fn, base_rows, set_fn, grid, encoder,
   batch_size=...) -> (pd_df, ice_df):
   - base_rows are natural-unit DataFrame rows.
   - set_fn(rows, value) returns modified natural rows; this is where coupled
     variables are recomputed.
   - metric_fn(encoded [B, S]) returns a DataFrame or array of metrics per row
     (e.g. log SR per class from HabitatLandscapePredictor + class_selection).
   - Returns the mean (PDP) and per-row (ICE) values, with grid values in
     natural units.
2. accumulated_local_effects(metric_fn, rows, variable, set_fn, encoder,
   n_bins=10) -> DataFrame. Quantile bins; local differences between bin edges;
   accumulated and centred.
3. binned_average(metric_fn, rows, variable, encoder, n_bins=10,
   zero_bin_threshold=0.1, min_n=50) -> the M-plot, mirroring the qmd's
   bin_weather.
4. Optional: h_statistic(metric_fn, rows, var_a, var_b, ...) (Friedman's H).
5. Tests: on a toy metric_fn with a known additive/interaction structure, PDP,
   ALE and H recover it; set_fn coupling is respected.

Stage 3, notebook (metric = log SR per class over the availability mask):
6. 1-D PDP + ICE, faceted by class (as facet_wrap(~veg_label) in the qmd;
   water omitted), with a rug of observed values:
   - hour (0–24, 30-min grid);
   - day of year (monthly grid);
   - rain;
   - wind speed;
   - wind direction (degrees, calm shown separately);
   - air temperature at fixed dew point;
   - RH at fixed temperature;
   - the "heat-stress" axis.
   Mark median sunrise and sunset on the hour plots.
7. ALE for air temp, wet bulb, dew point and RH. Then M-plots for every weather
   variable and for hour, labelled "observed-condition averages, comparable to
   the empirical plots".
8. 2-D PDP heatmaps per class: hour × wet bulb (heat-stress axis), hour × day of
   year, hour × rain. Use a diverging colour map centred on 0 log SR.
9. Optional: H for (hour, heat stress), (hour, yday), (hour, rain).
10. Save tidy CSVs (variable, value, class, pd, ice_id, ...) and PNGs to
    INFERENCE_DIR / "stage3". Print a short written summary of the strongest
    effects (largest log SR range per class and variable).

Done when: tests pass for at least two toy model shapes; all figures render; the
CSVs are saved; the runtime is within the budget logged in Stage 1.
Stop, report and ask before committing.
```

### Stage 4: Feature importance

```text
Read deepSSF_inference.md Sections 1–5 and the Progress log first.
Follow the Section 3 generality requirements for all package code.

Stage 4, package:
1. step_nll(model, spatial, scalars, bearing, targets) -> dict of per-step NLL
   for habitat, movement and joint (joint = -log_softmax(hab + move) at the
   target).
2. permutation_importance(model, dataset, groups, scalar_cols,
   raster_channel_names, n_repeats=5, batch_size=..., seed=..., device=...)
   -> DataFrame with columns group, head, baseline_nll, permuted_nll, increase,
   sd.
   - dataset is a MovementDataset (or a subset of one).
   - groups is a dict of name -> list of scalar column or raster channel names.
   - Shuffle a group by applying one random permutation of steps to all of the
     group's channels together.
   - For raster groups, swap those channels' windows between steps and keep each
     step's target.
3. Tests:
   - permuting a channel the toy model ignores (zeroed weights) gives ~0 increase;
   - a group permutation keeps within-group pairing;
   - deterministic under a fixed seed.

Stage 4, notebook:
4. Build a MovementDataset over a fixed-seed sample of 20k steps (from step_df
   after the window filter), using the Section 4.6 groups plus every individual
   scalar.
5. Plot grouped importance per head (horizontal bars with SD), then individual
   variables with the correlation caveat.
6. PDP-variance importance per class from the Stage 3 PDPs, as a heatmap of
   variable × class.
7. Save CSVs and PNGs to INFERENCE_DIR / "stage4". Write a short interpretation.

Done when: tests pass for at least two toy model shapes; the importance figures
exist; repeats give stable rankings (report the SD).
Stop, report and ask before committing.
```

### Stage 5: Movement kernel

```text
Read deepSSF_inference.md Sections 1–5 and the Progress log first.
Follow the Section 3 generality requirements for all package code.

Stage 5, package:
1. movement_parameters(model, spatial [B, C, D, D], scalars [B, S]) -> dict:
   - "raw" [B, P];
   - decoded tensors per component (gamma shapes, scales (m), weights; von Mises
     mean offsets, kappas, weights).
   Run model.scalar_grid_output -> cat -> conv_movement -> fcn_movement_all, as
   forward does. Decode with model.movement_grid_output's own _positive and the
   x500 scale factor read from the code path (not duplicated as magic numbers
   where avoidable). Validate the parameter count and raise a clear error for an
   unknown kernel layout.
2. kernel_summary(log_kernel [B, D, D], distance_grid, bearing_grid,
   bearing_tm1=None, quantiles=(0.25, 0.5, 0.75, 0.9), stay_radius=None,
   class_windows=None) -> DataFrame:
   - expected distance, SD, quantiles (from the cumulative mass sorted by
     distance), P(stay);
   - mass in the outermost ring (truncation flag);
   - turning mean resultant length and mean turning angle relative to
     bearing_tm1;
   - optionally the probability of the next step in each class (class_windows
     [B, K, D, D]).
   Works for a movement-only kernel or a joint kernel. Distance and bearing grids
   come from model.movement_grid_output buffers.
3. mixture_moments(decoded) -> the untruncated gamma-mixture mean/SD and the
   von Mises mixture resultant (i1e/i0e).
4. Tests:
   - movement_parameters reproduces forward()'s movement channel when passed
     through movement_grid_output;
   - kernel_summary on synthetic kernels (point mass, uniform) gives known
     values;
   - several toy shapes and windows.

Stage 5, notebook:
5. Calibration, on a fixed sample of observed steps (windows via crop_windows,
   observed scalars, bearing_tm1):
   - model-expected step length (movement-only and joint) by hour, against the
     observed mean/median sl_;
   - a randomised PIT histogram of observed step lengths under the joint kernel.
   Report before interpreting anything else.
6. Movement PDP/ICE (windows = real observed windows, stratified by starting
   class: floodplain sedgeland, eucalypt open forest, eucalypt woodland, dry
   rainforest, melaleuca classes pooled):
   - expected step length, P(stay) and turning resultant length;
   - against hour, day of year, rain, wind, the temperature/humidity axes and
     heat stress;
   - lines coloured by starting class.
   Also 2-D: hour × heat stress for expected step length, per starting class.
7. Next-step landcover probabilities (joint kernel) vs hour and heat stress, by
   starting class: "where does the next step go".
8. Example kernel plots for contrasting scenarios (e.g. floodplain at 06:00
   cool/dry vs 15:00 hot/humid): habitat, movement and joint windows side by
   side, with the observed step length distribution overlaid on the
   kernel-implied one. Sampling n steps is only for these illustrations.
9. Save CSVs and PNGs to INFERENCE_DIR / "stage5", with a short interpretation.

Done when: tests pass for at least two toy model shapes; calibration is reported;
the movement PDP figures exist.
Stop, report and ask before committing.
```

### Stage 6: Local interactive app

```text
Read deepSSF_inference.md Sections 1–5 and the Progress log first.

Stage 6 (local Streamlit app, in feral_pig_modelling/apps/):
1. Install streamlit and streamlit-folium into the deepssf env (pip). Don't use
   the old deepSSF-app env. Record the versions in the Progress log.
2. apps/habitat_selection_app.py. Use st.cache_resource for loading:
   - LAYER_PATHS, the checkpoint (params_from_checkpoint + load_checkpoint),
     ScalarEncoder.from_json(OUTPUT_DIR / "scalar_spec.json");
   - HabitatLandscapePredictor over the availability mask (or the whole
     landscape, toggle);
   - the observed fixes (thinned for display).
   The run directory is a sidebar text input defaulting to
   outputs/landcover/2026-11-17_1.
3. Inputs:
   - hour (0–24, 0.5 step); day of year (or a date picker);
   - rain; wind speed; wind direction (with a calm checkbox);
   - air temperature + RH, with dew point and wet bulb derived using the
     validated psychrometric helpers (move them into apps/weather_utils.py and
     import them in both the notebook and the app);
   - alternatively, "use the observed weather at <date-time>", picked from the
     CSV.
   Show out-of-range warnings from encoder.check_ranges().
4. Outputs:
   - a folium map with the habitat surface overlay (reuse add_heatmap_overlay,
     or the same reprojection code, with percentile colour limits; a toggle
     between per-scenario and fixed limits for comparing scenarios), observed
     fixes and a layer control;
   - a log SR bar chart per class (veg_colours);
   - the encoded scalar row in an expander.
5. Optional panel: click the map -> show the habitat, movement and joint kernels
   for a window centred on the clicked point, with kernel_summary numbers.
6. Keep the app pig-specific but thin: all model logic comes from deepssf
   functions. Add apps/README.md with run instructions:
   /opt/miniconda3/envs/deepssf/bin/streamlit run apps/habitat_selection_app.py

Done when: the app starts; changing an input updates the map in about 1 s on
MPS; the map overlay aligns with the observed fixes; there are no console
errors. Take a screenshot.
Stop, report and ask before committing.
```

### Stage 7: Static website version

```text
Read deepSSF_inference.md Sections 1–5 and the Progress log first.

Stage 7a (do first: precomputed maps on the website):
1. Pick a preset grid with the user before computing, e.g. hours
   {0,3,...,21} × seasons {early dry, late dry, early wet, late wet}
   (representative ydays) × weather presets {cool/dry, typical, hot/humid, rain}
   taken from real observed conditions for each season/hour.
   Budget: total assets ≤ ~30 MB. Choose the overlay crop (availability mask
   bounding box), resolution and format (WebP/PNG) to fit.
2. Compute the overlays with HabitatLandscapePredictor in Web Mercator (the same
   reprojection as add_heatmap_overlay), with one fixed colour scale across all
   presets so they're comparable. Save the images plus a manifest.json (preset ->
   file, bounds, inputs, log SR per class).
3. Build a standalone Leaflet HTML page with selectors/sliders that swap the
   image overlay, and a log SR bar chart per class from the manifest (plain JS;
   any charting library loaded from a CDN with a pinned version). Put it under
   outputs/landcover/<run>/web/. Add a Quarto page,
   deepssf_habitat_explorer.qmd, that embeds it via an iframe. Update
   _quarto.yml resources so the images and manifest are copied (the current
   pattern only copies outputs/**/*.html). Add the page and the inference
   notebook to the deepSSF sidebar.
4. Render only the affected pages (quarto render <file>) and check the docs/
   output loads locally (python -m http.server in docs/).

Stage 7b (optional: live model in the browser):
5. Export conv_habitat layers 2..N to ONNX (needs onnx + onnxruntime for
   checking). Ship the precomputed layer-1 raster part (float16, cropped to the
   mask) and the scalar-shift weights. In JS, encode the inputs (port the
   ScalarEncoder spec from scalar_spec.json), add the shift, run onnxruntime-web
   (WebGPU, falling back to WASM), normalise and colour. Check that the browser
   output matches Python for 3 scenarios (max abs diff < 1e-3). Watch the asset
   size; downsampling is NOT allowed (it changes what the CNN sees), only
   cropping.

Done when (7a): the page works on the local docs/ build with the overlay aligned
and the presets switching; the asset budget is met.
Stop, report and ask before committing or pushing.
```

### Stage 8 (optional): Simulation with arbitrary scalars

```text
Read deepSSF_inference.md Sections 1–5 and the Progress log first.
Follow the Section 3 generality requirements for all package code.

Stage 8, package:
1. Generalise make_simulation_inputs / simulate_trajectory /
   simulate_trajectories_batched / simulate_trajectories to accept an optional
   scalar_fn(step_index, hour, yday) -> full encoded scalar row (or a
   precomputed [n_steps, S] array). Keep the current 5-column behaviour as the
   default for backward compatibility. Validate the scalar count against the
   model's input channels.
2. Fix validate_next_step_probs's stale default scalar_cols (require it
   explicitly, or default from the dataset) and make the target pixel use floor,
   as in training. Note the behaviour change in CHANGELOG.
3. Tests: old 5-scalar calls are unchanged (seeded output identical); new
   13-scalar toy-model simulations run; scalar-count mismatch errors are clear.

Stage 8, notebook:
4. Simulate trajectories under weather scenarios (e.g. a typical dry-season day
   vs a hot/humid day, using observed half-hourly sequences from real dates via
   ScalarEncoder) from observed start points. Compare emergent space use
   (trajectory_heatmap), daily distance and landcover use with the observed data.

Done when: the full suite passes; scenario simulations run and are compared with
observed summaries.
Stop, report and ask before committing.
```

---

## 7. Progress log

| Date | Stage | Commits (package / analysis repo) | Notes (timings, deviations, decisions changed) |
|---|---|---|---|
| 2026-11-17 | 0 | not yet committed | Editable install done (`deepssf.__file__` → `deepSSF_package/src/deepssf/`); full suite 156 passed (150 before + 6 new). Toy shapes: (window 21, hab 1, move 2, 4 ch, 1 raster, 3 scalars) and (33, 3, 3, 16, 5, 7). **Timing baseline:** `predict_habitat_landscape`, full 1335 × 1715 landscape, MPS: **≈0.06 s per map** (two runs: mean 0.059 and 0.064 s, SD ≈0.001 s; first warm-up ≈0.4 s). Timings CSV: `inference/stage0/`. Deviations: the notebook keeps only the imports it uses (dropped networkx, base64, IPython display, Affine, LogNorm); the timing scenario is one observed step's inputs, not `make_scalar_row`. |
