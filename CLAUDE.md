# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Contract analysis of GPS-collared feral pigs in Kakadu National Park, NT. It runs
as a pipeline of analysis documents rather than as a package. There is no build, lint or
test suite. R (Quarto) handles data cleaning, covariate preparation and mapping. Python
(Jupyter) fits a deep learning step selection function (deepSSF), then uses it to simulate
movement and to estimate connectivity.

## Pipeline and data flow

The stages hand off through dated files in `data/`, `mapping/` and `outputs/`. **All three
directories are gitignored**, along with `*.pdf`, `*.pptx` and `andrew_20260822/`, so only
the code is under version control. Every path is relative to the repo root, so run things
from there.

1. **`data_preparation.qmd`** (R, `amt`): raw Lotek GPS CSV (`data/GPS_*.CSV`) → cleaned
   steps → `data/pig_all_steps_<YYYYMMDD>.csv`. CRS is GDA94 / Geoscience Australia
   Lambert, **EPSG:3112**. Step-start coordinates are `x1_`/`y1_`, time is `t1_`, and the ID
   column is `name`.
2. **`image_download_GEE.ipynb`** (Python, `ee`/`geemap`, separate `counting_waterholes`
   kernel): builds the AOI from the tracks and exports a Sentinel-2 median, NDVI/NDWI/NDMI
   indices and AlphaEarth embeddings to Google Drive. They are then moved by hand into
   `mapping/`.
3. **`data_exploration.qmd`** and **`data_exploration_east_alligator.qmd`** (R,
   `terra`/`sf`): exploration and plots. They also **write derived rasters** to `mapping/`
   (`DEM_3112*.tif`, `slope_3112*.tif`, the cropped `feral_pig_aoi_alphaearth_2025_crop.tif`,
   the water layer) and the covariate CSV `data/pig_all_steps_covs_subset_20260813.csv`.
   Near their ends they read simulation and betweenness TIFs back from
   `outputs/alphaearth/<run>/`, then crop, water-mask and log-transform them into `mapping/`
   for QGIS (`mapping/pig_mapping.qgz`). The `_east_alligator` version is the newer,
   actively edited copy, restricted to the East Alligator subset.
4. **`deepssf_piggies_train.ipynb`**: trains `ConvJointModel`. Its outputs go to
   `get_output_dir(base="outputs/alphaearth")`, which creates a new
   `outputs/alphaearth/<date>_<n>/` per run (checkpoints `best_model.pt`,
   `best_model_habitat.pt`, plus figures).
5. **`deepssf_piggies_predict.ipynb`**: loads a training run (the `OUTPUT_DIR` /
   `CKPT_PATH` cell; currently `outputs/alphaearth/2026-08-17_4`). It produces habitat
   selection surfaces, simulates trajectories (including releases at the aerial-survey
   locations in `data/KNP SS 250809_survey_pig.csv`), and builds heatmaps, folium maps and
   betweenness connectivity. All outputs go into that same run directory.
6. **`pig_predictions.qmd`** (R, `move2`/`moveVis`): animates the observed and simulated
   trajectories into `outputs/moveVis/`.

`deepssf_piggies.ipynb` is the older combined train+predict notebook, which was split into
the two notebooks above. Edit the split versions.

When one stage's output file name changes (they carry dates), update the hardcoded paths in
the downstream stages to match.

## The `deepssf` package lives elsewhere

The notebooks import nearly all model and simulation logic (`ConvJointModel`, `fit`,
`simulate_trajectories`, `trajectory_heatmap`, `load_environmental_layers`, and so on) from
`deepssf`. It is an **editable install** of the sibling repo
`../deepSSF_package/src/deepssf/`. To change library behaviour, edit it there. The
notebook cells mostly set parameters and call it.

- Python env: conda env `deepssf` (`/opt/miniconda3/envs/deepssf/bin/python`), Jupyter
  kernel `deepssf_kernel`. To check an import:
  `/opt/miniconda3/envs/deepssf/bin/python -c "import deepssf; print(deepssf.__file__)"`.
- `scipy` and `python-igraph` are **not installed** in that env. Betweenness uses pure
  `nx.betweenness_centrality`, so avoid any sparse-backed networkx call.
- In `load_environmental_layers()`, the `LAYER_PATHS` keys `s2` and `s2_dir` are reserved.
  Give a multi-band GeoTIFF any other key name.
- The training and prediction notebooks must agree on `LAYER_PATHS`, `SCALAR_COLS`,
  `WINDOW_SIZE` and `PIXEL_SIZE`, because the predict notebook rebuilds the model from the
  checkpoint (`params_from_checkpoint()`).

## Working with the notebooks

- `deepssf_piggies_predict.ipynb` (~390 MB) and `deepssf_piggies.ipynb` are too large for
  Read/NotebookEdit because of embedded outputs. `deepssf_piggies_train.ipynb` (~30 MB) is
  workable but heavy. Read cell sources with
  `json.load(open(f))['cells'][i]['source']`. To edit, rewrite the JSON with
  `json.dumps(nb, indent=1, sort_keys=True, ensure_ascii=False) + "\n"`, which round-trips
  byte for byte and avoids huge git diffs. Write to a temp file, verify that the untouched
  cells are unchanged, then `os.replace()`.
- Cells refer to each other by index (for example in `betweenness_connectivity.md`), so
  inserting cells shifts those references.
- `betweenness_connectivity.md` documents the betweenness section of the predict notebook:
  method, parameters, runtime and step-budget trade-offs, and sanity checks. Read it before
  touching that section. When testing, smoke-test against the small 1000-step trajectory
  CSV rather than the 4 GB one.

## Rendering

- Quarto: `quarto render data_preparation.qmd` (HTML; PDF formats are also declared). The
  docs use `paperpile.bib`. They are normally run interactively in RStudio
  (`feral_pig_modelling.Rproj`) or VS Code. The leaflet maps are thinned and drawn with
  `leafgl`, because ~246k SVG markers lock up the viewer.
- Notebooks run top to bottom in the `deepssf` kernel. Training picks `cuda`/`mps`/`cpu`
  via `get_device()`, and seeds are fixed at 42 so that runs are comparable.
