# Landscape connectivity via betweenness

Notes on the betweenness feature added to [`deepssf_piggies_predict.ipynb`](deepssf_piggies_predict.ipynb),
what it does, and how to re-run it.

## Why

The notebook already summarises simulated trajectories as a **heatmap** — simulated
locations per cell, i.e. *where the pigs were*. Betweenness answers a different question:
*which cells does movement have to pass through*. A large open area can hold many
locations and almost no betweenness; a narrow crossing can hold few locations and a great
deal of it. The two surfaces are related but not the same — on the 599-trajectory /
1,000-step run they correlate at only rho ~ 0.55 across the cells they share.

Following:

- Bastille-Rousseau, G., Douglas-Hamilton, I., Blake, S., Northrup, J. M., & Wittemyer, G.
  (2018). Applying network theory to animal movements to identify properties of landscape
  space use. *Ecological Applications*, 28(3), 854–864. <https://doi.org/10.1002/eap.1697>
- Hofmann, D. D., Cozzi, G., McNutt, J. W., Ozgul, A., & Behr, D. M. (2023). A three-step
  approach for assessing landscape connectivity via simulated dispersal: African wild dog
  case study. *Landscape Ecology*, 38(4), 981–998.
  <https://doi.org/10.1007/s10980-023-01602-4>

Adapted from Hofmann's
[`09_Betweenness.R`](https://github.com/DavidDHofmann/DispersalSimulation/blob/main/AWDCaseStudy/02_Analysis/09_Betweenness.R)
(R + `igraph`), reimplemented in Python + `networkx`.

## Method, and where each step lives

| # | Step (R script) | Notebook |
|---|---|---|
| 1 | Aggregate a reference raster; each cell centre is a node (`aggregate`, `r[] <- 1:ncell(r)`) | cell 73 — `bet_height`/`bet_width`/`bet_transform`/`N_NODES` |
| 2 | Densify each path, read off the cell under every point (`interpolatePath`, `extract`) | cell 73 — `path_cells()` |
| 3 | Cell-to-cell transitions, one per track per pair (`visitHist(singlecount = TRUE)`) | cell 73 — `transition_counts()` |
| 4 | Pool over tracks, weight `mean(count) / count` | cell 75 — `bet_weight` |
| 5 | Directed graph over **all** cells, weighted betweenness (`graph_from_data_frame`, `netMet`) | cell 75 — `bet_net`, `nx.betweenness_centrality` |
| 6 | Plot `sqrt(betweenness)` with `magma` | cell 77 |

Cells 79 (GeoTIFF) and 81 (folium layer) have no counterpart in the R script; they mirror
what the notebook already does with the heatmap.

The node grid is built with the same `ceil(shape / agg)` arithmetic and
`transform * Affine.scale(agg)` as `deepssf.trajectory_heatmap()`, so the betweenness
grid and a heatmap built at the same `agg` line up cell for cell.

Node ids run in raster order (`row * bet_width + col`), matching R's `r[] <- 1:ncell(r)`.

## Parameters

Set at the top of cell 73.

| Knob | Default | Why |
|---|---|---|
| `BETWEENNESS_DF` | `sim_scattered` | Whatever trajectories are in memory — by default the 599-trajectory run read from disk in cell 62. |
| `BETWEENNESS_AGG` | `25` | 25 x 20 m = **500 m cells** → 54 x 68 = **3,672 nodes**. Betweenness is all-pairs shortest paths, so the node count sets the runtime (see below). |
| `BETWEENNESS_BURN_IN` | `0` | Same meaning as the heatmap's `BURN_IN` — discard the first *n* steps, which cluster around the release sites. |
| `BETWEENNESS_MAX_STEPS` | `10_000` | Keep at most this many steps per trajectory. **The most consequential knob** — see *Step budget* below. |
| `BETWEENNESS_PERIOD` | `"full"` | `"full"`, `"wet"` or `"dry"`, via the same `filter_period()` / `SEASON_YDAYS` the heatmap uses (cell 63). |
| `BETWEENNESS_EPS` | `0.25` | Densification spacing in cell widths. Simulated steps are median ~26 m but reach ~714 m (p99 454 m), so a long step can jump over a 500 m cell; densifying catches it. Most steps get one sub-point, so it is nearly free. |
| `BETWEENNESS_K` | `None` | `None` = exact. An integer estimates betweenness from that many random pivot nodes — the escape hatch for a finer grid. |

### Resolution vs runtime

Measured on this landscape (1329 x 1692 at 20 m), exact weighted betweenness in networkx:

| `AGG` | Cell size | Nodes | Runtime |
|---|---|---|---|
| 50 | 1000 m | 918 | ~3 s |
| **25** | **500 m** | **3,672** | **~15–25 s** (6,825 edges); ~1 min at 10,000 steps (~21,000 edges) |
| 12 | 250 m | 15,651 | ~15–20 min — use `BETWEENNESS_K` |

### Step budget

Given enough steps every trajectory eventually visits most of the landscape, every
adjacent pair of cells picks up an edge from nearly every trajectory, the weights flatten
towards uniform, and betweenness ends up describing the *grid* rather than the *movement*.
Measured over all 59.9M locations of the 599-trajectory / 100,000-step run, on the 500 m
grid:

| `MAX_STEPS` | Edges | Cells in network (of 3,672) | Mean trajectories per edge |
|---|---|---|---|
| 1,000 | 7,096 | 1,658 (45%) | 5.3 |
| **10,000** | **20,821** | **3,348 (91%)** | **13.1** |
| 100,000 | 26,558 | 3,671 (99.97%) | 59.6 |

At 100,000 steps the network is effectively the whole raster — saturated. The default of
10,000 keeps good coverage while edges are still selective. Hofmann's script deals with
this by recomputing at several step budgets (`design$steps = c(125, 500, 2000)`) and
checking the pattern has settled; worth doing here too.

## Outputs

Written to `OUTPUT_DIR` (`outputs/alphaearth/2026-08-17_4`), with the parameters in the
stem as the heatmap products do:

```
betweenness_{N}traj_{cell}m_steps{MAX_STEPS}_{period}.png   # sqrt scale, magma, dpi=300
betweenness_{N}traj_{cell}m_steps{MAX_STEPS}_{period}.tif   # float32, nodata=nan
betweenness_map_{period}.html                               # folium, toggleable layer
```

e.g. `betweenness_599traj_500m_steps10000_full.tif`.

The GeoTIFF is **54 x 68 cells of 500 m, EPSG:3112**, origin `(-171300, -1415040)` —
the same origin as the covariate rasters and the heatmaps, so everything stacks in QGIS.
`nodata = NaN` marks cells no trajectory entered, keeping them distinct from cells that
are in the network but scored zero.

## Deviations from the R script

- **Self-transitions are dropped.** Densified points mostly stay in the same cell, so R's
  `visitHist` produces many `from == to` rows. They cannot affect shortest paths, but they
  *do* enter the `mean(TotalConnections)` that sets the edge weights. Here consecutive
  duplicates are collapsed before transitions are formed, so they never appear.
- **Steps that leave the raster break the chain.** Transitions touching an out-of-extent
  point are dropped, so a pig that wanders off the landscape and returns does not create a
  spurious long-range edge between the cells either side.
- **Unvisited cells are `NaN`, not `0`.**
- **Only betweenness is computed.** Hofmann's `netMet()` also offers `closeness` and
  `degree`. `degree` is a one-liner (`dict(bet_net.degree())`); `closeness` costs about the
  same as betweenness.
- `normalized=False` is passed to networkx to match igraph's unnormalised default, so the
  scores are raw shortest-path counts and are **not comparable across grids of different
  size**.

## Re-running

Requires these earlier cells to have been run: **9** (`raster_transform`), **17**
(`PIXEL_SIZE`), **27** (`landscape_tensors`), **56** or **62** (`sim_scattered`), **63**
(`filter_period`, `SEASON_YDAYS`), **67** (`raster_crs`, `save_raster`), and for the map,
**60** (`add_survey_points`, `N_MAP_TRAJECTORIES`). The betweenness cells reuse all of
them rather than recomputing.

**Smoke test first.** Point `BETWEENNESS_DF` at the small file
`outputs/alphaearth/2026-08-17_4/deepSSF_trajectories/deepSSF_pigs_599traj_1000steps_2026-08-21.csv`
(41.7 MB, 599k rows) before the 4.28 GB one — the whole section runs in well under a
minute. Transition building on the full 59.9M-row file takes a few minutes and is done one
trajectory at a time so memory stays bounded.

**Sanity checks that caught things while building this:**

- `(bet_height, bet_width) == (54, 68)` and `bet_transform` shares its origin with
  `raster_transform`.
- Consecutive distinct cells from `path_cells()` are always neighbours
  (`max(|Δrow|, |Δcol|) == 1`) — this is what verifies the densification.
- `(bet_src != bet_dst).all()` — no self-loops survived.
- Rank correlation with a heatmap at the same `agg` should be well below 1. If betweenness
  looks like the heatmap, the weighting or step budget is wrong.

## Environment

`networkx` (3.6.1) is the only requirement and is already in the `deepssf` conda env.
**`scipy` and `python-igraph` are not installed**, so stay on
`nx.betweenness_centrality` — it is pure-Python Brandes and does not touch
`scipy.sparse`. Anything sparse-backed (`nx.adjacency_matrix`, some centrality and flow
variants) will fail with an import error.

## Possible next steps

- Promote `path_cells()` / `transition_counts()` into `deepSSF_package`
  (`src/deepssf/simulate.py`) as a `trajectory_betweenness()` alongside
  `trajectory_heatmap()`, adding `networkx` to its `pyproject.toml`.
- Crop and water-mask the betweenness TIF the way
  [`data_exploration_east_alligator.qmd`](data_exploration_east_alligator.qmd) (~line 1312)
  already does for the heatmaps.
- Run the wet/dry seasons separately and difference them.
