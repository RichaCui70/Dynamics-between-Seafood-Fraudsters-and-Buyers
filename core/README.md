# `core/` — Model Engine

Shared numerical model and plotting primitives used by the Streamlit app and scenario tabs. Scenarios import from here; they do not redefine the map.

| File | Role |
|------|------|
| [`System.py`](System.py) | Discrete-time dynamical system (`DynamicalSystem`) |
| [`constants.py`](constants.py) | Default parameters, initial state, plot colors |
| [`plots.py`](plots.py) | Reusable Plotly figure builders |
| [`metrics.py`](metrics.py) | Summary metrics + Shapley φ_FP / φ_H for heatmaps |
| [`__init__.py`](__init__.py) | Package marker (empty) |

---

## `System.py` — `DynamicalSystem`

Core 4D map for seafood biomass `S`, fishing effort `E`, fraudster share `F`, and buyer fraud perception `FP`.

### Construction

```python
DynamicalSystem(params, state, equation_form="nondimensionalized" | "dimensionalized")
```

- `params` — dict of model coefficients (falls back to `DEFAULT_PARAMS`)
- `state` — `{'S', 'E', 'F', 'FP'}` (stored as `np.float128`)
- `equation_form` — which equation set `system_map()` advances

### State update & observables

| Method | Purpose |
|--------|---------|
| `system_map()` | One-step map + contemporaneous observables (prices, harvest, costs, …) |
| `generate_time_series(num_timesteps)` | Iterate `num_timesteps` steps; returns arrays for S, E, F, FP, prices, harvest, revenue/cost per effort |
| `seafood_state_*` / `effort_state_*` / `fraudster_state_*` / `fraud_perception_state_*` | Next-state equations (`_nondim` and `_dimensionalized` variants) |
| `catchability()`, `harvest()`, `market_price()`, `wholesale_price()`, … | Fraud-modulated bioeconomic quantities |

Dimensionalized form is what the Streamlit scenarios use. Nondimensionalized form keeps the same structure with scaled parameters (`nondim_params`). Bounds `F_min`/`F_max` and `FP_min`/`FP_max` are already dimensionless and pass through unchanged; both equation forms clip `F` and `FP` updates to those intervals (within `(0, 1)`).

### Stability analysis

| Method | Purpose |
|--------|---------|
| `find_fixed_point()` | Solve `G(x*) = x*` via `scipy.optimize.least_squares` (TRF), with orbit-mean + last-iterate candidates |
| `jacobian(state, h)` | 4×4 Jacobian by central finite differences |
| `stability_analysis()` | Fixed point → Jacobian → eigenvalues; spectral radius `ρ < 1` ⇒ locally stable |

---

## `constants.py`

| Export | Contents |
|--------|----------|
| `DEFAULT_PARAMS` | Full parameter set (`gamma_*`, elasticities, `r`, `K`, `F_threshold`, `q0`/`q1`, `pw0`/`pw1`, `c0`/`c1`, …) |
| `DEFAULT_INIT_STATE` | `{'S': 0.6, 'E': 0.3, 'F': 0.1, 'FP': 0.1}` — used by fraud scenarios |
| `VAR_COLORS` | Hex colors for S, E, F, FP, prices, harvest, revenue, cost |

Scenarios that need a no-fraud start (baseline tab, heatmap baselines) override `F`/`FP` to `0` themselves.

---

## `plots.py`

Plotly helpers shared by blast, prized-seafood, and EEZ tabs. Baseline builds its figures inline.

| Function | Figure |
|----------|--------|
| `plot_four_variable_time_series(...)` | Multi-column time series: row 1 = S / E / H; row 2 = F / FP |
| `plot_time_series_with_economics(...)` | Same plus market & wholesale prices (optional revenue/cost per effort row) |
| `plot_bifurcation(...)` | 2×2 attractor scatter of S*, E*, F*, FP* vs a sweep parameter |
| `plot_poincare_maps(...)` | Poincaré plots: `x(t)` vs `x(t+1)` for S and E (post-burn-in) |
| `plot_time_series_heatmap(...)` | One-row heatmaps from precomputed percent rows (`S̄`, `H̄`, `P̄ᵐ`, `φ_FP`, `φ_H`) |
| `HEATMAP_METRICS` | `['S̄', 'H̄', 'P̄ᵐ', 'φ_FP', 'φ_H']` — selectable heatmap rows in the UI |

Colors come from `VAR_COLORS`. Heatmap scales are diverging (`SEAFOOD_COLORSCALE` / `HARVEST_COLORSCALE`: green = better stock/harvest; `MARKET_PRICE_COLORSCALE` inverts polarity for `P̄ᵐ` / `φ_FP` / `φ_H`).

---

## `metrics.py`

Trajectory summary metrics used by scenario heatmaps.

| Export | Purpose |
|--------|---------|
| `compute_summary_metrics(...)` | Post-burn `S̄`, `H̄`, `P̄ᵐ` plus **per-timestep** Shapley `φ_FP`, `φ_H` (averaged; price units) for `Pᵐ(FP, H)` vs a reference |
| `build_heatmap_display_rows(...)` | From time series + baseline means → display percents for all five heatmap rows |
| `heatmap_display_percents(...)` | `S̄`/`H̄`/`P̄ᵐ` = % change vs baseline; `φ_FP`/`φ_H` = φ as % of baseline `P̄ᵐ` |

Shapley reference: `FP_ref = 0`, `H_ref` = no-fraud baseline harvest. Per-timestep then average so `φ_FP + φ_H = mean_t Pᵐ(FP_t, H_t) − Pᵐ(ref)`, which tracks the observed average price gap (unlike Shapley-of-means).
