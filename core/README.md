# `core/` — Model Engine

Shared numerical model and plotting primitives used by the Streamlit app and scenario tabs. Scenarios import from here; they do not redefine the map.

| File | Role |
|------|------|
| [`System.py`](System.py) | Discrete-time dynamical system (`DynamicalSystem`) |
| [`constants.py`](constants.py) | Default parameters, initial state, plot colors |
| [`plots.py`](plots.py) | Reusable Plotly figure builders |
| [`__init__.py`](__init__.py) | Package marker (empty) |

---

## `System.py` — `DynamicalSystem`

Core 4D map for seafood biomass `S`, fishing effort `E`, fraudster share `F`, and buyer fraud perception `FP`.

### Construction

```python
DynamicalSystem(params, state, type="nondimensionalized" | "dimensionalized")
```

- `params` — dict of model coefficients (falls back to `DEFAULT_PARAMS`)
- `state` — `{'S', 'E', 'F', 'FP'}` (stored as `np.float128`)
- `type` — which equation set `system_map()` advances

### State update & observables

| Method | Purpose |
|--------|---------|
| `system_map()` | One-step map + contemporaneous observables (prices, harvest, costs, …) |
| `time_series_plot(time)` | Iterate `time` steps; returns arrays for S, E, F, FP, prices, harvest, revenue/cost per effort |
| `seafood_state_*` / `effort_state_*` / `fraudster_state_*` / `p_fraudster_state_*` | Next-state equations (nondim and dimful variants) |
| `catchability()`, `harvest()`, `market_price()`, `wholesale_price()`, … | Fraud-modulated bioeconomic quantities |

Dimensionalized form is what the Streamlit scenarios use. Nondimensionalized form keeps the same structure with scaled parameters (`nondim_params`).

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
| `plot_4var_ts(...)` | Multi-column time series: row 1 = S / E / H; row 2 = F / FP |
| `plot_ts_with_economics(...)` | Same plus market & wholesale prices (optional revenue/cost per effort row) |
| `plot_bifurcation(...)` | 2×2 attractor scatter of S*, E*, F*, FP* vs a sweep parameter |
| `plot_return_maps(...)` | Poincaré plots: `x(t)` vs `x(t+1)` for S and E (post-burn) |
| `plot_ts_heatmap(...)` | One-row heatmaps of post-burn mean metrics (`S̄`, `H̄`, `P̄ᵐ`) as % change vs a baseline (or vs row mean) |
| `HEATMAP_METRICS` | `['S̄', 'H̄', 'P̄ᵐ']` — selectable heatmap rows in the UI |

Colors come from `VAR_COLORS`. Heatmap scales are diverging (green = better than baseline for stock/harvest; market-price polarity is inverted).
