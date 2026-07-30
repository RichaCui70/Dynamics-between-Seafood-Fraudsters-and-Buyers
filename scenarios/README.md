# `scenarios/` — Streamlit Research Tabs

Each scenario module is a Streamlit page section: analysis controls, cached simulations, and Plotly charts. All runs use `DynamicalSystem` in `"dimensionalized"` mode from [`core/`](../core/).

| File | Role |
|------|------|
| [`scenario_baseline.py`](scenario_baseline.py) | No-fraud S–E bioeconomics; focus parameter `r` |
| [`scenario_bf.py`](scenario_bf.py) | Blast / cyanide fishing; intensity `α` |
| [`scenario_ps.py`](scenario_ps.py) | Prized / protected seafood; price-premium `α` |
| [`scenario_eez.py`](scenario_eez.py) | EEZ non-enforcement; violation intensity `α` |
| [`_sys_params.py`](_sys_params.py) | Shared “System Parameters” slider UI |
| [`_status.py`](_status.py) | Header + compute-status indicator |
| [`__init__.py`](__init__.py) | Re-exports the four `scenario_*` entry points |

Plot builders live in [`core/plots.py`](../core/plots.py) (not under `scenarios/`).

---

## Shared UI helpers

### `_sys_params.py` — `system_parameters_ui(prefix, exclude=…)`

Expander with sliders for response speeds, market elasticities, fraud economics, and the starting state (`S₀`, `E₀`, `F₀`, `FP₀`). Parameter defaults come from `DEFAULT_PARAMS`; starting-state defaults come from `DEFAULT_INIT_STATE` unless a scenario supplies its own values. Returns a sorted `(key, val)` tuple used as a Streamlit cache key (`system_param_overrides`). Scenarios `exclude` parameters that `α` already drives (e.g. blast excludes `q1`, `pw1`, `c1`).

`initial_state_from_overrides(...)` converts the cached starting-value controls back to the state keys expected by `DynamicalSystem`. All time-series, bifurcation, Poincaré, and stability calculations use that state. No-fraud baseline runs preserve their scenario-specific `F₀ = FP₀ = 0` values.

### `_status.py`

- `scenario_header(title)` — header plus a status slot
- `status_indicator(slot, steps)` — spinner during compute, green check with hover tooltip listing completed steps

---

## Plot types (what you see in the app)

Fraud scenarios (BF, PS, EEZ) share the same tab layout. Baseline is S–E only.

### Time series

| Context | What it shows |
|---------|----------------|
| **Baseline** | Columns of selected `r` values. Row 1: S & H; row 2: E. No F/FP (held at 0). |
| **Scenario (vs α)** | Leading **Baseline** column (`α=0`, `F=FP=0`) plus columns for selected `α`. Tracks S, E, H, F, FP; BF/PS also show market & wholesale prices. |
| **Scenario (vs F_threshold)** | Same layout, sweeping detection threshold `F̂` at a held `α`. Shows how buyer awareness onset changes trajectories. |

**Scientific role:** compare transient and long-run dynamics as fraud intensity or detection threshold changes; baseline column anchors “honest market” behavior.

### Heatmaps (under time series — BF / PS / EEZ)

One-row Plotly heatmaps for post-burn metrics `S̄`, `H̄`, `P̄ᵐ`, `φ_FP`, `φ_H` (toggle with pills). Metrics are computed in `core.metrics`; the plot layer only renders.

- **`S̄` / `H̄` / `P̄ᵐ`:** % change vs the no-fraud baseline
- **`φ_FP` / `φ_H`:** per-timestep Shapley contributions of fraud perception and harvest to market price (averaged over post-burn), shown as % of baseline `P̄ᵐ`. Preferentially `φ_FP + φ_H ≈` the `P̄ᵐ` cell (same % of baseline scale).

Not a 2D α×F̂ grid — separate heatmaps under the **vs α** and **vs F_threshold** sweeps.

**Scientific role:** compact summary of welfare / stock / price impact relative to the honest baseline, plus how much of the price gap is attributed to awareness vs harvest.

### Bifurcation diagrams

Post-burn attractor points plotted against the sweep axis:

| Scenario | Sweep A | Sweep B |
|----------|---------|---------|
| Baseline | `r` → S*, E* only | — |
| BF / PS / EEZ | `α` → S*, E*, F*, FP* (held `F_threshold`) | `F_threshold` → same (held `α`) |

**Scientific role:** locate fixed points vs oscillations/chaos; mark where fraud or detection threshold opens new attractors. Baseline marks default `r`; PS marks `α = 0`.

### Poincaré / return maps

`x(t)` vs `x(t+1)` for S and E after burn-in (diagonal = fixed point). Same dual sweeps (vs α / vs F_threshold) as time series, including a baseline column in fraud scenarios.

**Scientific role:** visualize attractor geometry (period-1, cycles, strange attractors) without time clutter.

### Stability (prized seafood only)

Spectral radius `ρ = max|λᵢ|` vs `α` from `stability_analysis()`. Green: `ρ < 1` (stable); red: unstable. Horizontal line at `ρ = 1`.

---

## Threshold vs α (how the dual sweeps work)

There is **no single 2D α × F_threshold heatmap**. Instead each fraud scenario exposes two linked 1D analyses:

1. **vs α** — vary fraud/scenario intensity; hold `F_threshold` fixed (selectbox).
2. **vs F_threshold** — vary detection threshold; hold `α` fixed (selectbox).

Together they answer: *how bad does fraud get as intensity rises?* and *does raising the buyer detection threshold restore stability / stock?* Heatmaps and bifurcations follow the same split.

---

## Scenario modules

### `scenario_baseline.py` — Bioeconomic model (no fraud)

- Init: `F = FP = 0`. Focus parameter: intrinsic growth rate `r`.
- Cached: `baseline_time_series`, `baseline_bifurcation`.
- **Tabs:** Time Series · Bifurcation · Poincare  
  (figures built inline; does not use `core.plots` helpers.)

### `scenario_bf.py` — Blast / cyanide fishing

- Intensity `α ∈ [0, 1]` jointly sets destructive economics:
  - `q1 = q0 + α·0.33` ↑
  - `pw1 = pw0 − α·0.40` ↓
  - `c1 = c0 − α·0.80` ↓↓ (cost falls faster than wholesale price)
- Cached: `blast_time_series` / `blast_bifurcation` (and `*_vs_f_threshold` variants); `blast_baseline` / `blast_baseline_time_series` for heatmap & column baselines.
- **Tabs:** Time Series (with economics + heatmaps) · Bifurcation · Poincare — each with **vs α** / **vs F_threshold** subtabs.

### `scenario_ps.py` — Prized / protected seafood

- `α` drives illegal catch premium: `pw1 = pw0 + α·4`; gear unchanged (`c1 = c0`, `q1 = q0`).
- Same dual-sweep tabs as blast, plus **Stability** (`prized_spectral_sweep`).
- Time series use `plot_time_series_with_economics` (prices matter when premium rises).
- Cached helpers: `prized_time_series`, `prized_bifurcation`, `prized_baseline_time_series`, etc.

### `scenario_eez.py` — Non-enforcement of EEZ

- Outside-EEZ access: `q1 = q0 + α·0.23` ↑, `c1 = c0 + α·1.10` ↑; `pw1` stays at default.
- Same dual-sweep tabs as blast; time series use `plot_four_variable_time_series` (no extra price rows).
- Cached helpers: `eez_time_series`, `eez_bifurcation`, `eez_baseline_time_series`, etc.

---

## Quick reference: tabs × plots

| Tab | Baseline | Blast (BF) | Prized (PS) | EEZ |
|-----|----------|------------|-------------|-----|
| Time Series | S/H & E vs `r` | Economics TS + heatmap; vs α & vs F̂ | Economics TS + heatmap; vs α & vs F̂ | 4-var TS + heatmap; vs α & vs F̂ |
| Bifurcation | S*, E* vs `r` | 4-var vs α & vs F̂ | 4-var vs α & vs F̂ | 4-var vs α & vs F̂ |
| Poincare | S, E return maps vs `r` | vs α & vs F̂ (+ baseline col) | vs α & vs F̂ (+ baseline col) | vs α & vs F̂ (+ baseline col) |
| Stability | — | — | `ρ(α)` | — |
