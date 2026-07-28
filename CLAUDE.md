# Seafood Fraud Dynamical Systems — Project Context

## What This Project Is

A discrete-time dynamical systems model of seafood supply chain fraud. The model captures interactions between fish stocks, fishing effort, fraudulent wholesalers, and buyer fraud awareness. The core thesis: misinformation (fraud) breaks the natural feedback loops that would otherwise stabilize marine ecosystems, analogous to how information asymmetry destabilizes financial markets.

The project has a live Streamlit app: https://dynamics-between-seafood-fraudsters-and-buyers.streamlit.app/

---

## Repository Structure

| File | Purpose |
|------|---------|
| `System.py` | Core `DynamicalSystem` class — nondimensionalized and dimensionalized model, single-step mapping, time series generation, stability analysis |
| `app.py` | Streamlit web app |
| `text.py` | Markdown/LaTeX narrative displayed in the Streamlit app |
| `requirements.txt` | Pinned dependencies |

---

## The Four State Variables

| Variable | Symbol | Range | Description |
|----------|--------|-------|-------------|
| Seafood biomass | `S` | `(0, K]` | Fish stock, normalized to carrying capacity `K = 1` |
| Fishing effort | `E` | `(0, ∞)` | Magnitude of harvesting effort |
| Fraudster share | `F` | `[0, 1]` | Proportion of wholesalers engaged in fraud |
| Fraud perception | `FP` | `[0, 1]` | Proportion of buyers aware of fraudulent activity |

---

## Core Equations (Dimensionalized)

**Seafood & Effort:**
```
S_{t+1} = S_t * exp(γ_S * (r*(1 - S_t/K) - q_t*E_t))
E_{t+1} = E_t * exp(γ_E * (q_t * P^w_t * S_t - C_t))
```

**Harvest:**
```
H_t = q_t * E_t * S_t
```

**Seafood catchability**
```
q_t = (q1 - q0)*F_t + q0
```

**Fishing cost and wholesale price (fraud-modulated):**
```
C_t   = (C1 - C0)*F_t + C0
P^w_t = ((P^w1 - P^w0)*F_t + P^w0) / (γ_p * H_t)^ε_{s,w}
```

**Fraudster share (logistic map):**
```
F_{t+1} = F_t * exp(γ_F * (P^m_t - P^w_t)) / (1 + F_t * (exp(γ_F * (P^m_t - P^w_t)) - 1))
```

**Fraud perception (logistic map, threshold-triggered):**
```
FP_{t+1} = FP_t * exp(γ_FP * (F_t - F_hat)) / (1 + FP_t * (exp(γ_FP * (F_t - F_hat)) - 1))
```

**Market price:**
```
P^m_t = γ_M * sqrt((1 - FP_t)^ε_d / H_t^ε_{s,m})
```

---

## Default Parameters

```python
DEFAULT_PARAMS = {
    'gamma_m': 10.0,   # Market price scaling
    'gamma_p': 1.0,    # Wholesale price scaling
    'gamma_s': 1.0,    # Seafood response speed
    'gamma_e': 0.225,  # Effort response speed
    'gamma_f': 1.0,    # Fraudster response speed
    'gamma_fp': 10.0,  # Fraud perception response speed
    'e_d': 1.0,        # Demand elasticity (buyer sensitivity to perceived fraud)
    'e_sw': 0.95,      # Wholesale supply elasticity
    'e_sm': 1.0,       # Market supply elasticity
    'K': 1.0,          # Carrying capacity (normalized)
    'F_threshold': 0.5,# Fraud detection threshold (F_hat)
    'r': 0.225,        # Intrinsic growth rate of seafood biomass
    'q0': 0.07,        # Catchability at 0% fraud
    'q1': 0.15,        # Catchability at 100% fraud
    'pw0': 1.0,        # Wholesale price at 0% fraud
    'pw1': 0.81,       # Wholesale price at 100% fraud
    'c0': 0.9,         # Fishing cost at 0% fraud
    'c1': 0.153,       # Fishing cost at 100% fraud
}
```

---

## Key Parameters and Their Roles

| Parameter | Description |
|-----------|-------------|
| `r` | Intrinsic growth rate — primary bifurcation parameter in baseline scenario |
| `F_threshold` (`F_hat`) | Minimum fraud level buyers can perceive — governs onset of perception dynamics |
| `e_d` | Demand elasticity — buyer sensitivity to fraud perception; low = buyers still purchase despite fraud |
| `e_sw`, `e_sm` | Supply elasticities for wholesale and market prices |
| `pw1`, `c1` | Wholesale price and cost at full fraud — key parameters for prized/protected species and destructive fishing scenarios |
| `q1` | Catchability at full fraud — higher under blast/cyanide fishing |
| `alpha` | Fraudster influence |

---

## Research Scenarios

| # | Scenario | Key Parameter | What It Tests |
|---|----------|---------------|---------------|
| 1 | Baseline (no fraud) | `r` | Pure S–E bioeconomics; oscillations and chaos |
| 2 | Prized / protected seafood | Illegal catch commanding a price premium |
| 3 | Blast / cyanide fishing | Higher catchability + lower costs + lower wholesale price |
| 4 | EEZ non-enforcement | Outside-EEZ access — higher catchability, higher costs |

---

## Stability Analysis

The `DynamicalSystem` class has three key methods:

- **`find_fixed_point()`** — Uses `scipy.optimize.least_squares` with Trust Region Reflective (TRF) to find `x*` satisfying `G(x*) = x*`, with box constraints to keep variables in physical bounds. Uses orbit-mean + last-iterate two-candidate strategy to handle limit cycles and chaos.
- **`jacobian(state, h)`** — 4×4 Jacobian via central finite differences with optimal step size `h = ε^(1/3) * max(1, |x_i|)`.
- **`stability_analysis()`** — Finds fixed point, computes Jacobian, extracts eigenvalues via `numpy.linalg.eig`. Spectral radius `ρ < 1` → stable; `ρ > 1` → unstable. Transition at `ρ = 1` through complex conjugate pair = Neimark-Sacker bifurcation.

---

## Ecological / Theoretical Framing

- Humans are modeled as **apex predators** whose feeding behavior is mediated through economic systems and information flows rather than direct biological feedback.
- **Fraud breaks natural feedback loops** that would otherwise stabilize the ecosystem. In a transparent market, high fishing effort → stock depletion → lower profits → reduced effort (stabilizing). Fraud decouples this by artificially reducing costs and masking depletion signals.
- The model draws parallels to **financial instability theory**: fraud revelations can cascade like financial shocks; `F_hat` functions like a capital adequacy threshold; awareness spread mirrors disaster myopia dynamics in banking.
- Key policy levers modeled: awareness thresholds, fraud-effort feedback, tipping point identification, market tiering effects from verification systems.

---

## Conventions and Style

- State variables use a dict with keys `'S'`, `'E'`, `'F'`, `'FP'`
- The system can run in `"dimensionalized"` or `"nondimensionalized"` mode
- All state variables use `np.float128` for numerical precision
- Plots use matplotlib; the Streamlit app uses the same `DynamicalSystem` class
- Equations in `text.py` are raw LaTeX strings rendered by Streamlit
- All variables and functions should use snake_case
- Be descriptive with variable and function naming: Prioritize meaning over short names (e.g., use elapsed_time_seconds instead of just t or ets)
