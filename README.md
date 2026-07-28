# Dynamics between Seafood Fraudsters and Buyers

A discrete-time dynamical systems model exploring how seafood fraud propagates through supply chains and how buyer awareness, market forces, and fishing practices create feedback loops that either stabilize or collapse fish stocks.

## Website
Check out our website for more information and system simulations!</br>
https://dynamics-between-seafood-fraudsters-and-buyers.streamlit.app/

## Motivation

Seafood is one of the most heavily traded food commodities globally, supplying roughly a quarter of the world's animal protein intake. The complexity of global supply chains makes it difficult to ensure the authenticity and traceability of seafood products, creating opportunities for fraud — mislabelling, substitution, adulteration, and unethical harvesting. This project models the interactions between key players in the supply chain to understand when fraud self-corrects and when it spirals out of control.

## The Model

The system tracks four state variables over discrete time steps:

| Variable | Symbol | Range | Description |
|----------|--------|-------|-------------|
| **Seafood biomass** | $S$ | $(0, K]$ | Fish stock level, normalized to carrying capacity $K = 1$ |
| **Fishing effort** | $E$ | $(0, \infty)$ | Magnitude of harvesting effort by fishers |
| **Fraudster share** | $F$ | $[0, 1]$ | Proportion of wholesalers engaged in fraud |
| **Fraud perception** | $F^p$ | $[0, 1]$ | Proportion of buyers aware of fraudulent activity |

These interact through four coupled mechanisms:

1. **Seafood & Effort** — Effort grows when fishing is profitable (revenue > cost); higher effort depletes fish stocks, which recover via logistic growth at intrinsic rate $r$.
2. **Fishers & Wholesalers** — Fraudulent wholesalers enable illegal fishing methods, lowering costs and wholesale prices. Fraud grows when the gap between market price and wholesale price widens.
3. **Wholesalers & Buyers** — Buyers detect fraud when $F$ exceeds a threshold $\hat{F}$. Rising perception of fraud reduces demand and depresses market prices, squeezing fraudster margins.
4. **Feedback loop** — Lower market prices reduce the incentive for fraud, allowing perception to decay, demand to recover, and the cycle to potentially repeat.

The full equations and derivations are presented in the Streamlit app (sourced from `text.py`).

## Repository Structure

| File | Purpose |
|------|---------|
| `System.py` | Core `DynamicalSystem` class — implements both **nondimensionalized** and **dimensionalized** forms of the model, with methods for single-step mapping and time series generation |
| `app.py` | Streamlit web app — renders the model introduction and visualizations |
| `new_research.py` | Research script — runs 5 scenario experiments with time series, bifurcation diagrams, and return maps via matplotlib |
| `text.py` | Markdown narrative and LaTeX equations displayed in the Streamlit app |
| `requirements.txt` | Pinned Python dependencies |

## Research Scenarios

`new_research.py` and the Streamlit app investigate four scenarios using the dimensionalized model. Each scenario provides time series, bifurcation diagrams, and stability charts (spectral radius sweep):

| # | Scenario | Focus Parameter | What It Tests |
|---|----------|-----------------|---------------|
| 1 | **Baseline (no fraud)** | Intrinsic growth rate $r$ | Pure bioeconomic S–E dynamics; onset of oscillations and chaos as $r$ increases |
| 2 | **Prized / protected seafood** | Fraudster wholesale price $pw_1$ | What happens when illegal catch commands a premium over legal catch |
| 3 | **Blast / cyanide fishing** | Destruction intensity $\alpha$ | Joint effect of higher catchability, lower costs, and lower wholesale prices from destructive methods |
| 4 | **EEZ non-enforcement** | Violation intensity $\beta$ | Fishers access outside-EEZ waters — higher catchability but higher costs |

## Parameters

Key parameters and their roles:

| Parameter | Description |
|-----------|-------------|
| $\gamma_s, \gamma_e, \gamma_f, \gamma_{fp}$ | Response speeds for seafood, effort, fraudsters, and fraud perception |
| $\gamma_m, \gamma_p$ | Market price scaling and wholesale price scaling |
| $r$ | Intrinsic growth rate of seafood biomass |
| $K$ | Carrying capacity (normalized to 1) |
| $q_0, q_1$ | Catchability when fraud is 0% vs 100% (dimensionalized). $q_0, q_1$ not considered in nondimensionalized system; combined as just $q$ (todo) |
| $pw_0, pw_1$ | Wholesale price at 0% vs 100% fraud |
| $c_0, c_1$ | Fishing cost at 0% vs 100% fraud |
| $\varepsilon_d$ | Demand elasticity — buyer sensitivity to perceived fraud |
| $\varepsilon_{s,w}, \varepsilon_{s,m}$ | Supply elasticities for wholesale and market prices |
| $\hat{F}$ | Fraud detection threshold — minimum fraud level buyers can perceive |
| $F_{min}, F_{max}$ | Minimum and maximum level of fraud |
| $F^{p}_{min}, F^{p}_{max}$ | Minimum and maximum level of buyers whom are aware of fraud |

Default parameters (dimensionalized, from `System.py`):

```python
{
    'gamma_m': 10.0, 'gamma_p': 1.0, 'gamma_s': 1.0,
    'gamma_e': 0.225, 'gamma_f': 1.0, 'gamma_fp': 10.0,
    'e_d': 1.0, 'e_sw': 0.95, 'e_sm': 1.0,
    'K': 1.0, 'F_threshold': 0.5, 'r': 0.225,
    'q0': 0.07, 'q1': 0.15,
    'pw0': 1.0, 'pw1': 0.81,
    'c0': 0.9, 'c1': 0.153,
    'F_min': 0.0, 'F_max': 1.0,
    'FP_min': 0.0, 'FP_max': 1.0,
}
```

## Stability Analysis

The `DynamicalSystem` class includes numerical tools for local stability analysis of the 4D discrete map. Three methods work together:

### `find_fixed_point(initial_guess=None, warmup_steps=500, tol=1e-10)`

Finds a fixed point $x^*$ satisfying $G(x^*) = x^*$ by solving the root-finding problem $G(x) - x = 0$.

**Solver:** `scipy.optimize.least_squares` with the **Trust Region Reflective (TRF)** method. This was chosen over the previous `fsolve` (MINPACK `hybrd`) because TRF supports **box constraints**, which are essential for this system: $F$ and $F^p$ are proportions confined to $(0, 1)$, and $S$ must remain positive. Without bounds, unconstrained solvers can wander into physically meaningless regions (negative biomass, fraud fractions outside $[0,1]$) where the map produces `NaN`/`Inf`.

The bounds are:

$$S > 0, \quad E > 0, \quad 0 < F < 1, \quad 0 < F^p < 1$$

**Initial guess strategy — orbit mean with multi-candidate selection:**

A naive warm-start (simulating forward and using the last state) fails when the system is in a limit-cycle or chaotic regime: the last iterate sits on the oscillation rather than near a fixed point, and the solver drifts to a degenerate boundary equilibrium (e.g. $F \approx 0, F^p \approx 1, S \approx 0$) that is irrelevant to the dynamics.

To handle this, the solver uses a two-candidate approach:

1. **Orbit mean** — Simulate the map forward `warmup_steps` iterations and collect the last half of the trajectory. The component-wise mean of this tail approximates the center of the attractor. Even when the orbit oscillates wildly (e.g. $F$ bouncing between 0.03 and 0.97), the mean ($F \approx 0.5$) lands near the interior fixed point.
2. **Last iterate** — The final state of the warmup, which works well when the trajectory converges to a stable equilibrium.

Both candidates are passed to `least_squares` (with `max_nfev=5000`), and the best result is selected with a preference for **interior solutions** over boundary ones. A candidate is classified as a boundary solution if $S < 10^{-6}$, $F < 10^{-6}$, $F > 1 - 10^{-6}$, $F^p < 10^{-6}$, or $F^p > 1 - 10^{-6}$.

**Convergence:** A fixed point is considered converged when $\|G(x^*) - x^*\| < \texttt{tol}$ (default $10^{-10}$).

### `jacobian(state=None, h=None)`

Computes the $4 \times 4$ Jacobian matrix $J$ of the map at a given state using **central finite differences**:

$$J_{ji} = \frac{G_j(x + h \, e_i) - G_j(x - h \, e_i)}{2h}$$

The default perturbation $h = \varepsilon^{1/3} \cdot \max(1, |x_i|)$ where $\varepsilon \approx 2.2 \times 10^{-16}$ (float64 machine epsilon) gives $O(h^2)$ accuracy — the optimal trade-off between truncation and round-off error.

### `stability_analysis(initial_guess=None, warmup_steps=500, tol=1e-10)`

Orchestrates the full workflow: finds the fixed point, computes the Jacobian, and extracts eigenvalues via `numpy.linalg.eig` (LAPACK's `dgeev` — implicit QR iteration).

For a discrete-time map, the fixed point is **stable** when the spectral radius $\rho = \max_i |\lambda_i| < 1$ (all eigenvalues inside the unit circle) and **unstable** when $\rho > 1$. The transition at $\rho = 1$ through a complex conjugate pair is a **Neimark-Sacker bifurcation** (the discrete-time analog of a Hopf bifurcation).

If the fixed-point solver did not converge (residual above tolerance), the analysis reports `stable = False` with classification `"no fixed point found (solver did not converge)"` rather than trusting eigenvalues at a non-equilibrium point.

**Example:**

```python
from core.System import DynamicalSystem, DEFAULT_PARAMS
import numpy as np

params = DEFAULT_PARAMS.copy()
params.update({'pw1': 1.5, 'c1': 0.9, 'q1': 0.07})
state = {'S': np.float128(0.6), 'E': np.float128(0.3),
         'F': np.float128(0.1), 'FP': np.float128(0.1)}
sys = DynamicalSystem(params, state, "dimensionalized")

result = sys.stability_analysis()
print(f"Spectral radius: {result['spectral_radius']:.4f}")
print(f"Stable: {result['stable']}")
print(f"Classification: {result['classification']}")
```

## Getting Started

**Requirements:** Python 3.10+

```bash
pip install -r requirements.txt
```

**Run the Streamlit app:**

```bash
streamlit run app.py
```

**Run the research scenarios** (generates matplotlib figures interactively):

```bash
python new_research.py
```

Toggle individual scenarios on/off by changing the `if True` / `if False` guards at the top of each section in `new_research.py`.

## References

To be finished
