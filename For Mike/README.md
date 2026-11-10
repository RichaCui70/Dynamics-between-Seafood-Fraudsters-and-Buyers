# For Mike: low-value black market figures

Figures and the scripts that make them, for the Blast Fishing (Raja Ampat) section of the paper.

## Why there are two parameterizations

The default model uses a wholesale supply elasticity of ε_sw = 0.95. With that value, the wholesale price
P^w = P^w_base(F) / (γ_P·H)^ε_sw climbs steeply as harvest falls. That scarcity premium keeps harvesting
profitable in a depleted fishery, which looks more like a high-value black market than the low-value one we
describe (catchability q up, cost C down a lot, wholesale price P^w down).

The low-value runs lower ε_sw and rescale γ_P so the honest (no-fraud) equilibrium stays exactly the same:
γ_P = (H*₀)^(0.95/ε_sw − 1), where H*₀ = 0.0469 is the default honest harvest. Without this rescaling, the honest
fishery cannot break even at a low ε_sw.

| Set | ε_sw | γ_P | Where |
|---|---|---|---|
| Original | 0.95 | 1 | `figures/*.pdf` |
| First test | 0.05 | 1.23×10⁻²⁴ | `figures/true low value black market/*.pdf` |
| Recommended | 0.3 | 1.32×10⁻³ | `figures/true low value black market/tests/` |

0.3 sits well below the analytic threshold ε_sw = ε_sm/2 = 0.5. Above that threshold, scarcity squeezes the fraud
margin. Below it, results barely change with ε_sw, and γ_P stays interpretable: 1/γ_P ≈ 760 is the harvest at which
the wholesale price equals pw₀.

## Figures

- `figures/`: the current paper figures with the original parameters.
- `figures/true low value black market/`: the same figures with ε_sw = 0.05.
  - The margins figure and the flow diagram don't depend on ε_sw, so they are unchanged.
- `figures/true low value black market/tests/`:
  - `blast_fig_f_story_welfare_panels_multiples_esw_0.3_gammap_1.32e-3.pdf`: the story welfare panels with the
    recommended set.
  - `damaged_reef_price_decomposition_esw_0.95_vs_0.3.pdf`: splits the Damaged reef P̄ᵐ change into a buyer-perception
    term and a harvest term (Shapley), and shows F̄ and F̄^p against α.
  - `damaged_reef_affordability_map_r_vs_alpha_esw_0.95_vs_0.3.pdf`: affordability 1/P̄ᵐ over (α, r).
  - `fraud_incentive_scarcity_vs_perception_esw_0.95_vs_0.3.pdf`: wholesale and market price against harvest, and
    how aware buyers must be before fraud stops paying.

Main finding: under the low-value set, the Damaged reef affordability curve cannot drop below the honest market
for any r. Scarcity never makes fraud unprofitable, so buyer perception is the only brake on fraud. That keeps
F̄^p at 0.8–0.98, and the buyers' discount outweighs the price rise from lost harvest. The original panel's low
affordability came from the scarcity premium. In that case, wholesale price overtakes market price, fraud
collapses, and buyers never notice.

## Running the scripts

From the repository root, using the repo's `core/` package:

```bash
python "For Mike/blast_fishing_figures.py" --figures story_welfare_panels_multiples  # original figures
python "For Mike/mechanism_figures.py" --figures fig_2b fig_3 fig_revised_h1_single_panel
python "For Mike/true_low_value_figures.py"            # ε_sw = 0.05 set
python "For Mike/recommended_welfare_panels.py"        # ε_sw = 0.3 welfare panels
python "For Mike/damaged_reef_explainer_figures.py"    # three explainer figures
```

## Numerical caveat

Each fraud cycle drives the fraudster share F below machine precision, so F sits on the clip floor (machine
epsilon) for much of the time. The time-averaged market price P̄ᵐ and the fraud-cycle period therefore depend
partly on that floor:

- Moving the floor from 2.2×10⁻¹⁶ to 10⁻³⁰ changes P̄ᵐ by 10–50% and the cycle period by about 15%.
- S̄, H̄ and F̄ change by about 2% for α ≤ 0.5.
- Values near α = 1 are unreliable.

`np.longdouble` is float64 on Apple Silicon (machine epsilon 2.2×10⁻¹⁶) but 80-bit on x86 Linux (machine epsilon
1.1×10⁻¹⁹), so results can differ slightly between machines.
