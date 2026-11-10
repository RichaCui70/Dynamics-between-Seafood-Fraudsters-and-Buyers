"""Re-run the current (non-retired) paper figures with weak scarcity pricing.

The default parameterization uses eps_sw = 0.95, so the wholesale price
P^w = P^w_base(F) / (gamma_P H)^eps_sw climbs steeply as harvest falls. That
scarcity premium props up harvester revenue in a depleted fishery, which is
closer to a high-value black market than the low-value one we describe
(q up, C down a lot, P^w down). Here eps_sw = 0.05 (P^w almost flat in H).

Calibration: with eps_sw = 0.05 and gamma_P = 1 the honest fishery cannot break
even (q_0 P^w S < C_0 for every S <= K), so effort dies out and every
"x honest market" ratio divides by zero harvest. gamma_P is therefore rescaled
so the honest wholesale price at the honest equilibrium harvest H*_0 is
unchanged:

    (gamma_P H*_0)^0.05 = (H*_0)^0.95   =>   gamma_P = (H*_0)^(0.95 / 0.05 - 1).

The honest equilibrium (S*, E*, H*, P^w*, P^m*) is then identical to the
default model; only the price response to scarcity away from it is weakened,
and the relative wholesale-price discount of fraud (P^w_1 / P^w_0) is kept.

Figures are written as PDFs, under their original file names, to
``paper/figures/true low value black market/``.

Usage:
    python paper/true_low_value_figures.py
"""

from __future__ import annotations

import os
import sys

import matplotlib
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import blast_fishing_figures  # noqa: E402
import mechanism_figures  # noqa: E402
from blast_fishing_figures import DEFAULT_PARAMS, STORIES_BY_NAME, no_fraud_baseline  # noqa: E402

LOW_SCARCITY_PRICING_ELASTICITY = 0.05
OUTPUT_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures',
                                'true low value black market')

# Builders for the figures currently in paper/figures (not Retired/), with the
# file name each one has there.
FIGURES_TO_REBUILD = (
    (blast_fishing_figures.figure_boom_bust_anatomy,
     'blast_fig_d_boom_bust_anatomy_original'),
    (lambda: blast_fishing_figures.figure_story_welfare_panels(y_axis_as_multiple=True),
     'blast_fig_f_story_welfare_panels_multiples'),
    (blast_fishing_figures.figure_model_flow_diagram, 'blast_fig_g_model_flow_diagram'),
    (blast_fishing_figures.figure_black_market_margins, 'blast_fig_i_black_market_margins'),
    (mechanism_figures.figure_2b_iuu_payoff_advantage, 'mechanism_fig_2b_iuu_payoff_advantage'),
    (mechanism_figures.figure_3_consumer_response, 'mechanism_fig_3_consumer_response'),
    (mechanism_figures.figure_revised_hypothesis_single_panel,
     'mechanism_fig_revised_h1_single_panel'),
)


def calibrated_wholesale_price_scaling(new_elasticity: float) -> float:
    """gamma_P that keeps the honest equilibrium wholesale price at the default value."""
    honest_harvest = no_fraud_baseline(STORIES_BY_NAME['Baseline'])['harvest_mean']
    default_elasticity = DEFAULT_PARAMS['e_sw']
    return honest_harvest ** (default_elasticity / new_elasticity - 1)


def tidy_boom_bust_price_axis(figure):
    """Price now spans under two decades: label 1-2-5 ticks instead of minor 'a x 10^b' ones."""
    for axis in figure.axes:
        if axis.get_ylabel() == 'Price':
            axis.yaxis.set_major_locator(matplotlib.ticker.LogLocator(subs=(1.0, 2.0, 5.0)))
            axis.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def lift_fraud_always_pays_label(figure):
    """Move the label above the incentive curve, which now sits where the label was."""
    for axis in figure.axes:
        for text in axis.texts:
            if text.get_text() == 'Fraud always pays':
                text.set_position((0.5, 0.86))


# Layout fixes for annotations placed for the eps_sw = 0.95 results.
FIGURE_ADJUSTMENTS = {
    'blast_fig_d_boom_bust_anatomy_original': tidy_boom_bust_price_axis,
    'mechanism_fig_3_consumer_response': lift_fraud_always_pays_label,
}


def save_pdf_with_file_name(file_name: str):
    """save_figure replacement that writes one PDF under ``file_name``."""
    def save_pdf(figure, _original_file_stem: str):
        if file_name in FIGURE_ADJUSTMENTS:
            FIGURE_ADJUSTMENTS[file_name](figure)
        os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
        figure.savefig(os.path.join(OUTPUT_DIRECTORY, f'{file_name}.pdf'),
                       bbox_inches='tight', pad_inches=0.02)
        plt.close(figure)
        print(f'  wrote figures/true low value black market/{file_name}.pdf')
    return save_pdf


def main():
    # Story.params() copies DEFAULT_PARAMS at call time, so editing the shared
    # dict in place re-parameterizes every story in both figure modules.
    wholesale_price_scaling = calibrated_wholesale_price_scaling(LOW_SCARCITY_PRICING_ELASTICITY)
    DEFAULT_PARAMS['e_sw'] = LOW_SCARCITY_PRICING_ELASTICITY
    DEFAULT_PARAMS['gamma_p'] = wholesale_price_scaling
    print(f'eps_sw = {LOW_SCARCITY_PRICING_ELASTICITY}, gamma_P = {wholesale_price_scaling:.4e}')

    blast_fishing_figures.apply_paper_style()
    for build_figure, file_name in FIGURES_TO_REBUILD:
        print(f'Building {file_name} ...')
        pdf_saver = save_pdf_with_file_name(file_name)
        blast_fishing_figures.save_figure = pdf_saver
        mechanism_figures.save_figure = pdf_saver
        build_figure()


if __name__ == '__main__':
    main()
