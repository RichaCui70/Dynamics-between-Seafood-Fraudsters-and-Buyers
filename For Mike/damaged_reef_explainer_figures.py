"""Why the Damaged reef affordability curve flips with the wholesale supply elasticity.

Compares the original parameterization (eps_sw = 0.95, gamma_P = 1) with the
recommended low-value black market (eps_sw = 0.3, gamma_P calibrated so the
honest equilibrium is unchanged, see ``true_low_value_figures.py``):

    1. damaged_reef_price_decomposition   Pm-bar change split into the buyer
       perception and harvest terms (Shapley), plus F-bar and FP-bar, vs alpha.
    2. damaged_reef_affordability_map     1/Pm-bar over (alpha, r): no value of
       r brings affordability below the honest market under eps_sw = 0.3.
    3. fraud_incentive_threshold          how aware buyers must be before fraud
       stops paying, vs harvest: with eps_sw > eps_sm / 2 scarcity alone ends
       fraud; below it only buyer perception can.

PDFs go to ``paper/figures/true low value black market/tests/``.

Usage:
    python paper/damaged_reef_explainer_figures.py [--preview-directory DIR]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import blast_fishing_figures as figures  # noqa: E402
from blast_fishing_figures import (  # noqa: E402
    DEFAULT_INIT_STATE, DEFAULT_PARAMS, STORIES_BY_NAME, TEXT_WIDTH_INCHES, VAR_COLORS,
)
from true_low_value_figures import OUTPUT_DIRECTORY  # noqa: E402

TESTS_DIRECTORY = os.path.join(OUTPUT_DIRECTORY, 'tests')
DEFAULT_WHOLESALE_ELASTICITY = DEFAULT_PARAMS['e_sw']
RECOMMENDED_WHOLESALE_ELASTICITY = 0.3
HONEST_EQUILIBRIUM_HARVEST = figures.no_fraud_baseline(STORIES_BY_NAME['Baseline'])['harvest_mean']

# (e_sw, column title, line colour)
PARAMETERIZATIONS = (
    (DEFAULT_WHOLESALE_ELASTICITY, r'Original: $\epsilon_{s,w}=0.95,\ \gamma_P=1$', '#3d3c39'),
    (RECOMMENDED_WHOLESALE_ELASTICITY,
     r'Low-value: $\epsilon_{s,w}=0.3,\ \gamma_P=1.32{\times}10^{-3}$', '#2a78d6'),
)
DAMAGED_REEF_GROWTH_RATE = STORIES_BY_NAME['Damaged reef'].params()['r']
BASELINE_GROWTH_RATE = STORIES_BY_NAME['Baseline'].params()['r']
ALPHA_VALUES = np.linspace(0.0, 1.0, 101)
PRIMARY_TEXT_COLOR = '#0b0b0b'
SECONDARY_TEXT_COLOR = '#52514e'
NEUTRAL_LINE_COLOR = '#888780'
PERCEPTION_COLOR = VAR_COLORS['FP']
HARVEST_COLOR = VAR_COLORS['Harvest']


def use_wholesale_elasticity(elasticity: float):
    """Set e_sw and the gamma_P that keeps the default honest equilibrium wholesale price."""
    DEFAULT_PARAMS['e_sw'] = elasticity
    DEFAULT_PARAMS['gamma_p'] = HONEST_EQUILIBRIUM_HARVEST ** (
        DEFAULT_WHOLESALE_ELASTICITY / elasticity - 1)


def save_pdf(figure, file_name: str, preview_directory: str | None):
    os.makedirs(TESTS_DIRECTORY, exist_ok=True)
    figure.savefig(os.path.join(TESTS_DIRECTORY, f'{file_name}.pdf'), bbox_inches='tight',
                   pad_inches=0.02)
    if preview_directory:
        figure.savefig(os.path.join(preview_directory, f'{file_name}.png'), dpi=220,
                       bbox_inches='tight', pad_inches=0.02)
    plt.close(figure)
    print(f'  wrote figures/true low value black market/tests/{file_name}.pdf')


def style_alpha_axis(axis):
    axis.set_xlim(0, 1)
    axis.set_xticks([0, 0.5, 1], ['0', '0.5', '1'])
    axis.grid(axis='y', zorder=0)


# ---------------------------------------------------------------------------
# 1. Price decomposition and fraud shares vs alpha (Damaged reef)
# ---------------------------------------------------------------------------

def figure_price_decomposition(preview_directory=None):
    """Top: Pm-bar change = perception term + harvest term. Bottom: F-bar and FP-bar."""
    figure, axes = plt.subplots(
        2, 2, figsize=(TEXT_WIDTH_INCHES, 3.55), sharex=True, sharey='row',
        gridspec_kw=dict(hspace=0.16, wspace=0.08, height_ratios=(1.35, 1), left=0.13,
                         right=0.985, top=0.765, bottom=0.11),
    )
    for column, (elasticity, title, _) in enumerate(PARAMETERIZATIONS):
        use_wholesale_elasticity(elasticity)
        sweep = figures.blast_sweep(STORIES_BY_NAME['Damaged reef'], ALPHA_VALUES)
        price_axis, share_axis = axes[:, column]

        perception_term = sweep['phi_fp_percent']
        harvest_term = sweep['phi_h_percent']
        net_change = (sweep['market_price_mean_ratio'] - 1) * 100
        price_axis.fill_between(ALPHA_VALUES, 0, harvest_term, color=HARVEST_COLOR, alpha=0.45,
                                linewidth=0, zorder=2)
        price_axis.fill_between(ALPHA_VALUES, 0, perception_term, color=PERCEPTION_COLOR,
                                alpha=0.45, linewidth=0, zorder=2)
        price_axis.plot(ALPHA_VALUES, net_change, color=PRIMARY_TEXT_COLOR, linewidth=1.4,
                        zorder=4)
        price_axis.axhline(0, color=SECONDARY_TEXT_COLOR, linewidth=0.6, zorder=3)
        price_axis.set_title(title, pad=4, fontsize=7.5)
        style_alpha_axis(price_axis)

        share_axis.plot(ALPHA_VALUES, sweep['fraudster_mean'], color=VAR_COLORS['F'], zorder=3)
        share_axis.plot(ALPHA_VALUES, sweep['perception_mean'], color=VAR_COLORS['FP'],
                        linestyle=(0, (4, 1.5)), zorder=3)
        share_axis.axhline(DEFAULT_PARAMS['F_threshold'], color=NEUTRAL_LINE_COLOR,
                           linewidth=0.5, linestyle=(0, (2, 2)), zorder=1)
        share_axis.set_ylim(-0.04, 1.04)
        share_axis.set_xlabel(r'Blast intensity $\alpha$', labelpad=1)
        style_alpha_axis(share_axis)

    axes[1, 1].text(0.98, DEFAULT_PARAMS['F_threshold'] + 0.03, r'$\hat F$', ha='right',
                    va='bottom', fontsize=6.5, color=SECONDARY_TEXT_COLOR)
    axes[0, 0].text(0.55, 0.30, 'Buyers never notice:\nprice rises with scarcity',
                    transform=axes[0, 0].transAxes, ha='center', va='center', fontsize=6,
                    color=PRIMARY_TEXT_COLOR, linespacing=1.15)
    axes[0, 1].text(0.55, 0.90, 'Buyer discount outweighs\nthe scarcity premium',
                    transform=axes[0, 1].transAxes, ha='center', va='top', fontsize=6,
                    color=PRIMARY_TEXT_COLOR, linespacing=1.15)
    axes[1, 0].text(0.55, 0.62, 'Scarcity ends fraud', transform=axes[1, 0].transAxes,
                    ha='center', va='center', fontsize=6, color=PRIMARY_TEXT_COLOR)
    axes[1, 1].text(0.55, 0.30, r'Fraud pinned at $\hat F$; buyers aware',
                    transform=axes[1, 1].transAxes, ha='center', va='center', fontsize=6,
                    color=PRIMARY_TEXT_COLOR)
    axes[0, 0].set_ylabel(r'Change in $\bar P^m$' + '\n(% of honest market)', labelpad=2)
    axes[1, 0].set_ylabel('Time-mean share', labelpad=2)
    figure.align_ylabels(axes[:, 0])

    # Legend fills column by column: row 1 = price terms, row 2 = shares.
    figure.legend(
        handles=[
            Line2D([], [], color=PRIMARY_TEXT_COLOR, linewidth=1.4,
                   label=r'Net change in $\bar P^m$'),
            Line2D([], [], color=VAR_COLORS['F'], linewidth=1.4, label=r'Fraudsters $\bar F$'),
            Patch(facecolor=HARVEST_COLOR, alpha=0.45, label=r'Harvest term (less fish $\uparrow$)'),
            Line2D([], [], color=VAR_COLORS['FP'], linewidth=1.4, linestyle=(0, (4, 1.5)),
                   label=r'Fraud perception $\bar F^p$'),
            Patch(facecolor=PERCEPTION_COLOR, alpha=0.45,
                  label=r'Perception term (buyer discount $\downarrow$)'),
        ],
        loc='upper center', bbox_to_anchor=(0.56, 1.0), ncol=3, frameon=False,
        handlelength=1.8, columnspacing=1.0, fontsize=6.4,
    )
    figure.text(0.56, 0.84, rf'Damaged reef story ($r={DAMAGED_REEF_GROWTH_RATE:g}$); '
                'Shapley split of each time-mean price change',
                ha='center', va='bottom', fontsize=6.3, color=SECONDARY_TEXT_COLOR)
    save_pdf(figure, 'damaged_reef_price_decomposition_esw_0.95_vs_0.3', preview_directory)


# ---------------------------------------------------------------------------
# 2. Affordability over (alpha, r)
# ---------------------------------------------------------------------------

HONEST_DEPLETED_STOCK_LEVEL = 0.1   # honest S-bar_0 below this (x K): fished out before blasting


def affordability_grid(growth_rates, alpha_values) -> tuple[np.ndarray, np.ndarray]:
    """1/Pm-bar relative to each r's honest baseline on an (r, alpha) grid, and honest S-bar_0."""
    honest_params = DEFAULT_PARAMS.copy()
    honest_params.update(figures.blast_params(0.0))
    honest_params['r'] = growth_rates
    honest = figures.summarize_runs(honest_params, {**DEFAULT_INIT_STATE, 'F': 0.0, 'FP': 0.0})

    params = DEFAULT_PARAMS.copy()
    params.update(figures.blast_params(alpha_values[np.newaxis, :]))
    params['r'] = growth_rates[:, np.newaxis]
    blasted = figures.summarize_runs(params, DEFAULT_INIT_STATE)
    affordability = honest['market_price_mean'][:, np.newaxis] / blasted['market_price_mean']
    return affordability, honest['seafood_mean']


def figure_affordability_map(preview_directory=None):
    growth_rates = np.linspace(0.01, 0.25, 49)
    alpha_values = np.linspace(0.02, 1.0, 50)
    colormap = figures.diverging_change_colormap()
    log_limit = 1.0

    figure, axes = plt.subplots(
        1, 2, figsize=(TEXT_WIDTH_INCHES, 2.35), sharey=True,
        gridspec_kw=dict(wspace=0.08, left=0.11, right=0.86, top=0.80, bottom=0.17),
    )
    image = None
    for axis, (elasticity, title, _) in zip(axes, PARAMETERIZATIONS):
        use_wholesale_elasticity(elasticity)
        affordability, honest_stock = affordability_grid(growth_rates, alpha_values)
        log_affordability = np.log10(np.clip(affordability, 1e-3, 1e3))
        image = axis.pcolormesh(alpha_values, growth_rates, log_affordability, cmap=colormap,
                                vmin=-log_limit, vmax=log_limit, shading='nearest',
                                rasterized=True)
        axis.contour(alpha_values, growth_rates, log_affordability, levels=[0.0],
                     colors=PRIMARY_TEXT_COLOR, linewidths=0.8)
        is_depleted = honest_stock < HONEST_DEPLETED_STOCK_LEVEL
        if is_depleted.any():
            depleted_top = growth_rates[is_depleted].max() + 0.5 * np.diff(growth_rates)[0]
            axis.fill_between([alpha_values[0] - 0.01, alpha_values[-1] + 0.01],
                              growth_rates[0] - 0.005, depleted_top, facecolor='none',
                              edgecolor=SECONDARY_TEXT_COLOR, hatch='////', linewidth=0,
                              zorder=3)
            axis.text(0.5, (growth_rates[0] + depleted_top) / 2,
                      'Honest fishery already fished out\n'
                      rf'($\bar S_0 < {HONEST_DEPLETED_STOCK_LEVEL:g}K$): blasting changes nothing',
                      ha='center', va='center', fontsize=5.6, color=PRIMARY_TEXT_COLOR,
                      linespacing=1.1, zorder=4,
                      bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.9,
                                edgecolor='none'))
        for growth_rate, story_name in ((DAMAGED_REEF_GROWTH_RATE, 'Damaged reef'),
                                        (BASELINE_GROWTH_RATE, 'Baseline')):
            axis.axhline(growth_rate, color='white', linewidth=0.9, linestyle=(0, (3, 1.5)))
            axis.text(0.02, growth_rate + 0.004, story_name, ha='left', va='bottom',
                      fontsize=5.8, color=PRIMARY_TEXT_COLOR,
                      bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.8,
                                edgecolor='none'))
        half_step = 0.5 * np.diff(growth_rates)[0]
        axis.set_ylim(growth_rates[0] - half_step, growth_rates[-1] + half_step)
        axis.set_title(title, pad=4, fontsize=7.5)
        axis.set_xlabel(r'Blast intensity $\alpha$', labelpad=1)
        axis.set_xticks([0, 0.5, 1], ['0', '0.5', '1'])
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_color('#a9a7a0')
    axes[0].set_ylabel(r'Intrinsic growth rate $r$', labelpad=2)

    colorbar_axis = figure.add_axes([0.885, 0.20, 0.022, 0.57])
    colorbar = figure.colorbar(image, cax=colorbar_axis)
    colorbar_ticks = (0.1, 0.2, 0.5, 1, 2, 5, 10)
    colorbar.set_ticks([np.log10(tick) for tick in colorbar_ticks],
                       labels=[f'{tick:g}' for tick in colorbar_ticks])
    colorbar.outline.set_linewidth(0.4)
    colorbar.ax.tick_params(labelsize=6.5, length=2)
    colorbar.set_label(r'Affordability $1/\bar P^m$ (× honest)', fontsize=7, labelpad=2)
    figure.legend(
        handles=[Line2D([], [], color=PRIMARY_TEXT_COLOR, linewidth=0.8,
                        label='Affordability = honest market (orange side: worse)')],
        loc='upper center', bbox_to_anchor=(0.49, 1.0), frameon=False, handlelength=1.8,
        fontsize=6.6,
    )
    save_pdf(figure, 'damaged_reef_affordability_map_r_vs_alpha_esw_0.95_vs_0.3',
             preview_directory)


# ---------------------------------------------------------------------------
# 3. Perception needed to stop fraud vs harvest
# ---------------------------------------------------------------------------

def figure_fraud_incentive_threshold(preview_directory=None):
    """(a) Pm (unaware buyers) and Pw vs harvest; (b) perception that makes Delta P = 0.

    Delta P = Pm - Pw <= 0 once (1 - FP)^(eps_d/2) <= Pw / Pm|_{FP=0}, i.e.
    FP* = 1 - (Pw / Pm|_{FP=0})^(2/eps_d). Wholesale prices use F = F_hat = 0.5
    at alpha = 0.5 (P^w_base = 0.9). Markers: Damaged reef time means (H-bar, FP-bar).
    """
    harvest_values = np.logspace(-4, -1, 300)
    gamma_m, e_sm, e_d = DEFAULT_PARAMS['gamma_m'], DEFAULT_PARAMS['e_sm'], DEFAULT_PARAMS['e_d']
    fraud_share, alpha = DEFAULT_PARAMS['F_threshold'], 0.5
    blast = figures.blast_params(alpha)
    base_wholesale_price = (DEFAULT_PARAMS['pw0']
                            + (float(blast['pw1']) - DEFAULT_PARAMS['pw0']) * fraud_share)
    unaware_market_price = gamma_m * harvest_values ** (-e_sm / 2)

    figure, (price_axis, perception_axis) = plt.subplots(
        1, 2, figsize=(TEXT_WIDTH_INCHES, 2.45),
        gridspec_kw=dict(wspace=0.32, left=0.10, right=0.985, top=0.70, bottom=0.17),
    )
    price_axis.plot(harvest_values, unaware_market_price, color=VAR_COLORS['Pm'], linewidth=1.5,
                    zorder=4)
    price_axis.text(8e-3, 150,
                    '$P^m$, unaware\nbuyers ' + r'$\propto H^{-\epsilon_{s,m}/2}$', ha='left',
                    va='bottom', fontsize=5.8, color=PRIMARY_TEXT_COLOR, linespacing=1.1)

    sweep_alphas = np.array([0.1, 0.25, 0.5, 0.75, 1.0])
    for elasticity, title, color in PARAMETERIZATIONS:
        use_wholesale_elasticity(elasticity)
        wholesale_price = (base_wholesale_price
                           / (DEFAULT_PARAMS['gamma_p'] * harvest_values) ** elasticity)
        required_perception = np.clip(
            1 - (wholesale_price / unaware_market_price) ** (2 / e_d), 0, 1)
        price_axis.plot(harvest_values, wholesale_price, color=color, linewidth=1.3,
                        linestyle=(0, (4, 1.5)), zorder=3)
        perception_axis.plot(harvest_values, required_perception, color=color, linewidth=1.4,
                             zorder=3)

        sweep = figures.blast_sweep(STORIES_BY_NAME['Damaged reef'], sweep_alphas)
        perception_axis.scatter(sweep['harvest_mean'], sweep['perception_mean'], s=12,
                                color=color, edgecolors='white', linewidths=0.5, zorder=5)
        crossings = np.nonzero(np.diff(np.sign(unaware_market_price - wholesale_price)))[0]
        if len(crossings):
            crossing_harvest = harvest_values[crossings[0]]
            price_axis.axvline(crossing_harvest, color=color, linewidth=0.6,
                               linestyle=(0, (1, 1.5)), zorder=1)
            price_axis.text(crossing_harvest * 1.15, 2.2,
                            r'$P^w > P^m$ below' + '\nhere: fraud\nunprofitable', ha='left',
                            va='bottom', fontsize=5.5, color=color, linespacing=1.1)

    price_axis.set_xscale('log')
    price_axis.set_yscale('log')
    price_axis.set_xlim(harvest_values[0], harvest_values[-1])
    price_axis.set_ylim(2, 3000)
    price_axis.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda value, _: f'{value:g}'))
    price_axis.set_xlabel(r'Harvest $H$', labelpad=1)
    price_axis.set_ylabel('Price', labelpad=2)
    price_axis.set_title('(a) Wholesale vs market price', pad=3, fontsize=7.5)
    price_axis.grid(axis='y', zorder=0)

    perception_axis.set_xscale('log')
    perception_axis.set_xlim(harvest_values[0], harvest_values[-1])
    perception_axis.set_ylim(-0.04, 1.04)
    perception_axis.set_xlabel(r'Harvest $H$', labelpad=1)
    perception_axis.set_ylabel(r'Fraud perception $F^p$', labelpad=2)
    perception_axis.set_title('(b) Perception needed to stop fraud', pad=3, fontsize=7.5)
    perception_axis.grid(axis='y', zorder=0)
    perception_axis.text(0.04, 0.50, 'Above a curve: ' + r'$\Delta P<0$' + ',\nfraud recedes',
                         transform=perception_axis.transAxes, ha='left', va='center',
                         fontsize=5.8, color=SECONDARY_TEXT_COLOR, linespacing=1.15)

    figure.legend(
        handles=[
            Line2D([], [], color=PARAMETERIZATIONS[0][2], linewidth=1.4,
                   label=PARAMETERIZATIONS[0][1]),
            Line2D([], [], color=PARAMETERIZATIONS[1][2], linewidth=1.4,
                   label=PARAMETERIZATIONS[1][1]),
            Line2D([], [], color=SECONDARY_TEXT_COLOR, linewidth=1.3, linestyle=(0, (4, 1.5)),
                   label=r'Wholesale price $P^w$'),
            Line2D([], [], color=SECONDARY_TEXT_COLOR, marker='o', linestyle='none',
                   markersize=3.5, label=r'Damaged reef $(\bar H, \bar F^p)$, $\alpha$ = 0.1–1'),
        ],
        loc='upper center', bbox_to_anchor=(0.54, 1.0), ncol=2, frameon=False,
        handlelength=1.8, columnspacing=1.2, fontsize=6.4,
    )
    figure.text(0.54, 0.785, rf'$P^w$ at $F=\hat F={fraud_share:g}$, $\alpha={alpha:g}$; '
                r'$\epsilon_{s,m}/2 = 0.5$ separates the two cases',
                ha='center', va='bottom', fontsize=6.2, color=SECONDARY_TEXT_COLOR)
    save_pdf(figure, 'fraud_incentive_scarcity_vs_perception_esw_0.95_vs_0.3', preview_directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--preview-directory', default=None,
                        help='also write PNG previews here')
    arguments = parser.parse_args()
    figures.apply_paper_style()
    for build_figure in (figure_price_decomposition, figure_affordability_map,
                         figure_fraud_incentive_threshold):
        print(f'Building {build_figure.__name__} ...')
        build_figure(arguments.preview_directory)
    use_wholesale_elasticity(DEFAULT_WHOLESALE_ELASTICITY)


if __name__ == '__main__':
    main()
