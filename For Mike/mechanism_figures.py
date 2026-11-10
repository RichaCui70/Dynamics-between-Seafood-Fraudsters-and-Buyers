"""Mechanism figures for hypotheses H1-H3 of the AMMCS 2026 paper.

Each figure isolates one link in the feedback chain, using the same blast-fishing
parameterization (``blast_params``), four Raja Ampat stories, 2000-step horizon
and 1000-step transient as ``blast_fishing_figures.py``:

    Fig. 2A  (H1)  Fraud weakens the resource-effort feedback.
                   Fishing pressure  q E  and break-even stock  S_crit  vs alpha.
    Fig. 2B  (H2)  IUU harvesting pays only conditionally.
                   Payoff advantage  pi_I - pi_L  vs alpha, and its decomposition.
    Fig. 3   (H3)  Consumer awareness only matters when consumers can respond.
                   F, F^p, perception price multiplier and Delta P over time for
                   responsive (eps_d = 1) vs unresponsive (eps_d = 0) buyers.

Usage:
    python paper/mechanism_figures.py                     # all figures
    python paper/mechanism_figures.py --figures fig_3
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

from blast_fishing_figures import (  # noqa: E402
    DEFAULT_INIT_STATE,
    DEFAULT_PARAMS,
    STORIES_BY_NAME,
    Story,
    TEXT_WIDTH_INCHES,
    TOTAL_TIMESTEPS,
    TRANSIENT_TIMESTEPS,
    VAR_COLORS,
    apply_paper_style,
    blast_params,
    iterate_blast_map,
    no_fraud_baseline,
    record_time_series,
    save_figure,
    set_log_multiple_axis,
)

ALPHA_VALUES = np.linspace(0.0, 1.0, 101)
NEUTRAL_LINE_COLOR = '#888780'
SECONDARY_TEXT_COLOR = '#52514e'
PRIMARY_TEXT_COLOR = '#0b0b0b'

# Fig. 2B decomposition colours follow the app's VAR_COLORS semantics:
# catchability -> more harvest (gold), cost -> slate, wholesale price -> brown.
CATCHABILITY_GAIN_COLOR = VAR_COLORS['Harvest']
COST_SAVING_COLOR = VAR_COLORS['Cost']
PRICE_DISCOUNT_COLOR = VAR_COLORS['Pw']

FRAUD_SPREADS_COLOR = VAR_COLORS['F']
FRAUD_RECEDES_COLOR = '#2a78d6'


# ---------------------------------------------------------------------------
# Mechanism quantities
# ---------------------------------------------------------------------------

def break_even_stock_ratio(fraud_share, params: dict):
    """Break-even stock of a fleet with fraud share F, relative to an honest fleet.

    Solving  q P^w S_crit - C = 0  gives  S_crit = C (gamma_P H)^eps_sw / (q P^w_base),
    with P^w_base = P^w_0 + (P^w_1 - P^w_0) F. Two fleets selling into the same
    market share the supply term (gamma_P H)^eps_sw, so their ratio depends on
    the economics alone:

        S_crit(F) / S_crit(0) = [C(F) / C_0] * [q_0 P^w_0 / (q(F) P^w_base(F))].

    At F = 1 this is the IUU-to-legal break-even ratio; at the realized F_t it
    is the whole fleet's break-even relative to an honest fleet.
    """
    q0, pw0, c0 = params['q0'], params['pw0'], params['c0']
    catchability = q0 + (np.asarray(params['q1']) - q0) * fraud_share
    base_wholesale_price = pw0 + (np.asarray(params['pw1']) - pw0) * fraud_share
    fishing_cost = c0 + (np.asarray(params['c1']) - c0) * fraud_share
    return (fishing_cost / c0) * (q0 * pw0) / (catchability * base_wholesale_price)


def payoff_advantage_components(seafood, harvest, params: dict) -> dict:
    """Split  Delta pi_{I-L} = pi_I - pi_L  into three additive parts.

    With pi_k = q_k P^w_k S - C_k and P^w_k = P^w_{k,base} / (gamma_P H)^eps_sw,

        Delta pi = (q_1 - q_0) P^w_0 S / (gamma_P H)^eps_sw    catchability gain
                 + q_1 (P^w_1 - P^w_0) S / (gamma_P H)^eps_sw  price discount
                 + (C_0 - C_1)                                  cost saving.
    """
    q0, q1 = params['q0'], np.asarray(params['q1'])
    pw0, pw1 = params['pw0'], np.asarray(params['pw1'])
    c0, c1 = params['c0'], np.asarray(params['c1'])
    stock_per_supply = seafood / (params['gamma_p'] * harvest) ** params['e_sw']
    return {
        'catchability_gain': (q1 - q0) * pw0 * stock_per_supply,
        'price_discount': q1 * (pw1 - pw0) * stock_per_supply,
        'cost_saving': (c0 - c1) * np.ones_like(seafood),
        'legal_payoff': q0 * pw0 * stock_per_supply - c0,
    }


def mechanism_sweep(story, alpha_values=ALPHA_VALUES) -> dict:
    """Post-transient time means of the H1/H2 quantities over alpha for one story."""
    params = story.params()
    params.update(blast_params(alpha_values))
    sums = {}
    sample_count = 0
    for timestep, values in iterate_blast_map(params, DEFAULT_INIT_STATE, TOTAL_TIMESTEPS):
        if timestep < TRANSIENT_TIMESTEPS:
            continue
        seafood = values['S'].astype(np.float64)
        effort = values['E'].astype(np.float64)
        fraud_share = values['F'].astype(np.float64)
        harvest = values['H'].astype(np.float64)
        catchability = params['q0'] + (params['q1'] - params['q0']) * fraud_share
        wholesale_price = values['Pw'].astype(np.float64)
        # Elasticities of the resource-effort loop at the current state:
        #   stock -> effort:  d log E_{t+1} / d log S_t = gamma_E (1 - eps_sw) q P^w S
        #   effort -> stock: -d log S_{t+1} / d log E_t = gamma_S q E
        stock_to_effort = (params['gamma_e'] * (1 - params['e_sw'])
                           * catchability * wholesale_price * seafood)
        effort_to_stock = params['gamma_s'] * catchability * effort
        current = {
            'fishing_pressure': catchability * effort,
            'seafood': seafood,
            'fraud_share': fraud_share,
            'perception': values['FP'].astype(np.float64),
            'fleet_break_even_ratio': break_even_stock_ratio(fraud_share, params),
            'stock_to_effort_elasticity': stock_to_effort,
            'effort_to_stock_elasticity': effort_to_stock,
            'loop_gain': stock_to_effort * effort_to_stock,
            **payoff_advantage_components(seafood, harvest, params),
        }
        for key, value in current.items():
            sums[key] = sums.get(key, 0.0) + value
        sample_count += 1
    means = {key: value / sample_count for key, value in sums.items()}
    means['payoff_advantage'] = (means['catchability_gain'] + means['price_discount']
                                 + means['cost_saving'])
    return means


def style_alpha_axis(axis):
    axis.set_xlim(0, 1)
    axis.set_xticks([0, 0.25, 0.5, 0.75, 1], ['0', '0.25', '0.5', '0.75', '1'])
    axis.grid(axis='y', zorder=0)


def panel_letter(axis, letter: str):
    axis.text(-0.02, 1.02, f'({letter})', transform=axis.transAxes, ha='right', va='bottom',
              fontsize=8, fontweight='bold', color=PRIMARY_TEXT_COLOR)


# ---------------------------------------------------------------------------
# Fig. 2A (H1) — fishing pressure and break-even stock vs alpha
# ---------------------------------------------------------------------------

def figure_2a_fishing_pressure_and_break_even():
    """(a) Mean fishing pressure q E vs alpha (Baseline); (b) break-even stock vs alpha.

    Panel (a) marks the intrinsic growth rate r: at equilibrium q E = r (1 - S/K),
    so q E -> r means the stock is being fished towards zero. Panel (b) shows the
    break-even stock relative to an honest fleet facing the same market, which
    depends on the economics alone: for an all-IUU fleet (F = 1), and for the
    Baseline story's realized fraud mix (time mean of the ratio at F_t).
    """
    baseline_story = STORIES_BY_NAME['Baseline']
    baseline_sweep = mechanism_sweep(baseline_story)

    figure, (pressure_axis, break_even_axis) = plt.subplots(
        1, 2, figsize=(TEXT_WIDTH_INCHES, 2.1),
        gridspec_kw=dict(wspace=0.46, left=0.10, right=0.985, top=0.86, bottom=0.18),
    )

    pressure_axis.plot(ALPHA_VALUES, baseline_sweep['fishing_pressure'],
                       color=baseline_story.color, linewidth=1.4, zorder=3)
    growth_rate = baseline_story.params()['r']
    pressure_axis.axhline(growth_rate, color=NEUTRAL_LINE_COLOR, linewidth=0.6,
                          linestyle=(0, (2, 2)), zorder=1)
    pressure_axis.text(1.02, growth_rate, rf'$r = {growth_rate:g}$', ha='left', va='center',
                       fontsize=6.3, color=SECONDARY_TEXT_COLOR,
                       transform=pressure_axis.get_yaxis_transform(), clip_on=False)
    style_alpha_axis(pressure_axis)
    pressure_axis.set_ylim(0, 0.25)
    pressure_axis.set_title(r'Fishing pressure $\overline{qE}$', pad=3)
    pressure_axis.text(0.20, 0.155, 'Baseline story', ha='center', va='bottom',
                       fontsize=6.3, color=PRIMARY_TEXT_COLOR)
    pressure_axis.set_xlabel(r'Blast intensity $\alpha$', labelpad=1)
    pressure_axis.set_ylabel('Fraction of stock harvested per step', labelpad=2)
    pressure_axis.text(0.98, 0.04, r'$qE \to r$: stock fished to collapse',
                       transform=pressure_axis.transAxes, ha='right', va='bottom',
                       fontsize=6, color=SECONDARY_TEXT_COLOR)
    panel_letter(pressure_axis, 'a')

    all_iuu_ratio = break_even_stock_ratio(1.0, {**DEFAULT_PARAMS, **blast_params(ALPHA_VALUES)})
    break_even_axis.plot(ALPHA_VALUES, all_iuu_ratio, color=PRIMARY_TEXT_COLOR, linewidth=1.6,
                         zorder=3)
    break_even_axis.axhline(1.0, color=NEUTRAL_LINE_COLOR, linewidth=0.9, zorder=2)
    break_even_axis.plot(ALPHA_VALUES, baseline_sweep['fleet_break_even_ratio'],
                         color=baseline_story.color, linewidth=1.4, zorder=4)
    break_even_axis.text(0.75, 0.55, 'Baseline story\n(realized fraud mix)', ha='center',
                         va='top', fontsize=6.3, color=PRIMARY_TEXT_COLOR, linespacing=1.15)
    break_even_axis.text(0.78, 0.20, 'All-IUU fleet ($F{=}1$)', ha='center', va='bottom',
                         fontsize=6.3, color=PRIMARY_TEXT_COLOR)
    break_even_axis.text(0.98, 1.0, 'Honest fleet ($F{=}0$)', ha='right', va='bottom',
                         fontsize=6.3, color=SECONDARY_TEXT_COLOR)
    style_alpha_axis(break_even_axis)
    break_even_axis.set_ylim(0, 1.1)
    break_even_axis.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ['0', '25%', '50%', '75%', '100%'])
    break_even_axis.set_title(r'Break-even stock $S_{\rm crit}$', pad=3)
    break_even_axis.set_xlabel(r'Blast intensity $\alpha$', labelpad=1)
    break_even_axis.set_ylabel(r'% of honest fleet’s $S_{\rm crit}$', labelpad=2)
    panel_letter(break_even_axis, 'b')

    save_figure(figure, 'mechanism_fig_2a_fishing_pressure_break_even')


SCARCITY_REGIME_COLOR = '#a9a7a0'
SCARCITY_REGIME_PERCEPTION_LEVEL = 0.1   # mean F^p below this = buyers no longer brake fraud


def scarcity_regime_start(sweep: dict) -> float:
    """First alpha from which buyers stay unaware (mean F^p below the level) to alpha = 1.

    Beyond it, the Baseline story is in near-collapse busts and fraud is checked
    by stock scarcity (P^w overtakes P^m as H -> 0), not by buyer perception.
    """
    is_unaware = sweep['perception'] < SCARCITY_REGIME_PERCEPTION_LEVEL
    still_unaware_to_end = np.flip(np.logical_and.accumulate(np.flip(is_unaware)))
    if not still_unaware_to_end.any():
        return float('nan')
    return float(ALPHA_VALUES[np.argmax(still_unaware_to_end)])


def shade_scarcity_regime(axis, regime_start: float):
    if np.isfinite(regime_start):
        axis.axvspan(regime_start, 1.0, color=SCARCITY_REGIME_COLOR, alpha=0.18, linewidth=0,
                     zorder=0)


def figure_2a_collapse_regime():
    """Copy of Fig. 2A with the near-collapse regime shaded and the realized stock added.

    (a) Fishing pressure q E (Baseline). (b) Break-even stock relative to an
    honest fleet (all-IUU and Baseline realized mix) together with the realized
    stock S-bar / S-bar_0, all as % of the honest-market value. The shaded band
    is where buyers stay unaware and stock scarcity, not perception, limits fraud.
    """
    baseline_story = STORIES_BY_NAME['Baseline']
    baseline_sweep = mechanism_sweep(baseline_story)
    honest_stock = no_fraud_baseline(baseline_story)['seafood_mean']
    regime_start = scarcity_regime_start(baseline_sweep)

    figure, (pressure_axis, break_even_axis) = plt.subplots(
        1, 2, figsize=(TEXT_WIDTH_INCHES, 2.3),
        gridspec_kw=dict(wspace=0.46, left=0.10, right=0.985, top=0.78, bottom=0.16),
    )

    shade_scarcity_regime(pressure_axis, regime_start)
    pressure_axis.plot(ALPHA_VALUES, baseline_sweep['fishing_pressure'],
                       color=baseline_story.color, linewidth=1.4, zorder=3)
    growth_rate = baseline_story.params()['r']
    pressure_axis.axhline(growth_rate, color=NEUTRAL_LINE_COLOR, linewidth=0.6,
                          linestyle=(0, (2, 2)), zorder=1)
    pressure_axis.text(1.02, growth_rate, rf'$r = {growth_rate:g}$', ha='left', va='center',
                       fontsize=6.3, color=SECONDARY_TEXT_COLOR,
                       transform=pressure_axis.get_yaxis_transform(), clip_on=False)
    style_alpha_axis(pressure_axis)
    pressure_axis.set_ylim(0, 0.25)
    pressure_axis.set_title(r'Fishing pressure $\overline{qE}$', pad=3)
    pressure_axis.text(0.20, 0.155, 'Baseline story', ha='center', va='bottom',
                       fontsize=6.3, color=PRIMARY_TEXT_COLOR)
    pressure_axis.set_xlabel(r'Blast intensity $\alpha$', labelpad=1)
    pressure_axis.set_ylabel('Fraction of stock harvested per step', labelpad=2)
    pressure_axis.text(0.03, 0.04, r'$qE \to r$: stock fished to collapse',
                       transform=pressure_axis.transAxes, ha='left', va='bottom',
                       fontsize=6, color=SECONDARY_TEXT_COLOR)
    panel_letter(pressure_axis, 'a')

    shade_scarcity_regime(break_even_axis, regime_start)
    all_iuu_ratio = break_even_stock_ratio(1.0, {**DEFAULT_PARAMS, **blast_params(ALPHA_VALUES)})
    break_even_axis.plot(ALPHA_VALUES, all_iuu_ratio, color=PRIMARY_TEXT_COLOR, linewidth=1.6,
                         zorder=3)
    break_even_axis.axhline(1.0, color=NEUTRAL_LINE_COLOR, linewidth=0.9, zorder=2)
    break_even_axis.plot(ALPHA_VALUES, baseline_sweep['fleet_break_even_ratio'],
                         color=baseline_story.color, linewidth=1.4, zorder=4)
    break_even_axis.plot(ALPHA_VALUES, baseline_sweep['seafood'] / honest_stock,
                         color=baseline_story.color, linewidth=1.2, linestyle=(0, (3, 1.5)),
                         zorder=4)
    break_even_axis.text(0.36, 0.73, r'Baseline $S_{\rm crit}$ (realized mix)', ha='left',
                         va='bottom', fontsize=6.0, color=PRIMARY_TEXT_COLOR)
    realized_stock_ratio = baseline_sweep['seafood'] / honest_stock
    arrow_alpha = 0.52
    break_even_axis.annotate(
        r'Realized stock $\bar S/\bar S_0$',
        xy=(arrow_alpha, realized_stock_ratio[int(np.argmin(np.abs(ALPHA_VALUES - arrow_alpha)))]),
        xytext=(0.66, 0.38), ha='left', va='center', fontsize=6.0, color=PRIMARY_TEXT_COLOR,
        arrowprops=dict(arrowstyle='-', color=PRIMARY_TEXT_COLOR, linewidth=0.5,
                        shrinkA=1, shrinkB=1),
    )
    break_even_axis.text(0.13, 0.12, r'All-IUU $S_{\rm crit}$ ($F{=}1$)', ha='left',
                         va='center', fontsize=6.0, color=PRIMARY_TEXT_COLOR)
    break_even_axis.text(0.02, 1.0, 'Honest fleet ($F{=}0$)', ha='left', va='bottom',
                         fontsize=6.0, color=SECONDARY_TEXT_COLOR)
    style_alpha_axis(break_even_axis)
    break_even_axis.set_ylim(0, 1.1)
    break_even_axis.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ['0', '25%', '50%', '75%', '100%'])
    break_even_axis.set_title(r'Break-even stock vs realized stock', pad=3)
    break_even_axis.set_xlabel(r'Blast intensity $\alpha$', labelpad=1)
    break_even_axis.set_ylabel('% of honest-market value', labelpad=2)
    panel_letter(break_even_axis, 'b')

    figure.legend(
        handles=[Patch(facecolor=SCARCITY_REGIME_COLOR, alpha=0.35, linewidth=0,
                       label='Near-collapse busts: stock scarcity, not buyers, limits fraud '
                             rf'($\bar F^p < {SCARCITY_REGIME_PERCEPTION_LEVEL:g}$)')],
        loc='upper center', bbox_to_anchor=(0.53, 1.0), ncol=1, frameon=False,
        handlelength=1.6, fontsize=6.6,
    )
    save_figure(figure, 'mechanism_fig_2a_collapse_regime')


def figure_2a_feedback_strength():
    """Copy of Fig. 2A that measures the resource-effort feedback loop directly.

    Elasticities of the map at the current state, time-averaged (Baseline story):
        stock -> effort   d log E_{t+1} / d log S_t  = gamma_E (1 - eps_sw) q P^w S
        effort -> stock  -d log S_{t+1} / d log E_t  = gamma_S q E
        loop gain         product of the two (strength of the negative loop).
    (a) The depletion signal (stock -> effort) in absolute terms, with the honest
    market as reference; (b) all three indexed to the honest market (alpha = 0).
    """
    baseline_story = STORIES_BY_NAME['Baseline']
    baseline_sweep = mechanism_sweep(baseline_story)
    honest_index = 0   # ALPHA_VALUES[0] = 0: blast parameters equal the honest ones

    figure, (signal_axis, index_axis) = plt.subplots(
        1, 2, figsize=(TEXT_WIDTH_INCHES, 2.25),
        gridspec_kw=dict(wspace=0.42, left=0.125, right=0.985, top=0.86, bottom=0.17),
    )

    stock_to_effort = baseline_sweep['stock_to_effort_elasticity']
    signal_axis.plot(ALPHA_VALUES, stock_to_effort, color=baseline_story.color,
                     linewidth=1.4, zorder=3)
    signal_axis.axhline(stock_to_effort[honest_index], color=NEUTRAL_LINE_COLOR,
                        linewidth=0.9, zorder=2)
    signal_axis.text(0.98, stock_to_effort[honest_index] + 0.00004, 'Honest market',
                     ha='right', va='bottom', fontsize=6.0, color=SECONDARY_TEXT_COLOR)
    signal_axis.text(0.04, 0.05, r'$=\gamma_E\,(1-\epsilon_{s,w})\;qP^wS$',
                     transform=signal_axis.transAxes, ha='left', va='bottom', fontsize=6.3,
                     color=SECONDARY_TEXT_COLOR)
    style_alpha_axis(signal_axis)
    signal_axis.set_ylim(0.006, 0.011)
    signal_axis.set_yticks([0.006, 0.007, 0.008, 0.009, 0.010, 0.011])
    signal_axis.set_title('Depletion signal: stock → effort', pad=3)
    signal_axis.set_xlabel(r'Blast intensity $\alpha$', labelpad=1)
    signal_axis.set_ylabel(r'Elasticity $\partial \log E_{t+1} / \partial \log S_t$',
                           labelpad=2)
    panel_letter(signal_axis, 'a')

    loop_links = (
        ('effort_to_stock_elasticity', 'Effort → stock\n' + r'($\gamma_S\,qE$)',
         VAR_COLORS['E'], (0, (4, 1.5)), 0.78),
        ('loop_gain', 'Loop gain\n(product)', PRIMARY_TEXT_COLOR, '-', 0.80),
        ('stock_to_effort_elasticity', 'Stock → effort', baseline_story.color, '-', 0.30),
    )
    for key, label, color, line_style, label_alpha in loop_links:
        indexed = baseline_sweep[key] / baseline_sweep[key][honest_index]
        index_axis.plot(ALPHA_VALUES, indexed, color=color, linestyle=line_style,
                        linewidth=1.4, zorder=3)
        label_position = int(np.argmin(np.abs(ALPHA_VALUES - label_alpha)))
        is_below = key == 'stock_to_effort_elasticity'
        index_axis.text(label_alpha, indexed[label_position] * (0.90 if is_below else 1.08),
                        label, ha='center', va='top' if is_below else 'bottom', fontsize=6.0,
                        color=PRIMARY_TEXT_COLOR, linespacing=1.1)
    index_axis.axhline(1.0, color=NEUTRAL_LINE_COLOR, linewidth=0.6, zorder=1)
    style_alpha_axis(index_axis)
    index_axis.set_yscale('log')
    index_axis.set_ylim(0.5, 4.5)
    set_log_multiple_axis(index_axis, (0.5, 1, 2, 4))
    index_axis.set_title('Resource–effort loop', pad=3)
    index_axis.set_xlabel(r'Blast intensity $\alpha$', labelpad=1)
    index_axis.set_ylabel('× honest market', labelpad=2)
    panel_letter(index_axis, 'b')

    save_figure(figure, 'mechanism_fig_2a_feedback_strength')


# ---------------------------------------------------------------------------
# Fig. 2A, revised H1 — causal chain: cheaper catch -> lower stock -> weaker signal
# ---------------------------------------------------------------------------

# Parameter set where fraud persists (F = 1 at a steady fixed point for every
# alpha): low wholesale supply elasticity plus a cap on buyer perception.
PERSISTENT_FRAUD_STORY = Story('Persistent fraud', r'$\epsilon_{s,w}{=}0.4,\ F^p_{\max}{=}0.25$',
                               {'e_sw': 0.4, 'FP_max': 0.25}, '#2a78d6')


def figure_2a_revised_hypothesis(story=None, file_stem='mechanism_fig_2a_revised_h1'):
    """Three panels in causal order for the revised H1, one story vs alpha.

    (a) Cost per unit catch value kappa = C / (q P^w_base), relative to the honest
        market (the supply term (gamma_P H)^eps_sw cancels, so this is also the
        break-even stock relative to an honest fleet). Realized fraud mix and the
        all-IUU bound (F = 1).
    (b) Realized stock S-bar / S-bar_0: the biomass at which harvesting settles
        into break-even, since S* rises with kappa.
    (c) Depletion signal d log E_{t+1} / d log S_t (time mean = gamma_E (1 - eps_sw) C-bar)
        and fishing pressure q E, both relative to the honest market.
    The near-collapse regime (buyers unaware, scarcity limits fraud) is shaded.
    """
    story = story or STORIES_BY_NAME['Baseline']
    sweep = mechanism_sweep(story)
    honest_stock = no_fraud_baseline(story)['seafood_mean']
    regime_start = scarcity_regime_start(sweep)
    honest_index = 0
    story_color = story.color

    figure, (kappa_axis, stock_axis, feedback_axis) = plt.subplots(
        1, 3, figsize=(TEXT_WIDTH_INCHES, 2.3),
        gridspec_kw=dict(wspace=0.42, left=0.085, right=0.99, top=0.76, bottom=0.17),
    )
    for axis in (kappa_axis, stock_axis, feedback_axis):
        shade_scarcity_regime(axis, regime_start)
        style_alpha_axis(axis)
        axis.set_xticks([0, 0.5, 1], ['0', '0.5', '1'])
        axis.set_xlabel(r'Blast intensity $\alpha$', labelpad=1)
    percent_ticks = ([0, 0.25, 0.5, 0.75, 1.0], ['0', '25%', '50%', '75%', '100%'])

    # (a) Cause: fraud cheapens the catch.
    all_iuu_ratio = break_even_stock_ratio(1.0, {**DEFAULT_PARAMS, **blast_params(ALPHA_VALUES)})
    kappa_axis.axhline(1.0, color=NEUTRAL_LINE_COLOR, linewidth=0.9, zorder=2)
    kappa_axis.plot(ALPHA_VALUES, all_iuu_ratio, color=PRIMARY_TEXT_COLOR, linewidth=1.4,
                    zorder=3)
    kappa_axis.plot(ALPHA_VALUES, sweep['fleet_break_even_ratio'], color=story_color,
                    linewidth=1.4, zorder=4)
    kappa_axis.text(0.04, 1.02, 'Honest', ha='left', va='bottom', fontsize=5.8,
                    color=SECONDARY_TEXT_COLOR)
    kappa_axis.text(0.30, 0.05, 'All-IUU\n($F{=}1$)', ha='left', va='bottom', fontsize=5.8,
                    color=PRIMARY_TEXT_COLOR, linespacing=1.1)
    kappa_axis.set_ylim(0, 1.12)
    kappa_axis.set_yticks(*percent_ticks)
    kappa_axis.set_ylabel('% of honest market', labelpad=2)
    kappa_axis.set_title('(a) Fraud cheapens\nthe catch: ' + r'$C/(qP^w)$', pad=3, fontsize=7.5)

    # (b) Consequence: harvesting breaks even at a lower stock.
    stock_axis.axhline(1.0, color=NEUTRAL_LINE_COLOR, linewidth=0.9, zorder=2)
    stock_axis.plot(ALPHA_VALUES, sweep['seafood'] / honest_stock, color=story_color,
                    linewidth=1.4, zorder=4)
    stock_axis.set_ylim(0, 1.12)
    stock_axis.set_yticks(*percent_ticks)
    stock_axis.set_title('(b) …so fishing pays at\na lower stock: ' + r'$\bar S/\bar S_0$',
                         pad=3, fontsize=7.5)

    # (c) Feedback: the depletion signal weakens while pressure rises.
    draw_signal_and_pressure(feedback_axis, sweep, story_color)
    feedback_axis.set_ylabel('× honest market', labelpad=1)
    feedback_axis.set_title('(c) …the depletion signal\nweakens, pressure rises', pad=3,
                            fontsize=7.5)

    legend_handles = [
        Line2D([], [], color=story_color, linewidth=1.4,
               label=f'{story.name} story'),
        Line2D([], [], color=NEUTRAL_LINE_COLOR, linewidth=0.9, label='Honest market'),
    ]
    if np.isfinite(regime_start):
        legend_handles.append(Patch(facecolor=SCARCITY_REGIME_COLOR, alpha=0.35, linewidth=0,
                                    label='Near-collapse busts'))
    figure.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.53, 1.0),
                  ncol=len(legend_handles), frameon=False, handlelength=1.8, fontsize=6.6,
                  columnspacing=1.4)
    save_figure(figure, file_stem)


def value_at_alpha(series, alpha: float):
    return series[int(np.argmin(np.abs(ALPHA_VALUES - alpha)))]


def indexed_to_honest(sweep: dict, key: str):
    """Series divided by its alpha = 0 value (blast parameters equal the honest ones)."""
    return sweep[key] / sweep[key][0]


def set_adaptive_multiple_axis(axis, series_list, floor=0.4, ceiling=5.0):
    """Log 'x honest market' axis that always fits every series."""
    axis.set_yscale('log')
    lower_limit = min(floor, 0.8 * min(float(np.min(series)) for series in series_list))
    upper_limit = max(ceiling, 1.8 * max(float(np.max(series)) for series in series_list))
    axis.set_ylim(lower_limit, upper_limit)
    candidate_ticks = (0.05, 0.1, 0.25, 0.5, 1, 2, 4, 10, 30, 100, 300)
    set_log_multiple_axis(axis, tuple(
        tick for tick in candidate_ticks if lower_limit <= tick <= upper_limit))


def draw_signal_and_pressure(axis, sweep: dict, signal_color: str, label_alpha: float = 0.35):
    """Depletion signal and fishing pressure, both relative to the honest market."""
    signal_ratio = indexed_to_honest(sweep, 'stock_to_effort_elasticity')
    pressure_ratio = indexed_to_honest(sweep, 'fishing_pressure')
    axis.axhline(1.0, color=NEUTRAL_LINE_COLOR, linewidth=0.9, zorder=2)
    axis.plot(ALPHA_VALUES, pressure_ratio, color=VAR_COLORS['E'], linestyle=(0, (4, 1.5)),
              linewidth=1.4, zorder=3)
    axis.plot(ALPHA_VALUES, signal_ratio, color=signal_color, linewidth=1.4, zorder=4)
    set_adaptive_multiple_axis(axis, (signal_ratio, pressure_ratio))
    axis.text(label_alpha + 0.03, value_at_alpha(pressure_ratio, label_alpha),
              r'Pressure $qE$', ha='left', va='top', fontsize=5.8, color=PRIMARY_TEXT_COLOR)
    axis.text(label_alpha, value_at_alpha(signal_ratio, label_alpha) * 0.72,
              'Signal ' + r'$\propto \bar C$', ha='left', va='top', fontsize=5.8,
              color=PRIMARY_TEXT_COLOR)


def single_panel_legend(figure, story_color, regime_start, extra_handles=(), columns=None):
    legend_handles = list(extra_handles) + [
        Line2D([], [], color=NEUTRAL_LINE_COLOR, linewidth=0.9, label='Honest market')]
    if np.isfinite(regime_start):
        legend_handles.append(Patch(facecolor=SCARCITY_REGIME_COLOR, alpha=0.35, linewidth=0,
                                    label='Near-collapse busts'))
    figure.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.55, 1.0),
                  ncol=columns or len(legend_handles), frameon=False, handlelength=1.8,
                  fontsize=6.4, columnspacing=1.0, labelspacing=0.3)


SINGLE_PANEL_WIDTH_INCHES = 0.62 * TEXT_WIDTH_INCHES


def figure_signal_and_pressure(story=None, file_stem='mechanism_fig_signal_and_pressure'):
    """Standalone copy of Fig. 2A (revised) panel (c): signal and pressure vs alpha."""
    story = story or STORIES_BY_NAME['Baseline']
    sweep = mechanism_sweep(story)
    regime_start = scarcity_regime_start(sweep)

    figure, axis = plt.subplots(
        figsize=(SINGLE_PANEL_WIDTH_INCHES, 2.4),
        gridspec_kw=dict(left=0.17, right=0.97, top=0.75, bottom=0.16),
    )
    shade_scarcity_regime(axis, regime_start)
    style_alpha_axis(axis)
    draw_signal_and_pressure(axis, sweep, story.color)
    axis.set_xlabel(r'Blast intensity $\alpha$', labelpad=1)
    axis.set_ylabel('× honest market', labelpad=1)
    axis.set_title(f'Depletion signal weakens, pressure rises\n({story.name} story)', pad=3,
                   fontsize=7.5)
    single_panel_legend(figure, story.color, regime_start)
    save_figure(figure, file_stem)


def figure_revised_hypothesis_single_panel(story=None,
                                           file_stem='mechanism_fig_revised_h1_single_panel'):
    """All four clauses of the revised H1 on one 'x honest market' axis.

    Cost per unit catch value C/(q P^w) (cause), realized stock S-bar (biomass at
    which harvesting breaks even), depletion signal and fishing pressure q E.
    """
    story = story or STORIES_BY_NAME['Baseline']
    sweep = mechanism_sweep(story)
    honest_stock = no_fraud_baseline(story)['seafood_mean']
    regime_start = scarcity_regime_start(sweep)
    series = (
        # (values, colour, line style, legend label, direct label, label alpha, offset)
        (sweep['fleet_break_even_ratio'], VAR_COLORS['Cost'], '-',
         r'Cost per catch value $C/(qP^w)$', None, None, None),
        (indexed_to_honest(sweep, 'stock_to_effort_elasticity'), '#d6342e', (0, (5, 1.2)),
         r'Depletion signal $\propto \bar C$', None, None, None),
        (sweep['seafood'] / honest_stock, VAR_COLORS['S'], (0, (1.5, 1)),
         r'Stock $\bar S$', r'Stock $\bar S$', 0.47, 0.80),
        (indexed_to_honest(sweep, 'fishing_pressure'), VAR_COLORS['E'], (0, (4, 1.5)),
         r'Pressure $qE$', r'Pressure $qE$', 0.38, 0.93),
    )

    figure, axis = plt.subplots(
        figsize=(SINGLE_PANEL_WIDTH_INCHES, 2.75),
        gridspec_kw=dict(left=0.17, right=0.97, top=0.77, bottom=0.14),
    )
    shade_scarcity_regime(axis, regime_start)
    style_alpha_axis(axis)
    axis.axhline(1.0, color=NEUTRAL_LINE_COLOR, linewidth=0.9, zorder=2)
    for values, color, line_style, _, direct_label, label_alpha, offset in series:
        axis.plot(ALPHA_VALUES, values, color=color, linestyle=line_style, linewidth=1.4,
                  zorder=3)
        if direct_label:
            axis.text(label_alpha + 0.03, value_at_alpha(values, label_alpha) * offset,
                      direct_label, ha='left', va='top', fontsize=5.8,
                      color=PRIMARY_TEXT_COLOR)
    set_adaptive_multiple_axis(axis, [values for values, *_ in series])
    axis.set_xlabel(r'Blast intensity $\alpha$', labelpad=1)
    axis.set_ylabel('× honest market', labelpad=1)
    axis.set_title('Fraud weakens the depletion signal and raises pressure', pad=3, fontsize=7.5)
    series_handles = [Line2D([], [], color=color, linestyle=line_style, linewidth=1.4,
                             label=legend_label)
                      for _, color, line_style, legend_label, *_ in series]
    single_panel_legend(figure, story.color, regime_start, series_handles, columns=2)
    save_figure(figure, file_stem)


# ---------------------------------------------------------------------------
# Fig. 2B (H2) — IUU payoff advantage vs alpha
# ---------------------------------------------------------------------------

def figure_2b_iuu_payoff_advantage():
    """Decomposition of Delta pi_{I-L} vs alpha for the Baseline story.

    Both payoffs are evaluated at the realized post-transient market (S_t, H_t)
    and time-averaged. The catchability gain and cost saving are stacked above
    zero and the price discount below it, with the net advantage on top.
    """
    baseline = mechanism_sweep(STORIES_BY_NAME['Baseline'])

    figure, decomposition_axis = plt.subplots(
        figsize=(0.68 * TEXT_WIDTH_INCHES, 2.25),
        gridspec_kw=dict(left=0.15, right=0.98, top=0.90, bottom=0.17),
    )

    catchability_gain = baseline['catchability_gain']
    cost_saving = baseline['cost_saving']
    price_discount = baseline['price_discount']
    surface_gap = dict(edgecolor='white', linewidth=0.8)
    decomposition_axis.fill_between(ALPHA_VALUES, 0, cost_saving, color=COST_SAVING_COLOR,
                                    alpha=0.55, zorder=2, **surface_gap)
    decomposition_axis.fill_between(ALPHA_VALUES, cost_saving, cost_saving + catchability_gain,
                                    color=CATCHABILITY_GAIN_COLOR, alpha=0.55, zorder=2,
                                    **surface_gap)
    decomposition_axis.fill_between(ALPHA_VALUES, price_discount, 0, color=PRICE_DISCOUNT_COLOR,
                                    alpha=0.55, zorder=2, **surface_gap)
    decomposition_axis.plot(ALPHA_VALUES, baseline['payoff_advantage'],
                            color=PRIMARY_TEXT_COLOR, linewidth=1.4, zorder=4)
    decomposition_axis.axhline(0, color=SECONDARY_TEXT_COLOR, linewidth=0.6, zorder=3)

    def value_at(series, alpha):
        return series[int(np.argmin(np.abs(ALPHA_VALUES - alpha)))]

    direct_labels = (
        (0.80, value_at(cost_saving, 0.80) + 0.25 * value_at(catchability_gain, 0.80),
         r'Catchability gain $\uparrow q$'),
        (0.80, 0.5 * value_at(cost_saving, 0.80), r'Cost saving $\downarrow C$'),
        (0.80, 0.5 * value_at(price_discount, 0.80), r'Price discount $\downarrow P^w$'),
    )
    for x_position, y_position, text in direct_labels:
        decomposition_axis.text(x_position, y_position, text, ha='center', va='center',
                                fontsize=6, color=PRIMARY_TEXT_COLOR, zorder=5)
    net_label_alpha = 0.30
    decomposition_axis.annotate(
        r'Net $\Delta\pi_{I-L}$', xy=(net_label_alpha, value_at(baseline['payoff_advantage'],
                                                                 net_label_alpha)),
        xytext=(0.17, 1.75), fontsize=6.3, color=PRIMARY_TEXT_COLOR, ha='center',
        arrowprops=dict(arrowstyle='-', color=PRIMARY_TEXT_COLOR, linewidth=0.5,
                        shrinkA=1, shrinkB=1),
    )
    style_alpha_axis(decomposition_axis)
    decomposition_axis.set_title(r'IUU payoff advantage $\Delta\pi_{I-L}$ (Baseline story)',
                                 pad=3)
    decomposition_axis.set_xlabel(r'Blast intensity $\alpha$', labelpad=1)
    decomposition_axis.set_ylabel(r'$\pi_I - \pi_L$ (profit per unit effort)', labelpad=2)
    save_figure(figure, 'mechanism_fig_2b_iuu_payoff_advantage')


# ---------------------------------------------------------------------------
# Fig. 3 (H3) — responsive vs unresponsive consumers over time
# ---------------------------------------------------------------------------

def figure_3_consumer_response(alpha: float = 0.15, window_timesteps: int = 200):
    """Time series from the default initial state, Baseline vs No substitutes.

    Rows follow the causal chain: (1) F crosses F_hat and F^p rises;
    (2) the perception price multiplier (1 - F^p)^(eps_d / 2) falls only if
    eps_d > 0; (3) the fraud incentive Delta P = P^m - P^w, which sets
    logit(F_{t+1}) - logit(F_t) = gamma_F Delta P, turns negative only then.
    """
    consumer_cases = (
        (STORIES_BY_NAME['Baseline'], 'Responsive buyers', r'$\epsilon_d = 1$'),
        (STORIES_BY_NAME['No substitutes'], 'Unresponsive buyers (no substitutes)',
         r'$\epsilon_d = 0$'),
    )
    case_series = []
    for story, _, _ in consumer_cases:
        params = story.params()
        params.update({key: float(value) for key, value in blast_params(alpha).items()})
        series = record_time_series(params, DEFAULT_INIT_STATE, window_timesteps)
        series['multiplier'] = (1 - series['FP']) ** (params['e_d'] / 2)
        series['incentive'] = series['Pm'] - series['Pw']
        case_series.append((params, series))

    figure, axes = plt.subplots(
        3, 2, figsize=(TEXT_WIDTH_INCHES, 3.45), sharex=True, sharey='row',
        gridspec_kw=dict(hspace=0.16, wspace=0.08, left=0.15, right=0.985, top=0.79,
                         bottom=0.11),
    )
    incentive_limit = 1.15 * max(np.max(np.abs(series['incentive']))
                                 for _, series in case_series)

    for column, ((story, title, parameter_label), (params, series)) in enumerate(
            zip(consumer_cases, case_series)):
        share_axis, multiplier_axis, incentive_axis = axes[:, column]
        time_axis = np.arange(len(series['F']))
        fraud_recedes = series['incentive'] < 0

        for axis in axes[:, column]:
            axis.fill_between(time_axis, 0, 1, where=fraud_recedes, color=FRAUD_RECEDES_COLOR,
                              alpha=0.08, linewidth=0, transform=axis.get_xaxis_transform(),
                              step='mid', zorder=0)

        share_axis.plot(time_axis, series['F'], color=VAR_COLORS['F'], zorder=3)
        share_axis.plot(time_axis, series['FP'], color=VAR_COLORS['FP'],
                        linestyle=(0, (4, 1.5)), zorder=3)
        share_axis.axhline(params['F_threshold'], color=NEUTRAL_LINE_COLOR, linewidth=0.5,
                           linestyle=(0, (2, 2)), zorder=1)
        if column == 1:
            share_axis.text(time_axis[-1], params['F_threshold'] + 0.03, r'$\hat F$',
                            ha='right', va='bottom', fontsize=6.5,
                            color=SECONDARY_TEXT_COLOR)
        share_axis.set_ylim(-0.04, 1.04)
        share_axis.set_title(f'{title}, {parameter_label}', pad=4, fontsize=7.5)

        multiplier_axis.plot(time_axis, series['multiplier'], color=VAR_COLORS['Pm'], zorder=3)
        multiplier_axis.axhline(1.0, color=NEUTRAL_LINE_COLOR, linewidth=0.5, zorder=1)
        multiplier_axis.set_yscale('log')
        multiplier_axis.set_ylim(2e-3, 2.0)
        multiplier_axis.set_yticks([0.01, 0.1, 1], ['0.01', '0.1', '1'])
        multiplier_axis.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())

        incentive = series['incentive']
        incentive_axis.fill_between(time_axis, 0, incentive, where=incentive >= 0,
                                    color=FRAUD_SPREADS_COLOR, alpha=0.30, linewidth=0,
                                    interpolate=True, zorder=2)
        incentive_axis.fill_between(time_axis, 0, incentive, where=incentive < 0,
                                    color=FRAUD_RECEDES_COLOR, alpha=0.30, linewidth=0,
                                    interpolate=True, zorder=2)
        incentive_axis.plot(time_axis, incentive, color=PRIMARY_TEXT_COLOR, linewidth=1.0,
                            zorder=3)
        incentive_axis.axhline(0, color=SECONDARY_TEXT_COLOR, linewidth=0.6, zorder=1)
        incentive_axis.set_ylim(-incentive_limit, incentive_limit)
        incentive_axis.set_xlim(0, time_axis[-1])
        incentive_axis.set_xlabel('Time step', labelpad=1)

    axes[2, 1].text(0.5, 0.70, 'Fraud always pays', transform=axes[2, 1].transAxes,
                    ha='center', va='center', fontsize=6.3, color=PRIMARY_TEXT_COLOR)
    axes[1, 1].text(0.5, 0.55, 'Price ignores perception', transform=axes[1, 1].transAxes,
                    ha='center', va='center', fontsize=6.3, color=SECONDARY_TEXT_COLOR)

    axes[0, 0].set_ylabel('Share', labelpad=2)
    axes[1, 0].set_ylabel('Price multiplier\n' + r'$(1-F^p)^{\epsilon_d/2}$', labelpad=2)
    axes[2, 0].set_ylabel('Fraud incentive\n' + r'$\Delta P = P^m - P^w$', labelpad=2)
    figure.align_ylabels(axes[:, 0])

    figure.legend(
        handles=[
            Line2D([], [], color=VAR_COLORS['F'], linewidth=1.4, label=r'Fraudsters $F$'),
            Line2D([], [], color=VAR_COLORS['FP'], linewidth=1.4, linestyle=(0, (4, 1.5)),
                   label=r'Fraud perception $F^p$'),
            Patch(facecolor=FRAUD_SPREADS_COLOR, alpha=0.30,
                  label=r'$\Delta P > 0$: fraud spreads'),
            Patch(facecolor=FRAUD_RECEDES_COLOR, alpha=0.30,
                  label=r'$\Delta P < 0$: fraud recedes (shaded)'),
        ],
        loc='upper center', bbox_to_anchor=(0.56, 1.0), ncol=2, frameon=False,
        handlelength=2.0, columnspacing=1.2,
    )
    figure.text(0.56, 0.845, rf'Blast intensity $\alpha = {alpha}$, from '
                rf'$S_0={DEFAULT_INIT_STATE["S"]}$, $F_0=F^p_0={DEFAULT_INIT_STATE["F"]}$',
                ha='center', va='bottom', fontsize=6.3, color=SECONDARY_TEXT_COLOR)
    save_figure(figure, 'mechanism_fig_3_consumer_response')


FIGURE_BUILDERS = {
    'fig_2a': figure_2a_fishing_pressure_and_break_even,
    'fig_2a_collapse_regime': figure_2a_collapse_regime,
    'fig_2a_feedback_strength': figure_2a_feedback_strength,
    'fig_2a_revised_h1': figure_2a_revised_hypothesis,
    'fig_2a_revised_h1_persistent_fraud': lambda: figure_2a_revised_hypothesis(
        PERSISTENT_FRAUD_STORY, 'mechanism_fig_2a_revised_h1_persistent_fraud'),
    'fig_signal_and_pressure': figure_signal_and_pressure,
    'fig_revised_h1_single_panel': figure_revised_hypothesis_single_panel,
    'fig_2b': figure_2b_iuu_payoff_advantage,
    'fig_3': figure_3_consumer_response,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--figures', nargs='+', choices=sorted(FIGURE_BUILDERS),
                        default=list(FIGURE_BUILDERS))
    arguments = parser.parse_args()
    apply_paper_style()
    for figure_name in arguments.figures:
        print(f'Building {figure_name} ...')
        FIGURE_BUILDERS[figure_name]()


if __name__ == '__main__':
    main()
