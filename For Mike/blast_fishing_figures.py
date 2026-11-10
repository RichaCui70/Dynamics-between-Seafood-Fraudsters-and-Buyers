"""Prototype figures for the Blast Fishing section of the AMMCS 2026 paper.

Condenses the app's 4 x 20 grid of blast-fishing time series + heatmaps into a
handful of print-ready figures sized for the Springer ``svmult`` text block
(11.7 cm ~ 4.6 in wide). Every figure compares the four Raja Ampat stories from
the app's Blast Fishing "Plots" tab:

    Baseline        r = 0.225, eps_d = 1, F_hat = 0.5   (DEFAULT_PARAMS)
    Damaged reef    r = 0.08                            (low regrowth)
    No substitutes  eps_d = 0                           (buyers cannot exit)
    Aware buyers    F_hat = 0.05                        (low detection threshold)

on the three outcome metrics used throughout the paper: post-transient average
seafood stock S-bar, harvest H-bar and market price Pm-bar, each as % change vs
the story's own no-fraud, no-blast baseline (F = FP = 0, alpha = 0).

Simulation uses a vectorized copy of ``DynamicalSystem``'s dimensionalized map
(same long-double arithmetic and clipping) so whole alpha x F_hat grids run in
one pass. ``--verify`` checks it step-for-step against ``core.System``.

Usage:
    python paper/blast_fishing_figures.py                  # all figures
    python paper/blast_fishing_figures.py --figures policy_maps outcome_curves
    python paper/blast_fishing_figures.py --verify
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPOSITORY_ROOT)

from core.constants import DEFAULT_INIT_STATE, DEFAULT_PARAMS, VAR_COLORS  # noqa: E402
from core.metrics import shapley_fp_and_harvest  # noqa: E402

FIGURE_OUTPUT_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')

# Same horizon as the app's story tab: 2000 steps, first 1000 discarded.
TOTAL_TIMESTEPS = 2000
TRANSIENT_TIMESTEPS = 1000

# svmult text block width; figures are drawn at final print size.
TEXT_WIDTH_INCHES = 4.6

COLLAPSE_STOCK_LEVEL = 0.05       # S_min below this (x K) = near-collapse busts
STEADY_STOCK_VARIATION = 1e-3     # coefficient of variation below this = steady state

_LONG_DOUBLE = np.longdouble
_CLOSE_TO_ZERO = np.finfo(_LONG_DOUBLE).eps
_CLOSE_TO_ONE = 1 - np.finfo(_LONG_DOUBLE).epsneg


# ---------------------------------------------------------------------------
# Stories
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Story:
    """One Raja Ampat parameter story layered on DEFAULT_PARAMS."""
    name: str
    parameter_label: str
    param_overrides: dict = field(default_factory=dict)
    color: str = '#2a78d6'
    line_style: str = '-'
    marker: str = 'o'

    def params(self) -> dict:
        params = DEFAULT_PARAMS.copy()
        params.update(self.param_overrides)
        return params

    @property
    def legend_label(self) -> str:
        return f'{self.name} ({self.parameter_label})'


# Categorical slots 1-4 in fixed order (matches STORY_COLORS in scenario_bf).
# Orange and yellow sit close under CVD, so every story also has its own
# line style and marker for print / greyscale.
STORIES = (
    Story('Baseline', r'$r{=}0.225,\ \epsilon_d{=}1,\ \hat F{=}0.5$', {},
          '#2a78d6', '-', 'o'),
    Story('Damaged reef', r'$r{=}0.08$', {'r': 0.08},
          '#eb6834', (0, (5, 1.5)), 's'),
    Story('No substitutes', r'$\epsilon_d{=}0$', {'e_d': 0.0},
          '#1baf7a', (0, (1, 1)), '^'),
    Story('Aware buyers', r'$\hat F{=}0.05$', {'F_threshold': 0.05},
          '#eda100', (0, (4, 1, 1, 1)), 'D'),
)
STORIES_BY_NAME = {story.name: story for story in STORIES}

OUTCOME_METRICS = (
    ('seafood_mean', r'Seafood stock $\bar S$'),
    ('harvest_mean', r'Harvest $\bar H$'),
    ('market_price_mean', r'Market price $\bar P^m$'),
)

REGIME_STEADY, REGIME_CYCLING, REGIME_COLLAPSE = 0, 1, 2
REGIME_LABELS = ('Steady state', 'Boom–bust cycles', 'Near-collapse busts')
REGIME_COLORS = ('#e8e6e1', '#a9a7a0', '#3d3c39')

# Welfare framing used by Figure A (no-dynamics copy) and Figure F: every
# metric reads "up = better for people", so market price is shown as
# affordability = baseline Pm-bar / Pm-bar (the log-axis mirror of the price
# change; e.g. price +900% -> affordability -90%). The alpha = 0 point is the
# honest-market baseline itself (0% by definition), matching the "Baseline"
# column of the app's time-series heatmaps. Line colours follow the app's
# VAR_COLORS (stock = blue, harvest = gold, market price = orange).
# (metric key, panel title, legend label, invert?, line colour)
WELFARE_METRICS = (
    ('seafood_mean', r'Seafood stock $\bar S$', r'Seafood stock $\bar S$', False,
     VAR_COLORS['S']),
    ('harvest_mean', r'Food security $\bar H$', r'Food security $\bar H$', False,
     VAR_COLORS['Harvest']),
    ('market_price_mean', r'Food affordability $1/\bar P^m$',
     r'Food affordability $1/\bar P^m$', True, VAR_COLORS['Pm']),
)
WELFARE_Y_FLOOR, WELFARE_Y_CEILING = 0.075, 11

WORSE_THAN_BASELINE_COLOR = '#d6342e'
WORSE_THAN_BASELINE_TEXT_COLOR = '#a3231e'


def blast_params(alpha) -> dict:
    """Blast economics at intensity alpha (vectorized ``scenario_bf._blast_params``)."""
    alpha = np.asarray(alpha, dtype=np.float64)
    return {
        'q1': DEFAULT_PARAMS['q0'] + alpha * 0.33,
        'pw1': DEFAULT_PARAMS['pw0'] - alpha * 0.40,
        'c1': DEFAULT_PARAMS['c0'] - alpha * 0.80,
    }


# ---------------------------------------------------------------------------
# Vectorized dimensionalized map (mirrors core.System.DynamicalSystem)
# ---------------------------------------------------------------------------

def _logistic_share_update(share, exponent, share_min, share_max):
    """F / FP logistic update with DynamicalSystem's clipping and 0/1 fixed points."""
    with np.errstate(over='ignore', invalid='ignore'):
        growth_factor = np.exp(exponent)
        next_share = share * growth_factor / (1 + share * (growth_factor - 1))
        next_share = np.where(np.isfinite(growth_factor), next_share, 1.0)
    lower = np.maximum(_CLOSE_TO_ZERO, share_min)
    upper = np.minimum(_CLOSE_TO_ONE, share_max)
    next_share = np.clip(next_share, lower, upper)
    is_fixed_point = (share == 0.0) | (share == 1.0)
    return np.where(is_fixed_point, share, next_share)


def iterate_blast_map(params: dict, initial_state: dict, total_timesteps: int):
    """Yield state + observables at t = 0..total_timesteps for a batch of runs.

    Any parameter may be an array; all arrays broadcast to one batch shape.

    Yields:
        (timestep, dict) with long-double arrays for S, E, F, FP and the
        contemporaneous harvest H, market price Pm and wholesale price Pw.
    """
    p = {key: np.asarray(value, dtype=_LONG_DOUBLE) for key, value in params.items()}
    batch_shape = np.broadcast_shapes(*(value.shape for value in p.values()))
    seafood, effort, fraudsters, perception = (
        np.broadcast_to(np.asarray(initial_state[key], dtype=_LONG_DOUBLE), batch_shape).copy()
        for key in ('S', 'E', 'F', 'FP')
    )
    for timestep in range(total_timesteps + 1):
        catchability = p['q0'] + (p['q1'] - p['q0']) * fraudsters
        harvest = np.maximum(catchability * effort * seafood, _CLOSE_TO_ZERO)
        market_price = np.maximum(
            p['gamma_m'] * np.sqrt((1 - perception) ** p['e_d'] / harvest ** p['e_sm']),
            _CLOSE_TO_ZERO,
        )
        wholesale_price = np.maximum(
            (fraudsters * (p['pw1'] - p['pw0']) + p['pw0'])
            / (p['gamma_p'] * harvest) ** p['e_sw'],
            _CLOSE_TO_ZERO,
        )
        fishing_cost = fraudsters * (p['c1'] - p['c0']) + p['c0']
        yield timestep, {
            'S': seafood, 'E': effort, 'F': fraudsters, 'FP': perception,
            'H': harvest, 'Pm': market_price, 'Pw': wholesale_price,
        }
        with np.errstate(over='ignore'):
            next_seafood = np.maximum(
                seafood * np.exp(p['gamma_s'] * (p['r'] * (1 - seafood / p['K'])
                                                 - catchability * effort)),
                _CLOSE_TO_ZERO,
            )
            next_effort = np.maximum(
                effort * np.exp(p['gamma_e'] * (catchability * wholesale_price * seafood
                                                - fishing_cost)),
                _CLOSE_TO_ZERO,
            )
        next_fraudsters = _logistic_share_update(
            fraudsters, p['gamma_f'] * (market_price - wholesale_price), p['F_min'], p['F_max'],
        )
        next_perception = _logistic_share_update(
            perception, p['gamma_fp'] * (fraudsters - p['F_threshold']),
            p['FP_min'], p['FP_max'],
        )
        seafood, effort, fraudsters, perception = (
            next_seafood, next_effort, next_fraudsters, next_perception,
        )


def record_time_series(params: dict, initial_state: dict, total_timesteps: int,
                       record_from_timestep: int = 0) -> dict:
    """Full float64 time series (t >= record_from_timestep) for each variable."""
    recorded = {key: [] for key in ('S', 'E', 'F', 'FP', 'H', 'Pm', 'Pw')}
    for timestep, values in iterate_blast_map(params, initial_state, total_timesteps):
        if timestep >= record_from_timestep:
            for key in recorded:
                recorded[key].append(values[key].astype(np.float64))
    return {key: np.array(series) for key, series in recorded.items()}


def summarize_runs(params: dict, initial_state: dict, *, harvest_reference=None,
                   total_timesteps: int = TOTAL_TIMESTEPS,
                   transient_timesteps: int = TRANSIENT_TIMESTEPS) -> dict:
    """Post-transient summary statistics, accumulated online (no full series kept).

    When ``harvest_reference`` is given, also accumulates the per-timestep
    Shapley split of Pm into fraud perception (phi_FP) and harvest (phi_H)
    against (FP = 0, H = harvest_reference), as in ``core.metrics``.
    """
    sums = {}
    seafood_min = seafood_max = None
    sample_count = 0
    for timestep, values in iterate_blast_map(params, initial_state, total_timesteps):
        if timestep < transient_timesteps:
            continue
        seafood = values['S'].astype(np.float64)
        current = {
            'seafood': seafood,
            'seafood_squared': seafood ** 2,
            'harvest': values['H'].astype(np.float64),
            'market_price': values['Pm'].astype(np.float64),
            'fraudsters': values['F'].astype(np.float64),
            'perception': values['FP'].astype(np.float64),
        }
        if harvest_reference is not None:
            current['phi_fp'], current['phi_h'] = shapley_fp_and_harvest(
                current['perception'], current['harvest'], 0.0, harvest_reference,
                params['gamma_m'], params['e_d'], params['e_sm'],
            )
        for key, value in current.items():
            sums[key] = sums.get(key, 0.0) + value
        seafood_min = seafood if seafood_min is None else np.minimum(seafood_min, seafood)
        seafood_max = seafood if seafood_max is None else np.maximum(seafood_max, seafood)
        sample_count += 1

    means = {key: value / sample_count for key, value in sums.items()}
    seafood_std = np.sqrt(np.maximum(means['seafood_squared'] - means['seafood'] ** 2, 0.0))
    seafood_variation = seafood_std / np.maximum(means['seafood'], 1e-12)
    regime = np.where(
        seafood_variation < STEADY_STOCK_VARIATION, REGIME_STEADY,
        np.where(seafood_min < COLLAPSE_STOCK_LEVEL, REGIME_COLLAPSE, REGIME_CYCLING),
    )
    summary = {
        'seafood_mean': means['seafood'],
        'harvest_mean': means['harvest'],
        'market_price_mean': means['market_price'],
        'fraudster_mean': means['fraudsters'],
        'perception_mean': means['perception'],
        'seafood_min': seafood_min,
        'seafood_max': seafood_max,
        'regime': regime,
    }
    if harvest_reference is not None:
        summary['phi_fp'] = means['phi_fp']
        summary['phi_h'] = means['phi_h']
    return summary


def no_fraud_baseline(story: Story) -> dict:
    """The story's honest-market reference: alpha = 0, F = FP = 0."""
    params = story.params()
    params.update(blast_params(0.0))
    initial_state = {**DEFAULT_INIT_STATE, 'F': 0.0, 'FP': 0.0}
    summary = summarize_runs(params, initial_state)
    return {key: float(value) for key, value in summary.items()}


def blast_sweep(story: Story, alpha_values, f_threshold_values=None) -> dict:
    """Summaries over alpha (and optionally F_hat) for one story, plus % changes.

    Returns the ``summarize_runs`` dict with extra keys ``<metric>_ratio`` (ratio
    to the story's no-fraud baseline) and ``phi_fp_percent`` / ``phi_h_percent``
    (Shapley contributions as % of the baseline Pm-bar).
    """
    baseline = no_fraud_baseline(story)
    params = story.params()
    alpha_values = np.asarray(alpha_values, dtype=np.float64)
    if f_threshold_values is not None:
        alpha_grid, f_threshold_grid = np.meshgrid(alpha_values, f_threshold_values)
        params.update(blast_params(alpha_grid))
        params['F_threshold'] = f_threshold_grid
    else:
        params.update(blast_params(alpha_values))
    summary = summarize_runs(params, DEFAULT_INIT_STATE,
                             harvest_reference=baseline['harvest_mean'])
    for metric, _ in OUTCOME_METRICS:
        summary[f'{metric}_ratio'] = summary[metric] / baseline[metric]
    summary['phi_fp_percent'] = summary['phi_fp'] / baseline['market_price_mean'] * 100
    summary['phi_h_percent'] = summary['phi_h'] / baseline['market_price_mean'] * 100
    summary['baseline'] = baseline
    return summary


def story_time_series(story: Story, alpha: float, window_timesteps: int) -> tuple[dict, dict]:
    """Post-transient time series window for one story at one alpha, plus its baseline."""
    params = story.params()
    params.update(blast_params(alpha))
    series = record_time_series(
        params, DEFAULT_INIT_STATE, TRANSIENT_TIMESTEPS + window_timesteps,
        record_from_timestep=TRANSIENT_TIMESTEPS,
    )
    return series, no_fraud_baseline(story)


# ---------------------------------------------------------------------------
# Shared styling helpers
# ---------------------------------------------------------------------------

def apply_paper_style():
    """Times-like serif text at 8 pt to sit beside mathptmx body text."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['STIXGeneral', 'Times New Roman', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'font.size': 8,
        'axes.titlesize': 8,
        'axes.labelsize': 8,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
        'axes.linewidth': 0.6,
        'axes.edgecolor': '#52514e',
        'axes.labelcolor': '#0b0b0b',
        'xtick.color': '#52514e',
        'ytick.color': '#52514e',
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.major.size': 2.5,
        'ytick.major.size': 2.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': False,
        'grid.color': '#e5e4e0',
        'grid.linewidth': 0.5,
        'lines.linewidth': 1.4,
        'savefig.dpi': 300,
        'figure.dpi': 150,
        'pdf.fonttype': 42,
    })


PERCENT_TICKS = (-90, -75, -50, 0, 100, 300, 900)


def percent_tick_label(percent: float) -> str:
    return '0' if percent == 0 else f'{percent:+g}%'


def set_log_percent_axis(axis, percent_ticks=PERCENT_TICKS, which='y'):
    """Log-ratio axis whose ticks read as % change (ratio 1 = no change)."""
    tick_ratios = [1 + percent / 100 for percent in percent_ticks]
    tick_labels = [percent_tick_label(percent) for percent in percent_ticks]
    if which == 'y':
        axis.set_yscale('log')
        axis.set_yticks(tick_ratios, tick_labels)
        axis.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    else:
        axis.set_xscale('log')
        axis.set_xticks(tick_ratios, tick_labels)
        axis.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())


MULTIPLE_TICKS = (0.1, 0.2, 0.5, 1, 2, 5, 10)


def set_log_multiple_axis(axis, multiple_ticks=MULTIPLE_TICKS):
    """Log-ratio y-axis whose ticks read as multiples of the baseline (1 = no change)."""
    axis.set_yticks(multiple_ticks, [f'{multiple:g}' for multiple in multiple_ticks])
    axis.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())


def diverging_change_colormap():
    """Orange (decrease) -> neutral grey -> blue (increase), on log ratio."""
    return LinearSegmentedColormap.from_list('decrease_increase', [
        '#6e2a0c', '#c4511f', '#f0a07a', '#e8e6e1', '#9cc0ea', '#2a78d6', '#0e3a73',
    ])


def save_figure(figure, file_stem: str):
    os.makedirs(FIGURE_OUTPUT_DIRECTORY, exist_ok=True)
    figure.savefig(os.path.join(FIGURE_OUTPUT_DIRECTORY, f'{file_stem}.pdf'),
                   bbox_inches='tight', pad_inches=0.02)
    plt.close(figure)
    print(f'  wrote figures/{file_stem}.pdf')


def welfare_ratio(sweep: dict, metric: str, is_inverted: bool, alpha_values) -> np.ndarray:
    """Ratio to the no-fraud baseline, inverted for affordability (1 / price).

    At alpha = 0 the ratio is pinned to 1: the no-blast point is the honest
    market (F = FP = 0), as in the app heatmaps' Baseline column, rather than
    the alpha = 0 run with fraudsters present (whose price is already lower).
    """
    ratios = np.clip(sweep[f'{metric}_ratio'], 0.05, 12)
    ratios = np.where(np.asarray(alpha_values) == 0.0, 1.0, ratios)
    return 1.0 / ratios if is_inverted else ratios


def style_welfare_axis(axis):
    """Log % axis with the red 'worse than baseline' band below zero."""
    axis.axhspan(WELFARE_Y_FLOOR, 1.0, color=WORSE_THAN_BASELINE_COLOR, alpha=0.09,
                 linewidth=0, zorder=0)
    axis.text(0.03, 0.03, 'Worse than\nbaseline', transform=axis.transAxes, ha='left',
              va='bottom', linespacing=1.1, fontsize=5.5,
              color=WORSE_THAN_BASELINE_TEXT_COLOR, zorder=2)
    axis.axhline(1.0, color='#888780', linewidth=0.6, zorder=1)
    axis.grid(axis='y', zorder=0)
    set_log_percent_axis(axis)
    axis.set_ylim(WELFARE_Y_FLOOR, WELFARE_Y_CEILING)
    axis.set_xlim(0, 1)
    axis.set_xticks([0, 0.5, 1], ['0', '0.5', '1'])


# ---------------------------------------------------------------------------
# Figure A — outcome curves vs alpha, with regime ribbon
# ---------------------------------------------------------------------------

def figure_outcome_curves(include_dynamics_ribbon: bool = True):
    """Three panels (S, H, Pm) vs alpha, one solid line per story.

    Replaces the alpha-sweep time-series grid: each line is the post-transient
    mean. With ``include_dynamics_ribbon``, a ribbon underneath says *how* each
    story gets there (steady state, boom-bust cycles, or busts that nearly wipe
    out the stock); without it the figure is the three outcome panels only.
    """
    alpha_values = np.linspace(0.0, 1.0, 101)
    sweeps = {story.name: blast_sweep(story, alpha_values) for story in STORIES}

    if include_dynamics_ribbon:
        figure = plt.figure(figsize=(TEXT_WIDTH_INCHES, 2.95))
        grid = figure.add_gridspec(2, 3, height_ratios=(3.3, 1.0), hspace=0.42, wspace=0.42,
                                   left=0.09, right=0.99, top=0.83, bottom=0.11)
    else:
        figure = plt.figure(figsize=(TEXT_WIDTH_INCHES, 2.2))
        grid = figure.add_gridspec(1, 3, wspace=0.42,
                                   left=0.09, right=0.99, top=0.77, bottom=0.15)
    outcome_axes = [figure.add_subplot(grid[0, column]) for column in range(3)]

    for axis, (metric, title, _, is_inverted, _) in zip(outcome_axes, WELFARE_METRICS):
        style_welfare_axis(axis)
        for story in STORIES:
            axis.plot(alpha_values,
                      welfare_ratio(sweeps[story.name], metric, is_inverted, alpha_values),
                      color=story.color, linewidth=1.3, zorder=3)
        axis.set_title(title, pad=3)
    outcome_axes[1].set_xlabel(r'Blast intensity $\alpha$', labelpad=1)
    outcome_axes[0].set_ylabel('% change vs no-fraud baseline', labelpad=1)

    if include_dynamics_ribbon:
        ribbon_axis = figure.add_subplot(grid[1, :])
        regime_colormap = ListedColormap(REGIME_COLORS)
        regime_rows = np.array([sweeps[story.name]['regime'] for story in STORIES])
        ribbon_axis.imshow(regime_rows, aspect='auto', cmap=regime_colormap, vmin=-0.5,
                           vmax=2.5, interpolation='nearest',
                           extent=(alpha_values[0] - 0.005, alpha_values[-1] + 0.005,
                                   len(STORIES), 0))
        for row_index in range(1, len(STORIES)):
            ribbon_axis.axhline(row_index, color='white', linewidth=1.2)
        ribbon_axis.set_yticks(np.arange(len(STORIES)) + 0.5,
                               [story.name for story in STORIES], fontsize=6.5)
        for tick_label in ribbon_axis.get_yticklabels():
            tick_label.set_color('#0b0b0b')
        for row_index, story in enumerate(STORIES):
            ribbon_axis.plot([-0.022, -0.006], [row_index + 0.5] * 2, color=story.color,
                             linewidth=1.6, clip_on=False, solid_capstyle='butt')
        ribbon_axis.tick_params(axis='y', length=0, pad=9)
        ribbon_axis.set_xlim(0, 1)
        ribbon_axis.set_xticks([0, 0.25, 0.5, 0.75, 1], ['0', '0.25', '0.5', '0.75', '1'])
        ribbon_axis.set_xlabel(r'Blast intensity $\alpha$', labelpad=1)
        ribbon_axis.set_title(r'Long-run dynamics of the stock $S$', pad=2, fontsize=7,
                              loc='left', color='#52514e')
        for spine in ribbon_axis.spines.values():
            spine.set_visible(False)
        ribbon_axis.legend(
            handles=[Patch(facecolor=color, edgecolor='#a9a7a0', linewidth=0.4, label=label)
                     for color, label in zip(REGIME_COLORS, REGIME_LABELS)],
            loc='upper center', bbox_to_anchor=(0.5, -0.75), ncol=3, frameon=False,
            handlelength=1.4, columnspacing=1.2,
        )

    figure.legend(
        handles=[Line2D([], [], color=story.color, linewidth=1.3, label=story.legend_label)
                 for story in STORIES],
        loc='upper center', bbox_to_anchor=(0.53, 1.0), ncol=2, frameon=False,
        handlelength=2.6, columnspacing=1.0, labelspacing=0.3,
    )
    save_figure(figure, 'blast_fig_a_outcome_curves' if include_dynamics_ribbon
                else 'blast_fig_a_outcome_curves_no_dynamics')


# ---------------------------------------------------------------------------
# Figure F — one panel per story, three welfare lines (companion to Figure A)
# ---------------------------------------------------------------------------

def figure_story_welfare_panels(y_axis_as_multiple: bool = False):
    """Four panels (one per story), each with S-bar, H-bar and 1/Pm-bar vs alpha.

    The transpose of the no-dynamics Figure A: same data, same log axis and
    red 'worse than baseline' band, but grouped by story so each panel shows
    how stock, food security and affordability trade off within one scenario.
    With ``y_axis_as_multiple`` the ticks read as multiples of the baseline
    (1 = no change, 0.5 = half, 2 = double) instead of % change.
    """
    alpha_values = np.linspace(0.0, 1.0, 101)
    sweeps = {story.name: blast_sweep(story, alpha_values) for story in STORIES}

    figure, axes = plt.subplots(
        1, len(STORIES), figsize=(TEXT_WIDTH_INCHES, 2.25), sharey=True,
        gridspec_kw=dict(wspace=0.10, left=0.11, right=0.995, top=0.72, bottom=0.15),
    )
    for axis, story in zip(axes, STORIES):
        style_welfare_axis(axis)
        for metric, _, _, is_inverted, line_color in WELFARE_METRICS:
            axis.plot(alpha_values,
                      welfare_ratio(sweeps[story.name], metric, is_inverted, alpha_values),
                      color=line_color, linewidth=1.3, zorder=3)
        axis.set_title(f'{story.name}\n{story.parameter_label}', pad=3, fontsize=7.5,
                       linespacing=1.3)
        if y_axis_as_multiple:
            set_log_multiple_axis(axis)
        axis.tick_params(axis='y', labelleft=axis is axes[0])
    axes[0].set_ylabel('× no-fraud baseline' if y_axis_as_multiple
                       else '% change vs no-fraud baseline', labelpad=1)
    figure.supxlabel(r'Blast intensity $\alpha$', y=0.005, fontsize=8)

    figure.legend(
        handles=[Line2D([], [], color=line_color, linewidth=1.3, label=legend_label)
                 for _, _, legend_label, _, line_color in WELFARE_METRICS],
        loc='upper center', bbox_to_anchor=(0.55, 1.0), ncol=3, frameon=False,
        handlelength=2.0, columnspacing=1.4,
    )
    save_figure(figure, 'blast_fig_f_story_welfare_panels_multiples' if y_axis_as_multiple
                else 'blast_fig_f_story_welfare_panels')


# ---------------------------------------------------------------------------
# Figure B — alpha x F_hat policy maps
# ---------------------------------------------------------------------------

def figure_policy_maps():
    """3 x 3 heatmaps of % change over (alpha, F_hat).

    Rows: Baseline, Damaged reef, No substitutes. Columns: S-bar, H-bar, Pm-bar.
    The F_hat axis subsumes the Aware-buyers story (dashed line at F_hat = 0.05)
    and the app's "vs F_threshold" sweeps. Black contour = the near-collapse
    frontier (stock dips below 5% of K during busts); dots = oscillatory
    (non-steady) long-run dynamics.
    """
    alpha_values = np.linspace(0.0, 1.0, 81)
    f_threshold_values = np.linspace(0.05, 0.95, 73)
    map_stories = [STORIES_BY_NAME[name] for name in ('Baseline', 'Damaged reef', 'No substitutes')]
    sweeps = {story.name: blast_sweep(story, alpha_values, f_threshold_values)
              for story in map_stories}

    colormap = diverging_change_colormap()
    log_limit = 1.0  # +-1 decade: -90% ... +900%
    figure, axes = plt.subplots(
        3, 3, figsize=(TEXT_WIDTH_INCHES, 4.15), sharex=True, sharey=True,
        gridspec_kw=dict(hspace=0.12, wspace=0.10, left=0.15, right=0.86, top=0.94, bottom=0.13),
    )
    image = None
    for row_index, story in enumerate(map_stories):
        sweep = sweeps[story.name]
        for column_index, (metric, title) in enumerate(OUTCOME_METRICS):
            axis = axes[row_index, column_index]
            log_ratio = np.log10(np.clip(sweep[f'{metric}_ratio'], 1e-3, 1e3))
            image = axis.pcolormesh(alpha_values, f_threshold_values, log_ratio, cmap=colormap,
                                    vmin=-log_limit, vmax=log_limit, shading='nearest',
                                    rasterized=True)
            axis.contour(alpha_values, f_threshold_values,
                         (sweep['regime'] == REGIME_COLLAPSE).astype(float), levels=[0.5],
                         colors='#0b0b0b', linewidths=0.8)
            axis.contour(alpha_values, f_threshold_values, log_ratio, levels=[0.0],
                         colors='#52514e', linewidths=0.5, linestyles=[(0, (2, 1.5))])
            oscillating = sweep['regime'] != REGIME_STEADY
            stride_alpha, stride_f = 6, 6
            alpha_grid, f_grid = np.meshgrid(alpha_values, f_threshold_values)
            axis.scatter(alpha_grid[::stride_f, ::stride_alpha][oscillating[::stride_f, ::stride_alpha]],
                         f_grid[::stride_f, ::stride_alpha][oscillating[::stride_f, ::stride_alpha]],
                         s=0.7, color='#3d3c39', alpha=0.55, linewidths=0, rasterized=True)
            if story.name == 'Baseline':
                for f_threshold, line_story in ((0.5, story), (0.05, STORIES_BY_NAME['Aware buyers'])):
                    axis.axhline(f_threshold, color=line_story.color, linewidth=1.0,
                                 linestyle=(0, (3, 1.5)))
            if row_index == 0:
                axis.set_title(title, pad=3)
            if column_index == 0:
                axis.set_ylabel(story.name, labelpad=2, fontsize=8, color='#0b0b0b')
                axis.set_yticks([0.05, 0.5, 0.95], ['0.05', '0.5', '0.95'])
            if row_index == 2:
                axis.set_xlabel(r'Blast intensity $\alpha$', labelpad=1)
                axis.set_xticks([0, 0.5, 1], ['0', '0.5', '1'])
            for spine in axis.spines.values():
                spine.set_visible(True)
                spine.set_color('#a9a7a0')

    annotation_box = dict(boxstyle='round,pad=0.12', facecolor='white', alpha=0.85,
                          edgecolor='none')
    axes[0, 0].text(0.03, 0.09, r'Aware buyers $\hat F{=}0.05$', ha='left', va='bottom',
                    fontsize=5.5, color='#0b0b0b', bbox=annotation_box)
    axes[0, 0].text(0.03, 0.54, r'Baseline $\hat F{=}0.5$', ha='left', va='bottom',
                    fontsize=5.5, color='#0b0b0b', bbox=annotation_box)
    figure.text(0.015, 0.535, r'Buyer detection threshold $\hat F$', rotation=90,
                ha='left', va='center', fontsize=8)

    colorbar_axis = figure.add_axes([0.885, 0.30, 0.022, 0.50])
    colorbar = figure.colorbar(image, cax=colorbar_axis)
    colorbar_ticks = (-90, -75, -50, 0, 100, 300, 900)
    colorbar.set_ticks([np.log10(1 + percent / 100) for percent in colorbar_ticks],
                       labels=[percent_tick_label(percent) for percent in colorbar_ticks])
    colorbar.outline.set_linewidth(0.4)
    colorbar.ax.tick_params(labelsize=6.5, length=2)
    colorbar.set_label('Change vs no-fraud baseline', fontsize=7, labelpad=2)
    figure.legend(
        handles=[
            Line2D([], [], color='#0b0b0b', linewidth=0.8, label='Near-collapse frontier'),
            Line2D([], [], color='#52514e', linewidth=0.5, linestyle=(0, (2, 1.5)),
                   label='No change'),
            Line2D([], [], color='#0b0b0b', marker='.', linestyle='none', markersize=2.5,
                   alpha=0.6, label='Oscillating'),
        ],
        loc='lower center', bbox_to_anchor=(0.5, -0.01), ncol=3, frameon=False,
        handlelength=1.8, columnspacing=1.4,
    )
    save_figure(figure, 'blast_fig_b_policy_maps')


# ---------------------------------------------------------------------------
# Figure C — who sets the price? Shapley decomposition vs alpha
# ---------------------------------------------------------------------------

def figure_price_decomposition():
    """Per story: Pm-bar change split into buyer exit (phi_FP) and supply (phi_H).

    Positive parts stack upward, negative parts downward; the black line is
    the total (= Pm-bar change). The y-axis is linear and clipped at +300% so
    the low-alpha regime stays readable; overflow is labelled.
    """
    alpha_values = np.linspace(0.0, 1.0, 101)
    sweeps = {story.name: blast_sweep(story, alpha_values) for story in STORIES}
    perception_color, harvest_color = VAR_COLORS['FP'], VAR_COLORS['Harvest']
    y_min, y_max = -100, 300

    figure, axes = plt.subplots(
        1, 4, figsize=(TEXT_WIDTH_INCHES, 1.75), sharey=True,
        gridspec_kw=dict(wspace=0.12, left=0.11, right=0.995, top=0.78, bottom=0.22),
    )
    for axis, story in zip(axes, STORIES):
        sweep = sweeps[story.name]
        phi_perception, phi_harvest = sweep['phi_fp_percent'], sweep['phi_h_percent']
        total = phi_perception + phi_harvest
        positive_perception = np.maximum(phi_perception, 0)
        negative_perception = np.minimum(phi_perception, 0)
        axis.fill_between(alpha_values, 0, positive_perception, color=perception_color,
                          alpha=0.85, linewidth=0)
        axis.fill_between(alpha_values, positive_perception,
                          positive_perception + np.maximum(phi_harvest, 0),
                          color=harvest_color, alpha=0.85, linewidth=0)
        axis.fill_between(alpha_values, 0, negative_perception, color=perception_color,
                          alpha=0.85, linewidth=0)
        axis.fill_between(alpha_values, negative_perception,
                          negative_perception + np.minimum(phi_harvest, 0),
                          color=harvest_color, alpha=0.85, linewidth=0)
        axis.plot(alpha_values, np.clip(total, y_min, y_max), color='#0b0b0b', linewidth=1.0)
        axis.axhline(0, color='#52514e', linewidth=0.5)
        axis.set_ylim(y_min, y_max)
        axis.set_xlim(0, 1)
        axis.set_xticks([0, 0.5, 1], ['0', '0.5', '1'])
        axis.set_xlabel(r'$\alpha$', labelpad=0)
        axis.grid(axis='y')
        axis.set_title(story.name, pad=3, color='#0b0b0b')
        axis.plot(0.04, 1.08, transform=axis.transAxes, marker=story.marker, color=story.color,
                  markersize=3.5, clip_on=False, markeredgecolor='white', markeredgewidth=0.4)
        if total[-1] > y_max:
            axis.annotate(f'{total[-1]:+.0f}%\nAt ' + r'$\alpha{=}1$', xy=(1.0, y_max),
                          xytext=(0.97, y_max * 0.93), ha='right', va='top', fontsize=6,
                          color='#0b0b0b')
    axes[0].set_yticks([-100, 0, 100, 200, 300], ['−100%', '0', '+100%', '+200%', '+300%'])
    axes[0].set_ylabel(r'$\Delta\bar P^m$ (% of baseline)', labelpad=1)
    figure.legend(
        handles=[
            Patch(facecolor=perception_color, alpha=0.85, label=r'Buyer awareness $\varphi_{F^p}$'),
            Patch(facecolor=harvest_color, alpha=0.85, label=r'Supply / harvest $\varphi_H$'),
            Line2D([], [], color='#0b0b0b', linewidth=1.0, label=r'Total $\Delta\bar P^m$'),
        ],
        loc='upper center', bbox_to_anchor=(0.55, 1.02), ncol=3, frameon=False,
        handlelength=1.6, columnspacing=1.4,
    )
    save_figure(figure, 'blast_fig_c_price_decomposition')


# ---------------------------------------------------------------------------
# Figure D — anatomy of one fraud boom-bust cycle (mechanism explainer)
# ---------------------------------------------------------------------------

def figure_boom_bust_anatomy(alpha: float = 0.15, window_timesteps: int = 130):
    """Stacked panels for the Baseline story over ~2 fraud cycles.

    (a) S and H relative to the honest baseline; (b) F and FP; (c) Pm and Pw.
    Background shading marks when fraudsters are the majority (F > 1/2) and
    numbered markers walk through one cycle.
    """
    story = STORIES_BY_NAME['Baseline']
    series, baseline = story_time_series(story, alpha, window_timesteps)
    time_axis = np.arange(len(series['S']))

    figure, axes = plt.subplots(
        3, 1, figsize=(TEXT_WIDTH_INCHES, 3.0), sharex=True,
        gridspec_kw=dict(hspace=0.14, left=0.11, right=0.79, top=0.86, bottom=0.12),
    )
    fraud_majority = series['F'] > 0.5
    for axis in axes:
        axis.fill_between(time_axis, 0, 1, where=fraud_majority,
                          color=VAR_COLORS['F'], alpha=0.25, linewidth=0,
                          transform=axis.get_xaxis_transform(), step='mid')

    stock_axis, share_axis, price_axis = axes
    stock_axis.plot(time_axis, series['S'] / baseline['seafood_mean'], color=VAR_COLORS['S'],
                    label=r'Stock $S/\bar S_0$')
    stock_axis.plot(time_axis, series['H'] / baseline['harvest_mean'], color=VAR_COLORS['Harvest'],
                    linestyle=(0, (4, 1.5)), label=r'Harvest $H/\bar H_0$')
    stock_axis.axhline(1, color='#888780', linewidth=0.5)
    stock_axis.set_ylabel('Relative to\nhonest market', labelpad=1)

    share_axis.plot(time_axis, series['F'], color=VAR_COLORS['F'], label=r'Fraudsters $F$')
    share_axis.plot(time_axis, series['FP'], color=VAR_COLORS['FP'], linestyle=(0, (4, 1.5)),
                    label=r'Fraud perception $F^p$')
    share_axis.axhline(DEFAULT_PARAMS['F_threshold'], color='#888780', linewidth=0.5,
                       linestyle=(0, (2, 2)))
    share_axis.text(time_axis[-1], DEFAULT_PARAMS['F_threshold'] + 0.03, r'$\hat F$',
                    ha='right', va='bottom', fontsize=6.5, color='#52514e')
    share_axis.set_ylim(-0.04, 1.04)
    share_axis.set_ylabel('Share', labelpad=1)

    price_axis.plot(time_axis, series['Pm'], color=VAR_COLORS['Pm'], label=r'Market $P^m$')
    price_axis.plot(time_axis, series['Pw'], color=VAR_COLORS['Pw'], linestyle=(0, (4, 1.5)),
                    label=r'Wholesale $P^w$')
    price_axis.axhline(baseline['market_price_mean'], color='#888780', linewidth=0.5)
    price_axis.text(time_axis[-1], baseline['market_price_mean'] * 1.15, r'Honest $\bar P^m_0$',
                    ha='right', va='bottom', fontsize=6, color='#52514e')
    price_axis.set_yscale('log')
    price_axis.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda value, _: f'{value:g}'))
    price_axis.set_ylabel('Price', labelpad=1)
    price_axis.set_xlabel(f'Time step after transient (Baseline story, ' + r'$\alpha$'
                          + f' = {alpha})', labelpad=1)
    price_axis.set_xlim(0, time_axis[-1])

    legend_handles = [handle for axis in axes for handle in axis.get_legend_handles_labels()[0]]
    figure.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.45, 1.0),
                  ncol=3, frameon=False, handlelength=2.0, columnspacing=1.2,
                  labelspacing=0.3)

    # Numbered walk through the first complete fraud cycle.
    fraud_rises = np.nonzero(np.diff(fraud_majority.astype(int)) == 1)[0]
    if len(fraud_rises):
        cycle_start = fraud_rises[0] + 1
        after_start = slice(cycle_start, None)
        perception_rise = cycle_start + int(np.argmax(series['FP'][after_start] > 0.5))
        fraud_fall = cycle_start + int(np.argmax(~fraud_majority[after_start]))
        stock_bottom = cycle_start + int(np.argmin(series['S'][cycle_start:fraud_fall + 30]))
        steps = [
            (share_axis, cycle_start, series['F'][cycle_start], 'Fraud takes over'),
            (share_axis, perception_rise, series['FP'][perception_rise], 'Buyers notice'),
            (price_axis, perception_rise + 3, series['Pm'][perception_rise + 3], 'Buyers exit'),
            (share_axis, fraud_fall, series['F'][fraud_fall], 'Fraud unprofitable'),
            (stock_axis, stock_bottom, series['S'][stock_bottom] / baseline['seafood_mean'],
             'Stock bottoms out'),
        ]
        for number, (axis, timestep, value, _) in enumerate(steps, 1):
            axis.annotate(str(number), xy=(timestep, value), fontsize=6, ha='center',
                          va='center', color='white', fontweight='bold',
                          bbox=dict(boxstyle='circle,pad=0.18', facecolor='#0b0b0b',
                                    edgecolor='none'), zorder=10)
        for number, (*_, text) in enumerate(steps, 1):
            label_height = 0.74 - (number - 1) * 0.085
            figure.text(0.815, label_height, str(number), fontsize=6, ha='center', va='center',
                        color='white', fontweight='bold',
                        bbox=dict(boxstyle='circle,pad=0.18', facecolor='#0b0b0b',
                                  edgecolor='none'))
            figure.text(0.835, label_height, text, fontsize=6.5, va='center', color='#0b0b0b')
        figure.text(0.80, 0.81, 'One fraud cycle', fontsize=7, color='#52514e', va='center')
        figure.text(0.80, 0.27, 'Shaded: fraudsters\nare the majority', fontsize=6,
                    color='#52514e', va='top', linespacing=1.2)
    save_figure(figure, 'blast_fig_d_boom_bust_anatomy')


# ---------------------------------------------------------------------------
# Figure E — condensed time-series matrix (replaces the 4 x 20 grid)
# ---------------------------------------------------------------------------

def figure_time_series_matrix(alpha_values=(0.15, 0.55, 1.0), window_timesteps: int = 400):
    """Rows = stories, columns = alpha. Each cell is a stock sparkline.

    Blue line: S(t)/K. Red band along the top: fraudster share F(t) as
    opacity. Badge: S-bar / H-bar / Pm-bar % change vs the story's baseline,
    averaged over the full post-transient window (only the first
    ``window_timesteps`` are drawn).
    """
    figure, axes = plt.subplots(
        len(STORIES), len(alpha_values), figsize=(TEXT_WIDTH_INCHES, 3.55),
        sharex=True, sharey=True,
        gridspec_kw=dict(hspace=0.16, wspace=0.06, left=0.17, right=0.99, top=0.91, bottom=0.10),
    )
    fraud_colormap = LinearSegmentedColormap.from_list('fraud', ['#ffffff', VAR_COLORS['F']])
    for row_index, story in enumerate(STORIES):
        for column_index, alpha in enumerate(alpha_values):
            axis = axes[row_index, column_index]
            full_series, baseline = story_time_series(
                story, alpha, TOTAL_TIMESTEPS - TRANSIENT_TIMESTEPS,
            )
            series = {key: values[:window_timesteps + 1] for key, values in full_series.items()}
            time_axis = np.arange(len(series['S']))
            axis.imshow(series['F'][np.newaxis, :], aspect='auto', cmap=fraud_colormap,
                        vmin=0, vmax=1, extent=(0, time_axis[-1], 1.12, 1.0),
                        interpolation='nearest')
            axis.axhline(baseline['seafood_mean'], color='#888780', linewidth=0.5,
                         linestyle=(0, (2, 2)))
            axis.fill_between(time_axis, 0, series['S'], color=VAR_COLORS['S'], alpha=0.12,
                              linewidth=0)
            axis.plot(time_axis, series['S'], color=VAR_COLORS['S'], linewidth=0.8)
            changes = [
                (full_series['S'].mean() / baseline['seafood_mean'] - 1) * 100,
                (full_series['H'].mean() / baseline['harvest_mean'] - 1) * 100,
                (full_series['Pm'].mean() / baseline['market_price_mean'] - 1) * 100,
            ]
            badge = (r'$\bar S$' + f' {changes[0]:+.0f}%   ' + r'$\bar H$' + f' {changes[1]:+.0f}%   '
                     + r'$\bar P^m$' + f' {changes[2]:+.0f}%')
            axis.text(0.5, 0.80, badge, transform=axis.transAxes, ha='center', va='top',
                      fontsize=5.3, color='#0b0b0b',
                      bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.8,
                                edgecolor='none'))
            axis.set_ylim(0, 1.12)
            axis.set_xlim(0, time_axis[-1])
            axis.set_yticks([0, 0.5, 1], ['0', '.5', '1'])
            axis.set_xticks([0, 200, 400], ['0', '200', ''])
            if row_index == 0:
                axis.set_title(r'$\alpha$' + f' = {alpha}', pad=3)
            if column_index == 0:
                axis.set_ylabel(story.name, rotation=0, ha='right', va='center', labelpad=4,
                                color='#0b0b0b')
            if row_index == len(STORIES) - 1:
                axis.set_xlabel('Time step', labelpad=1)
    figure.legend(
        handles=[
            Line2D([], [], color=VAR_COLORS['S'], linewidth=0.8, label=r'Stock $S/K$'),
            Line2D([], [], color='#888780', linewidth=0.5, linestyle=(0, (2, 2)),
                   label=r'Honest-market $\bar S_0$'),
            Patch(facecolor=VAR_COLORS['F'], label=r'Top strip: fraudster share $F$ (opacity)'),
        ],
        loc='upper center', bbox_to_anchor=(0.57, 1.0), ncol=3, frameon=False,
        handlelength=1.6, columnspacing=1.2,
    )
    save_figure(figure, 'blast_fig_e_time_series_matrix')


# ---------------------------------------------------------------------------
# Figure G — model flow diagram (paper version of slide 5)
# ---------------------------------------------------------------------------

def _draw_flow_arrow(axis, start, end, curvature, label_phrase, label_math, label_position,
                     label_vertical_alignment):
    """Curved arrow between two points with a two-line label (phrase + model term)."""
    from matplotlib.patches import FancyArrowPatch
    axis.add_patch(FancyArrowPatch(
        start, end, connectionstyle=f'arc3,rad={curvature}', arrowstyle='-|>',
        mutation_scale=7, linewidth=0.9, color='#0b0b0b', shrinkA=0, shrinkB=0,
    ))
    label_x, label_y = label_position
    axis.text(label_x, label_y, label_phrase + '\n' + label_math, ha='center',
              va=label_vertical_alignment, fontsize=6.3, color='#0b0b0b', linespacing=1.35)


def figure_model_flow_diagram():
    """SIR-style flow diagram of the four-state map, coloured as in the slides.

    Boxes are the state variables (S, E, F, F^p) in their VAR_COLORS. Each
    arrow carries the model term through which the source variable enters the
    target's update, including the response speed gamma of the target equation:

        S self-loop   gamma_S r (1 - S_t/K)              logistic regrowth
        E -> S        gamma_S q(F_t) E_t                  fishing mortality
        S -> E        gamma_E q(F_t) P^w(F_t, H_t) S_t              catch revenue (stock feedback)
        E -> F        H_t = q(F_t) E_t S_t                          supply sets P^w and P^m
        F -> E        q(F_t), C(F_t), P^w(F_t, H_t)                 fraud shifts catchability, cost, price
        F -> F^p      gamma_FP (F_t - F_hat)                        detection
        F^p -> F      gamma_F (P^m(F^p_t, H_t) - P^w(F_t, H_t))     buyer demand
    """
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    figure_height = 2.2
    figure = plt.figure(figsize=(TEXT_WIDTH_INCHES, figure_height))
    axis = figure.add_axes([0, 0, 1, 1])
    axis.set_xlim(0, TEXT_WIDTH_INCHES)
    axis.set_ylim(0, figure_height)
    axis.set_aspect('equal')
    axis.axis('off')

    box_width, box_height, box_center_y = 0.84, 0.56, 0.95
    box_specs = (
        ('S', 0.80, 'Seafood stock', r'$S_t$', VAR_COLORS['S']),
        ('E', 1.92, 'Harvesting effort', r'$E_t$', VAR_COLORS['E']),
        ('F', 3.04, 'Fraudulent\nwholesalers', r'$F_t$', VAR_COLORS['F']),
        ('FP', 4.16 - 0.04, 'Fraud perception\n(buyers)', r'$F^p_t$', VAR_COLORS['FP']),
    )
    box_center_x = {}
    for key, center_x, name, symbol, color in box_specs:
        box_center_x[key] = center_x
        axis.add_patch(FancyBboxPatch(
            (center_x - box_width / 2, box_center_y - box_height / 2), box_width, box_height,
            boxstyle='round,pad=0,rounding_size=0.07', facecolor=color, edgecolor='none',
        ))
        axis.text(center_x, box_center_y + 0.07, name, ha='center', va='center', fontsize=6.6,
                  color='white', fontweight='bold', linespacing=1.1)
        axis.text(center_x, box_center_y - 0.17, symbol, ha='center', va='center',
                  fontsize=9.5, color='white')

    box_top = box_center_y + box_height / 2
    box_bottom = box_center_y - box_height / 2
    arrow_offset = 0.17   # arrows leave / enter this far from the box centre
    # arc3 bends relative to the arrow's direction, so flip the sign for
    # right-to-left arrows to keep top arcs above and bottom arcs below.
    arc_curvature = 0.55
    top_label_y, bottom_label_y = box_top + 0.38, box_bottom - 0.33

    # Forward arcs over the top, feedback arcs underneath.
    flow_arrows = (
        ('S', 'E', 'top', 'Catch revenue', r'$\gamma_E\, q(F_t)\, P^w(F_t, H_t)\, S_t$'),
        ('E', 'F', 'top', 'Sell catch', r'$H_t = q(F_t)\, E_t S_t$'),
        ('F', 'FP', 'top', 'Detection of fraud', r'$\gamma_{F^p}\,(F_t - \hat F)$'),
        ('E', 'S', 'bottom', 'Harvest seafood', r'$\gamma_S\, q(F_t)\, E_t$'),
        ('F', 'E', 'bottom', 'IUU fishing opportunities',
         r'$q(F_t),\ C(F_t),\ P^w(F_t, H_t)$'),
        ('FP', 'F', 'bottom', 'Demand for seafood',
         r'$\gamma_F\,\left(P^m(F^p_t, H_t) - P^w(F_t, H_t)\right)$'),
    )
    for source, target, side, phrase, math in flow_arrows:
        source_x, target_x = box_center_x[source], box_center_x[target]
        direction = np.sign(target_x - source_x)
        label_x = (source_x + target_x) / 2
        if side == 'top':
            start = (source_x + direction * arrow_offset, box_top)
            end = (target_x - direction * arrow_offset, box_top)
            _draw_flow_arrow(axis, start, end, -arc_curvature * direction, phrase, math,
                             (label_x, top_label_y), 'bottom')
        else:
            start = (source_x + direction * arrow_offset, box_bottom)
            end = (target_x - direction * arrow_offset, box_bottom)
            _draw_flow_arrow(axis, start, end, arc_curvature * direction, phrase, math,
                             (label_x, bottom_label_y), 'top')

    # Logistic regrowth: self-loop on the left side of the seafood box.
    seafood_left = box_center_x['S'] - box_width / 2
    axis.add_patch(FancyArrowPatch(
        (seafood_left, box_center_y + 0.13), (seafood_left, box_center_y - 0.13),
        connectionstyle='arc3,rad=1.9', arrowstyle='-|>', mutation_scale=7, linewidth=0.9,
        color='#0b0b0b', shrinkA=0, shrinkB=0,
    ))
    axis.text(seafood_left - 0.13, box_center_y + 0.36, 'Reproduce\n' + r'$\gamma_S\, r\,(1 - S_t/K)$',
              ha='center', va='bottom', fontsize=6.3, color='#0b0b0b', linespacing=1.35)

    # Subsystem headings with thin rules spanning their boxes.
    heading_y = figure_height - 0.04
    for heading, first_key, last_key in (('Bioeconomic subsystem', 'S', 'E'),
                                         ('Behavioural subsystem', 'F', 'FP')):
        left = box_center_x[first_key] - box_width / 2
        right = box_center_x[last_key] + box_width / 2
        axis.text((left + right) / 2, heading_y, heading, ha='center', va='top', fontsize=7.5,
                  color='#0b0b0b', fontweight='bold')
        axis.plot([left, right], [heading_y - 0.17] * 2, color='#a9a7a0', linewidth=0.6)

    save_figure(figure, 'blast_fig_g_model_flow_diagram')


# ---------------------------------------------------------------------------
# Figure H — full dependency network of the map
# ---------------------------------------------------------------------------

# Direct dependencies of the dimensionalized map (source -> target). Derived
# quantities: q(F), C(F), H = q E S, P^w(F, H), P^m(F^p, H). Composed through
# them, the state-to-state pattern equals the Jacobian's non-zero structure:
#   S' <- S, E, F    E' <- S, E, F    F' <- S, E, F, F^p    F^p' <- F, F^p
DEPENDENCY_EDGES = (
    # (source, target, parameter label, arc curvature)
    ('S', 'S', r'$\gamma_S,\ r,\ K$', 0.0),
    ('E', 'E', r'$\gamma_E$', 0.0),
    ('F', 'F', r'$\gamma_F$', 0.0),
    ('FP', 'FP', r'$\gamma_{F^p}$', 0.0),
    ('E', 'S', None, 0.25),
    ('S', 'E', None, 0.25),
    ('q', 'S', None, 0.0),
    ('q', 'E', None, 0.0),
    ('C', 'E', None, 0.0),
    ('Pw', 'E', None, 0.0),
    ('S', 'H', None, 0.0),
    ('E', 'H', None, 0.0),
    ('q', 'H', None, 0.0),
    ('F', 'q', r'$q_0,\ q_1$', 0.0),
    ('F', 'C', r'$C_0,\ C_1$', 0.0),
    ('F', 'Pw', r'$P^w_0,\ P^w_1$', -0.2),
    ('H', 'Pw', r'$\gamma_P,\ \epsilon_{s,w}$', 0.0),
    ('H', 'Pm', r'$\gamma_M,\ \epsilon_{s,m}$', 0.0),
    ('FP', 'Pm', r'$\epsilon_d$', 0.0),
    ('Pw', 'F', None, -0.2),
    ('Pm', 'F', None, 0.0),
    ('F', 'FP', r'$\hat F$', 0.25),
)


# Algebraic definitions drawn inside the derived-quantity nodes of the
# equation version of Figure H (same forms as core.System). Detectability and
# harvester profitability are named concepts with no symbol of their own, so
# they show only their expression; 'D' and 'Profit' are internal node keys.
DERIVED_QUANTITY_EQUATIONS = {
    'H': r'$H = q\,E\,S$',
    'Pm': r'$P^m = \gamma_M \sqrt{\frac{(1 - F^p)^{\epsilon_d}}{H^{\epsilon_{s,m}}}}$',
    'Pw': r'$P^w = \frac{(P^w_1 - P^w_0)\,F + P^w_0}{(\gamma_P H)^{\epsilon_{s,w}}}$',
    'C': r'$C = (C_1 - C_0)\,F + C_0$',
    'q': r'$q = (q_1 - q_0)\,F + q_0$',
    'D': r'$\gamma_{F^p}\,(F - \hat F)$',
    'Profit': r'$\gamma_E\,(q\,P^w S - C)$',
}
# Concept nodes that exist only in the equation version.
EQUATION_NETWORK_EXTRA_NODES = {
    'D': (0, 0, 'derived', r'$D$', 'Detectability', '#52514e', None),
    'Profit': (0, 0, 'derived', r'$\pi$', 'Harvester profitability', '#52514e', None),
}
# The equation version has its own layout: taller, with the wide equation
# boxes stacked in two centre columns between the state variables.
EQUATION_NETWORK_FIGURE_HEIGHT = 3.9
EQUATION_NETWORK_POSITIONS = {
    'S': (0.62, 3.45), 'FP': (3.98, 3.45), 'E': (0.62, 0.45), 'F': (3.98, 0.45),
    'H': (1.75, 3.45), 'Pm': (2.82, 3.45),
    'q': (1.72, 2.55), 'Pw': (2.80, 2.20),
    'Profit': (1.60, 1.50), 'C': (2.65, 0.60),
    'D': (3.98, 2.10),
}
# Edges of the equation version. Detectability splits F -> F^p, and harvester
# profitability collects every input of the effort equation (S, q, P^w, C). The
# gammas and F_hat they contain are therefore dropped from the E and F^p loops.
# Negative curvature bends an arrow to the left of its direction of travel.
EQUATION_NETWORK_EDGES = (
    ('S', 'S', r'$\gamma_S,\ r,\ K$', 0.0),
    ('E', 'E', None, 0.0),
    ('F', 'F', r'$\gamma_F$', 0.0),
    ('FP', 'FP', None, 0.0),
    ('E', 'S', None, 0.0),
    ('q', 'S', None, 0.0),
    ('S', 'H', None, 0.0),
    ('E', 'H', None, -0.22),
    ('q', 'H', None, 0.0),
    ('F', 'q', None, 0.0),
    ('F', 'C', None, 0.0),
    ('F', 'Pw', None, -0.2),
    ('H', 'Pw', None, 0.0),
    ('H', 'Pm', None, 0.0),
    ('FP', 'Pm', None, 0.0),
    ('Pw', 'F', None, -0.2),
    ('Pm', 'F', None, 0.0),
    ('F', 'D', None, 0.0),
    ('D', 'FP', None, 0.0),
    ('S', 'Profit', None, 0.0),
    ('q', 'Profit', None, 0.0),
    ('Pw', 'Profit', None, 0.0),
    ('C', 'Profit', None, 0.0),
    ('Profit', 'E', None, 0.0),
)


def figure_model_dependency_network(include_equations: bool = False):
    """Every direct dependency in the map as one network.

    Rounded boxes are the four state variables; circles are the derived
    quantities they are computed through. Each arrow means "the source appears
    in the target's equation" and is coloured by its source, so any variable's
    influence can be traced by colour. Self-loops mark that every state's
    update depends on its own current value. Small labels name the parameters
    that act on that link (each state's response speed gamma sits on its loop).

    With ``include_equations``, catchability, harvest, market price, wholesale
    price and cost become boxes holding their defining equations, and two
    concept boxes are added: detectability (between F and F^p) and harvester
    profitability (collecting the inputs of the effort equation). Arrow labels
    for parameters that those boxes already show are then omitted.
    """
    from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

    figure_width = TEXT_WIDTH_INCHES
    figure_height = EQUATION_NETWORK_FIGURE_HEIGHT if include_equations else 3.25
    figure = plt.figure(figsize=(figure_width, figure_height))
    axis = figure.add_axes([0, 0, 1, 1])
    axis.set_xlim(0, figure_width)
    axis.set_ylim(0, figure_height)
    axis.set_aspect('equal')
    axis.axis('off')

    state_box_width, state_box_height = 0.80, 0.46
    derived_radius = 0.19
    nodes = {
        # key: (x, y, kind, symbol, name, colour, name placement)
        'S': (0.62, 2.78, 'state', r'$S$', 'Seafood stock', VAR_COLORS['S'], None),
        'E': (0.62, 0.50, 'state', r'$E$', 'Harvesting effort', VAR_COLORS['E'], None),
        'F': (3.98, 0.50, 'state', r'$F$', 'Fraudsters', VAR_COLORS['F'], None),
        'FP': (3.98, 2.78, 'state', r'$F^p$', 'Fraud perception', VAR_COLORS['FP'], None),
        'H': (1.80, 2.78, 'derived', r'$H$', 'Harvest', VAR_COLORS['Harvest'], 'above'),
        'q': (1.80, 1.48, 'derived', r'$q$', 'Catchability', '#52514e', 'right'),
        'C': (2.40, 0.50, 'derived', r'$C$', 'Fishing cost', VAR_COLORS['Cost'], 'below'),
        'Pw': (2.85, 1.62, 'derived', r'$P^w$', 'Wholesale\nprice', VAR_COLORS['Pw'], 'above'),
        'Pm': (2.98, 2.78, 'derived', r'$P^m$', 'Market price', VAR_COLORS['Pm'], 'above'),
    }
    equation_nodes = set(DERIVED_QUANTITY_EQUATIONS) if include_equations else set()
    dependency_edges = EQUATION_NETWORK_EDGES if include_equations else DEPENDENCY_EDGES
    if include_equations:
        nodes.update(EQUATION_NETWORK_EXTRA_NODES)
        for key, (x, y) in EQUATION_NETWORK_POSITIONS.items():
            kind = 'equation' if key in equation_nodes else nodes[key][2]
            nodes[key] = (x, y, kind) + nodes[key][3:]
    renderer = figure.canvas.get_renderer()
    node_patches = {}
    for key, (x, y, kind, symbol, name, color, name_placement) in nodes.items():
        if kind == 'equation':
            # Name (small italic) over the equation, boxed to fit the text.
            equation_text = axis.text(x, y - 0.05, DERIVED_QUANTITY_EQUATIONS[key],
                                      ha='center', va='center', fontsize=6.6,
                                      color='#0b0b0b', zorder=4)
            name_text = axis.text(x, y, name.replace('\n', ' '), ha='center', va='top',
                                  fontsize=5.8, color='#52514e', style='italic', zorder=4)
            to_data = axis.transData.inverted()
            text_extent = equation_text.get_window_extent(renderer).transformed(to_data)
            name_width = name_text.get_window_extent(renderer).transformed(to_data).width
            name_text_height = 0.13
            box_padding = 0.06
            box_width = max(text_extent.width, name_width) + 2 * box_padding
            box_left = x - box_width / 2
            box_bottom = text_extent.y0 - box_padding
            box_height = text_extent.height + name_text_height + 1.5 * box_padding
            patch = FancyBboxPatch(
                (box_left, box_bottom), box_width, box_height,
                boxstyle='round,pad=0,rounding_size=0.05', facecolor='white',
                edgecolor=color, linewidth=1.3, zorder=3,
            )
            axis.add_patch(patch)
            name_text.set_y(box_bottom + box_height - 0.035)
        elif kind == 'state':
            patch = FancyBboxPatch(
                (x - state_box_width / 2, y - state_box_height / 2),
                state_box_width, state_box_height,
                boxstyle='round,pad=0,rounding_size=0.06', facecolor=color, edgecolor='none',
                zorder=3,
            )
            axis.add_patch(patch)
            axis.text(x, y + 0.07, symbol, ha='center', va='center', fontsize=10,
                      color='white', zorder=4)
            axis.text(x, y - 0.14, name, ha='center', va='center', fontsize=5.8,
                      color='white', fontweight='bold', zorder=4)
        else:
            patch = Circle((x, y), derived_radius, facecolor='white', edgecolor=color,
                           linewidth=1.3, zorder=3)
            axis.add_patch(patch)
            axis.text(x, y, symbol, ha='center', va='center', fontsize=8.5, color='#0b0b0b',
                      zorder=4)
            name_offsets = {
                'above': (0, derived_radius + 0.05, 'center', 'bottom'),
                'below': (0, -derived_radius - 0.05, 'center', 'top'),
                'left': (-derived_radius - 0.06, 0, 'right', 'center'),
                'right': (derived_radius + 0.06, 0, 'left', 'center'),
            }
            dx, dy, ha, va = name_offsets[name_placement]
            axis.text(x + dx, y + dy, name, ha=ha, va=va, fontsize=5.8, color='#52514e',
                      style='italic', linespacing=1.05, zorder=4)
        node_patches[key] = patch

    # Self-loops sit on the outer side of each state box.
    self_loop_side = {'S': 'left', 'E': 'left', 'F': 'right', 'FP': 'right'}
    parameter_label_style = dict(fontsize=6, color='#0b0b0b', zorder=6,
                                 bbox=dict(boxstyle='round,pad=0.08', facecolor='white',
                                           edgecolor='none', alpha=0.9))
    for source, target, parameter_label, curvature in dependency_edges:
        color = nodes[source][5]
        if source == target:
            x, y = nodes[source][:2]
            side = self_loop_side[source]
            edge_x = x - state_box_width / 2 if side == 'left' else x + state_box_width / 2
            start, end = (edge_x, y + 0.10), (edge_x, y - 0.10)
            loop_curvature = 2.0 if side == 'left' else -2.0
            axis.add_patch(FancyArrowPatch(
                start, end, connectionstyle=f'arc3,rad={loop_curvature}', arrowstyle='-|>',
                mutation_scale=6, linewidth=0.9, color=color, shrinkA=0, shrinkB=0, zorder=2,
            ))
            if parameter_label:
                label_x = edge_x - 0.05 if side == 'left' else edge_x + 0.05
                axis.text(label_x, y + state_box_height / 2 + 0.06, parameter_label,
                          ha='right' if side == 'left' else 'left', va='bottom',
                          **parameter_label_style)
            continue
        source_x, source_y = nodes[source][:2]
        target_x, target_y = nodes[target][:2]
        if target in equation_nodes:
            parameter_label = None
        axis.add_patch(FancyArrowPatch(
            (source_x, source_y), (target_x, target_y),
            connectionstyle=f'arc3,rad={curvature}', arrowstyle='-|>', mutation_scale=6,
            linewidth=0.9, color=color, patchA=node_patches[source],
            patchB=node_patches[target], shrinkA=1, shrinkB=1.5, zorder=2,
        ))
        if parameter_label:
            # Label sits on the arc's midpoint (arc3 control point is offset by
            # curvature x chord length, perpendicular to the chord).
            chord_x, chord_y = target_x - source_x, target_y - source_y
            midpoint_x = (source_x + target_x) / 2 + curvature / 2 * chord_y
            midpoint_y = (source_y + target_y) / 2 - curvature / 2 * chord_x
            axis.text(midpoint_x, midpoint_y, parameter_label, ha='center', va='center',
                      **parameter_label_style)

    derived_shape = 'Outlined: derived quantities.' if include_equations else \
        'Circles: derived quantities.'
    axis.text(figure_width / 2, 0.02,
              f'Filled boxes: state variables.   {derived_shape}   '
              'Arrow: source enters the target\'s equation (coloured by source).',
              ha='center', va='bottom', fontsize=6, color='#52514e')
    save_figure(figure, 'blast_fig_h_model_dependency_network_equations' if include_equations
                else 'blast_fig_h_model_dependency_network')


# ---------------------------------------------------------------------------
# Figure I — black-market margins (paper version of slides 14-15)
# ---------------------------------------------------------------------------

PRIZED_PRICE_PREMIUM_SCALE = 4.0   # scenario_ps.PRICE_PREMIUM_SCALE

# Figure I parameterizations. Low value = DEFAULT_PARAMS (P^w down, C down a lot,
# q up). High value = roughly its mirror image (P^w up a lot, C up, q down):
# the price and cost changes swap magnitudes, so the margin grows by a similar
# but not identical amount (0.10 -> 0.66 low, 0.10 -> 0.72 high) and both
# panels share one y-axis. The high-value set is a proposal for the paper;
# the app's prized-seafood scenario uses prized_params.
LOW_VALUE_BLACK_MARKET_PARAMS = {key: DEFAULT_PARAMS[key] for key in ('pw1', 'c1', 'q1')}
HIGH_VALUE_BLACK_MARKET_PARAMS = {'pw1': 1.80, 'c1': 1.08, 'q1': 0.035}


def prized_params(alpha) -> dict:
    """Prized-seafood economics at premium alpha (vectorized ``scenario_ps._prized_params``)."""
    alpha = np.asarray(alpha, dtype=np.float64)
    return {
        'q1': DEFAULT_PARAMS['q0'] + 0.0 * alpha,
        'pw1': DEFAULT_PARAMS['pw0'] + alpha * PRIZED_PRICE_PREMIUM_SCALE,
        'c1': DEFAULT_PARAMS['c0'] + 0.0 * alpha,
    }


def figure_black_market_margins():
    """Wholesale price and fishing cost vs fraudster share, for both black markets.

    Plots the fraud-modulated terms of the model,
        P^w(F) = P^w_0 + (P^w_1 - P^w_0) F   (wholesale price at gamma_P H = 1)
        C(F)   = C_0   + (C_1   - C_0)   F,
    for the low-value (DEFAULT_PARAMS) and high-value (mirror-image) black
    markets on a shared y-axis. The margin P^w - C is shaded, with its honest
    (F = 0) and full-fraud (F = 1) values marked; the parameter values used are
    printed under each panel.
    """
    pw0, c0, q0 = DEFAULT_PARAMS['pw0'], DEFAULT_PARAMS['c0'], DEFAULT_PARAMS['q0']
    fraud_share = np.linspace(0.0, 1.0, 101)
    honest_line = rf'$P^w_0 = {pw0:.2f},\ \ C_0 = {c0:.2f},\ \ q_0 = {q0:.2f}$'

    def two_or_three_decimals(value):
        return f'{value:.2f}' if round(value, 2) == round(value, 3) else f'{value:.3f}'

    def full_fraud_line(params):
        pw1, c1, q1 = (two_or_three_decimals(params[key]) for key in ('pw1', 'c1', 'q1'))
        return rf'$P^w_1 = {pw1},\ \ C_1 = {c1},\ \ q_1 = {q1}$'

    markets = (
        ('Low-value black market', 'Fraudsters negotiate lower prices',
         r'$P^w\!\downarrow\quad C\!\downarrow\!\downarrow\quad q\!\uparrow$',
         LOW_VALUE_BLACK_MARKET_PARAMS),
        ('High-value black market', 'Harvesters negotiate higher prices',
         r'$P^w\!\uparrow\!\uparrow\quad C\!\uparrow\quad q\!\downarrow$',
         HIGH_VALUE_BLACK_MARKET_PARAMS),
    )
    price_color, cost_color = VAR_COLORS['Pw'], VAR_COLORS['Cost']
    margin_color = VAR_COLORS['E']
    y_limits = (0.0, 1.9)

    figure, axes = plt.subplots(
        1, 2, figsize=(TEXT_WIDTH_INCHES, 2.75), sharey=True,
        gridspec_kw=dict(wspace=0.12, left=0.10, right=0.98, top=0.79, bottom=0.30),
    )
    for axis, (title, subtitle, arrows, params) in zip(axes, markets):
        wholesale_price = pw0 + (params['pw1'] - pw0) * fraud_share
        fishing_cost = c0 + (params['c1'] - c0) * fraud_share
        axis.plot(fraud_share, wholesale_price, color=price_color, linewidth=1.5, zorder=3)
        axis.plot(fraud_share, fishing_cost, color=cost_color, linewidth=1.5, zorder=3)
        axis.fill_between(fraud_share, fishing_cost, wholesale_price,
                          color=margin_color, alpha=0.15, linewidth=0, zorder=1)
        for share_index, horizontal_alignment, offset in ((0, 'left', 0.03),
                                                          (-1, 'right', -0.03)):
            low, high = fishing_cost[share_index], wholesale_price[share_index]
            x_position = fraud_share[share_index] + offset
            # Tiny margins get a plain bar: arrowheads would overlap.
            is_tiny = (high - low) < 0.12 * (y_limits[1] - y_limits[0])
            axis.annotate('', xy=(x_position, high), xytext=(x_position, low),
                          arrowprops=dict(arrowstyle='-' if is_tiny else '<->',
                                          color=margin_color, linewidth=0.9,
                                          shrinkA=0, shrinkB=0),
                          zorder=4)
            axis.text(x_position + (0.025 if share_index == 0 else -0.025),
                      (low + high) / 2, f'{high - low:.2f}',
                      ha=horizontal_alignment, va='center', fontsize=6.3,
                      color='#0b0b0b', zorder=5,
                      bbox=dict(boxstyle='round,pad=0.1', facecolor='white',
                                edgecolor='none', alpha=0.85))
        axis.set_xlim(0, 1)
        axis.set_ylim(*y_limits)
        axis.set_xticks([0, 0.5, 1], ['0', '0.5', '1'])
        axis.set_xlabel(r'Fraudster share $F$  (0 = honest, 1 = all fraud)', labelpad=1)
        axis.grid(axis='y')
        axis.set_title(f'{title}\n', pad=2, fontsize=8, fontweight='bold')
        axis.text(0.5, 1.035, f'{subtitle}:  {arrows}', transform=axis.transAxes,
                  ha='center', va='bottom', fontsize=6.6, color='#52514e')
        axis.text(0.0, -0.27, honest_line + '\n' + full_fraud_line(params),
                  transform=axis.transAxes, ha='left', va='top', fontsize=6.2,
                  color='#0b0b0b', linespacing=1.45)
    axes[0].set_ylabel('Price  /  cost', labelpad=2)

    figure.legend(
        handles=[
            Line2D([], [], color=price_color, linewidth=1.5, label=r'Wholesale price $P^w(F)$'),
            Line2D([], [], color=cost_color, linewidth=1.5, label=r'Fishing cost $C(F)$'),
            Patch(facecolor=margin_color, alpha=0.15, label=r'Margin $P^w - C$'),
        ],
        loc='upper center', bbox_to_anchor=(0.54, 1.0), ncol=3, frameon=False,
        handlelength=1.5, columnspacing=1.4, fontsize=6.6,
    )
    save_figure(figure, 'blast_fig_i_black_market_margins')


# ---------------------------------------------------------------------------
# Verification against core.System
# ---------------------------------------------------------------------------

def verify_against_dynamical_system(timesteps: int = 600):
    """Fail loudly if the vectorized map drifts from DynamicalSystem."""
    from core.System import DynamicalSystem

    print('Verifying vectorized map against core.System.DynamicalSystem ...')
    worst_difference = 0.0
    for story in STORIES:
        for alpha in (0.0, 0.15, 0.55, 1.0):
            params = story.params()
            params.update({key: float(value) for key, value in blast_params(alpha).items()})
            state = {key: np.longdouble(value) for key, value in DEFAULT_INIT_STATE.items()}
            reference = DynamicalSystem(params, state, 'dimensionalized').generate_time_series(
                timesteps,
            )
            vectorized = record_time_series(params, DEFAULT_INIT_STATE, timesteps)
            for reference_key, key in (('Seafood', 'S'), ('Effort', 'E'),
                                       ('Fraudsters', 'F'), ('Perception of Fraud', 'FP')):
                difference = np.max(np.abs(reference[reference_key].astype(np.float64)
                                           - vectorized[key]))
                worst_difference = max(worst_difference, float(difference))
    print(f'  worst |difference| over S, E, F, FP: {worst_difference:.3e}')
    if worst_difference > 1e-9:
        raise SystemExit('Vectorized map does not match DynamicalSystem.')
    print('  OK')


FIGURE_BUILDERS = {
    'outcome_curves': figure_outcome_curves,
    'outcome_curves_no_dynamics': lambda: figure_outcome_curves(include_dynamics_ribbon=False),
    'story_welfare_panels': figure_story_welfare_panels,
    'story_welfare_panels_multiples': lambda: figure_story_welfare_panels(y_axis_as_multiple=True),
    'model_flow_diagram': figure_model_flow_diagram,
    'model_dependency_network': figure_model_dependency_network,
    'model_dependency_network_equations':
        lambda: figure_model_dependency_network(include_equations=True),
    'black_market_margins': figure_black_market_margins,
    'policy_maps': figure_policy_maps,
    'price_decomposition': figure_price_decomposition,
    'boom_bust_anatomy': figure_boom_bust_anatomy,
    'time_series_matrix': figure_time_series_matrix,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--figures', nargs='+', choices=sorted(FIGURE_BUILDERS),
                        default=list(FIGURE_BUILDERS))
    parser.add_argument('--verify', action='store_true',
                        help='check the vectorized map against DynamicalSystem first')
    arguments = parser.parse_args()

    if arguments.verify:
        verify_against_dynamical_system()
    apply_paper_style()
    for figure_name in arguments.figures:
        print(f'Building {figure_name} ...')
        FIGURE_BUILDERS[figure_name]()


if __name__ == '__main__':
    main()
