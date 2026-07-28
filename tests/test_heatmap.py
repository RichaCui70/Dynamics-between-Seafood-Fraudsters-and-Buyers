"""
Test suite for plot_time_series_heatmap — verifying the prompt requirements:

  1.  Returns a list of Plotly Figures (one per active metric) when inputs are valid.
  2.  Y-label of each figure = the corresponding metric pill (S̄, H̄, P̄ᵐ).
  3.  Pills control which metrics appear: each selected metric → one figure.
  4.  X-axis columns = the same param_vals that were selected for the time series.
  5.  When a param value is removed (unselected), its column disappears.
  6.  Column COUNT matches param_vals count (proxy for "same-width" alignment).
  7.  Cell values are signed % deviation from row mean (format: +X.X% / -X.X%).
  8.  Averages use post-burn data (burn_in_fraction=0.6 by default).
  9.  Seafood/Harvest use red→white→green; market price polarity is inverted.
 10.  Per-metric symmetric normalisation: z colour values in [0, 1] with 0%→0.5.
 11.  Metric display order follows HEATMAP_METRICS canonical order.
 12.  Edge cases: empty metrics → None, empty params → None, single column/row.
"""

import re
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import plotly.graph_objects as go
from core.plots import (
    plot_time_series_heatmap,
    HEATMAP_METRICS,
    SEAFOOD_COLORSCALE,
    HARVEST_COLORSCALE,
    MARKET_PRICE_COLORSCALE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ts(n: int = 400, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    return {
        'Seafood':      rng.uniform(0.2, 0.8, n),
        'Harvest':      rng.uniform(0.01, 0.1, n),
        'Market Price': rng.uniform(0.5, 5.0, n),
        'Effort':       rng.uniform(0.1, 1.0, n),  # extra key — must be ignored
    }


PARAM_VALS = [1.0, 2.0, 3.0, 5.0]
TS_DICT = {pv: _make_ts(seed=int(pv * 10)) for pv in PARAM_VALS}

_PCT_RE = re.compile(r'^[+-]\d+\.\d+%$')


def _get_figs(ts_dict=None, param_vals=None, label='pw₁', metrics=None):
    return plot_time_series_heatmap(
        ts_dict  if ts_dict    is not None else TS_DICT,
        param_vals if param_vals is not None else PARAM_VALS,
        label,
        metrics  if metrics    is not None else HEATMAP_METRICS,
    )


# ---------------------------------------------------------------------------
# Requirement 1 — returns a list of go.Figure
# ---------------------------------------------------------------------------

def test_returns_list_of_figures():
    result = _get_figs()
    assert isinstance(result, list)
    assert all(isinstance(f, go.Figure) for f in result)


def test_list_length_matches_active_metrics():
    for n in range(1, len(HEATMAP_METRICS) + 1):
        active = HEATMAP_METRICS[:n]
        figs = _get_figs(metrics=active)
        assert len(figs) == n, f"Expected {n} figures, got {len(figs)}"


# ---------------------------------------------------------------------------
# Requirement 12 (edge) — empty inputs → None
# ---------------------------------------------------------------------------

def test_returns_none_for_empty_metrics():
    assert _get_figs(metrics=[]) is None


def test_returns_none_for_empty_param_vals():
    assert _get_figs(ts_dict={}, param_vals=[]) is None


# ---------------------------------------------------------------------------
# Requirement 2 — each figure's y label = its metric
# ---------------------------------------------------------------------------

def test_each_figure_y_label_matches_metric():
    for n in range(1, len(HEATMAP_METRICS) + 1):
        active = HEATMAP_METRICS[:n]
        figs = _get_figs(metrics=active)
        for fig, pill in zip(figs, active):
            assert list(fig.data[0].y) == [pill]


# ---------------------------------------------------------------------------
# Requirement 3 — pill selection controls which figures are produced
# ---------------------------------------------------------------------------

def test_skipping_metric_removes_its_figure():
    all_figs   = _get_figs(metrics=HEATMAP_METRICS)
    two_figs   = _get_figs(metrics=['S̄', 'P̄ᵐ'])
    one_fig    = _get_figs(metrics=['H̄'])
    assert len(all_figs) == 3
    assert len(two_figs) == 2
    assert len(one_fig)  == 1
    assert list(one_fig[0].data[0].y) == ['H̄']


# ---------------------------------------------------------------------------
# Requirement 4 & 6 — x-axis columns = param_vals; count must match
# ---------------------------------------------------------------------------

def test_x_count_matches_param_vals():
    figs = _get_figs()
    for fig in figs:
        assert len(fig.data[0].x) == len(PARAM_VALS)


def test_x_labels_are_string_of_param_vals():
    figs = _get_figs()
    expected = [str(v) for v in PARAM_VALS]
    for fig in figs:
        assert list(fig.data[0].x) == expected


# ---------------------------------------------------------------------------
# Requirement 5 — removing a param value removes its column
# ---------------------------------------------------------------------------

def test_column_disappears_when_param_removed():
    subset = PARAM_VALS[:2]
    full_figs = _get_figs()
    sub_figs  = _get_figs(param_vals=subset)
    for f in full_figs:
        assert len(f.data[0].x) == len(PARAM_VALS)
    for f in sub_figs:
        assert len(f.data[0].x) == len(subset)


def test_column_values_match_selected_params():
    subset = [1.0, 5.0]
    figs = _get_figs(param_vals=subset)
    for fig in figs:
        assert list(fig.data[0].x) == ['1.0', '5.0']


# ---------------------------------------------------------------------------
# Requirement 7 — cell text is signed % deviation from row mean
# ---------------------------------------------------------------------------

def test_text_values_are_signed_percent():
    figs = _get_figs()
    for fig in figs:
        for cell in fig.data[0].text[0]:
            assert _PCT_RE.match(cell), f"Cell '{cell}' is not a signed-% string"


def test_pct_deviations_sum_to_zero():
    """Sum of % deviations across a row must be 0 (property of mean-centring)."""
    figs = _get_figs()
    for fig in figs:
        pcts = [float(c.rstrip('%')) for c in fig.data[0].text[0]]
        assert abs(sum(pcts)) < 1e-6 * len(pcts), (
            f"Deviations don't sum to 0: {pcts}"
        )


def test_higher_avg_gives_positive_pct():
    """Parameter value with higher post-burn average should show positive %."""
    controlled = {
        1.0: {'Seafood': np.full(400, 0.3), 'Harvest': np.full(400, 0.3), 'Market Price': np.full(400, 0.3)},
        2.0: {'Seafood': np.full(400, 0.7), 'Harvest': np.full(400, 0.7), 'Market Price': np.full(400, 0.7)},
    }
    figs = plot_time_series_heatmap(controlled, [1.0, 2.0], 'x', ['S̄'])
    texts = figs[0].data[0].text[0]  # [pct_for_1.0, pct_for_2.0]
    assert texts[0].startswith('-'), f"Lower avg (pv=1.0) should be negative, got {texts[0]}"
    assert texts[1].startswith('+'), f"Higher avg (pv=2.0) should be positive, got {texts[1]}"


# ---------------------------------------------------------------------------
# Requirement 8 — averages use post-burn data (burn_in_fraction=0.6 by default)
# ---------------------------------------------------------------------------

def test_burn_fraction_applied():
    """pv=1.0 has pre-burn=100, post-burn=0.1 → should show negative % vs pv=2.0 (const=1)."""
    series_length, burn_in_steps = 400, int(400 * 0.6)
    biased = np.concatenate([
        np.full(burn_in_steps, 100.0),
        np.full(series_length - burn_in_steps, 0.1),
    ])
    const  = np.full(series_length, 1.0)
    biased_dict = {
        1.0: {'Seafood': biased, 'Harvest': biased, 'Market Price': biased},
        2.0: {'Seafood': const,  'Harvest': const,  'Market Price': const},
    }
    figs = plot_time_series_heatmap(biased_dict, [1.0, 2.0], 'x', ['S̄'])
    pct_low = figs[0].data[0].text[0][0]  # pv=1.0
    assert pct_low.startswith('-'), (
        f"pv=1.0 post-burn avg ≈ 0.1 should be below mean — got {pct_low}"
    )


def test_custom_burn_fraction_zero():
    """burn_in_fraction=0.0 → full series is used; biased series avg >> const series avg."""
    series_length, burn_in_steps = 400, int(400 * 0.6)
    biased = np.concatenate([
        np.full(burn_in_steps, 100.0),
        np.full(series_length - burn_in_steps, 0.1),
    ])
    const  = np.full(series_length, 1.0)
    biased_dict = {
        1.0: {'Seafood': biased, 'Harvest': biased, 'Market Price': biased},
        2.0: {'Seafood': const,  'Harvest': const,  'Market Price': const},
    }
    figs = plot_time_series_heatmap(
        biased_dict, [1.0, 2.0], 'x', ['S̄'], burn_in_fraction=0.0,
    )
    pct_high = figs[0].data[0].text[0][0]  # pv=1.0 — now has very high full-series avg
    assert pct_high.startswith('+'), (
        f"pv=1.0 full-series avg ≈ 60 should be above mean — got {pct_high}"
    )


# ---------------------------------------------------------------------------
# Requirement 9 — Seafood/Harvest red↔green; market price polarity inverted
# ---------------------------------------------------------------------------

def _cs_colors(cs):
    """Extract just the color strings from a colorscale (handles list or tuple entries)."""
    return [entry[1] for entry in cs]


def test_seafood_uses_green_red_colorscale():
    figs = _get_figs(metrics=['S̄'])
    cs = figs[0].data[0].colorscale
    assert _cs_colors(cs) == _cs_colors(SEAFOOD_COLORSCALE), f"Seafood colorscale wrong: {cs}"


def test_harvest_uses_green_red_colorscale():
    figs = _get_figs(metrics=['H̄'])
    cs = figs[0].data[0].colorscale
    assert _cs_colors(cs) == _cs_colors(HARVEST_COLORSCALE), f"Harvest colorscale wrong: {cs}"


def test_market_price_uses_inverted_colorscale():
    figs = _get_figs(metrics=['P̄ᵐ'])
    cs = figs[0].data[0].colorscale
    assert _cs_colors(cs) == _cs_colors(MARKET_PRICE_COLORSCALE), f"Market Price colorscale wrong: {cs}"


# ---------------------------------------------------------------------------
# Requirement 10 — symmetric normalisation: z values in [0, 1]; 0% → 0.5
# ---------------------------------------------------------------------------

def test_z_values_in_unit_interval():
    figs = _get_figs()
    for fig in figs:
        for v in fig.data[0].z[0]:
            assert -1e-9 <= v <= 1 + 1e-9, f"z value {v} outside [0, 1]"


def test_zero_deviation_maps_to_half():
    """When all param values have the same post-burn avg, every z must equal 0.5."""
    const_dict = {pv: {k: np.full(400, 0.5) for k in ['Seafood', 'Harvest', 'Market Price']}
                  for pv in PARAM_VALS}
    figs = plot_time_series_heatmap(const_dict, PARAM_VALS, 'x', HEATMAP_METRICS)
    for fig in figs:
        for v in fig.data[0].z[0]:
            assert abs(v - 0.5) < 1e-9, f"Expected 0.5 for zero deviation, got {v}"


def test_positive_deviation_above_half():
    """param value with higher avg → z > 0.5 (positive deviation maps above midpoint)."""
    controlled = {
        1.0: {'Seafood': np.full(400, 0.2), 'Harvest': np.full(400, 0.2), 'Market Price': np.full(400, 0.2)},
        2.0: {'Seafood': np.full(400, 0.8), 'Harvest': np.full(400, 0.8), 'Market Price': np.full(400, 0.8)},
    }
    figs = plot_time_series_heatmap(controlled, [1.0, 2.0], 'x', HEATMAP_METRICS)
    for fig in figs:
        z = fig.data[0].z[0]
        assert z[0] < 0.5, f"Lower param (pv=1.0) should map below 0.5, got {z[0]}"
        assert z[1] > 0.5, f"Higher param (pv=2.0) should map above 0.5, got {z[1]}"


# ---------------------------------------------------------------------------
# Requirement 11 — canonical HEATMAP_METRICS order preserved
# ---------------------------------------------------------------------------

def test_metric_order_follows_canonical_order():
    reversed_metrics = list(reversed(HEATMAP_METRICS))
    figs = _get_figs(metrics=reversed_metrics)
    for fig, expected_pill in zip(figs, HEATMAP_METRICS):
        assert fig.data[0].y[0] == expected_pill


# ---------------------------------------------------------------------------
# Edge cases — single column / single metric
# ---------------------------------------------------------------------------

def test_single_param_value():
    single = {PARAM_VALS[0]: _make_ts()}
    figs = plot_time_series_heatmap(single, [PARAM_VALS[0]], 'pw₁', HEATMAP_METRICS)
    assert figs is not None
    for fig in figs:
        assert len(fig.data[0].x) == 1
        assert abs(fig.data[0].z[0][0] - 0.5) < 1e-9  # zero deviation → 0.5


def test_single_metric():
    figs = _get_figs(metrics=['H̄'])
    assert figs is not None and len(figs) == 1
    assert figs[0].data[0].y[0] == 'H̄'


# ---------------------------------------------------------------------------
# HEATMAP_METRICS constant sanity
# ---------------------------------------------------------------------------

def test_heatmap_metrics_constant():
    assert HEATMAP_METRICS == ['S̄', 'H̄', 'P̄ᵐ']
    assert len(HEATMAP_METRICS) == 3
