"""
Test suite for plot_time_series_heatmap — render-only heatmap figures.

  1.  Returns a list of Plotly Figures (one per active metric) when inputs are valid.
  2.  Y-label of each figure = the corresponding metric pill.
  3.  Pills control which metrics appear: each selected metric → one figure.
  4.  X-axis columns = the same param_vals that were selected for the time series.
  5.  When a param value is removed (unselected), its column disappears.
  6.  Column COUNT matches param_vals count (proxy for "same-width" alignment).
  7.  Cell values are signed % strings (format: +X.X% / -X.X%).
  8.  Seafood/Harvest use red→white→green; market price / φ polarity inverted.
  9.  Per-metric symmetric normalisation: z colour values in [0, 1] with 0%→0.5.
 10.  Metric display order follows HEATMAP_METRICS canonical order.
 11.  Edge cases: empty metrics → None, empty params → None, single column/row.
"""

import re
import sys
import os
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


PARAM_VALS = [1.0, 2.0, 3.0, 5.0]

# Deterministic percents: column i gets (i - mean_index) * 10
_MEAN_INDEX = (len(PARAM_VALS) - 1) / 2.0
PERCENT_BY_METRIC = {
    metric: [(i - _MEAN_INDEX) * 10.0 for i in range(len(PARAM_VALS))]
    for metric in HEATMAP_METRICS
}

_PCT_RE = re.compile(r'^[+-]\d+\.\d+%$')


def _get_figs(percent_by_metric=None, param_vals=None, label='pw₁', metrics=None):
    param_vals = PARAM_VALS if param_vals is None else param_vals
    metrics = HEATMAP_METRICS if metrics is None else metrics
    if percent_by_metric is None:
        percent_by_metric = {
            metric: [(i - (len(param_vals) - 1) / 2.0) * 10.0 for i in range(len(param_vals))]
            for metric in HEATMAP_METRICS
        }
    return plot_time_series_heatmap(percent_by_metric, param_vals, label, metrics)


def test_returns_list_of_figures():
    result = _get_figs()
    assert isinstance(result, list)
    assert all(isinstance(f, go.Figure) for f in result)


def test_list_length_matches_active_metrics():
    for n in range(1, len(HEATMAP_METRICS) + 1):
        active = HEATMAP_METRICS[:n]
        figs = _get_figs(metrics=active)
        assert len(figs) == n, f"Expected {n} figures, got {len(figs)}"


def test_returns_none_for_empty_metrics():
    assert _get_figs(metrics=[]) is None


def test_returns_none_for_empty_param_vals():
    assert _get_figs(param_vals=[]) is None


def test_each_figure_y_label_matches_metric():
    for n in range(1, len(HEATMAP_METRICS) + 1):
        active = HEATMAP_METRICS[:n]
        figs = _get_figs(metrics=active)
        for fig, pill in zip(figs, active):
            assert list(fig.data[0].y) == [pill]


def test_skipping_metric_removes_its_figure():
    all_figs = _get_figs(metrics=HEATMAP_METRICS)
    two_figs = _get_figs(metrics=['S̄', 'P̄ᵐ'])
    one_fig = _get_figs(metrics=['H̄'])
    assert len(all_figs) == len(HEATMAP_METRICS)
    assert len(two_figs) == 2
    assert len(one_fig) == 1
    assert list(one_fig[0].data[0].y) == ['H̄']


def test_x_count_matches_param_vals():
    figs = _get_figs()
    for fig in figs:
        assert len(fig.data[0].x) == len(PARAM_VALS)


def test_x_labels_are_string_of_param_vals():
    figs = _get_figs()
    expected = [str(v) for v in PARAM_VALS]
    for fig in figs:
        assert list(fig.data[0].x) == expected


def test_column_disappears_when_param_removed():
    subset = PARAM_VALS[:2]
    full_figs = _get_figs()
    sub_figs = _get_figs(param_vals=subset)
    for f in full_figs:
        assert len(f.data[0].x) == len(PARAM_VALS)
    for f in sub_figs:
        assert len(f.data[0].x) == len(subset)


def test_column_values_match_selected_params():
    subset = [1.0, 5.0]
    figs = _get_figs(param_vals=subset)
    for fig in figs:
        assert list(fig.data[0].x) == ['1.0', '5.0']


def test_text_values_are_signed_percent():
    figs = _get_figs()
    for fig in figs:
        for cell in fig.data[0].text[0]:
            assert _PCT_RE.match(cell), f"Cell '{cell}' is not a signed-% string"


def test_renders_precomputed_percents():
    percents = {
        metric: [-20.0, 10.0]
        for metric in HEATMAP_METRICS
    }
    figs = plot_time_series_heatmap(percents, [1.0, 2.0], 'x', ['S̄'])
    assert figs[0].data[0].text[0] == ['-20.0%', '+10.0%']


def _cs_colors(cs):
    return [entry[1] for entry in cs]


def test_seafood_uses_green_red_colorscale():
    figs = _get_figs(metrics=['S̄'])
    cs = figs[0].data[0].colorscale
    assert _cs_colors(cs) == _cs_colors(SEAFOOD_COLORSCALE)


def test_harvest_uses_green_red_colorscale():
    figs = _get_figs(metrics=['H̄'])
    cs = figs[0].data[0].colorscale
    assert _cs_colors(cs) == _cs_colors(HARVEST_COLORSCALE)


def test_market_price_uses_inverted_colorscale():
    figs = _get_figs(metrics=['P̄ᵐ'])
    cs = figs[0].data[0].colorscale
    assert _cs_colors(cs) == _cs_colors(MARKET_PRICE_COLORSCALE)


def test_phi_metrics_use_market_price_colorscale():
    for metric in ('φ_FP', 'φ_H'):
        figs = _get_figs(metrics=[metric])
        cs = figs[0].data[0].colorscale
        assert _cs_colors(cs) == _cs_colors(MARKET_PRICE_COLORSCALE)


def test_z_values_in_unit_interval():
    figs = _get_figs()
    for fig in figs:
        for v in fig.data[0].z[0]:
            assert -1e-9 <= v <= 1 + 1e-9, f"z value {v} outside [0, 1]"


def test_zero_percent_maps_to_half():
    zeros = {metric: [0.0] * len(PARAM_VALS) for metric in HEATMAP_METRICS}
    figs = plot_time_series_heatmap(zeros, PARAM_VALS, 'x', HEATMAP_METRICS)
    for fig in figs:
        for v in fig.data[0].z[0]:
            assert abs(v - 0.5) < 1e-9, f"Expected 0.5 for 0%, got {v}"


def test_positive_percent_above_half():
    percents = {metric: [-10.0, 20.0] for metric in HEATMAP_METRICS}
    figs = plot_time_series_heatmap(percents, [1.0, 2.0], 'x', ['S̄', 'P̄ᵐ'])
    for fig in figs:
        z = fig.data[0].z[0]
        assert z[0] < 0.5
        assert z[1] > 0.5


def test_metric_order_follows_canonical_order():
    reversed_metrics = list(reversed(HEATMAP_METRICS))
    figs = _get_figs(metrics=reversed_metrics)
    for fig, expected_pill in zip(figs, HEATMAP_METRICS):
        assert fig.data[0].y[0] == expected_pill


def test_single_param_value():
    percents = {metric: [0.0] for metric in HEATMAP_METRICS}
    figs = plot_time_series_heatmap(percents, [PARAM_VALS[0]], 'pw₁', HEATMAP_METRICS)
    assert figs is not None
    for fig in figs:
        assert len(fig.data[0].x) == 1
        assert abs(fig.data[0].z[0][0] - 0.5) < 1e-9


def test_single_metric():
    figs = _get_figs(metrics=['H̄'])
    assert figs is not None and len(figs) == 1
    assert figs[0].data[0].y[0] == 'H̄'


def test_heatmap_metrics_constant():
    assert HEATMAP_METRICS == ['S̄', 'H̄', 'P̄ᵐ', 'φ_FP', 'φ_H']
    assert len(HEATMAP_METRICS) == 5


def test_mismatched_percent_length_raises():
    bad = {metric: [0.0] for metric in HEATMAP_METRICS}  # length 1, params length 2
    with pytest.raises(ValueError):
        plot_time_series_heatmap(bad, [1.0, 2.0], 'x', ['S̄'])
