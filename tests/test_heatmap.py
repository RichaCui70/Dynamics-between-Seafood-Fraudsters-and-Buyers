"""
Test suite for plot_ts_heatmap — verifying the prompt requirements:

  1. Heatmap appears under the time series (function returns a Figure).
  2. Y-axis rows = selected subset of [S̄, H̄, P̄ᵐ]; pills control which rows appear.
  3. X-axis columns = the same param_vals that were selected for the time series.
  4. When a param value is removed (unselected), its column disappears.
  5. Column COUNT matches param_vals count (proxy for "same width" alignment).
  6. Averages use post-burn data (burn_frac=0.6 by default).
  7. Edge cases: empty metrics → None, empty params → None, single column, single row.
  8. Per-row normalisation keeps colour values in [0, 1].
  9. Cell text values are formatted to 3 decimal places.
 10. Metric display order follows HEATMAP_METRICS canonical order regardless of
     the order the caller passes active_metrics.
"""

import re
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import plotly.graph_objects as go
from scenarios.plots import plot_ts_heatmap, HEATMAP_METRICS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ts(n: int = 400, seed: int = 42) -> dict:
    """Minimal fake time-series dict with the three heatmap-relevant keys."""
    rng = np.random.default_rng(seed)
    return {
        'Seafood':      rng.uniform(0.2, 0.8, n),
        'Harvest':      rng.uniform(0.01, 0.1, n),
        'Market Price': rng.uniform(0.5, 5.0, n),
        'Effort':       rng.uniform(0.1, 1.0, n),  # extra key — must be ignored
    }


PARAM_VALS = [1.0, 2.0, 3.0, 5.0]
TS_DICT = {pv: _make_ts(seed=int(pv * 10)) for pv in PARAM_VALS}


# ---------------------------------------------------------------------------
# Requirement 1 — function returns a plotly Figure when inputs are valid
# ---------------------------------------------------------------------------

def test_returns_figure():
    fig = plot_ts_heatmap(TS_DICT, PARAM_VALS, 'pw₁', HEATMAP_METRICS)
    assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# Requirement 7 (edge) — empty inputs → None (no crash, no empty figure)
# ---------------------------------------------------------------------------

def test_returns_none_for_empty_metrics():
    result = plot_ts_heatmap(TS_DICT, PARAM_VALS, 'pw₁', [])
    assert result is None


def test_returns_none_for_empty_param_vals():
    result = plot_ts_heatmap({}, [], 'pw₁', HEATMAP_METRICS)
    assert result is None


# ---------------------------------------------------------------------------
# Requirement 2 — y-axis rows match active_metrics (pills control rows)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_metrics", [1, 2, 3])
def test_y_count_matches_active_metrics(n_metrics):
    active = HEATMAP_METRICS[:n_metrics]
    fig = plot_ts_heatmap(TS_DICT, PARAM_VALS, 'pw₁', active)
    assert len(fig.data[0].y) == n_metrics, (
        f"Expected {n_metrics} y-rows, got {len(fig.data[0].y)}"
    )


def test_y_labels_match_active_metrics():
    active = ['S̄', 'P̄ᵐ']  # skip H̄
    fig = plot_ts_heatmap(TS_DICT, PARAM_VALS, 'pw₁', active)
    assert list(fig.data[0].y) == ['S̄', 'P̄ᵐ']


# ---------------------------------------------------------------------------
# Requirement 3 & 5 — x-axis columns = param_vals; count must match
# ---------------------------------------------------------------------------

def test_x_count_matches_param_vals():
    fig = plot_ts_heatmap(TS_DICT, PARAM_VALS, 'pw₁', HEATMAP_METRICS)
    assert len(fig.data[0].x) == len(PARAM_VALS), (
        f"Expected {len(PARAM_VALS)} x-columns, got {len(fig.data[0].x)}"
    )


def test_x_labels_are_string_of_param_vals():
    fig = plot_ts_heatmap(TS_DICT, PARAM_VALS, 'pw₁', HEATMAP_METRICS)
    expected = [str(v) for v in PARAM_VALS]
    assert list(fig.data[0].x) == expected


# ---------------------------------------------------------------------------
# Requirement 4 — removing a param value removes its column
# ---------------------------------------------------------------------------

def test_column_disappears_when_param_removed():
    subset = PARAM_VALS[:2]
    fig_full = plot_ts_heatmap(TS_DICT, PARAM_VALS,  'pw₁', HEATMAP_METRICS)
    fig_sub  = plot_ts_heatmap(TS_DICT, subset,       'pw₁', HEATMAP_METRICS)
    assert len(fig_full.data[0].x) == len(PARAM_VALS)
    assert len(fig_sub.data[0].x)  == len(subset)


def test_column_values_match_selected_params():
    """The x labels of the subset figure must be exactly the subset param vals."""
    subset = [1.0, 5.0]
    fig = plot_ts_heatmap(TS_DICT, subset, 'pw₁', HEATMAP_METRICS)
    assert list(fig.data[0].x) == ['1.0', '5.0']


# ---------------------------------------------------------------------------
# Requirement 6 — averages use post-burn data (default burn_frac=0.6)
# ---------------------------------------------------------------------------

def test_burn_fraction_applied():
    """First 60% of ts = 100.0, last 40% = 0.1. Post-burn avg must be near 0.1."""
    n = 400
    burn = int(n * 0.6)
    biased = np.concatenate([np.full(burn, 100.0), np.full(n - burn, 0.1)])
    biased_dict = {1.0: {
        'Seafood':      biased.copy(),
        'Harvest':      biased.copy(),
        'Market Price': biased.copy(),
    }}
    fig = plot_ts_heatmap(biased_dict, [1.0], 'pw₁', ['S̄'])
    text_val = float(fig.data[0].text[0][0])
    assert text_val < 1.0, (
        f"Expected post-burn avg ≈ 0.1, got {text_val:.4f}. "
        "Burn fraction may not be applied."
    )


def test_custom_burn_fraction():
    """burn_frac=0.0 → average includes the full series (100.0 ≈ average of full biased array)."""
    n = 400
    burn = int(n * 0.6)
    biased = np.concatenate([np.full(burn, 100.0), np.full(n - burn, 0.1)])
    biased_dict = {1.0: {
        'Seafood':      biased.copy(),
        'Harvest':      biased.copy(),
        'Market Price': biased.copy(),
    }}
    fig = plot_ts_heatmap(biased_dict, [1.0], 'pw₁', ['S̄'], burn_frac=0.0)
    text_val = float(fig.data[0].text[0][0])
    # full avg ≈ 0.6*100 + 0.4*0.1 = 60.04
    assert text_val > 50.0, (
        f"Expected full-series avg ≈ 60, got {text_val:.4f}."
    )


# ---------------------------------------------------------------------------
# Requirement 8 — per-row normalisation: z colour values in [0, 1]
# ---------------------------------------------------------------------------

def test_per_row_normalisation_range():
    fig = plot_ts_heatmap(TS_DICT, PARAM_VALS, 'pw₁', HEATMAP_METRICS)
    z = fig.data[0].z
    for i, row in enumerate(z):
        assert min(row) >= -1e-9, f"Row {i} has colour value below 0"
        assert max(row) <= 1 + 1e-9, f"Row {i} has colour value above 1"


def test_per_row_normalisation_independent():
    """Different metrics can have very different raw scales; each row must span [0, 1]."""
    extreme_dict = {
        pv: {
            'Seafood':      np.full(400, 0.5 + pv * 0.01),   # tiny variance
            'Harvest':      np.full(400, pv * 1000),           # huge values
            'Market Price': np.full(400, 1.0 / pv),
        }
        for pv in PARAM_VALS
    }
    fig = plot_ts_heatmap(extreme_dict, PARAM_VALS, 'x', HEATMAP_METRICS)
    z = fig.data[0].z
    for row in z:
        assert min(row) >= -1e-9
        assert max(row) <= 1 + 1e-9


# ---------------------------------------------------------------------------
# Requirement 9 — cell text is formatted to 3 decimal places
# ---------------------------------------------------------------------------

_FLOAT_3DP = re.compile(r'^\d+\.\d{3}$')

def test_text_values_are_3dp_floats():
    fig = plot_ts_heatmap(TS_DICT, PARAM_VALS, 'pw₁', HEATMAP_METRICS)
    text = fig.data[0].text
    for row in text:
        for cell in row:
            assert _FLOAT_3DP.match(cell), (
                f"Cell text '{cell}' is not a 3-decimal-place float."
            )


# ---------------------------------------------------------------------------
# Requirement 10 — canonical HEATMAP_METRICS order preserved
# ---------------------------------------------------------------------------

def test_metric_order_follows_canonical_order():
    """Active_metrics passed in reverse order must still appear in HEATMAP_METRICS order."""
    reversed_metrics = list(reversed(HEATMAP_METRICS))
    fig = plot_ts_heatmap(TS_DICT, PARAM_VALS, 'pw₁', reversed_metrics)
    assert list(fig.data[0].y) == HEATMAP_METRICS


# ---------------------------------------------------------------------------
# Edge case — single column (single param value)
# ---------------------------------------------------------------------------

def test_single_param_value():
    single = {PARAM_VALS[0]: _make_ts()}
    fig = plot_ts_heatmap(single, [PARAM_VALS[0]], 'pw₁', HEATMAP_METRICS)
    assert fig is not None
    assert len(fig.data[0].x) == 1
    # All colour values should be 0.0 (only one point → normalised to min = max)
    for row in fig.data[0].z:
        assert abs(row[0]) < 1e-9 or abs(row[0] - 0.0) < 1e-9


# ---------------------------------------------------------------------------
# Edge case — single metric
# ---------------------------------------------------------------------------

def test_single_metric():
    fig = plot_ts_heatmap(TS_DICT, PARAM_VALS, 'pw₁', ['H̄'])
    assert fig is not None
    assert len(fig.data[0].y) == 1
    assert fig.data[0].y[0] == 'H̄'


# ---------------------------------------------------------------------------
# HEATMAP_METRICS constant sanity
# ---------------------------------------------------------------------------

def test_heatmap_metrics_constant():
    assert HEATMAP_METRICS == ['S̄', 'H̄', 'P̄ᵐ']
    assert len(HEATMAP_METRICS) == 3
