"""Tests for pseudo-arclength fixed-point continuation."""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.System import DynamicalSystem
from core.constants import DEFAULT_PARAMS, DEFAULT_INIT_STATE
from core.continuation import continue_fixed_point


def _params_at_f_threshold(mu):
    """Continuation along F_threshold with all other params at defaults (α = 0)."""
    params = DEFAULT_PARAMS.copy()
    params['F_threshold'] = float(mu)
    return params


def test_continuation_seeds_and_returns_finite_branch():
    branch = continue_fixed_point(
        _params_at_f_threshold,
        (0.2, 0.8),
        initial_state_defaults=DEFAULT_INIT_STATE,
        num_points=40,
        warmup_steps=300,
    )
    assert branch['seed_found']
    assert len(branch['param_values']) >= 5
    for key in ('S', 'E', 'F', 'FP', 'spectral_radius'):
        assert np.all(np.isfinite(branch[key]))
    assert branch['stable'].dtype == bool


def test_continuation_param_values_within_range():
    range_min, range_max = 0.15, 0.85
    branch = continue_fixed_point(
        _params_at_f_threshold,
        (range_min, range_max),
        initial_state_defaults=DEFAULT_INIT_STATE,
        num_points=40,
        warmup_steps=300,
    )
    assert branch['seed_found']
    # Allow a tiny numerical overshoot; points should land on / inside the interval.
    assert float(np.min(branch['param_values'])) >= range_min - 1e-6
    assert float(np.max(branch['param_values'])) <= range_max + 1e-6


def test_continuation_residuals_small_at_sampled_points():
    branch = continue_fixed_point(
        _params_at_f_threshold,
        (0.25, 0.75),
        initial_state_defaults=DEFAULT_INIT_STATE,
        num_points=30,
        warmup_steps=300,
        tol=1e-10,
    )
    assert branch['seed_found']
    sample_indices = np.linspace(0, len(branch['param_values']) - 1, num=5, dtype=int)
    for index in sample_indices:
        mu = float(branch['param_values'][index])
        state = {
            'S': np.float128(branch['S'][index]),
            'E': np.float128(branch['E'][index]),
            'F': np.float128(branch['F'][index]),
            'FP': np.float128(branch['FP'][index]),
        }
        system = DynamicalSystem(_params_at_f_threshold(mu), state, "dimensionalized")
        mapped = system._evaluate_map_vector([
            float(state['S']), float(state['E']),
            float(state['F']), float(state['FP']),
        ])
        residual_norm = float(np.linalg.norm(
            mapped - np.array([
                float(state['S']), float(state['E']),
                float(state['F']), float(state['FP']),
            ])
        ))
        assert residual_norm < 1e-6, f"Residual {residual_norm} at μ={mu}"


def test_continuation_matches_independent_fixed_point():
    branch = continue_fixed_point(
        _params_at_f_threshold,
        (0.3, 0.7),
        initial_state_defaults=DEFAULT_INIT_STATE,
        num_points=40,
        warmup_steps=400,
    )
    assert branch['seed_found']

    for target_mu in (0.35, 0.55):
        # Nearest continuation sample to the spot-check parameter.
        nearest = int(np.argmin(np.abs(branch['param_values'] - target_mu)))
        cont_state = {
            'S': float(branch['S'][nearest]),
            'E': float(branch['E'][nearest]),
            'F': float(branch['F'][nearest]),
            'FP': float(branch['FP'][nearest]),
        }
        cont_mu = float(branch['param_values'][nearest])

        system = DynamicalSystem(
            _params_at_f_threshold(cont_mu),
            {k: np.float128(v) for k, v in DEFAULT_INIT_STATE.items()},
            "dimensionalized",
        )
        independent = system.find_fixed_point(
            initial_guess=cont_state, warmup_steps=1, tol=1e-10,
        )
        assert independent['converged']
        for key in ('S', 'E', 'F', 'FP'):
            assert independent['fixed_point'][key] == pytest.approx(
                cont_state[key], rel=1e-3, abs=1e-3,
            )
