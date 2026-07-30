"""Shared bifurcation-parameter UI and continuation runners for fraud scenarios."""

import numpy as np
import streamlit as st

from core.System import DynamicalSystem
from core.constants import DEFAULT_PARAMS
from core.continuation import continue_fixed_point
from ._sys_params import PARAM_GROUPS, initial_state_from_overrides

# Keys that are box constraints or initial conditions — not dynamical axes.
_EXCLUDED_PARAM_KEYS = frozenset({
    'F_min', 'F_max', 'FP_min', 'FP_max', 'S0', 'E0', 'F0', 'FP0',
})


def _build_continuation_param_bounds():
    """Flatten PARAM_GROUPS into a {key: (min, max, step, label)} lookup."""
    bounds = {
        'alpha': (0.0, 1.0, 0.05, 'α (scenario intensity)'),
    }
    for _group_name, params in PARAM_GROUPS:
        for key, label, slider_min, slider_max, slider_step in params:
            if key in _EXCLUDED_PARAM_KEYS:
                continue
            bounds[key] = (slider_min, slider_max, slider_step, label)
    return bounds


CONTINUATION_PARAM_BOUNDS = _build_continuation_param_bounds()


def bifurcation_parameter_ui(prefix: str, default_key: str):
    """
    Render a bifurcation-parameter dropdown and range slider.

    Args:
        prefix: Unique Streamlit widget-key prefix.
        default_key: Default parameter key (``'alpha'`` or ``'F_threshold'``).

    Returns:
        ``(param_key, (range_min, range_max))``.
    """
    option_keys = list(CONTINUATION_PARAM_BOUNDS.keys())
    ordered_keys = []
    if default_key in option_keys:
        ordered_keys.append(default_key)
    if 'alpha' in option_keys and 'alpha' not in ordered_keys:
        ordered_keys.append('alpha')
    for key in option_keys:
        if key not in ordered_keys:
            ordered_keys.append(key)

    labels = {
        key: CONTINUATION_PARAM_BOUNDS[key][3] for key in ordered_keys
    }
    param_key = st.selectbox(
        "Bifurcation parameter",
        ordered_keys,
        index=0,
        format_func=lambda key: labels[key],
        key=f"{prefix}_bif_param",
    )
    slider_min, slider_max, slider_step, _label = CONTINUATION_PARAM_BOUNDS[param_key]
    default_range = (float(slider_min), float(slider_max))
    if param_key == 'alpha':
        default_range = (0.0, 1.0)
    elif param_key == 'F_threshold':
        default_range = (0.1, 1.0)

    param_range = st.slider(
        f"{labels[param_key]} range",
        float(slider_min), float(slider_max),
        default_range, float(slider_step),
        key=f"{prefix}_bif_range",
    )
    return param_key, (float(param_range[0]), float(param_range[1]))


def resolve_params_at(
    param_key,
    alpha_setter,
    held_alpha,
    held_f_threshold,
    base_params,
):
    """
    Build a ``mu -> full params dict`` closure for continuation / bifurcation sweeps.

    Args:
        param_key: Which parameter is being swept (``'alpha'`` or a ``DEFAULT_PARAMS`` key).
        alpha_setter: Scenario callable ``alpha -> dict`` of composite overrides
            (e.g. ``_blast_params``).
        held_alpha: α held fixed when ``param_key != 'alpha'``.
        held_f_threshold: F̂ held fixed when ``param_key != 'F_threshold'``.
        base_params: Already-merged ``DEFAULT_PARAMS`` + UI overrides.

    Returns:
        Callable ``mu -> dict``.
    """
    def params_at(mu):
        params = dict(base_params)
        if param_key == 'alpha':
            params.update(alpha_setter(float(mu)))
            params['F_threshold'] = float(held_f_threshold)
        else:
            params.update(alpha_setter(float(held_alpha)))
            if param_key != 'F_threshold':
                params['F_threshold'] = float(held_f_threshold)
            params[param_key] = float(mu)
        return params

    return params_at


def param_axis_label(param_key: str) -> str:
    """Return a short axis label for the chosen bifurcation parameter."""
    if param_key == 'alpha':
        return 'α (scenario intensity)'
    if param_key in CONTINUATION_PARAM_BOUNDS:
        return CONTINUATION_PARAM_BOUNDS[param_key][3]
    return param_key


def _strip_starting_value_keys(params: dict) -> dict:
    """Remove S0/E0/F0/FP0 keys that are not DynamicalSystem parameters."""
    cleaned = dict(params)
    for state_key in ('S0', 'E0', 'F0', 'FP0'):
        cleaned.pop(state_key, None)
    return cleaned


def run_continuation_branch(
    param_key,
    param_range,
    resolution,
    alpha_setter,
    held_alpha,
    held_f_threshold,
    system_param_overrides=(),
):
    """
    Run pseudo-arclength continuation along ``param_key``.

    Args:
        param_key: Continuation / sweep parameter.
        param_range: ``(min, max)`` interval.
        resolution: Approximate number of accepted points per direction.
        alpha_setter: Scenario callable ``alpha -> dict``.
        held_alpha: α held when not sweeping α.
        held_f_threshold: F̂ held when not sweeping F̂.
        system_param_overrides: Sorted ``(key, val)`` tuple from the UI.

    Returns:
        Branch dict from ``continue_fixed_point``.
    """
    base_params = DEFAULT_PARAMS.copy()
    if system_param_overrides:
        base_params.update(dict(system_param_overrides))
    base_params = _strip_starting_value_keys(base_params)

    params_at = resolve_params_at(
        param_key, alpha_setter, held_alpha, held_f_threshold, base_params,
    )
    initial_state = initial_state_from_overrides(system_param_overrides)
    return continue_fixed_point(
        params_at,
        param_range,
        initial_state_defaults=initial_state,
        num_points=resolution,
        equation_form="dimensionalized",
    )


def run_attractor_bifurcation(
    param_key,
    param_range,
    resolution,
    bifurcation_timesteps,
    burn_in_fraction,
    alpha_setter,
    held_alpha,
    held_f_threshold,
    system_param_overrides=(),
):
    """
    Sweep an attractor diagram along ``param_key`` (orbit samples after burn-in).

    Args:
        param_key: Sweep parameter.
        param_range: ``(min, max)`` interval.
        resolution: Number of sweep grid points.
        bifurcation_timesteps: Orbit length per grid point.
        burn_in_fraction: Fraction of the orbit discarded as transient.
        alpha_setter: Scenario callable ``alpha -> dict``.
        held_alpha: α held when not sweeping α.
        held_f_threshold: F̂ held when not sweeping F̂.
        system_param_overrides: Sorted ``(key, val)`` tuple from the UI.

    Returns:
        Tuple of arrays ``(mu, S, E, F, FP)`` of attractor samples.
    """
    base_params = DEFAULT_PARAMS.copy()
    if system_param_overrides:
        base_params.update(dict(system_param_overrides))
    base_params = _strip_starting_value_keys(base_params)
    params_at = resolve_params_at(
        param_key, alpha_setter, held_alpha, held_f_threshold, base_params,
    )
    initial_state = initial_state_from_overrides(system_param_overrides)
    mu_values = np.linspace(float(param_range[0]), float(param_range[1]), resolution)
    burn_in_steps = int(bifurcation_timesteps * burn_in_fraction)

    bif_mu, bif_seafood, bif_effort, bif_fraudsters, bif_perception = [], [], [], [], []
    for mu in mu_values:
        params = params_at(float(mu))
        state = {key: np.float128(value) for key, value in initial_state.items()}
        system = DynamicalSystem(params, state, "dimensionalized")
        time_series = system.generate_time_series(num_timesteps=bifurcation_timesteps)
        seafood_attractor = time_series['Seafood'][burn_in_steps:].astype(np.float64)
        effort_attractor = time_series['Effort'][burn_in_steps:].astype(np.float64)
        fraudsters_attractor = time_series['Fraudsters'][burn_in_steps:].astype(np.float64)
        perception_attractor = (
            time_series['Perception of Fraud'][burn_in_steps:].astype(np.float64)
        )
        attractor_length = len(seafood_attractor)
        bif_mu.extend([float(mu)] * attractor_length)
        bif_seafood.extend(seafood_attractor.tolist())
        bif_effort.extend(effort_attractor.tolist())
        bif_fraudsters.extend(fraudsters_attractor.tolist())
        bif_perception.extend(perception_attractor.tolist())
    return (
        np.array(bif_mu), np.array(bif_seafood), np.array(bif_effort),
        np.array(bif_fraudsters), np.array(bif_perception),
    )


def show_continuation_diagnostics(branch, param_label: str):
    """Surface seed / early-termination warnings under a bifurcation figure."""
    if not branch.get('seed_found', False) or len(branch.get('param_values', [])) == 0:
        st.warning(
            f"Continuation could not seed a fixed point anywhere in the "
            f"requested {param_label} range."
        )
        return
    early = []
    if branch.get('forward_terminated_early'):
        early.append('forward')
    if branch.get('backward_terminated_early'):
        early.append('backward')
    if early:
        st.caption(
            f"Continuation terminated early in the {' / '.join(early)} direction(s) "
            f"before covering the full {param_label} range "
            f"(fold at a boundary, Newton failure, or step-size underflow)."
        )
