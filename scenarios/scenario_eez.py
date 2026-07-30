import streamlit as st
import numpy as np
import plotly.graph_objects as go
from core.System import DynamicalSystem

from core.constants import DEFAULT_PARAMS
from core.plots import plot_four_variable_time_series, plot_bifurcation, plot_poincare_maps, plot_time_series_heatmap, HEATMAP_METRICS
from core.metrics import build_heatmap_display_rows, market_price_params_from_overrides
from ._status import scenario_header, status_indicator
from ._sys_params import initial_state_from_overrides, system_parameters_ui
from ._continuation_ui import (
    bifurcation_parameter_ui,
    param_axis_label,
    run_attractor_bifurcation,
    run_continuation_branch,
    show_continuation_diagnostics,
)


F_THRESHOLD_OPTIONS = [0.05, 0.25, 0.5, 0.75, 0.95]
ALPHA_OPTIONS = [0.0, 0.10, 0.15, 0.25, 0.40, 0.55, 0.70, 0.85, 1.00]


def _eez_params(alpha: float) -> dict:
    return {
        'q1': float(DEFAULT_PARAMS['q0'] + alpha * 0.23),
        'c1': float(DEFAULT_PARAMS['c0'] + alpha * 1.10),
    }


@st.cache_data(show_spinner=False)
def eez_time_series(alpha: float, f_threshold: float, simulation_timesteps: int,
                    system_param_overrides: tuple = ()) -> dict:
    params = DEFAULT_PARAMS.copy()
    if system_param_overrides:
        params.update(dict(system_param_overrides))
    params.update(_eez_params(alpha))
    params['F_threshold'] = f_threshold
    initial_state = initial_state_from_overrides(system_param_overrides)
    state = {k: np.float128(v) for k, v in initial_state.items()}
    system = DynamicalSystem(params, state, "dimensionalized")
    time_series = system.generate_time_series(num_timesteps=simulation_timesteps)
    return {k: v.astype(np.float64) for k, v in time_series.items()}


@st.cache_data(show_spinner=False)
def eez_time_series_vs_f_threshold(alpha_held: float, f_threshold: float,
                                   simulation_timesteps: int,
                                   system_param_overrides: tuple = ()) -> dict:
    params = DEFAULT_PARAMS.copy()
    if system_param_overrides:
        params.update(dict(system_param_overrides))
    params.update(_eez_params(alpha_held))
    params['F_threshold'] = f_threshold
    initial_state = initial_state_from_overrides(system_param_overrides)
    state = {k: np.float128(v) for k, v in initial_state.items()}
    system = DynamicalSystem(params, state, "dimensionalized")
    time_series = system.generate_time_series(num_timesteps=simulation_timesteps)
    return {k: v.astype(np.float64) for k, v in time_series.items()}


@st.cache_data(show_spinner=False)
def eez_bifurcation(param_key: str, range_min: float, range_max: float,
                    resolution: int, bifurcation_timesteps: int,
                    burn_in_fraction: float, held_alpha: float,
                    held_f_threshold: float,
                    system_param_overrides: tuple = ()) -> tuple:
    """Attractor bifurcation sweep along an arbitrary parameter."""
    return run_attractor_bifurcation(
        param_key, (range_min, range_max), resolution, bifurcation_timesteps,
        burn_in_fraction, _eez_params, held_alpha, held_f_threshold,
        system_param_overrides,
    )


@st.cache_data(show_spinner=False)
def eez_continuation(param_key: str, range_min: float, range_max: float,
                     resolution: int, held_alpha: float,
                     held_f_threshold: float,
                     system_param_overrides: tuple = ()) -> dict:
    """Pseudo-arclength fixed-point continuation along an arbitrary parameter."""
    return run_continuation_branch(
        param_key, (range_min, range_max), resolution, _eez_params,
        held_alpha, held_f_threshold, system_param_overrides,
    )


@st.cache_data(show_spinner=False)
def eez_baseline(simulation_timesteps: int, system_param_overrides: tuple = ()) -> dict:
    """Baseline: no fraud (F=0, FP=0), standard parameters, no EEZ violation (alpha=0)."""
    params = DEFAULT_PARAMS.copy()
    if system_param_overrides:
        params.update(dict(system_param_overrides))
    params.update(_eez_params(0.0))
    params.update({'F_threshold': 0.5})
    initial_state = initial_state_from_overrides(system_param_overrides)
    state = {k: np.float128(v) for k, v in initial_state.items()}
    state['F'] = np.float128(0.0)
    state['FP'] = np.float128(0.0)
    system = DynamicalSystem(params, state, "dimensionalized")
    time_series = system.generate_time_series(num_timesteps=simulation_timesteps)
    burn_in_steps = int(simulation_timesteps * 0.6)
    return {
        'Seafood': float(np.mean(time_series['Seafood'][burn_in_steps:])),
        'Harvest': float(np.mean(time_series['Harvest'][burn_in_steps:])),
        'Market Price': float(np.mean(time_series['Market Price'][burn_in_steps:])),
    }


@st.cache_data(show_spinner=False)
def eez_baseline_time_series(simulation_timesteps: int,
                             system_param_overrides: tuple = ()) -> dict:
    """Full time series at alpha=0, F=0, FP=0 for the fixed baseline column."""
    params = DEFAULT_PARAMS.copy()
    if system_param_overrides:
        params.update(dict(system_param_overrides))
    params.update(_eez_params(0.0))
    params.update({'F_threshold': 0.5})
    initial_state = initial_state_from_overrides(system_param_overrides)
    state = {k: np.float128(v) for k, v in initial_state.items()}
    state['F'] = np.float128(0.0)
    state['FP'] = np.float128(0.0)
    system = DynamicalSystem(params, state, "dimensionalized")
    time_series = system.generate_time_series(num_timesteps=simulation_timesteps)
    return {k: v.astype(np.float64) for k, v in time_series.items()}


def scenario_eez():
    status_slot = scenario_header("Scenario 3 — Non-Enforcement of EEZ")
    st.caption(
        "Fishers access outside-EEZ waters: q₁↑ (more fish), c₁↑ (higher cost). "
        f"pw₁ stays at default ({DEFAULT_PARAMS['pw1']}). "
        "Focus parameter: EEZ violation intensity α ∈ [0, 1]."
    )

    with st.expander("Analysis Parameters", expanded=False):
        col_vs_alpha, col_vs_f_threshold = st.columns(2, gap="large")

        with col_vs_alpha:
            st.markdown("#### vs α")
            st.markdown("**Time Series & Poincare**")
            simulation_timesteps_alpha = st.slider(
                "Time period", 100, 1000, 400, 50, key="eez_simA",
            )
            selected_alphas = st.multiselect(
                "α values", ALPHA_OPTIONS,
                default=[0.15, 0.40, 0.70, 1.00], key="eez_av",
            )
            f_threshold_for_alpha_sweep = st.selectbox(
                "F_threshold", F_THRESHOLD_OPTIONS,
                index=F_THRESHOLD_OPTIONS.index(0.5), key="eez_ftA",
            )
            st.markdown("**Bifurcation & Continuation**")
            bifurcation_timesteps_alpha = st.slider(
                "Iteration length", 100, 1000, 300, 50, key="eez_bifA_iter",
            )
            bifurcation_resolution_alpha = st.slider(
                "Attractor resolution", 50, 500, 200, 50, key="eez_resA",
            )
            continuation_resolution_alpha = st.slider(
                "Continuation resolution", 40, 300, 100, 20, key="eez_contA",
            )
            bif_param_alpha, bif_range_alpha = bifurcation_parameter_ui(
                "eezA", default_key="alpha",
            )

        with col_vs_f_threshold:
            st.markdown("#### vs F_threshold")
            st.markdown("**Time Series & Poincare**")
            simulation_timesteps_ft = st.slider(
                "Time period", 100, 1000, 400, 50, key="eez_simB",
            )
            selected_f_thresholds = st.multiselect(
                "F_threshold values", F_THRESHOLD_OPTIONS,
                default=[0.25, 0.5, 0.75, 0.95], key="eez_ftv",
            )
            alpha_held = st.selectbox(
                "α (held)", ALPHA_OPTIONS,
                index=ALPHA_OPTIONS.index(0.55), key="eez_ahold",
            )
            st.markdown("**Bifurcation & Continuation**")
            bifurcation_timesteps_ft = st.slider(
                "Iteration length", 100, 1000, 300, 50, key="eez_bifB_iter",
            )
            bifurcation_resolution_ft = st.slider(
                "Attractor resolution", 50, 500, 200, 50, key="eez_resB",
            )
            continuation_resolution_ft = st.slider(
                "Continuation resolution", 40, 300, 100, 20, key="eez_contB",
            )
            bif_param_ft, bif_range_ft = bifurcation_parameter_ui(
                "eezB", default_key="F_threshold",
            )

    system_param_overrides = system_parameters_ui("eez", exclude={'q1', 'c1'})

    if not selected_alphas:
        st.warning("Select at least one *α* value.")
        return
    if not selected_f_thresholds:
        st.warning("Select at least one *F_threshold* value.")
        return

    selected_alphas = sorted(selected_alphas)
    selected_f_thresholds = sorted(selected_f_thresholds)
    burn_in_steps_alpha = int(simulation_timesteps_alpha * 0.6)
    burn_in_steps_ft = int(simulation_timesteps_ft * 0.6)

    alpha_column_labels = [
        f'α={alpha}  (q₁={_eez_params(alpha)["q1"]:.2f}, c₁={_eez_params(alpha)["c1"]:.2f})'
        for alpha in selected_alphas
    ]
    held_alpha_label = (
        f'α={alpha_held}  '
        f'(q₁={_eez_params(alpha_held)["q1"]:.2f}, '
        f'c₁={_eez_params(alpha_held)["c1"]:.2f})'
    )

    tab_time_series, tab_bifurcation, tab_poincare = st.tabs(
        ["Time Series", "Bifurcation", "Poincare"]
    )

    with tab_time_series:
        with status_indicator(status_slot, [
            "Running time-series simulations (α sweep)",
            "Running time-series simulations (F_threshold sweep)",
            "Computing baseline (no fraud, no EEZ violation)",
        ]):
            time_series_by_alpha = {
                alpha: eez_time_series(
                    float(alpha), float(f_threshold_for_alpha_sweep),
                    simulation_timesteps_alpha, system_param_overrides,
                )
                for alpha in selected_alphas
            }
            baseline_time_series = eez_baseline_time_series(
                simulation_timesteps_alpha, system_param_overrides,
            )
            time_axis_alpha = np.arange(simulation_timesteps_alpha + 1)
            time_series_by_f_threshold = {
                f_threshold: eez_time_series_vs_f_threshold(
                    float(alpha_held), float(f_threshold),
                    simulation_timesteps_ft, system_param_overrides,
                )
                for f_threshold in selected_f_thresholds
            }
            time_axis_ft = np.arange(simulation_timesteps_ft + 1)
            baseline_means = eez_baseline(
                simulation_timesteps_alpha, system_param_overrides,
            )

        heatmap_metrics = st.pills(
            "Heatmap rows", HEATMAP_METRICS, default=HEATMAP_METRICS,
            selection_mode="multi", key="eez_hm_m",
        )
        st.caption(
            "Heatmap: S̄/H̄/P̄ᵐ = % change vs baseline; "
            "φ_FP/φ_H = Shapley contribution as % of baseline P̄ᵐ "
            "(no fraud, no fraud perception, no EEZ violation)"
        )
        gamma_m, e_d, e_sm = market_price_params_from_overrides(system_param_overrides)
        subtab_vs_alpha, subtab_vs_f_threshold = st.tabs(["vs α", "vs F_threshold"])
        with subtab_vs_alpha:
            time_series_with_baseline = {'Baseline': baseline_time_series, **time_series_by_alpha}
            param_values_with_baseline = ['Baseline'] + selected_alphas
            column_labels = ['Baseline (α=0, F=FP=0)'] + alpha_column_labels
            fig = plot_four_variable_time_series(
                time_series_with_baseline, time_axis_alpha, param_values_with_baseline, 'α',
                f'EEZ Non-Enforcement — Time Series by Violation Intensity   '
                f'(F_threshold={f_threshold_for_alpha_sweep}  |  q₁↑  c₁↑  |  '
                f'pw₁={DEFAULT_PARAMS["pw1"]} default)',
            )
            for index, label in enumerate(column_labels):
                fig.layout.annotations[index].text = label
            st.plotly_chart(fig, width='stretch')
            if heatmap_metrics:
                percent_by_metric = build_heatmap_display_rows(
                    time_series_with_baseline, param_values_with_baseline,
                    gamma_m=gamma_m, e_d=e_d, e_sm=e_sm,
                    baseline_means=baseline_means,
                )
                heatmap_figs = plot_time_series_heatmap(
                    percent_by_metric, param_values_with_baseline, 'α',
                    heatmap_metrics,
                )
                if heatmap_figs:
                    for heatmap_fig in heatmap_figs:
                        st.plotly_chart(heatmap_fig, width='stretch')
        with subtab_vs_f_threshold:
            baseline_time_series_ft = eez_baseline_time_series(
                simulation_timesteps_ft, system_param_overrides,
            )
            time_series_ft_with_baseline = {
                'Baseline': baseline_time_series_ft, **time_series_by_f_threshold,
            }
            f_threshold_values_with_baseline = ['Baseline'] + selected_f_thresholds
            f_threshold_column_labels = (
                ['Baseline (F=FP=0)'] + [str(f_threshold) for f_threshold in selected_f_thresholds]
            )
            fig = plot_four_variable_time_series(
                time_series_ft_with_baseline, time_axis_ft,
                f_threshold_values_with_baseline, 'F_threshold',
                f'EEZ Non-Enforcement — Time Series as F_threshold Increases   '
                f'(held {held_alpha_label}  |  pw₁={DEFAULT_PARAMS["pw1"]} default)',
            )
            for index, label in enumerate(f_threshold_column_labels):
                fig.layout.annotations[index].text = label
            st.plotly_chart(fig, width='stretch')
            if heatmap_metrics:
                percent_by_metric = build_heatmap_display_rows(
                    time_series_ft_with_baseline, f_threshold_values_with_baseline,
                    gamma_m=gamma_m, e_d=e_d, e_sm=e_sm,
                    baseline_means=baseline_means,
                )
                heatmap_figs = plot_time_series_heatmap(
                    percent_by_metric, f_threshold_values_with_baseline,
                    'F_threshold', heatmap_metrics,
                )
                if heatmap_figs:
                    for heatmap_fig in heatmap_figs:
                        st.plotly_chart(heatmap_fig, width='stretch')

    with tab_bifurcation:
        with status_indicator(status_slot, [
            "Computing bifurcation diagram (panel A)",
            "Computing bifurcation diagram (panel B)",
            "Continuing fixed points (panel A)",
            "Continuing fixed points (panel B)",
        ]):
            bif_mu_a, bif_seafood, bif_effort, bif_fraudsters, bif_perception = eez_bifurcation(
                bif_param_alpha, float(bif_range_alpha[0]), float(bif_range_alpha[1]),
                bifurcation_resolution_alpha, bifurcation_timesteps_alpha, 0.6,
                float(alpha_held), float(f_threshold_for_alpha_sweep),
                system_param_overrides,
            )
            bif_mu_b, bif_seafood_ft, bif_effort_ft, bif_fraudsters_ft, bif_perception_ft = (
                eez_bifurcation(
                    bif_param_ft, float(bif_range_ft[0]), float(bif_range_ft[1]),
                    bifurcation_resolution_ft, bifurcation_timesteps_ft, 0.6,
                    float(alpha_held), float(f_threshold_for_alpha_sweep),
                    system_param_overrides,
                )
            )
            branch_a = eez_continuation(
                bif_param_alpha, float(bif_range_alpha[0]), float(bif_range_alpha[1]),
                continuation_resolution_alpha, float(alpha_held),
                float(f_threshold_for_alpha_sweep), system_param_overrides,
            )
            branch_b = eez_continuation(
                bif_param_ft, float(bif_range_ft[0]), float(bif_range_ft[1]),
                continuation_resolution_ft, float(alpha_held),
                float(f_threshold_for_alpha_sweep), system_param_overrides,
            )

        axis_label_a = param_axis_label(bif_param_alpha)
        axis_label_b = param_axis_label(bif_param_ft)
        bif_subtab_alpha, bif_subtab_ft = st.tabs(["vs α", "vs F_threshold"])
        with bif_subtab_alpha:
            fig = plot_bifurcation(
                bif_mu_a, bif_seafood, bif_effort, bif_fraudsters, bif_perception,
                xlabel=axis_label_a,
                title=(
                    f'Bifurcation over {axis_label_a}   '
                    f'(held F_threshold={f_threshold_for_alpha_sweep}, α={alpha_held} when not sweeping α)'
                ),
                continuation_branch=branch_a,
            )
            st.plotly_chart(fig, width='stretch')
            show_continuation_diagnostics(branch_a, axis_label_a)
        with bif_subtab_ft:
            fig = plot_bifurcation(
                bif_mu_b, bif_seafood_ft, bif_effort_ft,
                bif_fraudsters_ft, bif_perception_ft,
                xlabel=axis_label_b,
                title=f'Bifurcation over {axis_label_b}   (held {held_alpha_label})',
                continuation_branch=branch_b,
            )
            st.plotly_chart(fig, width='stretch')
            show_continuation_diagnostics(branch_b, axis_label_b)

    with tab_poincare:
        with status_indicator(status_slot, [
            "Running time-series simulations (α sweep)",
            "Running time-series simulations (F_threshold sweep)",
        ]):
            time_series_by_alpha = {
                alpha: eez_time_series(
                    float(alpha), float(f_threshold_for_alpha_sweep),
                    simulation_timesteps_alpha, system_param_overrides,
                )
                for alpha in selected_alphas
            }
            baseline_time_series = eez_baseline_time_series(
                simulation_timesteps_alpha, system_param_overrides,
            )
            time_series_by_f_threshold = {
                f_threshold: eez_time_series_vs_f_threshold(
                    float(alpha_held), float(f_threshold),
                    simulation_timesteps_ft, system_param_overrides,
                )
                for f_threshold in selected_f_thresholds
            }

        poincare_vs_alpha, poincare_vs_ft = st.tabs(["vs α", "vs F_threshold"])
        with poincare_vs_alpha:
            time_series_with_baseline = {'Baseline': baseline_time_series, **time_series_by_alpha}
            param_values_with_baseline = ['Baseline'] + selected_alphas
            column_labels = ['Baseline (α=0, F=FP=0)'] + alpha_column_labels
            fig = plot_poincare_maps(
                time_series_with_baseline, param_values_with_baseline, 'α', burn_in_steps_alpha,
            )
            for index, label in enumerate(column_labels):
                fig.layout.annotations[index].text = label
            st.plotly_chart(fig, width='stretch')
        with poincare_vs_ft:
            baseline_time_series_ft = eez_baseline_time_series(
                simulation_timesteps_ft, system_param_overrides,
            )
            time_series_ft_with_baseline = {
                'Baseline': baseline_time_series_ft, **time_series_by_f_threshold,
            }
            f_threshold_values_with_baseline = ['Baseline'] + selected_f_thresholds
            f_threshold_column_labels = (
                ['Baseline (F=FP=0)'] + [str(f_threshold) for f_threshold in selected_f_thresholds]
            )
            fig = plot_poincare_maps(
                time_series_ft_with_baseline, f_threshold_values_with_baseline,
                'F_threshold', burn_in_steps_ft,
            )
            for index, label in enumerate(f_threshold_column_labels):
                fig.layout.annotations[index].text = label
            st.plotly_chart(fig, width='stretch')
