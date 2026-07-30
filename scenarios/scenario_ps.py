import streamlit as st
import numpy as np
import plotly.graph_objects as go
from core.System import DynamicalSystem

from core.constants import DEFAULT_PARAMS
from core.plots import plot_four_variable_time_series, plot_time_series_with_economics, plot_bifurcation, plot_poincare_maps, plot_time_series_heatmap, HEATMAP_METRICS
from core.metrics import build_heatmap_display_rows, market_price_params_from_overrides
from ._status import scenario_header, status_indicator
from ._sys_params import initial_state_from_overrides, system_parameters_ui


F_THRESHOLD_OPTIONS = [0.05, 0.25, 0.5, 0.75, 0.95]
ALPHA_OPTIONS = [0.0, 0.10, 0.15, 0.25, 0.40, 0.55, 0.70, 0.85, 1.00]

PRICE_PREMIUM_SCALE = 4.0


def _prized_params(alpha: float) -> dict:
    return {
        'pw1': float(DEFAULT_PARAMS['pw0'] + alpha * PRICE_PREMIUM_SCALE),
        'c1': DEFAULT_PARAMS['c0'],
        'q1': DEFAULT_PARAMS['q0'],
    }


@st.cache_data(show_spinner=False)
def prized_time_series(alpha: float, f_threshold: float, simulation_timesteps: int,
                       system_param_overrides: tuple = ()) -> dict:
    params = DEFAULT_PARAMS.copy()
    if system_param_overrides:
        params.update(dict(system_param_overrides))
    params.update(_prized_params(alpha))
    params['F_threshold'] = f_threshold
    initial_state = initial_state_from_overrides(system_param_overrides)
    state = {k: np.float128(v) for k, v in initial_state.items()}
    system = DynamicalSystem(params, state, "dimensionalized")
    time_series = system.generate_time_series(num_timesteps=simulation_timesteps)
    return {k: v.astype(np.float64) for k, v in time_series.items()}


@st.cache_data(show_spinner=False)
def prized_bifurcation(alpha_min: float, alpha_max: float, resolution: int,
                       bifurcation_timesteps: int, burn_in_fraction: float,
                       f_threshold: float,
                       system_param_overrides: tuple = ()) -> tuple:
    alpha_values = np.linspace(alpha_min, alpha_max, resolution)
    burn_in_steps = int(bifurcation_timesteps * burn_in_fraction)
    bif_alpha, bif_seafood, bif_effort, bif_fraudsters, bif_perception = [], [], [], [], []
    for alpha in alpha_values:
        params = DEFAULT_PARAMS.copy()
        if system_param_overrides:
            params.update(dict(system_param_overrides))
        params.update(_prized_params(float(alpha)))
        params['F_threshold'] = f_threshold
        initial_state = initial_state_from_overrides(system_param_overrides)
        state = {k: np.float128(v) for k, v in initial_state.items()}
        system = DynamicalSystem(params, state, "dimensionalized")
        time_series = system.generate_time_series(num_timesteps=bifurcation_timesteps)
        seafood_attractor = time_series['Seafood'][burn_in_steps:].astype(np.float64)
        effort_attractor = time_series['Effort'][burn_in_steps:].astype(np.float64)
        fraudsters_attractor = time_series['Fraudsters'][burn_in_steps:].astype(np.float64)
        perception_attractor = time_series['Perception of Fraud'][burn_in_steps:].astype(np.float64)
        attractor_length = len(seafood_attractor)
        bif_alpha.extend([float(alpha)] * attractor_length)
        bif_seafood.extend(seafood_attractor.tolist())
        bif_effort.extend(effort_attractor.tolist())
        bif_fraudsters.extend(fraudsters_attractor.tolist())
        bif_perception.extend(perception_attractor.tolist())
    return (
        np.array(bif_alpha), np.array(bif_seafood), np.array(bif_effort),
        np.array(bif_fraudsters), np.array(bif_perception),
    )


@st.cache_data(show_spinner=False)
def prized_time_series_vs_f_threshold(alpha_held: float, f_threshold: float,
                                      simulation_timesteps: int,
                                      system_param_overrides: tuple = ()) -> dict:
    params = DEFAULT_PARAMS.copy()
    if system_param_overrides:
        params.update(dict(system_param_overrides))
    params.update(_prized_params(alpha_held))
    params['F_threshold'] = f_threshold
    initial_state = initial_state_from_overrides(system_param_overrides)
    state = {k: np.float128(v) for k, v in initial_state.items()}
    system = DynamicalSystem(params, state, "dimensionalized")
    time_series = system.generate_time_series(num_timesteps=simulation_timesteps)
    return {k: v.astype(np.float64) for k, v in time_series.items()}


@st.cache_data(show_spinner=False)
def prized_bifurcation_vs_f_threshold(alpha_held: float, f_threshold_min: float,
                                      f_threshold_max: float, resolution: int,
                                      bifurcation_timesteps: int, burn_in_fraction: float,
                                      system_param_overrides: tuple = ()) -> tuple:
    f_threshold_values = np.linspace(f_threshold_min, f_threshold_max, resolution)
    burn_in_steps = int(bifurcation_timesteps * burn_in_fraction)
    bif_f_threshold, bif_seafood, bif_effort, bif_fraudsters, bif_perception = [], [], [], [], []
    for f_threshold in f_threshold_values:
        params = DEFAULT_PARAMS.copy()
        if system_param_overrides:
            params.update(dict(system_param_overrides))
        params.update({
            **_prized_params(alpha_held), 'F_threshold': float(f_threshold),
        })
        initial_state = initial_state_from_overrides(system_param_overrides)
        state = {k: np.float128(v) for k, v in initial_state.items()}
        system = DynamicalSystem(params, state, "dimensionalized")
        time_series = system.generate_time_series(num_timesteps=bifurcation_timesteps)
        seafood_attractor = time_series['Seafood'][burn_in_steps:].astype(np.float64)
        effort_attractor = time_series['Effort'][burn_in_steps:].astype(np.float64)
        fraudsters_attractor = time_series['Fraudsters'][burn_in_steps:].astype(np.float64)
        perception_attractor = time_series['Perception of Fraud'][burn_in_steps:].astype(np.float64)
        attractor_length = len(seafood_attractor)
        bif_f_threshold.extend([float(f_threshold)] * attractor_length)
        bif_seafood.extend(seafood_attractor.tolist())
        bif_effort.extend(effort_attractor.tolist())
        bif_fraudsters.extend(fraudsters_attractor.tolist())
        bif_perception.extend(perception_attractor.tolist())
    return (
        np.array(bif_f_threshold), np.array(bif_seafood), np.array(bif_effort),
        np.array(bif_fraudsters), np.array(bif_perception),
    )


@st.cache_data(show_spinner=False)
def prized_spectral_sweep(alpha_min: float, alpha_max: float, resolution: int,
                          system_param_overrides: tuple = ()) -> tuple:
    alpha_values = np.linspace(alpha_min, alpha_max, resolution)
    spectral_radii = np.empty(resolution)
    for index, alpha in enumerate(alpha_values):
        params = DEFAULT_PARAMS.copy()
        if system_param_overrides:
            params.update(dict(system_param_overrides))
        params.update(_prized_params(float(alpha)))
        initial_state = initial_state_from_overrides(system_param_overrides)
        state = {k: np.float128(v) for k, v in initial_state.items()}
        system = DynamicalSystem(params, state, "dimensionalized")
        result = system.stability_analysis()
        spectral_radii[index] = result['spectral_radius']
    return alpha_values.astype(np.float64), spectral_radii.astype(np.float64)


@st.cache_data(show_spinner=False)
def prized_baseline(simulation_timesteps: int, system_param_overrides: tuple = ()) -> dict:
    """Baseline: no fraud (F=0, FP=0), standard parameters, no premium (alpha=0)."""
    params = DEFAULT_PARAMS.copy()
    if system_param_overrides:
        params.update(dict(system_param_overrides))
    params.update(_prized_params(0.0))
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
def prized_baseline_time_series(simulation_timesteps: int,
                                system_param_overrides: tuple = ()) -> dict:
    """Full time series at alpha=0, F=0, FP=0 for the fixed baseline column."""
    params = DEFAULT_PARAMS.copy()
    if system_param_overrides:
        params.update(dict(system_param_overrides))
    params.update(_prized_params(0.0))
    params.update({'F_threshold': 0.5})
    initial_state = initial_state_from_overrides(system_param_overrides)
    state = {k: np.float128(v) for k, v in initial_state.items()}
    state['F'] = np.float128(0.0)
    state['FP'] = np.float128(0.0)
    system = DynamicalSystem(params, state, "dimensionalized")
    time_series = system.generate_time_series(num_timesteps=simulation_timesteps)
    return {k: v.astype(np.float64) for k, v in time_series.items()}


def scenario_ps():
    status_slot = scenario_header("Scenario 2 — Prized / Protected Seafood")
    st.caption(
        f"α drives a price premium for protected species: pw₁ = pw₀ + α·{PRICE_PREMIUM_SCALE:.0f}. "
        f"Same gear: c₁ = c₀, q₁ = q₀. Focus parameter: α."
    )

    with st.expander("Analysis Parameters", expanded=False):
        col_vs_alpha, col_vs_f_threshold = st.columns(2, gap="large")

        with col_vs_alpha:
            st.markdown("#### vs α")
            st.markdown("**Time Series & Poincare**")
            simulation_timesteps_alpha = st.slider(
                "Time period", 100, 1000, 400, 50, key="ps_simA",
            )
            selected_alphas = st.multiselect(
                "α values", ALPHA_OPTIONS,
                default=[0.15, 0.40, 0.70, 1.00], key="ps_a",
            )
            f_threshold_for_alpha_sweep = st.selectbox(
                "F_threshold", F_THRESHOLD_OPTIONS,
                index=F_THRESHOLD_OPTIONS.index(0.5), key="ps_ftA",
            )
            st.markdown("**Bifurcation**")
            bifurcation_timesteps_alpha = st.slider(
                "Iteration length", 100, 1000, 300, 50, key="ps_bifA_iter",
            )
            bifurcation_resolution_alpha = st.slider(
                "Resolution", 50, 500, 200, 50, key="ps_resA",
            )
            alpha_range = st.slider(
                "α range", 0.0, 1.0, (0.0, 1.0), 0.05,
                key="ps_rng",
            )

        with col_vs_f_threshold:
            st.markdown("#### vs F_threshold")
            st.markdown("**Time Series & Poincare**")
            simulation_timesteps_ft = st.slider(
                "Time period", 100, 1000, 400, 50, key="ps_simB",
            )
            selected_f_thresholds = st.multiselect(
                "F_threshold values", F_THRESHOLD_OPTIONS,
                default=[0.25, 0.5, 0.75, 0.95], key="ps_ftv",
            )
            alpha_held = st.selectbox(
                "α (held)", ALPHA_OPTIONS,
                index=ALPHA_OPTIONS.index(0.40), key="ps_a_hold",
            )
            st.markdown("**Bifurcation**")
            bifurcation_timesteps_ft = st.slider(
                "Iteration length", 100, 1000, 300, 50, key="ps_bifB_iter",
            )
            bifurcation_resolution_ft = st.slider(
                "Resolution", 50, 500, 200, 50, key="ps_resB",
            )
            f_threshold_range = st.slider(
                "F_threshold range", 0.0, 1.0, (0.1, 1.0), 0.05, key="ps_ftrng",
            )

    system_param_overrides = system_parameters_ui("ps", exclude={'pw1'})

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

    tab_time_series, tab_bifurcation, tab_poincare, tab_stability = st.tabs(
        ["Time Series", "Bifurcation", "Poincare", "Stability"]
    )

    with tab_time_series:
        with status_indicator(status_slot, [
            "Running time-series simulations (α sweep)",
            "Running time-series simulations (F_threshold sweep)",
            "Computing baseline (no fraud)",
        ]):
            time_series_by_alpha = {
                alpha: prized_time_series(
                    float(alpha), float(f_threshold_for_alpha_sweep),
                    simulation_timesteps_alpha, system_param_overrides,
                )
                for alpha in selected_alphas
            }
            baseline_time_series = prized_baseline_time_series(
                simulation_timesteps_alpha, system_param_overrides,
            )
            time_axis_alpha = np.arange(simulation_timesteps_alpha + 1)
            time_series_by_f_threshold = {
                f_threshold: prized_time_series_vs_f_threshold(
                    float(alpha_held), float(f_threshold),
                    simulation_timesteps_ft, system_param_overrides,
                )
                for f_threshold in selected_f_thresholds
            }
            time_axis_ft = np.arange(simulation_timesteps_ft + 1)
            baseline_means = prized_baseline(
                simulation_timesteps_alpha, system_param_overrides,
            )

        alpha_column_labels = [
            f'α={alpha}  (pw₁={_prized_params(alpha)["pw1"]:.2f})'
            for alpha in selected_alphas
        ]
        held_alpha_label = (
            f'α={alpha_held}  '
            f'(pw₁={_prized_params(alpha_held)["pw1"]:.2f})'
        )

        heatmap_metrics = st.pills(
            "Heatmap rows", HEATMAP_METRICS, default=HEATMAP_METRICS,
            selection_mode="multi", key="ps_hm_m",
        )
        st.caption(
            "Heatmap: S̄/H̄/P̄ᵐ = % change vs baseline; "
            "φ_FP/φ_H = Shapley contribution as % of baseline P̄ᵐ "
            "(no fraud, no fraud perception)"
        )
        gamma_m, e_d, e_sm = market_price_params_from_overrides(system_param_overrides)
        subtab_vs_alpha, subtab_vs_f_threshold = st.tabs(["vs α", "vs F_threshold"])
        with subtab_vs_alpha:
            time_series_with_baseline = {'Baseline': baseline_time_series, **time_series_by_alpha}
            param_values_with_baseline = ['Baseline'] + selected_alphas
            column_labels = ['Baseline (α=0, F=FP=0)'] + alpha_column_labels
            fig = plot_time_series_with_economics(
                time_series_with_baseline, time_axis_alpha, param_values_with_baseline, 'α',
                f'Prized Seafood — Time Series as α Increases   '
                f'(c₁=c₀={DEFAULT_PARAMS["c0"]},  q₁=q₀={DEFAULT_PARAMS["q0"]},  F_threshold={f_threshold_for_alpha_sweep})',
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
            baseline_time_series_ft = prized_baseline_time_series(
                simulation_timesteps_ft, system_param_overrides,
            )
            time_series_ft_with_baseline = {
                'Baseline': baseline_time_series_ft, **time_series_by_f_threshold,
            }
            f_threshold_values_with_baseline = ['Baseline'] + selected_f_thresholds
            f_threshold_column_labels = (
                ['Baseline (F=FP=0)'] + [str(f_threshold) for f_threshold in selected_f_thresholds]
            )
            fig = plot_time_series_with_economics(
                time_series_ft_with_baseline, time_axis_ft,
                f_threshold_values_with_baseline, 'F_threshold',
                f'Prized Seafood — Time Series as F_threshold Increases   '
                f'(held {held_alpha_label},  c₁=c₀={DEFAULT_PARAMS["c0"]},  q₁=q₀={DEFAULT_PARAMS["q0"]})',
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
            "Computing bifurcation diagram (α sweep)",
            "Computing bifurcation diagram (F_threshold sweep)",
        ]):
            bif_alpha, bif_seafood, bif_effort, bif_fraudsters, bif_perception = prized_bifurcation(
                float(alpha_range[0]), float(alpha_range[1]),
                bifurcation_resolution_alpha, bifurcation_timesteps_alpha, 0.6,
                float(f_threshold_for_alpha_sweep), system_param_overrides,
            )
            (
                bif_f_threshold, bif_seafood_ft, bif_effort_ft,
                bif_fraudsters_ft, bif_perception_ft,
            ) = prized_bifurcation_vs_f_threshold(
                float(alpha_held), float(f_threshold_range[0]), float(f_threshold_range[1]),
                bifurcation_resolution_ft, bifurcation_timesteps_ft, 0.6, system_param_overrides,
            )

        bif_subtab_alpha, bif_subtab_ft = st.tabs(["vs α", "vs F_threshold"])
        with bif_subtab_alpha:
            fig = plot_bifurcation(
                bif_alpha, bif_seafood, bif_effort, bif_fraudsters, bif_perception,
                xlabel='α (price premium intensity)',
                title=f'Bifurcation Diagram over α   (F_threshold={f_threshold_for_alpha_sweep})',
                vline_x=0.0, vline_label='α = 0 (no premium)',
            )
            st.plotly_chart(fig, width='stretch')
        with bif_subtab_ft:
            fig = plot_bifurcation(
                bif_f_threshold, bif_seafood_ft, bif_effort_ft,
                bif_fraudsters_ft, bif_perception_ft,
                xlabel='F_threshold',
                title=f'Bifurcation Diagram over F_threshold   (held {held_alpha_label})',
            )
            st.plotly_chart(fig, width='stretch')

    with tab_poincare:
        with status_indicator(status_slot, [
            "Running time-series simulations (α sweep)",
            "Running time-series simulations (F_threshold sweep)",
        ]):
            time_series_by_alpha = {
                alpha: prized_time_series(
                    float(alpha), float(f_threshold_for_alpha_sweep),
                    simulation_timesteps_alpha, system_param_overrides,
                )
                for alpha in selected_alphas
            }
            baseline_time_series = prized_baseline_time_series(
                simulation_timesteps_alpha, system_param_overrides,
            )
            time_series_by_f_threshold = {
                f_threshold: prized_time_series_vs_f_threshold(
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
            baseline_time_series_ft = prized_baseline_time_series(
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

    with tab_stability:
        with status_indicator(status_slot, ["Computing stability sweep"]):
            alpha_sweep, spectral_radii = prized_spectral_sweep(0.0, 1.0, 100, system_param_overrides)

        finite_mask = np.isfinite(spectral_radii)
        alpha_finite, spectral_radii_finite = alpha_sweep[finite_mask], spectral_radii[finite_mask]
        stable_mask = spectral_radii_finite < 1.0
        y_axis_cap = (
            max(float(np.max(spectral_radii_finite[spectral_radii_finite < 50])) * 1.1, 2.0)
            if np.any(spectral_radii_finite < 50) else 5.0
        )
        spectral_radii_plot = np.clip(spectral_radii_finite, 0, y_axis_cap)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=alpha_finite[stable_mask], y=spectral_radii_plot[stable_mask],
            mode='markers', marker=dict(color='#2E8B57', size=6),
            name='Stable (ρ < 1)',
        ))
        fig.add_trace(go.Scatter(
            x=alpha_finite[~stable_mask], y=spectral_radii_plot[~stable_mask],
            mode='markers', marker=dict(color='#DC143C', size=6),
            name='Unstable (ρ ≥ 1)',
        ))
        fig.add_hline(y=1.0, line_dash='dash', line_color='gray',
                      annotation_text='ρ = 1 (stability boundary)')
        fig.add_vline(
            x=0.0, line_dash='dot', line_color='steelblue',
            annotation_text='α = 0 (no premium)',
            annotation_position='top right',
        )
        fig.update_layout(
            height=600,
            title_text=(
                f'Spectral Radius vs α — Fixed-Point Stability   '
                f'(c₁=c₀={DEFAULT_PARAMS["c0"]},  q₁=q₀={DEFAULT_PARAMS["q0"]},  F_threshold={DEFAULT_PARAMS["F_threshold"]})'
            ),
            xaxis_title='Destruction Intensity (α)',
            yaxis_title='Spectral Radius  ρ = max|λᵢ|',
            yaxis_range=[0, y_axis_cap],
            margin=dict(t=60, b=40),
            legend=dict(yanchor='top', y=0.99, xanchor='right', x=0.99),
        )
        st.plotly_chart(fig, width='stretch')
