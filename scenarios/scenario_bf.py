import streamlit as st
import numpy as np
from core.System import DynamicalSystem

from core.constants import DEFAULT_INIT_STATE, DEFAULT_PARAMS
from core.plots import plot_bifurcation, plot_poincare_maps, plot_time_series_with_economics, plot_time_series_heatmap, HEATMAP_METRICS
from ._status import scenario_header, status_indicator
from ._sys_params import system_parameters_ui


F_THRESHOLD_OPTIONS = [0.05, 0.25, 0.5, 0.75, 0.95]
ALPHA_OPTIONS = [0.0, 0.10, 0.15, 0.25, 0.40, 0.55, 0.70, 0.85, 1.00]


def _blast_params(alpha: float) -> dict:
    return {
        'q1': float(DEFAULT_PARAMS['q0'] + alpha * 0.33),
        'pw1': float(DEFAULT_PARAMS['pw0'] - alpha * 0.40),
        'c1': float(DEFAULT_PARAMS['c0'] - alpha * 0.80),
    }


@st.cache_data(show_spinner=False)
def blast_time_series(alpha: float, f_threshold: float, simulation_timesteps: int,
                      system_param_overrides: tuple = ()) -> dict:
    params = DEFAULT_PARAMS.copy()
    if system_param_overrides:
        params.update(dict(system_param_overrides))
    params.update(_blast_params(alpha))
    params['F_threshold'] = f_threshold
    state = {k: np.float128(v) for k, v in DEFAULT_INIT_STATE.items()}
    system = DynamicalSystem(params, state, "dimensionalized")
    time_series = system.generate_time_series(num_timesteps=simulation_timesteps)
    return {k: v.astype(np.float64) for k, v in time_series.items()}


@st.cache_data(show_spinner=False)
def blast_bifurcation(alpha_min: float, alpha_max: float, resolution: int,
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
        params.update(_blast_params(float(alpha)))
        params['F_threshold'] = f_threshold
        state = {k: np.float128(v) for k, v in DEFAULT_INIT_STATE.items()}
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
def blast_time_series_vs_f_threshold(alpha_held: float, f_threshold: float,
                                     simulation_timesteps: int,
                                     system_param_overrides: tuple = ()) -> dict:
    params = DEFAULT_PARAMS.copy()
    if system_param_overrides:
        params.update(dict(system_param_overrides))
    params.update(_blast_params(alpha_held))
    params['F_threshold'] = f_threshold
    state = {k: np.float128(v) for k, v in DEFAULT_INIT_STATE.items()}
    system = DynamicalSystem(params, state, "dimensionalized")
    time_series = system.generate_time_series(num_timesteps=simulation_timesteps)
    return {k: v.astype(np.float64) for k, v in time_series.items()}


@st.cache_data(show_spinner=False)
def blast_bifurcation_vs_f_threshold(alpha_held: float, f_threshold_min: float,
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
        params.update(_blast_params(alpha_held))
        params['F_threshold'] = float(f_threshold)
        state = {k: np.float128(v) for k, v in DEFAULT_INIT_STATE.items()}
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
def blast_baseline(simulation_timesteps: int, system_param_overrides: tuple = ()) -> dict:
    """Baseline: no fraud (F=0, FP=0), standard parameters, no blast (alpha=0)."""
    params = DEFAULT_PARAMS.copy()
    if system_param_overrides:
        params.update(dict(system_param_overrides))
    params.update(_blast_params(0.0))
    params.update({'F_threshold': 0.5})
    state = {k: np.float128(v) for k, v in DEFAULT_INIT_STATE.items()}
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
def blast_baseline_time_series(simulation_timesteps: int,
                               system_param_overrides: tuple = ()) -> dict:
    """Full time series at alpha=0, F=0, FP=0 for the fixed baseline column."""
    params = DEFAULT_PARAMS.copy()
    if system_param_overrides:
        params.update(dict(system_param_overrides))
    params.update(_blast_params(0.0))
    params.update({'F_threshold': 0.5})
    state = {k: np.float128(v) for k, v in DEFAULT_INIT_STATE.items()}
    state['F'] = np.float128(0.0)
    state['FP'] = np.float128(0.0)
    system = DynamicalSystem(params, state, "dimensionalized")
    time_series = system.generate_time_series(num_timesteps=simulation_timesteps)
    return {k: v.astype(np.float64) for k, v in time_series.items()}


def scenario_bf():
    status_slot = scenario_header("Scenario 1 — Blast / Cyanide Fishing")
    st.caption(
        "Destructive methods: q₁↑  pw₁↓  c₁↓↓ (cost drops much more than price). "
        "A single destruction intensity α ∈ [0, 1] jointly scales all three."
    )

    with st.expander("Analysis Parameters", expanded=False):
        col_vs_alpha, col_vs_f_threshold = st.columns(2, gap="large")

        with col_vs_alpha:
            st.markdown("#### vs α")
            st.markdown("**Time Series & Poincare**")
            simulation_timesteps_alpha = st.slider(
                "Time period", 100, 1000, 400, 50, key="bf_simA",
            )
            selected_alphas = st.multiselect(
                "α values", ALPHA_OPTIONS,
                default=[0.15, 0.40, 0.70, 1.00], key="bf_av",
            )
            f_threshold_for_alpha_sweep = st.selectbox(
                "F_threshold", F_THRESHOLD_OPTIONS,
                index=F_THRESHOLD_OPTIONS.index(0.5), key="bf_ftA",
            )
            st.markdown("**Bifurcation**")
            bifurcation_timesteps_alpha = st.slider(
                "Iteration length", 100, 1000, 300, 50, key="bf_bifA_iter",
            )
            bifurcation_resolution_alpha = st.slider(
                "Resolution", 50, 500, 200, 50, key="bf_resA",
            )
            alpha_range = st.slider(
                "α range", 0.0, 1.0, (0.0, 1.0), 0.05, key="bf_rng",
            )

        with col_vs_f_threshold:
            st.markdown("#### vs F_threshold")
            st.markdown("**Time Series & Poincare**")
            simulation_timesteps_ft = st.slider(
                "Time period", 100, 1000, 400, 50, key="bf_simB",
            )
            selected_f_thresholds = st.multiselect(
                "F_threshold values", F_THRESHOLD_OPTIONS,
                default=[0.25, 0.5, 0.75, 0.95], key="bf_ftv",
            )
            alpha_held = st.selectbox(
                "α (held)", ALPHA_OPTIONS,
                index=ALPHA_OPTIONS.index(0.55), key="bf_ahold",
            )
            st.markdown("**Bifurcation**")
            bifurcation_timesteps_ft = st.slider(
                "Iteration length", 100, 1000, 300, 50, key="bf_bifB_iter",
            )
            bifurcation_resolution_ft = st.slider(
                "Resolution", 50, 500, 200, 50, key="bf_resB",
            )
            f_threshold_range = st.slider(
                "F_threshold range", 0.0, 1.0, (0.1, 1.0), 0.05, key="bf_ftrng",
            )

    system_param_overrides = system_parameters_ui("bf", exclude={'q1', 'pw1', 'c1'})

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
        f'α={alpha}  (q₁={_blast_params(alpha)["q1"]:.2f}, '
        f'pw₁={_blast_params(alpha)["pw1"]:.2f}, '
        f'c₁={_blast_params(alpha)["c1"]:.2f})'
        for alpha in selected_alphas
    ]
    held_alpha_label = (
        f'α={alpha_held}  '
        f'(q₁={_blast_params(alpha_held)["q1"]:.2f}, '
        f'pw₁={_blast_params(alpha_held)["pw1"]:.2f}, '
        f'c₁={_blast_params(alpha_held)["c1"]:.2f})'
    )

    tab_time_series, tab_bifurcation, tab_poincare = st.tabs(
        ["Time Series", "Bifurcation", "Poincare"]
    )

    with tab_time_series:
        with status_indicator(status_slot, [
            "Running time-series simulations (α sweep)",
            "Running time-series simulations (F_threshold sweep)",
            "Computing baseline (no fraud, no blast)",
        ]):
            time_series_by_alpha = {
                alpha: blast_time_series(
                    float(alpha), float(f_threshold_for_alpha_sweep),
                    simulation_timesteps_alpha, system_param_overrides,
                )
                for alpha in selected_alphas
            }
            baseline_time_series = blast_baseline_time_series(
                simulation_timesteps_alpha, system_param_overrides,
            )
            time_axis_alpha = np.arange(simulation_timesteps_alpha + 1)
            time_series_by_f_threshold = {
                f_threshold: blast_time_series_vs_f_threshold(
                    float(alpha_held), float(f_threshold),
                    simulation_timesteps_ft, system_param_overrides,
                )
                for f_threshold in selected_f_thresholds
            }
            time_axis_ft = np.arange(simulation_timesteps_ft + 1)
            baseline_means = blast_baseline(
                simulation_timesteps_alpha, system_param_overrides,
            )

        heatmap_metrics = st.pills(
            "Heatmap rows", HEATMAP_METRICS, default=HEATMAP_METRICS,
            selection_mode="multi", key="bf_hm_m",
        )
        st.caption("Heatmap: % change vs. baseline (no fraud, no fraud perception, no blast)")
        subtab_vs_alpha, subtab_vs_f_threshold = st.tabs(["vs α", "vs F_threshold"])
        with subtab_vs_alpha:
            time_series_with_baseline = {'Baseline': baseline_time_series, **time_series_by_alpha}
            param_values_with_baseline = ['Baseline'] + selected_alphas
            column_labels = ['Baseline (α=0, F=FP=0)'] + alpha_column_labels
            fig = plot_time_series_with_economics(
                time_series_with_baseline, time_axis_alpha, param_values_with_baseline, 'α',
                f'Blast Fishing — Time Series by Destruction Intensity   '
                f'(F_threshold={f_threshold_for_alpha_sweep}  |  q₁↑  pw₁↓  c₁↓↓)',
            )
            for index, label in enumerate(column_labels):
                fig.layout.annotations[index].text = label
            st.plotly_chart(fig, width='stretch')
            if heatmap_metrics:
                heatmap_figs = plot_time_series_heatmap(
                    time_series_with_baseline, param_values_with_baseline, 'α',
                    heatmap_metrics, baseline_dict=baseline_means,
                )
                if heatmap_figs:
                    for heatmap_fig in heatmap_figs:
                        st.plotly_chart(heatmap_fig, width='stretch')
        with subtab_vs_f_threshold:
            baseline_time_series_ft = blast_baseline_time_series(
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
                f'Blast Fishing — Time Series as F_threshold Increases   '
                f'(held {held_alpha_label})',
            )
            for index, label in enumerate(f_threshold_column_labels):
                fig.layout.annotations[index].text = label
            st.plotly_chart(fig, width='stretch')
            if heatmap_metrics:
                heatmap_figs = plot_time_series_heatmap(
                    time_series_ft_with_baseline, f_threshold_values_with_baseline,
                    'F_threshold', heatmap_metrics, baseline_dict=baseline_means,
                )
                if heatmap_figs:
                    for heatmap_fig in heatmap_figs:
                        st.plotly_chart(heatmap_fig, width='stretch')

    with tab_bifurcation:
        with status_indicator(status_slot, [
            "Computing bifurcation diagram (α sweep)",
            "Computing bifurcation diagram (F_threshold sweep)",
        ]):
            bif_alpha, bif_seafood, bif_effort, bif_fraudsters, bif_perception = blast_bifurcation(
                float(alpha_range[0]), float(alpha_range[1]),
                bifurcation_resolution_alpha, bifurcation_timesteps_alpha, 0.6,
                float(f_threshold_for_alpha_sweep), system_param_overrides,
            )
            (
                bif_f_threshold, bif_seafood_ft, bif_effort_ft,
                bif_fraudsters_ft, bif_perception_ft,
            ) = blast_bifurcation_vs_f_threshold(
                float(alpha_held), float(f_threshold_range[0]), float(f_threshold_range[1]),
                bifurcation_resolution_ft, bifurcation_timesteps_ft, 0.6, system_param_overrides,
            )

        bif_subtab_alpha, bif_subtab_ft = st.tabs(["vs α", "vs F_threshold"])
        with bif_subtab_alpha:
            fig = plot_bifurcation(
                bif_alpha, bif_seafood, bif_effort, bif_fraudsters, bif_perception,
                xlabel='Destruction Intensity (α)',
                title='Bifurcation over α   '
                      f'(F_threshold={f_threshold_for_alpha_sweep}  |  α=0 → honest  |  α=1 → q₁=0.40, pw₁=0.60, c₁=0.10)',
            )
            st.plotly_chart(fig, width='stretch')
        with bif_subtab_ft:
            fig = plot_bifurcation(
                bif_f_threshold, bif_seafood_ft, bif_effort_ft,
                bif_fraudsters_ft, bif_perception_ft,
                xlabel='F_threshold',
                title=f'Bifurcation over F_threshold   (held {held_alpha_label})',
            )
            st.plotly_chart(fig, width='stretch')

    with tab_poincare:
        with status_indicator(status_slot, [
            "Running time-series simulations (α sweep)",
            "Running time-series simulations (F_threshold sweep)",
        ]):
            time_series_by_alpha = {
                alpha: blast_time_series(
                    float(alpha), float(f_threshold_for_alpha_sweep),
                    simulation_timesteps_alpha, system_param_overrides,
                )
                for alpha in selected_alphas
            }
            baseline_time_series = blast_baseline_time_series(
                simulation_timesteps_alpha, system_param_overrides,
            )
            time_series_by_f_threshold = {
                f_threshold: blast_time_series_vs_f_threshold(
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
            baseline_time_series_ft = blast_baseline_time_series(
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
