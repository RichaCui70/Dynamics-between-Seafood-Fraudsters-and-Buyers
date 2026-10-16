import streamlit as st
import numpy as np
from core.System import DynamicalSystem

from core.constants import DEFAULT_PARAMS
from core.plots import plot_bifurcation, plot_poincare_maps, plot_time_series_with_economics, plot_time_series_heatmap, HEATMAP_METRICS
from core.plots import (
    STORY_FRAUD_METRICS,
    STORY_OUTCOME_METRICS,
    plot_story_outcome_curves,
    plot_story_phase_portraits,
    plot_story_time_series_strip,
)
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
# Fixed categorical order: baseline, damaged reef, no substitutes, aware buyers
STORY_COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
STORY_TOTAL_TIMESTEPS = 2000
STORY_TRANSIENT_TIMESTEPS = 1000


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
    initial_state = initial_state_from_overrides(system_param_overrides)
    state = {k: np.longdouble(v) for k, v in initial_state.items()}
    system = DynamicalSystem(params, state, "dimensionalized")
    time_series = system.generate_time_series(num_timesteps=simulation_timesteps)
    return {k: v.astype(np.float64) for k, v in time_series.items()}


@st.cache_data(show_spinner=False)
def blast_time_series_vs_f_threshold(alpha_held: float, f_threshold: float,
                                     simulation_timesteps: int,
                                     system_param_overrides: tuple = ()) -> dict:
    params = DEFAULT_PARAMS.copy()
    if system_param_overrides:
        params.update(dict(system_param_overrides))
    params.update(_blast_params(alpha_held))
    params['F_threshold'] = f_threshold
    initial_state = initial_state_from_overrides(system_param_overrides)
    state = {k: np.longdouble(v) for k, v in initial_state.items()}
    system = DynamicalSystem(params, state, "dimensionalized")
    time_series = system.generate_time_series(num_timesteps=simulation_timesteps)
    return {k: v.astype(np.float64) for k, v in time_series.items()}


@st.cache_data(show_spinner=False)
def blast_bifurcation(param_key: str, range_min: float, range_max: float,
                      resolution: int, bifurcation_timesteps: int,
                      burn_in_fraction: float, held_alpha: float,
                      held_f_threshold: float,
                      system_param_overrides: tuple = ()) -> tuple:
    """Attractor bifurcation sweep along an arbitrary parameter."""
    return run_attractor_bifurcation(
        param_key, (range_min, range_max), resolution, bifurcation_timesteps,
        burn_in_fraction, _blast_params, held_alpha, held_f_threshold,
        system_param_overrides,
    )


@st.cache_data(show_spinner=False)
def blast_continuation(param_key: str, range_min: float, range_max: float,
                       resolution: int, held_alpha: float,
                       held_f_threshold: float,
                       system_param_overrides: tuple = ()) -> dict:
    """Pseudo-arclength fixed-point continuation along an arbitrary parameter."""
    return run_continuation_branch(
        param_key, (range_min, range_max), resolution, _blast_params,
        held_alpha, held_f_threshold, system_param_overrides,
    )


@st.cache_data(show_spinner=False)
def blast_baseline(simulation_timesteps: int, system_param_overrides: tuple = ()) -> dict:
    """Baseline: no fraud (F=0, FP=0), standard parameters, no blast (alpha=0)."""
    params = DEFAULT_PARAMS.copy()
    if system_param_overrides:
        params.update(dict(system_param_overrides))
    params.update(_blast_params(0.0))
    params.update({'F_threshold': 0.5})
    initial_state = initial_state_from_overrides(system_param_overrides)
    state = {k: np.longdouble(v) for k, v in initial_state.items()}
    state['F'] = np.longdouble(0.0)
    state['FP'] = np.longdouble(0.0)
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
    initial_state = initial_state_from_overrides(system_param_overrides)
    state = {k: np.longdouble(v) for k, v in initial_state.items()}
    state['F'] = np.longdouble(0.0)
    state['FP'] = np.longdouble(0.0)
    system = DynamicalSystem(params, state, "dimensionalized")
    time_series = system.generate_time_series(num_timesteps=simulation_timesteps)
    return {k: v.astype(np.float64) for k, v in time_series.items()}


def _story_overrides(system_param_overrides: tuple, **story_values) -> tuple:
    """System overrides with one story's parameter changes applied on top."""
    overrides = dict(system_param_overrides)
    overrides.update(story_values)
    return tuple(sorted(overrides.items()))


def _post_burn_mean(series, burn_in_steps: int) -> float:
    return float(np.mean(series[burn_in_steps:]))


@st.cache_data(show_spinner=False)
def blast_story_outcome_curves(alpha_values: tuple, f_threshold: float,
                               simulation_timesteps: int, burn_in_steps: int,
                               story_overrides: tuple) -> dict:
    """Post-transient outcomes at each α for one story.

    S̄, H̄, P̄ᵐ are ratios to the story's own no-fraud baseline. F̄ and F̄P̄ are
    ratios to the story's own α = 0 run, since the no-fraud baseline has
    F = FP = 0; a zero α = 0 mean gives NaN (a gap in the curve).
    """
    baseline_time_series = blast_baseline_time_series(simulation_timesteps, story_overrides)
    no_blast_time_series = blast_time_series(
        0.0, f_threshold, simulation_timesteps, story_overrides,
    )
    outcomes_by_metric = {
        metric: [] for metric, _ in STORY_OUTCOME_METRICS + STORY_FRAUD_METRICS
    }
    for alpha in alpha_values:
        time_series = blast_time_series(
            float(alpha), f_threshold, simulation_timesteps, story_overrides,
        )
        for metric, _ in STORY_OUTCOME_METRICS:
            outcomes_by_metric[metric].append(
                _post_burn_mean(time_series[metric], burn_in_steps)
                / _post_burn_mean(baseline_time_series[metric], burn_in_steps)
            )
        for metric, _ in STORY_FRAUD_METRICS:
            no_blast_mean = _post_burn_mean(no_blast_time_series[metric], burn_in_steps)
            outcomes_by_metric[metric].append(
                _post_burn_mean(time_series[metric], burn_in_steps) / no_blast_mean
                if no_blast_mean > 1e-9 else np.nan
            )
    return {metric: np.array(values) for metric, values in outcomes_by_metric.items()}


def _story_plots_tab(status_slot, system_param_overrides: tuple,
                     f_threshold_for_alpha_sweep: float):
    """Plots tab: the four Raja Ampat stories compared in three figures."""
    with st.expander("Story Parameters", expanded=False):
        st.caption(
            "Four Raja Ampat stories on top of the System Parameters below. "
            "Baseline uses the vs α F_threshold; each other story changes one parameter. "
            "Means (S̄, H̄, P̄ᵐ, F̄, F̄P̄) are taken after the transient."
        )
        col_low_r, col_low_e_d, col_low_f_threshold = st.columns(3, gap="large")
        with col_low_r:
            damaged_reef_r = st.number_input(
                "Damaged reef — r", 0.01, 1.0, 0.08, 0.005,
                format="%.3f", key="bf_story_r",
            )
        with col_low_e_d:
            no_substitutes_e_d = st.number_input(
                "No substitutes — ε_d", 0.0, 3.0, 0.0, 0.05,
                format="%.2f", key="bf_story_ed",
            )
        with col_low_f_threshold:
            aware_buyers_f_threshold = st.selectbox(
                "Aware buyers — F_threshold", F_THRESHOLD_OPTIONS,
                index=F_THRESHOLD_OPTIONS.index(0.05), key="bf_story_ft",
            )
        (col_total, col_transient, col_resolution,
         col_strip_alpha, col_phase_alphas) = st.columns(5, gap="large")
        with col_total:
            story_total_timesteps = st.slider(
                "Total time steps", 500, 5000, STORY_TOTAL_TIMESTEPS, 250,
                key="bf_story_total",
            )
        with col_transient:
            story_transient_timesteps = st.slider(
                "Transient (discarded)", 0, 4750, STORY_TRANSIENT_TIMESTEPS, 250,
                key="bf_story_transient",
            )
        with col_resolution:
            story_alpha_resolution = st.slider(
                "α resolution (outcome curves)", 11, 101, 51, 10, key="bf_story_res",
            )
        with col_strip_alpha:
            strip_alpha = st.selectbox(
                "α (time series strip)", ALPHA_OPTIONS,
                index=ALPHA_OPTIONS.index(0.55), key="bf_story_alpha",
            )
        with col_phase_alphas:
            phase_alphas = st.multiselect(
                "α values (phase portraits)", ALPHA_OPTIONS,
                default=[0.15, 0.55], key="bf_story_phase",
            )

    if story_transient_timesteps >= story_total_timesteps:
        st.warning("The transient must be shorter than the total number of time steps.")
        return

    system_values = dict(system_param_overrides)
    stories = [
        (
            f"Baseline  (r={system_values['r']:g}, ε_d={system_values['e_d']:g}, "
            f"F̂={f_threshold_for_alpha_sweep})",
            system_param_overrides, float(f_threshold_for_alpha_sweep),
        ),
        (
            f"Damaged reef  (r={damaged_reef_r:g})",
            _story_overrides(system_param_overrides, r=float(damaged_reef_r)),
            float(f_threshold_for_alpha_sweep),
        ),
        (
            f"No substitutes  (ε_d={no_substitutes_e_d:g})",
            _story_overrides(system_param_overrides, e_d=float(no_substitutes_e_d)),
            float(f_threshold_for_alpha_sweep),
        ),
        (
            f"Aware buyers  (F̂={aware_buyers_f_threshold})",
            system_param_overrides, float(aware_buyers_f_threshold),
        ),
    ]
    story_labels = [label for label, _, _ in stories]
    story_timesteps = int(story_total_timesteps)
    story_burn_in_steps = int(story_transient_timesteps)
    alpha_grid = tuple(float(alpha) for alpha in np.linspace(0.0, 1.0, story_alpha_resolution))

    with status_indicator(status_slot, [
        "Sweeping α for each story (outcome curves)",
        "Running story time series at the shared α",
        "Running trajectories for phase portraits",
    ]):
        outcomes_by_story = {
            label: blast_story_outcome_curves(
                alpha_grid, f_threshold, story_timesteps, story_burn_in_steps, overrides,
            )
            for label, overrides, f_threshold in stories
        }
        strip_time_series_by_story = {
            label: blast_time_series(
                float(strip_alpha), f_threshold, story_timesteps, overrides,
            )
            for label, overrides, f_threshold in stories
        }
        strip_baseline_by_story = {
            label: blast_baseline_time_series(story_timesteps, overrides)
            for label, overrides, _ in stories
        }
        phase_trajectories_by_alpha = {}
        phase_fraud_means_by_alpha = {}
        for alpha in sorted(phase_alphas):
            phase_trajectories_by_alpha[alpha] = {}
            phase_fraud_means_by_alpha[alpha] = {}
            for label, overrides, f_threshold in stories:
                time_series = blast_time_series(
                    float(alpha), f_threshold, story_timesteps, overrides,
                )
                phase_trajectories_by_alpha[alpha][label] = (
                    time_series['Seafood'][story_burn_in_steps:],
                    time_series['Effort'][story_burn_in_steps:],
                )
                phase_fraud_means_by_alpha[alpha][label] = (
                    _post_burn_mean(time_series['Fraudsters'], story_burn_in_steps),
                    _post_burn_mean(time_series['Perception of Fraud'], story_burn_in_steps),
                )
        no_fraud_time_series = blast_baseline_time_series(
            story_timesteps, system_param_overrides,
        )

    st.plotly_chart(
        plot_story_outcome_curves(
            np.array(alpha_grid), outcomes_by_story, story_labels, STORY_COLORS,
        ),
        width='stretch',
    )

    outcome_badges = {}
    for label in story_labels:
        strip_series = strip_time_series_by_story[label]
        badge_parts = []
        for (metric, _), symbol in zip(STORY_OUTCOME_METRICS, ['S̄', 'H̄', 'P̄ᵐ']):
            percent_change = (
                _post_burn_mean(strip_series[metric], story_burn_in_steps)
                / _post_burn_mean(strip_baseline_by_story[label][metric], story_burn_in_steps)
                - 1
            ) * 100
            badge_parts.append(f'{symbol} {percent_change:+.0f}%')
        mean_fraudsters = _post_burn_mean(strip_series['Fraudsters'], story_burn_in_steps)
        mean_perception = _post_burn_mean(strip_series['Perception of Fraud'], story_burn_in_steps)
        outcome_badges[label] = (
            '   '.join(badge_parts)
            + f'<br>F̄ {mean_fraudsters:.2f}   F̄P̄ {mean_perception:.2f}'
        )
    st.plotly_chart(
        plot_story_time_series_strip(
            strip_time_series_by_story, story_labels, STORY_COLORS,
            np.arange(story_timesteps + 1), story_burn_in_steps, outcome_badges,
            f'Same blast intensity (α = {strip_alpha}), four Raja Ampat stories   '
            f'(grey = {story_burn_in_steps}-step transient)',
        ),
        width='stretch',
    )

    if phase_trajectories_by_alpha:
        st.plotly_chart(
            plot_story_phase_portraits(
                phase_trajectories_by_alpha, phase_fraud_means_by_alpha,
                story_labels, STORY_COLORS,
                (float(no_fraud_time_series['Seafood'][-1]),
                 float(no_fraud_time_series['Effort'][-1])),
            ),
            width='stretch',
        )
    else:
        st.info("Select at least one α value for the phase portraits.")


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
            st.markdown("**Bifurcation & Continuation**")
            bifurcation_timesteps_alpha = st.slider(
                "Iteration length", 100, 1000, 300, 50, key="bf_bifA_iter",
            )
            bifurcation_resolution_alpha = st.slider(
                "Attractor resolution", 50, 500, 200, 50, key="bf_resA",
            )
            continuation_resolution_alpha = st.slider(
                "Continuation resolution", 40, 300, 100, 20, key="bf_contA",
            )
            bif_param_alpha, bif_range_alpha = bifurcation_parameter_ui(
                "bfA", default_key="alpha",
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
            st.markdown("**Bifurcation & Continuation**")
            bifurcation_timesteps_ft = st.slider(
                "Iteration length", 100, 1000, 300, 50, key="bf_bifB_iter",
            )
            bifurcation_resolution_ft = st.slider(
                "Attractor resolution", 50, 500, 200, 50, key="bf_resB",
            )
            continuation_resolution_ft = st.slider(
                "Continuation resolution", 40, 300, 100, 20, key="bf_contB",
            )
            bif_param_ft, bif_range_ft = bifurcation_parameter_ui(
                "bfB", default_key="F_threshold",
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

    tab_time_series, tab_bifurcation, tab_poincare, tab_plots = st.tabs(
        ["Time Series", "Bifurcation", "Poincare", "Plots"]
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
        st.caption(
            "Heatmap: S̄/H̄/P̄ᵐ = % change vs baseline; "
            "φ_FP/φ_H = Shapley contribution as % of baseline P̄ᵐ "
            "(no fraud, no fraud perception, no blast)"
        )
        gamma_m, e_d, e_sm = market_price_params_from_overrides(system_param_overrides)
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
            bif_mu_a, bif_seafood, bif_effort, bif_fraudsters, bif_perception = blast_bifurcation(
                bif_param_alpha, float(bif_range_alpha[0]), float(bif_range_alpha[1]),
                bifurcation_resolution_alpha, bifurcation_timesteps_alpha, 0.6,
                float(alpha_held), float(f_threshold_for_alpha_sweep),
                system_param_overrides,
            )
            bif_mu_b, bif_seafood_ft, bif_effort_ft, bif_fraudsters_ft, bif_perception_ft = (
                blast_bifurcation(
                    bif_param_ft, float(bif_range_ft[0]), float(bif_range_ft[1]),
                    bifurcation_resolution_ft, bifurcation_timesteps_ft, 0.6,
                    float(alpha_held), float(f_threshold_for_alpha_sweep),
                    system_param_overrides,
                )
            )
            branch_a = blast_continuation(
                bif_param_alpha, float(bif_range_alpha[0]), float(bif_range_alpha[1]),
                continuation_resolution_alpha, float(alpha_held),
                float(f_threshold_for_alpha_sweep), system_param_overrides,
            )
            branch_b = blast_continuation(
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

    with tab_plots:
        _story_plots_tab(status_slot, system_param_overrides, f_threshold_for_alpha_sweep)
