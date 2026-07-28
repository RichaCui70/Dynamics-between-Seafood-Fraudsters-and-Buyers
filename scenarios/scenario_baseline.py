import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from core.System import DynamicalSystem

from core.constants import VAR_COLORS, DEFAULT_PARAMS
from ._status import scenario_header, status_indicator
from ._sys_params import system_parameters_ui

NO_FRAUD_INIT_STATE = {'S': 0.6, 'E': 0.3, 'F': 0.0, 'FP': 0.0}

@st.cache_data(show_spinner=False)
def baseline_time_series(growth_rate: float, simulation_timesteps: int,
                         system_param_overrides: tuple = ()) -> dict:
    params = DEFAULT_PARAMS.copy()
    if system_param_overrides:
        params.update(dict(system_param_overrides))
    params['r'] = growth_rate
    state = {k: np.float128(v) for k, v in NO_FRAUD_INIT_STATE.items()}
    system = DynamicalSystem(params, state, "dimensionalized")
    time_series = system.generate_time_series(num_timesteps=simulation_timesteps)
    return {k: v.astype(np.float64) for k, v in time_series.items()}


@st.cache_data(show_spinner=False)
def baseline_bifurcation(growth_rate_min: float, growth_rate_max: float, resolution: int,
                         bifurcation_timesteps: int, burn_in_fraction: float,
                         system_param_overrides: tuple = ()) -> tuple:
    growth_rate_values = np.linspace(growth_rate_min, growth_rate_max, resolution)
    burn_in_steps = int(bifurcation_timesteps * burn_in_fraction)
    bif_growth_rates, bif_seafood, bif_effort = [], [], []
    for growth_rate in growth_rate_values:
        params = DEFAULT_PARAMS.copy()
        if system_param_overrides:
            params.update(dict(system_param_overrides))
        params['r'] = float(growth_rate)
        state = {k: np.float128(v) for k, v in NO_FRAUD_INIT_STATE.items()}
        system = DynamicalSystem(params, state, "dimensionalized")
        time_series = system.generate_time_series(num_timesteps=bifurcation_timesteps)
        seafood_attractor = time_series['Seafood'][burn_in_steps:].astype(np.float64)
        effort_attractor = time_series['Effort'][burn_in_steps:].astype(np.float64)
        attractor_length = len(seafood_attractor)
        bif_growth_rates.extend([float(growth_rate)] * attractor_length)
        bif_seafood.extend(seafood_attractor.tolist())
        bif_effort.extend(effort_attractor.tolist())
    return np.array(bif_growth_rates), np.array(bif_seafood), np.array(bif_effort)


def scenario_baseline():
    status_slot = scenario_header("Baseline — Bioeconomic Model (No Fraud)")
    st.caption(
        "F = 0, FP = 0 throughout. The system reduces to Seafood (S) vs "
        "Effort (E) only. Focus parameter: intrinsic growth rate *r*."
    )

    with st.expander("Analysis Parameters", expanded=False):
        col_sim, col_resolution, col_range = st.columns(3)
        with col_sim:
            simulation_timesteps = st.slider(
                "Simulation length", 100, 1000, 300, 50, key="baseline_sim",
            )
        with col_resolution:
            bifurcation_resolution = st.slider(
                "Bifurcation resolution", 50, 500, 250, 50, key="baseline_res",
            )
        with col_range:
            growth_rate_range = st.slider(
                "r sweep range", 0.1, 6.0, (0.1, 4.0), 0.1, key="baseline_rng",
            )
        selected_growth_rates = st.multiselect(
            "r values for time series & poincare",
            [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.75, 4.0, 5.0],
            default=[0.5, 1.5, 2.5, 3.75],
            key="baseline_rv",
        )

    system_param_overrides = system_parameters_ui("baseline")

    if not selected_growth_rates:
        st.warning("Select at least one *r* value.")
        return

    selected_growth_rates = sorted(selected_growth_rates)
    num_growth_rates = len(selected_growth_rates)
    burn_in_steps = int(simulation_timesteps * 0.6)

    tab_time_series, tab_bifurcation, tab_poincare = st.tabs(
        ["Time Series", "Bifurcation", "Poincare"]
    )

    with tab_time_series:
        with status_indicator(status_slot, ["Running time-series simulations"]):
            time_series_by_r = {
                growth_rate: baseline_time_series(
                    float(growth_rate), simulation_timesteps, system_param_overrides,
                )
                for growth_rate in selected_growth_rates
            }
            time_axis = np.arange(simulation_timesteps + 1)

        fig = make_subplots(
            rows=2, cols=num_growth_rates,
            subplot_titles=[f'r = {growth_rate}' for growth_rate in selected_growth_rates]
                           + [''] * num_growth_rates,
            shared_xaxes=True, vertical_spacing=0.10, horizontal_spacing=0.05,
        )
        for col, growth_rate in enumerate(selected_growth_rates, 1):
            series = time_series_by_r[growth_rate]
            fig.add_trace(go.Scatter(
                x=time_axis, y=series['Seafood'], mode='lines',
                line=dict(color=VAR_COLORS['S'], width=1.5),
                name='Seafood (S)', legendgroup='S', showlegend=(col == 1),
            ), row=1, col=col)
            fig.add_trace(go.Scatter(
                x=time_axis, y=series['Harvest'], mode='lines',
                line=dict(color=VAR_COLORS['Harvest'], width=1.5),
                name='Harvest (H)', legendgroup='H', showlegend=(col == 1),
            ), row=1, col=col)
            fig.add_trace(go.Scatter(
                x=time_axis, y=series['Effort'], mode='lines',
                line=dict(color=VAR_COLORS['E'], width=1.5),
                name='Effort (E)', legendgroup='E', showlegend=(col == 1),
            ), row=2, col=col)
        fig.update_yaxes(title_text='S / H', row=1, col=1)
        fig.update_yaxes(title_text='Effort (E)', row=2, col=1)
        fig.update_yaxes(rangemode='tozero')
        fig.update_xaxes(title_text='Time', row=2)
        fig.update_layout(
            height=600,
            title_y=1.0,
            title_text='Baseline (No Fraud) — Time Series as r Increases',
            legend=dict(orientation='h', yanchor='bottom', y=1.06),
            margin=dict(t=80, b=40),
        )
        st.plotly_chart(fig, width='stretch')

    with tab_bifurcation:
        with status_indicator(status_slot, ["Computing bifurcation diagram"]):
            bif_growth_rates, bif_seafood, bif_effort = baseline_bifurcation(
                float(growth_rate_range[0]), float(growth_rate_range[1]),
                bifurcation_resolution, 300, 0.6, system_param_overrides,
            )

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Seafood S*', 'Effort E*'],
            horizontal_spacing=0.08,
        )
        fig.add_trace(go.Scattergl(
            x=bif_growth_rates, y=bif_seafood, mode='markers',
            marker=dict(color=VAR_COLORS['S'], size=2, opacity=0.4),
            showlegend=False,
        ), row=1, col=1)
        fig.add_trace(go.Scattergl(
            x=bif_growth_rates, y=bif_effort, mode='markers',
            marker=dict(color=VAR_COLORS['E'], size=2, opacity=0.4),
            showlegend=False,
        ), row=1, col=2)
        default_growth_rate = DEFAULT_PARAMS['r']
        fig.add_vline(
            x=default_growth_rate, line_dash='dash', line_color='gray',
            annotation_text=f'Default r = {default_growth_rate}',
            annotation_position='top right', row=1, col=1,
        )
        fig.add_vline(
            x=default_growth_rate, line_dash='dash', line_color='gray', row=1, col=2,
        )
        fig.update_xaxes(title_text='Intrinsic Growth Rate (r)')
        fig.update_layout(
            height=600,
            title_text='Bifurcation Diagram over r (No Fraud)',
            margin=dict(t=60, b=40),
        )
        st.plotly_chart(fig, width='stretch')

    with tab_poincare:
        with status_indicator(status_slot, ["Running time-series simulations"]):
            time_series_by_r = {
                growth_rate: baseline_time_series(
                    float(growth_rate), simulation_timesteps, system_param_overrides,
                )
                for growth_rate in selected_growth_rates
            }

        fig = make_subplots(
            rows=2, cols=num_growth_rates,
            subplot_titles=[f'r = {growth_rate}' for growth_rate in selected_growth_rates]
                           + [''] * num_growth_rates,
            vertical_spacing=0.12, horizontal_spacing=0.05,
        )
        for col, growth_rate in enumerate(selected_growth_rates, 1):
            series = time_series_by_r[growth_rate]
            for row, (var_name, color) in enumerate([
                ('Seafood', VAR_COLORS['S']), ('Effort', VAR_COLORS['E']),
            ], 1):
                values = series[var_name]
                values_t, values_t_plus_1 = values[burn_in_steps:-1], values[burn_in_steps + 1:]
                fig.add_trace(go.Scattergl(
                    x=values_t, y=values_t_plus_1, mode='markers',
                    marker=dict(color=color, size=2, opacity=0.6),
                    showlegend=False,
                ), row=row, col=col)
                axis_min = float(min(values_t.min(), values_t_plus_1.min())) * 0.9
                axis_max = float(max(values_t.max(), values_t_plus_1.max())) * 1.1
                fig.add_trace(go.Scatter(
                    x=[axis_min, axis_max], y=[axis_min, axis_max], mode='lines',
                    line=dict(color='black', width=0.8, dash='dash'),
                    showlegend=False,
                ), row=row, col=col)
        fig.update_yaxes(title_text='S(t+1)', row=1, col=1)
        fig.update_yaxes(title_text='E(t+1)', row=2, col=1)
        fig.update_layout(
            height=600,
            title_text='Poincare — x(t) vs x(t+1) (attractor only)',
            margin=dict(t=60, b=40),
        )
        st.plotly_chart(fig, width='stretch')
