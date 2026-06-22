import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from System import DynamicalSystem, DEFAULT_PARAMS

from .constants import S1_COLORS, S1_NO_FRAUD
from ._status import scenario_header, status_indicator
from ._sys_params import sys_params_ui


@st.cache_data(show_spinner=False)
def s1_time_series(r_val: float, sim_time: int, sys_params: tuple = ()) -> dict:
    p = DEFAULT_PARAMS.copy()
    if sys_params:
        p.update(dict(sys_params))
    p['r'] = r_val
    state = {k: np.float128(v) for k, v in S1_NO_FRAUD.items()}
    sys = DynamicalSystem(p, state, "dimensionalized")
    ts = sys.time_series_plot(time=sim_time)
    return {k: v.astype(np.float64) for k, v in ts.items()}


@st.cache_data(show_spinner=False)
def s1_bifurcation(r_min: float, r_max: float, resolution: int,
                   bif_time: int, burn_frac: float,
                   sys_params: tuple = ()) -> tuple:
    r_sweep = np.linspace(r_min, r_max, resolution)
    burn = int(bif_time * burn_frac)
    br_r, br_S, br_E = [], [], []
    for rv in r_sweep:
        p = DEFAULT_PARAMS.copy()
        if sys_params:
            p.update(dict(sys_params))
        p['r'] = float(rv)
        state = {k: np.float128(v) for k, v in S1_NO_FRAUD.items()}
        sys = DynamicalSystem(p, state, "dimensionalized")
        ts = sys.time_series_plot(time=bif_time)
        s_att = ts['Seafood'][burn:].astype(np.float64)
        e_att = ts['Effort'][burn:].astype(np.float64)
        n = len(s_att)
        br_r.extend([float(rv)] * n)
        br_S.extend(s_att.tolist())
        br_E.extend(e_att.tolist())
    return np.array(br_r), np.array(br_S), np.array(br_E)


def scenario_1():
    status_slot = scenario_header("Scenario 1 — Baseline Bioeconomic Model (No Fraud)")
    st.caption(
        "F = 0, FP = 0 throughout. The system reduces to Seafood (S) vs "
        "Effort (E) only. Focus parameter: intrinsic growth rate *r*."
    )

    with st.expander("Analysis Parameters", expanded=False):
        _c1, _c2, _c3 = st.columns(3)
        with _c1:
            s1_sim = st.slider("Simulation length", 100, 1000, 300, 50, key="s1_sim")
        with _c2:
            s1_res = st.slider("Bifurcation resolution", 50, 500, 250, 50, key="s1_res")
        with _c3:
            s1_rng = st.slider("r sweep range", 0.1, 6.0, (0.1, 4.0), 0.1, key="s1_rng")
        s1_r_vals = st.multiselect(
            "r values for time series & poincare",
            [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.75, 4.0, 5.0],
            default=[0.5, 1.5, 2.5, 3.75],
            key="s1_rv",
        )

    sys_t = sys_params_ui("s1")

    if not s1_r_vals:
        st.warning("Select at least one *r* value.")
        return

    s1_r_vals = sorted(s1_r_vals)
    _N = len(s1_r_vals)
    _burn = int(s1_sim * 0.6)

    tab_ts, tab_bif, tab_rm = st.tabs(
        ["Time Series", "Bifurcation", "Poincare"]
    )

    with tab_ts:
        with status_indicator(status_slot, ["Running time-series simulations"]):
            ts1 = {rv: s1_time_series(float(rv), s1_sim, sys_t) for rv in s1_r_vals}
            t1 = np.arange(s1_sim + 1)

        fig = make_subplots(
            rows=2, cols=_N,
            subplot_titles=[f'r = {rv}' for rv in s1_r_vals] + [''] * _N,
            shared_xaxes=True, vertical_spacing=0.10, horizontal_spacing=0.05,
        )
        for col, rv in enumerate(s1_r_vals, 1):
            d = ts1[rv]
            fig.add_trace(go.Scatter(
                x=t1, y=d['Seafood'], mode='lines',
                line=dict(color=S1_COLORS['S'], width=1.5),
                name='Seafood (S)', legendgroup='S', showlegend=(col == 1),
            ), row=1, col=col)
            fig.add_trace(go.Scatter(
                x=t1, y=d['Effort'], mode='lines',
                line=dict(color=S1_COLORS['E'], width=1.5),
                name='Effort (E)', legendgroup='E', showlegend=(col == 1),
            ), row=2, col=col)
        fig.update_yaxes(title_text='Seafood (S)', row=1, col=1)
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

    with tab_bif:
        with status_indicator(status_slot, ["Computing bifurcation diagram"]):
            br_r, br_S, br_E = s1_bifurcation(
                float(s1_rng[0]), float(s1_rng[1]), s1_res, 300, 0.6, sys_t,
            )

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Seafood S*', 'Effort E*'],
            horizontal_spacing=0.08,
        )
        fig.add_trace(go.Scattergl(
            x=br_r, y=br_S, mode='markers',
            marker=dict(color=S1_COLORS['S'], size=2, opacity=0.4),
            showlegend=False,
        ), row=1, col=1)
        fig.add_trace(go.Scattergl(
            x=br_r, y=br_E, mode='markers',
            marker=dict(color=S1_COLORS['E'], size=2, opacity=0.4),
            showlegend=False,
        ), row=1, col=2)
        _def_r = DEFAULT_PARAMS['r']
        fig.add_vline(
            x=_def_r, line_dash='dash', line_color='gray',
            annotation_text=f'Default r = {_def_r}',
            annotation_position='top right', row=1, col=1,
        )
        fig.add_vline(x=_def_r, line_dash='dash', line_color='gray', row=1, col=2)
        fig.update_xaxes(title_text='Intrinsic Growth Rate (r)')
        fig.update_layout(
            height=600,
            title_text='Bifurcation Diagram over r (No Fraud)',
            margin=dict(t=60, b=40),
        )
        st.plotly_chart(fig, width='stretch')

    with tab_rm:
        with status_indicator(status_slot, ["Running time-series simulations"]):
            ts1 = {rv: s1_time_series(float(rv), s1_sim, sys_t) for rv in s1_r_vals}

        fig = make_subplots(
            rows=2, cols=_N,
            subplot_titles=[f'r = {rv}' for rv in s1_r_vals] + [''] * _N,
            vertical_spacing=0.12, horizontal_spacing=0.05,
        )
        for col, rv in enumerate(s1_r_vals, 1):
            d = ts1[rv]
            for row, (var, clr) in enumerate([
                ('Seafood', S1_COLORS['S']), ('Effort', S1_COLORS['E']),
            ], 1):
                x = d[var]
                x_t, x_tp1 = x[_burn:-1], x[_burn + 1:]
                fig.add_trace(go.Scattergl(
                    x=x_t, y=x_tp1, mode='markers',
                    marker=dict(color=clr, size=2, opacity=0.6),
                    showlegend=False,
                ), row=row, col=col)
                lo = float(min(x_t.min(), x_tp1.min())) * 0.9
                hi = float(max(x_t.max(), x_tp1.max())) * 1.1
                fig.add_trace(go.Scatter(
                    x=[lo, hi], y=[lo, hi], mode='lines',
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
