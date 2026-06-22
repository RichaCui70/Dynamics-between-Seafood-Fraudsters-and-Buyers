import streamlit as st
import numpy as np
import plotly.graph_objects as go
from System import DynamicalSystem, DEFAULT_PARAMS

from .constants import _PW0, _C0, _Q0, FULL_INIT
from .plots import plot_4var_ts, plot_ts_with_economics, plot_bifurcation, plot_return_maps, plot_ts_heatmap, HEATMAP_METRICS
from ._status import scenario_header, status_indicator
from ._sys_params import sys_params_ui


_FT_OPTIONS = [0.05, 0.25, 0.5, 0.75, 0.95]
_PW_HOLD_OPTIONS = [
    0.25, 0.5, 0.75, 1.0, 1.25, 1.50, 1.75, 2.00, 2.50, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00,
]


@st.cache_data(show_spinner=False)
def s2_time_series(pw1_val: float, ft_val: float, sim_time: int,
                   sys_params: tuple = ()) -> dict:
    p = DEFAULT_PARAMS.copy()
    if sys_params:
        p.update(dict(sys_params))
    p.update({'pw1': pw1_val, 'c1': _C0, 'q1': _Q0, 'F_threshold': ft_val})
    state = {k: np.float128(v) for k, v in FULL_INIT.items()}
    sys = DynamicalSystem(p, state, "dimensionalized")
    ts = sys.time_series_plot(time=sim_time)
    return {k: v.astype(np.float64) for k, v in ts.items()}


@st.cache_data(show_spinner=False)
def s2_bifurcation(pw_min: float, pw_max: float, resolution: int,
                   bif_time: int, burn_frac: float, ft_val: float,
                   sys_params: tuple = ()) -> tuple:
    pw_sweep = np.linspace(pw_min, pw_max, resolution)
    burn = int(bif_time * burn_frac)
    bp_p, bp_S, bp_E = [], [], []
    for pw in pw_sweep:
        p = DEFAULT_PARAMS.copy()
        if sys_params:
            p.update(dict(sys_params))
        p.update({'pw1': float(pw), 'c1': _C0, 'q1': _Q0, 'F_threshold': ft_val})
        state = {k: np.float128(v) for k, v in FULL_INIT.items()}
        sys = DynamicalSystem(p, state, "dimensionalized")
        ts = sys.time_series_plot(time=bif_time)
        s_att = ts['Seafood'][burn:].astype(np.float64)
        e_att = ts['Effort'][burn:].astype(np.float64)
        n = len(s_att)
        bp_p.extend([float(pw)] * n)
        bp_S.extend(s_att.tolist())
        bp_E.extend(e_att.tolist())
    return np.array(bp_p), np.array(bp_S), np.array(bp_E)


@st.cache_data(show_spinner=False)
def s2_time_series_ft(pw1_hold: float, ft_val: float, sim_time: int,
                      sys_params: tuple = ()) -> dict:
    p = DEFAULT_PARAMS.copy()
    if sys_params:
        p.update(dict(sys_params))
    p.update({'pw1': pw1_hold, 'c1': _C0, 'q1': _Q0, 'F_threshold': ft_val})
    state = {k: np.float128(v) for k, v in FULL_INIT.items()}
    sys = DynamicalSystem(p, state, "dimensionalized")
    ts = sys.time_series_plot(time=sim_time)
    return {k: v.astype(np.float64) for k, v in ts.items()}


@st.cache_data(show_spinner=False)
def s2_bifurcation_ft(pw1_hold: float, ft_min: float, ft_max: float,
                      resolution: int, bif_time: int, burn_frac: float,
                      sys_params: tuple = ()) -> tuple:
    ft_sweep = np.linspace(ft_min, ft_max, resolution)
    burn = int(bif_time * burn_frac)
    bf_f, bf_S, bf_E = [], [], []
    for ft in ft_sweep:
        p = DEFAULT_PARAMS.copy()
        if sys_params:
            p.update(dict(sys_params))
        p.update({
            'pw1': pw1_hold, 'c1': _C0, 'q1': _Q0, 'F_threshold': float(ft),
        })
        state = {k: np.float128(v) for k, v in FULL_INIT.items()}
        sys = DynamicalSystem(p, state, "dimensionalized")
        ts = sys.time_series_plot(time=bif_time)
        s_att = ts['Seafood'][burn:].astype(np.float64)
        e_att = ts['Effort'][burn:].astype(np.float64)
        n = len(s_att)
        bf_f.extend([float(ft)] * n)
        bf_S.extend(s_att.tolist())
        bf_E.extend(e_att.tolist())
    return np.array(bf_f), np.array(bf_S), np.array(bf_E)


@st.cache_data(show_spinner=False)
def s2_spectral_sweep(pw_min: float, pw_max: float, resolution: int,
                      sys_params: tuple = ()) -> tuple:
    pw_vals = np.linspace(pw_min, pw_max, resolution)
    rho_vals = np.empty(resolution)
    for i, pw in enumerate(pw_vals):
        p = DEFAULT_PARAMS.copy()
        if sys_params:
            p.update(dict(sys_params))
        p.update({'pw1': float(pw), 'c1': _C0, 'q1': _Q0})
        state = {k: np.float128(v) for k, v in FULL_INIT.items()}
        sys = DynamicalSystem(p, state, "dimensionalized")
        result = sys.stability_analysis()
        rho_vals[i] = result['spectral_radius']
    return pw_vals.astype(np.float64), rho_vals.astype(np.float64)


def scenario_2():
    status_slot = scenario_header("Scenario 2 — Prized / Protected Seafood")
    st.caption(
        f"Fraudsters pay a premium for protected species (pw₁ > pw₀). "
        f"Same gear, same waters: c₁ = c₀ = {_C0}, q₁ = q₀ = {_Q0}. "
        f"Focus parameter: pw₁."
    )

    with st.expander("Analysis Parameters", expanded=False):
        colA, colB = st.columns(2, gap="large")

        with colA:
            st.markdown("#### vs pw₁")
            st.markdown("**Time Series & Poincare**")
            s2_simA = st.slider("Time period", 100, 1000, 400, 50, key="s2_simA")
            s2_pw_vals = st.multiselect(
                "pw₁ values", _PW_HOLD_OPTIONS,
                default=[1.0, 1.50, 2.00, 5.00, 8.00], key="s2_pw",
            )
            s2_ft_A = st.selectbox(
                "F_threshold", _FT_OPTIONS,
                index=_FT_OPTIONS.index(0.5), key="s2_ftA",
            )
            st.markdown("**Bifurcation**")
            s2_bifA_iter = st.slider(
                "Iteration length", 100, 1000, 300, 50, key="s2_bifA_iter",
            )
            s2_resA = st.slider(
                "Resolution", 50, 500, 200, 50, key="s2_resA",
            )
            s2_rng = st.slider(
                "pw₁ range", float(_PW0), 8.0, (float(_PW0), 5.0), 0.1,
                key="s2_rng",
            )

        with colB:
            st.markdown("#### vs F_threshold")
            st.markdown("**Time Series & Poincare**")
            s2_simB = st.slider("Time period", 100, 1000, 400, 50, key="s2_simB")
            s2_ft_vals = st.multiselect(
                "F_threshold values", _FT_OPTIONS,
                default=[0.25, 0.5, 0.75, 0.95], key="s2_ftv",
            )
            s2_pw_hold = st.selectbox(
                "pw₁ (held)", _PW_HOLD_OPTIONS,
                index=_PW_HOLD_OPTIONS.index(2.00), key="s2_pwhold",
            )
            st.markdown("**Bifurcation**")
            s2_bifB_iter = st.slider(
                "Iteration length", 100, 1000, 300, 50, key="s2_bifB_iter",
            )
            s2_resB = st.slider(
                "Resolution", 50, 500, 200, 50, key="s2_resB",
            )
            s2_ft_rng = st.slider(
                "F_threshold range", 0.0, 1.0, (0.1, 1.0), 0.05, key="s2_ftrng",
            )

    sys_t = sys_params_ui("s2")

    if not s2_pw_vals:
        st.warning("Select at least one *pw₁* value.")
        return
    if not s2_ft_vals:
        st.warning("Select at least one *F_threshold* value.")
        return

    s2_pw_vals = sorted(s2_pw_vals)
    s2_ft_vals = sorted(s2_ft_vals)
    _burnA = int(s2_simA * 0.6)
    _burnB = int(s2_simB * 0.6)

    tab_ts, tab_bif, tab_rm, tab_stab = st.tabs(
        ["Time Series", "Bifurcation", "Poincare", "Stability"]
    )

    with tab_ts:
        with status_indicator(status_slot, [
            "Running time-series simulations (pw₁ sweep)",
            "Running time-series simulations (F_threshold sweep)",
        ]):
            ts2 = {pw: s2_time_series(float(pw), float(s2_ft_A), s2_simA, sys_t) for pw in s2_pw_vals}
            t2_A = np.arange(s2_simA + 1)
            ts2_ft = {
                ft: s2_time_series_ft(float(s2_pw_hold), float(ft), s2_simB, sys_t)
                for ft in s2_ft_vals
            }
            t2_B = np.arange(s2_simB + 1)

        hm_metrics = st.pills(
            "Heatmap rows", HEATMAP_METRICS, default=HEATMAP_METRICS,
            selection_mode="multi", key="s2_hm_m",
        )
        tsA, tsB = st.tabs(["vs pw₁", "vs F_threshold"])
        with tsA:
            fig = plot_ts_with_economics(
                ts2, t2_A, s2_pw_vals, 'pw₁',
                f'Prized Seafood — Time Series as pw₁ Increases   '
                f'(c₁=c₀={_C0},  q₁=q₀={_Q0},  F_threshold={s2_ft_A})',
            )
            st.plotly_chart(fig, width='stretch')
            if hm_metrics:
                hm = plot_ts_heatmap(ts2, s2_pw_vals, 'pw₁', hm_metrics)
                if hm:
                    for hm_fig in hm:
                        st.plotly_chart(hm_fig, width='stretch')
        with tsB:
            fig = plot_ts_with_economics(
                ts2_ft, t2_B, s2_ft_vals, 'F_threshold',
                f'Prized Seafood — Time Series as F_threshold Increases   '
                f'(pw₁={s2_pw_hold},  c₁=c₀={_C0},  q₁=q₀={_Q0})',
            )
            st.plotly_chart(fig, width='stretch')
            if hm_metrics:
                hm = plot_ts_heatmap(ts2_ft, s2_ft_vals, 'F_threshold', hm_metrics)
                if hm:
                    for hm_fig in hm:
                        st.plotly_chart(hm_fig, width='stretch')

    with tab_bif:
        with status_indicator(status_slot, [
            "Computing bifurcation diagram (pw₁ sweep)",
            "Computing bifurcation diagram (F_threshold sweep)",
        ]):
            bp_p, bp_S, bp_E = s2_bifurcation(
                float(s2_rng[0]), float(s2_rng[1]), s2_resA, s2_bifA_iter, 0.6,
                float(s2_ft_A), sys_t,
            )
            bf_f, bf_S, bf_E = s2_bifurcation_ft(
                float(s2_pw_hold), float(s2_ft_rng[0]), float(s2_ft_rng[1]),
                s2_resB, s2_bifB_iter, 0.6, sys_t,
            )

        bifA, bifB = st.tabs(["vs pw₁", "vs F_threshold"])
        with bifA:
            fig = plot_bifurcation(
                bp_p, bp_S, bp_E,
                xlabel='pw₁ (black-market wholesale price)',
                title=f'Bifurcation Diagram over pw₁   (F_threshold={s2_ft_A})',
                vline_x=float(_PW0), vline_label=f'pw₀ = {_PW0}',
            )
            st.plotly_chart(fig, width='stretch')
        with bifB:
            fig = plot_bifurcation(
                bf_f, bf_S, bf_E,
                xlabel='F_threshold',
                title=f'Bifurcation Diagram over F_threshold   (pw₁={s2_pw_hold})',
            )
            st.plotly_chart(fig, width='stretch')

    with tab_rm:
        with status_indicator(status_slot, [
            "Running time-series simulations (pw₁ sweep)",
            "Running time-series simulations (F_threshold sweep)",
        ]):
            ts2 = {pw: s2_time_series(float(pw), float(s2_ft_A), s2_simA, sys_t) for pw in s2_pw_vals}
            ts2_ft = {
                ft: s2_time_series_ft(float(s2_pw_hold), float(ft), s2_simB, sys_t)
                for ft in s2_ft_vals
            }

        rmA, rmB = st.tabs(["vs pw₁", "vs F_threshold"])
        with rmA:
            fig = plot_return_maps(ts2, s2_pw_vals, 'pw₁', _burnA)
            st.plotly_chart(fig, width='stretch')
        with rmB:
            fig = plot_return_maps(ts2_ft, s2_ft_vals, 'F_threshold', _burnB)
            st.plotly_chart(fig, width='stretch')

    with tab_stab:
        with status_indicator(status_slot, ["Computing stability sweep"]):
            s2_pw_sweep, s2_rho = s2_spectral_sweep(0.05, 5.0, 100, sys_t)

        finite = np.isfinite(s2_rho)
        pw_fin, rho_fin = s2_pw_sweep[finite], s2_rho[finite]
        stable_mask = rho_fin < 1.0
        y_cap = max(float(np.max(rho_fin[rho_fin < 50])) * 1.1, 2.0) if np.any(rho_fin < 50) else 5.0
        rho_plot = np.clip(rho_fin, 0, y_cap)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pw_fin[stable_mask], y=rho_plot[stable_mask],
            mode='markers', marker=dict(color='#2E8B57', size=6),
            name='Stable (ρ < 1)',
        ))
        fig.add_trace(go.Scatter(
            x=pw_fin[~stable_mask], y=rho_plot[~stable_mask],
            mode='markers', marker=dict(color='#DC143C', size=6),
            name='Unstable (ρ ≥ 1)',
        ))
        fig.add_hline(y=1.0, line_dash='dash', line_color='gray',
                      annotation_text='ρ = 1 (stability boundary)')
        fig.update_layout(
            height=600,
            title_text=(
                f'Spectral Radius vs pw₁ — Fixed-Point Stability   '
                f'(c₁=c₀={_C0},  q₁=q₀={_Q0},  F_threshold={DEFAULT_PARAMS["F_threshold"]})'
            ),
            xaxis_title='pw₁ (black-market wholesale price)',
            yaxis_title='Spectral Radius  ρ = max|λᵢ|',
            yaxis_range=[0, y_cap],
            margin=dict(t=60, b=40),
            legend=dict(yanchor='top', y=0.99, xanchor='right', x=0.99),
        )
        st.plotly_chart(fig, width='stretch')
