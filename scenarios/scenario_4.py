import streamlit as st
import numpy as np
import plotly.graph_objects as go
from System import DynamicalSystem, DEFAULT_PARAMS

from .constants import _C0, _Q0, FULL_INIT
from .plots import plot_4var_ts, plot_bifurcation, plot_return_maps, plot_ts_heatmap, HEATMAP_METRICS
from ._status import scenario_header, status_indicator
from ._sys_params import sys_params_ui


_FT_OPTIONS = [0.05, 0.25, 0.5, 0.75, 0.95]
_BETA_HOLD_OPTIONS = [0.0, 0.10, 0.15, 0.25, 0.40, 0.55, 0.70, 0.85, 1.00]


def _eez_params(beta: float) -> dict:
    return {
        'q1': float(_Q0 + beta * 0.23),
        'c1': float(_C0 + beta * 1.10),
    }


@st.cache_data(show_spinner=False)
def s4_time_series(beta_val: float, ft_val: float, sim_time: int,
                   sys_params: tuple = ()) -> dict:
    p = DEFAULT_PARAMS.copy()
    if sys_params:
        p.update(dict(sys_params))
    p.update(_eez_params(beta_val))
    p['F_threshold'] = ft_val
    state = {k: np.float128(v) for k, v in FULL_INIT.items()}
    sys = DynamicalSystem(p, state, "dimensionalized")
    ts = sys.time_series_plot(time=sim_time)
    return {k: v.astype(np.float64) for k, v in ts.items()}


@st.cache_data(show_spinner=False)
def s4_bifurcation(b_min: float, b_max: float, resolution: int,
                   bif_time: int, burn_frac: float, ft_val: float,
                   sys_params: tuple = ()) -> tuple:
    b_sweep = np.linspace(b_min, b_max, resolution)
    burn = int(bif_time * burn_frac)
    bb_b, bb_S, bb_E = [], [], []
    for bv in b_sweep:
        p = DEFAULT_PARAMS.copy()
        if sys_params:
            p.update(dict(sys_params))
        p.update(_eez_params(float(bv)))
        p['F_threshold'] = ft_val
        state = {k: np.float128(v) for k, v in FULL_INIT.items()}
        sys = DynamicalSystem(p, state, "dimensionalized")
        ts = sys.time_series_plot(time=bif_time)
        s_att = ts['Seafood'][burn:].astype(np.float64)
        e_att = ts['Effort'][burn:].astype(np.float64)
        n = len(s_att)
        bb_b.extend([float(bv)] * n)
        bb_S.extend(s_att.tolist())
        bb_E.extend(e_att.tolist())
    return np.array(bb_b), np.array(bb_S), np.array(bb_E)


@st.cache_data(show_spinner=False)
def s4_time_series_ft(beta_hold: float, ft_val: float, sim_time: int,
                      sys_params: tuple = ()) -> dict:
    p = DEFAULT_PARAMS.copy()
    if sys_params:
        p.update(dict(sys_params))
    p.update(_eez_params(beta_hold))
    p['F_threshold'] = ft_val
    state = {k: np.float128(v) for k, v in FULL_INIT.items()}
    sys = DynamicalSystem(p, state, "dimensionalized")
    ts = sys.time_series_plot(time=sim_time)
    return {k: v.astype(np.float64) for k, v in ts.items()}


@st.cache_data(show_spinner=False)
def s4_bifurcation_ft(beta_hold: float, ft_min: float, ft_max: float,
                      resolution: int, bif_time: int, burn_frac: float,
                      sys_params: tuple = ()) -> tuple:
    ft_sweep = np.linspace(ft_min, ft_max, resolution)
    burn = int(bif_time * burn_frac)
    bf_f, bf_S, bf_E = [], [], []
    for ft in ft_sweep:
        p = DEFAULT_PARAMS.copy()
        if sys_params:
            p.update(dict(sys_params))
        p.update(_eez_params(beta_hold))
        p['F_threshold'] = float(ft)
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
def s4_stability_heatmap(c1_min: float, c1_max: float,
                         q1_min: float, q1_max: float,
                         resolution: int,
                         sys_params: tuple = ()) -> tuple:
    c1_arr = np.linspace(c1_min, c1_max, resolution)
    q1_arr = np.linspace(q1_min, q1_max, resolution)
    stable_grid = np.full((resolution, resolution), np.nan)
    for i, q1 in enumerate(q1_arr):
        for j, c1 in enumerate(c1_arr):
            p = DEFAULT_PARAMS.copy()
            if sys_params:
                p.update(dict(sys_params))
            p.update({'c1': float(c1), 'q1': float(q1)})
            state = {k: np.float128(v) for k, v in FULL_INIT.items()}
            sys = DynamicalSystem(p, state, "dimensionalized")
            result = sys.stability_analysis()
            stable_grid[i, j] = 1.0 if result['stable'] else 0.0
    return c1_arr.astype(np.float64), q1_arr.astype(np.float64), stable_grid


def scenario_4():
    status_slot = scenario_header("Scenario 4 — Non-Enforcement of EEZ")
    st.caption(
        "Fishers access outside-EEZ waters: q₁↑ (more fish), c₁↑ (higher cost). "
        f"pw₁ stays at default ({DEFAULT_PARAMS['pw1']}). "
        "Focus parameter: EEZ violation intensity β ∈ [0, 1]."
    )

    with st.expander("Analysis Parameters", expanded=False):
        colA, colB = st.columns(2, gap="large")

        with colA:
            st.markdown("#### vs β")
            st.markdown("**Time Series & Poincare**")
            s4_simA = st.slider("Time period", 100, 1000, 400, 50, key="s4_simA")
            s4_b_vals = st.multiselect(
                "β values", _BETA_HOLD_OPTIONS,
                default=[0.15, 0.40, 0.70, 1.00], key="s4_bv",
            )
            s4_ft_A = st.selectbox(
                "F_threshold", _FT_OPTIONS,
                index=_FT_OPTIONS.index(0.5), key="s4_ftA",
            )
            st.markdown("**Bifurcation**")
            s4_bifA_iter = st.slider(
                "Iteration length", 100, 1000, 300, 50, key="s4_bifA_iter",
            )
            s4_resA = st.slider(
                "Resolution", 50, 500, 200, 50, key="s4_resA",
            )
            s4_rng = st.slider(
                "β range", 0.0, 1.0, (0.0, 1.0), 0.05, key="s4_rng",
            )

        with colB:
            st.markdown("#### vs F_threshold")
            st.markdown("**Time Series & Poincare**")
            s4_simB = st.slider("Time period", 100, 1000, 400, 50, key="s4_simB")
            s4_ft_vals = st.multiselect(
                "F_threshold values", _FT_OPTIONS,
                default=[0.25, 0.5, 0.75, 0.95], key="s4_ftv",
            )
            s4_b_hold = st.selectbox(
                "β (held)", _BETA_HOLD_OPTIONS,
                index=_BETA_HOLD_OPTIONS.index(0.55), key="s4_bhold",
            )
            st.markdown("**Bifurcation**")
            s4_bifB_iter = st.slider(
                "Iteration length", 100, 1000, 300, 50, key="s4_bifB_iter",
            )
            s4_resB = st.slider(
                "Resolution", 50, 500, 200, 50, key="s4_resB",
            )
            s4_ft_rng = st.slider(
                "F_threshold range", 0.0, 1.0, (0.1, 1.0), 0.05, key="s4_ftrng",
            )

    sys_t = sys_params_ui("s4")

    if not s4_b_vals:
        st.warning("Select at least one *β* value.")
        return
    if not s4_ft_vals:
        st.warning("Select at least one *F_threshold* value.")
        return

    s4_b_vals = sorted(s4_b_vals)
    s4_ft_vals = sorted(s4_ft_vals)
    _burnA = int(s4_simA * 0.6)
    _burnB = int(s4_simB * 0.6)

    ep_labels = [
        f'β={b}  (q₁={_eez_params(b)["q1"]:.2f}, c₁={_eez_params(b)["c1"]:.2f})'
        for b in s4_b_vals
    ]
    hold_tag = (
        f'β={s4_b_hold}  '
        f'(q₁={_eez_params(s4_b_hold)["q1"]:.2f}, '
        f'c₁={_eez_params(s4_b_hold)["c1"]:.2f})'
    )

    tab_ts, tab_bif, tab_rm = st.tabs(
        ["Time Series", "Bifurcation", "Poincare"]
    )

    with tab_ts:
        with status_indicator(status_slot, [
            "Running time-series simulations (β sweep)",
            "Running time-series simulations (F_threshold sweep)",
        ]):
            ts4 = {b: s4_time_series(float(b), float(s4_ft_A), s4_simA, sys_t) for b in s4_b_vals}
            t4_A = np.arange(s4_simA + 1)
            ts4_ft = {
                ft: s4_time_series_ft(float(s4_b_hold), float(ft), s4_simB, sys_t)
                for ft in s4_ft_vals
            }
            t4_B = np.arange(s4_simB + 1)

        hm_metrics = st.pills(
            "Heatmap rows", HEATMAP_METRICS, default=HEATMAP_METRICS,
            selection_mode="multi", key="s4_hm_m",
        )
        tsA, tsB = st.tabs(["vs β", "vs F_threshold"])
        with tsA:
            fig = plot_4var_ts(
                ts4, t4_A, s4_b_vals, 'β',
                f'EEZ Non-Enforcement — Time Series by Violation Intensity   '
                f'(F_threshold={s4_ft_A}  |  q₁↑  c₁↑  |  '
                f'pw₁={DEFAULT_PARAMS["pw1"]} default)',
            )
            for i, lbl in enumerate(ep_labels):
                fig.layout.annotations[i].text = lbl
            st.plotly_chart(fig, width='stretch')
            if hm_metrics:
                hm = plot_ts_heatmap(ts4, s4_b_vals, 'β', hm_metrics)
                if hm:
                    for hm_fig in hm:
                        st.plotly_chart(hm_fig, width='stretch')
        with tsB:
            fig = plot_4var_ts(
                ts4_ft, t4_B, s4_ft_vals, 'F_threshold',
                f'EEZ Non-Enforcement — Time Series as F_threshold Increases   '
                f'(held {hold_tag}  |  pw₁={DEFAULT_PARAMS["pw1"]} default)',
            )
            st.plotly_chart(fig, width='stretch')
            if hm_metrics:
                hm = plot_ts_heatmap(ts4_ft, s4_ft_vals, 'F_threshold', hm_metrics)
                if hm:
                    for hm_fig in hm:
                        st.plotly_chart(hm_fig, width='stretch')

    with tab_bif:
        with status_indicator(status_slot, [
            "Computing bifurcation diagram (β sweep)",
            "Computing bifurcation diagram (F_threshold sweep)",
        ]):
            bb_b, bb_S, bb_E = s4_bifurcation(
                float(s4_rng[0]), float(s4_rng[1]), s4_resA, s4_bifA_iter, 0.6,
                float(s4_ft_A), sys_t,
            )
            bf_f, bf_S, bf_E = s4_bifurcation_ft(
                float(s4_b_hold), float(s4_ft_rng[0]), float(s4_ft_rng[1]),
                s4_resB, s4_bifB_iter, 0.6, sys_t,
            )

        bifA, bifB = st.tabs(["vs β", "vs F_threshold"])
        with bifA:
            fig = plot_bifurcation(
                bb_b, bb_S, bb_E,
                xlabel='EEZ Violation Intensity (β)',
                title='Bifurcation over β   '
                      f'(F_threshold={s4_ft_A}  |  β=0 → honest  |  β=1 → q₁=0.30, c₁=2.00)',
            )
            st.plotly_chart(fig, width='stretch')
        with bifB:
            fig = plot_bifurcation(
                bf_f, bf_S, bf_E,
                xlabel='F_threshold',
                title=f'Bifurcation over F_threshold   (held {hold_tag})',
            )
            st.plotly_chart(fig, width='stretch')

    with tab_rm:
        with status_indicator(status_slot, [
            "Running time-series simulations (β sweep)",
            "Running time-series simulations (F_threshold sweep)",
        ]):
            ts4 = {b: s4_time_series(float(b), float(s4_ft_A), s4_simA, sys_t) for b in s4_b_vals}
            ts4_ft = {
                ft: s4_time_series_ft(float(s4_b_hold), float(ft), s4_simB, sys_t)
                for ft in s4_ft_vals
            }

        rmA, rmB = st.tabs(["vs β", "vs F_threshold"])
        with rmA:
            fig = plot_return_maps(ts4, s4_b_vals, 'β', _burnA)
            st.plotly_chart(fig, width='stretch')
        with rmB:
            fig = plot_return_maps(ts4_ft, s4_ft_vals, 'F_threshold', _burnB)
            st.plotly_chart(fig, width='stretch')
