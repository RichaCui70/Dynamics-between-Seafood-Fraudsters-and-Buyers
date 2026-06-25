import streamlit as st
import numpy as np
import plotly.graph_objects as go
from System import DynamicalSystem, DEFAULT_PARAMS

from .constants import _C0, _Q0, FULL_INIT
from .plots import plot_4var_ts, plot_bifurcation, plot_return_maps, plot_ts_heatmap, HEATMAP_METRICS
from ._status import scenario_header, status_indicator
from ._sys_params import sys_params_ui


_FT_OPTIONS = [0.05, 0.25, 0.5, 0.75, 0.95]
_ALPHA_HOLD_OPTIONS = [0.0, 0.10, 0.15, 0.25, 0.40, 0.55, 0.70, 0.85, 1.00]


def _eez_params(alpha: float) -> dict:
    return {
        'q1': float(_Q0 + alpha * 0.23),
        'c1': float(_C0 + alpha * 1.10),
    }


@st.cache_data(show_spinner=False)
def eez_time_series(alpha_val: float, ft_val: float, sim_time: int,
                    sys_params: tuple = ()) -> dict:
    p = DEFAULT_PARAMS.copy()
    if sys_params:
        p.update(dict(sys_params))
    p.update(_eez_params(alpha_val))
    p['F_threshold'] = ft_val
    state = {k: np.float128(v) for k, v in FULL_INIT.items()}
    sys = DynamicalSystem(p, state, "dimensionalized")
    ts = sys.time_series_plot(time=sim_time)
    return {k: v.astype(np.float64) for k, v in ts.items()}


@st.cache_data(show_spinner=False)
def eez_bifurcation(a_min: float, a_max: float, resolution: int,
                    bif_time: int, burn_frac: float, ft_val: float,
                    sys_params: tuple = ()) -> tuple:
    a_sweep = np.linspace(a_min, a_max, resolution)
    burn = int(bif_time * burn_frac)
    ba_a, ba_S, ba_E = [], [], []
    for av in a_sweep:
        p = DEFAULT_PARAMS.copy()
        if sys_params:
            p.update(dict(sys_params))
        p.update(_eez_params(float(av)))
        p['F_threshold'] = ft_val
        state = {k: np.float128(v) for k, v in FULL_INIT.items()}
        sys = DynamicalSystem(p, state, "dimensionalized")
        ts = sys.time_series_plot(time=bif_time)
        s_att = ts['Seafood'][burn:].astype(np.float64)
        e_att = ts['Effort'][burn:].astype(np.float64)
        n = len(s_att)
        ba_a.extend([float(av)] * n)
        ba_S.extend(s_att.tolist())
        ba_E.extend(e_att.tolist())
    return np.array(ba_a), np.array(ba_S), np.array(ba_E)


@st.cache_data(show_spinner=False)
def eez_time_series_ft(alpha_hold: float, ft_val: float, sim_time: int,
                       sys_params: tuple = ()) -> dict:
    p = DEFAULT_PARAMS.copy()
    if sys_params:
        p.update(dict(sys_params))
    p.update(_eez_params(alpha_hold))
    p['F_threshold'] = ft_val
    state = {k: np.float128(v) for k, v in FULL_INIT.items()}
    sys = DynamicalSystem(p, state, "dimensionalized")
    ts = sys.time_series_plot(time=sim_time)
    return {k: v.astype(np.float64) for k, v in ts.items()}


@st.cache_data(show_spinner=False)
def eez_bifurcation_ft(alpha_hold: float, ft_min: float, ft_max: float,
                       resolution: int, bif_time: int, burn_frac: float,
                       sys_params: tuple = ()) -> tuple:
    ft_sweep = np.linspace(ft_min, ft_max, resolution)
    burn = int(bif_time * burn_frac)
    bf_f, bf_S, bf_E = [], [], []
    for ft in ft_sweep:
        p = DEFAULT_PARAMS.copy()
        if sys_params:
            p.update(dict(sys_params))
        p.update(_eez_params(alpha_hold))
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
def eez_stability_heatmap(c1_min: float, c1_max: float,
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


@st.cache_data(show_spinner=False)
def eez_baseline(sim_time: int, sys_params: tuple = ()) -> dict:
    """Baseline: no fraud (F=0, FP=0), standard parameters, no EEZ violation (alpha=0)."""
    p = DEFAULT_PARAMS.copy()
    if sys_params:
        p.update(dict(sys_params))
    p.update(_eez_params(0.0))
    p.update({'F_threshold': 0.5})
    state = {k: np.float128(v) for k, v in FULL_INIT.items()}
    state['F'] = np.float128(0.0)
    state['FP'] = np.float128(0.0)
    sys = DynamicalSystem(p, state, "dimensionalized")
    ts = sys.time_series_plot(time=sim_time)
    burn = int(sim_time * 0.6)
    return {
        'Seafood': float(np.mean(ts['Seafood'][burn:])),
        'Harvest': float(np.mean(ts['Harvest'][burn:])),
        'Market Price': float(np.mean(ts['Market Price'][burn:])),
    }


@st.cache_data(show_spinner=False)
def eez_baseline_ts(sim_time: int, sys_params: tuple = ()) -> dict:
    """Full time series at alpha=0, F=0, FP=0 for the fixed baseline column."""
    p = DEFAULT_PARAMS.copy()
    if sys_params:
        p.update(dict(sys_params))
    p.update(_eez_params(0.0))
    p.update({'F_threshold': 0.5})
    state = {k: np.float128(v) for k, v in FULL_INIT.items()}
    state['F'] = np.float128(0.0)
    state['FP'] = np.float128(0.0)
    sys = DynamicalSystem(p, state, "dimensionalized")
    ts = sys.time_series_plot(time=sim_time)
    return {k: v.astype(np.float64) for k, v in ts.items()}


def scenario_eez():
    status_slot = scenario_header("Scenario 3 — Non-Enforcement of EEZ")
    st.caption(
        "Fishers access outside-EEZ waters: q₁↑ (more fish), c₁↑ (higher cost). "
        f"pw₁ stays at default ({DEFAULT_PARAMS['pw1']}). "
        "Focus parameter: EEZ violation intensity α ∈ [0, 1]."
    )

    with st.expander("Analysis Parameters", expanded=False):
        colA, colB = st.columns(2, gap="large")

        with colA:
            st.markdown("#### vs α")
            st.markdown("**Time Series & Poincare**")
            eez_simA = st.slider("Time period", 100, 1000, 400, 50, key="eez_simA")
            eez_a_vals = st.multiselect(
                "α values", _ALPHA_HOLD_OPTIONS,
                default=[0.15, 0.40, 0.70, 1.00], key="eez_av",
            )
            eez_ft_A = st.selectbox(
                "F_threshold", _FT_OPTIONS,
                index=_FT_OPTIONS.index(0.5), key="eez_ftA",
            )
            st.markdown("**Bifurcation**")
            eez_bifA_iter = st.slider(
                "Iteration length", 100, 1000, 300, 50, key="eez_bifA_iter",
            )
            eez_resA = st.slider(
                "Resolution", 50, 500, 200, 50, key="eez_resA",
            )
            eez_rng = st.slider(
                "α range", 0.0, 1.0, (0.0, 1.0), 0.05, key="eez_rng",
            )

        with colB:
            st.markdown("#### vs F_threshold")
            st.markdown("**Time Series & Poincare**")
            eez_simB = st.slider("Time period", 100, 1000, 400, 50, key="eez_simB")
            eez_ft_vals = st.multiselect(
                "F_threshold values", _FT_OPTIONS,
                default=[0.25, 0.5, 0.75, 0.95], key="eez_ftv",
            )
            eez_a_hold = st.selectbox(
                "α (held)", _ALPHA_HOLD_OPTIONS,
                index=_ALPHA_HOLD_OPTIONS.index(0.55), key="eez_ahold",
            )
            st.markdown("**Bifurcation**")
            eez_bifB_iter = st.slider(
                "Iteration length", 100, 1000, 300, 50, key="eez_bifB_iter",
            )
            eez_resB = st.slider(
                "Resolution", 50, 500, 200, 50, key="eez_resB",
            )
            eez_ft_rng = st.slider(
                "F_threshold range", 0.0, 1.0, (0.1, 1.0), 0.05, key="eez_ftrng",
            )

    sys_t = sys_params_ui("eez", exclude={'q1', 'c1'})

    if not eez_a_vals:
        st.warning("Select at least one *α* value.")
        return
    if not eez_ft_vals:
        st.warning("Select at least one *F_threshold* value.")
        return

    eez_a_vals = sorted(eez_a_vals)
    eez_ft_vals = sorted(eez_ft_vals)
    _burnA = int(eez_simA * 0.6)
    _burnB = int(eez_simB * 0.6)

    ep_labels = [
        f'α={a}  (q₁={_eez_params(a)["q1"]:.2f}, c₁={_eez_params(a)["c1"]:.2f})'
        for a in eez_a_vals
    ]
    hold_tag = (
        f'α={eez_a_hold}  '
        f'(q₁={_eez_params(eez_a_hold)["q1"]:.2f}, '
        f'c₁={_eez_params(eez_a_hold)["c1"]:.2f})'
    )

    tab_ts, tab_bif, tab_rm = st.tabs(
        ["Time Series", "Bifurcation", "Poincare"]
    )

    with tab_ts:
        with status_indicator(status_slot, [
            "Running time-series simulations (α sweep)",
            "Running time-series simulations (F_threshold sweep)",
            "Computing baseline (no fraud, no EEZ violation)",
        ]):
            ts4 = {a: eez_time_series(float(a), float(eez_ft_A), eez_simA, sys_t) for a in eez_a_vals}
            ts_bl = eez_baseline_ts(eez_simA, sys_t)
            t4_A = np.arange(eez_simA + 1)
            ts4_ft = {
                ft: eez_time_series_ft(float(eez_a_hold), float(ft), eez_simB, sys_t)
                for ft in eez_ft_vals
            }
            t4_B = np.arange(eez_simB + 1)
            eez_baseline_vals = eez_baseline(eez_simA, sys_t)

        hm_metrics = st.pills(
            "Heatmap rows", HEATMAP_METRICS, default=HEATMAP_METRICS,
            selection_mode="multi", key="eez_hm_m",
        )
        st.caption("Heatmap: % change vs. baseline (no fraud, no fraud perception, no EEZ violation)")
        tsA, tsB = st.tabs(["vs α", "vs F_threshold"])
        with tsA:
            _ts_full = {'Baseline': ts_bl, **ts4}
            _vals_full = ['Baseline'] + eez_a_vals
            _all_labels = ['Baseline (α=0, F=FP=0)'] + ep_labels
            fig = plot_4var_ts(
                _ts_full, t4_A, _vals_full, 'α',
                f'EEZ Non-Enforcement — Time Series by Violation Intensity   '
                f'(F_threshold={eez_ft_A}  |  q₁↑  c₁↑  |  '
                f'pw₁={DEFAULT_PARAMS["pw1"]} default)',
            )
            for i, lbl in enumerate(_all_labels):
                fig.layout.annotations[i].text = lbl
            st.plotly_chart(fig, width='stretch')
            if hm_metrics:
                hm = plot_ts_heatmap(_ts_full, _vals_full, 'α', hm_metrics, baseline_dict=eez_baseline_vals)
                if hm:
                    for hm_fig in hm:
                        st.plotly_chart(hm_fig, width='stretch')
        with tsB:
            ts_bl_B = eez_baseline_ts(eez_simB, sys_t)
            _ts_full_ft = {'Baseline': ts_bl_B, **ts4_ft}
            _vals_full_ft = ['Baseline'] + eez_ft_vals
            _ft_labels = ['Baseline (F=FP=0)'] + [str(ft) for ft in eez_ft_vals]
            fig = plot_4var_ts(
                _ts_full_ft, t4_B, _vals_full_ft, 'F_threshold',
                f'EEZ Non-Enforcement — Time Series as F_threshold Increases   '
                f'(held {hold_tag}  |  pw₁={DEFAULT_PARAMS["pw1"]} default)',
            )
            for i, lbl in enumerate(_ft_labels):
                fig.layout.annotations[i].text = lbl
            st.plotly_chart(fig, width='stretch')
            if hm_metrics:
                hm = plot_ts_heatmap(_ts_full_ft, _vals_full_ft, 'F_threshold', hm_metrics, baseline_dict=eez_baseline_vals)
                if hm:
                    for hm_fig in hm:
                        st.plotly_chart(hm_fig, width='stretch')

    with tab_bif:
        with status_indicator(status_slot, [
            "Computing bifurcation diagram (α sweep)",
            "Computing bifurcation diagram (F_threshold sweep)",
        ]):
            ba_a, ba_S, ba_E = eez_bifurcation(
                float(eez_rng[0]), float(eez_rng[1]), eez_resA, eez_bifA_iter, 0.6,
                float(eez_ft_A), sys_t,
            )
            bf_f, bf_S, bf_E = eez_bifurcation_ft(
                float(eez_a_hold), float(eez_ft_rng[0]), float(eez_ft_rng[1]),
                eez_resB, eez_bifB_iter, 0.6, sys_t,
            )

        bifA, bifB = st.tabs(["vs α", "vs F_threshold"])
        with bifA:
            fig = plot_bifurcation(
                ba_a, ba_S, ba_E,
                xlabel='EEZ Violation Intensity (α)',
                title='Bifurcation over α   '
                      f'(F_threshold={eez_ft_A}  |  α=0 → honest  |  α=1 → q₁=0.30, c₁=2.00)',
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
            "Running time-series simulations (α sweep)",
            "Running time-series simulations (F_threshold sweep)",
        ]):
            ts4 = {a: eez_time_series(float(a), float(eez_ft_A), eez_simA, sys_t) for a in eez_a_vals}
            ts_bl = eez_baseline_ts(eez_simA, sys_t)
            ts4_ft = {
                ft: eez_time_series_ft(float(eez_a_hold), float(ft), eez_simB, sys_t)
                for ft in eez_ft_vals
            }

        rmA, rmB = st.tabs(["vs α", "vs F_threshold"])
        with rmA:
            _ts_full = {'Baseline': ts_bl, **ts4}
            _vals_full = ['Baseline'] + eez_a_vals
            _all_labels = ['Baseline (α=0, F=FP=0)'] + ep_labels
            fig = plot_return_maps(_ts_full, _vals_full, 'α', _burnA)
            for i, lbl in enumerate(_all_labels):
                fig.layout.annotations[i].text = lbl
            st.plotly_chart(fig, width='stretch')
        with rmB:
            ts_bl_B = eez_baseline_ts(eez_simB, sys_t)
            _ts_full_ft = {'Baseline': ts_bl_B, **ts4_ft}
            _vals_full_ft = ['Baseline'] + eez_ft_vals
            _ft_labels = ['Baseline (F=FP=0)'] + [str(ft) for ft in eez_ft_vals]
            fig = plot_return_maps(_ts_full_ft, _vals_full_ft, 'F_threshold', _burnB)
            for i, lbl in enumerate(_ft_labels):
                fig.layout.annotations[i].text = lbl
            st.plotly_chart(fig, width='stretch')
