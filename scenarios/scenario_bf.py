import streamlit as st
import numpy as np
from core.System import DynamicalSystem

from core.constants import DEFAULT_INIT_STATE, DEFAULT_PARAMS
from core.plots import plot_bifurcation, plot_return_maps, plot_ts_with_economics, plot_ts_heatmap, HEATMAP_METRICS
from ._status import scenario_header, status_indicator
from ._sys_params import sys_params_ui


_FT_OPTIONS = [0.05, 0.25, 0.5, 0.75, 0.95]
_ALPHA_HOLD_OPTIONS = [0.0, 0.10, 0.15, 0.25, 0.40, 0.55, 0.70, 0.85, 1.00]


def _blast_params(alpha: float) -> dict:
    return {
        'q1': float(DEFAULT_PARAMS['q0'] + alpha * 0.33),
        'pw1': float(DEFAULT_PARAMS['pw0'] - alpha * 0.40),
        'c1': float(DEFAULT_PARAMS['c0'] - alpha * 0.80),
    }


@st.cache_data(show_spinner=False)
def bf_time_series(alpha_val: float, ft_val: float, sim_time: int,
                   sys_params: tuple = ()) -> dict:
    p = DEFAULT_PARAMS.copy()
    if sys_params:
        p.update(dict(sys_params))
    p.update(_blast_params(alpha_val))
    p['F_threshold'] = ft_val
    state = {k: np.float128(v) for k, v in DEFAULT_INIT_STATE.items()}
    sys = DynamicalSystem(p, state, "dimensionalized")
    ts = sys.time_series_plot(time=sim_time)
    return {k: v.astype(np.float64) for k, v in ts.items()}


@st.cache_data(show_spinner=False)
def bf_bifurcation(a_min: float, a_max: float, resolution: int,
                   bif_time: int, burn_frac: float, ft_val: float,
                   sys_params: tuple = ()) -> tuple:
    a_sweep = np.linspace(a_min, a_max, resolution)
    burn = int(bif_time * burn_frac)
    ba_a, ba_S, ba_E, ba_F, ba_FP = [], [], [], [], []
    for av in a_sweep:
        p = DEFAULT_PARAMS.copy()
        if sys_params:
            p.update(dict(sys_params))
        p.update(_blast_params(float(av)))
        p['F_threshold'] = ft_val
        state = {k: np.float128(v) for k, v in DEFAULT_INIT_STATE.items()}
        sys = DynamicalSystem(p, state, "dimensionalized")
        ts = sys.time_series_plot(time=bif_time)
        s_att = ts['Seafood'][burn:].astype(np.float64)
        e_att = ts['Effort'][burn:].astype(np.float64)
        f_att = ts['Fraudsters'][burn:].astype(np.float64)
        fp_att = ts['Perception of Fraud'][burn:].astype(np.float64)
        n = len(s_att)
        ba_a.extend([float(av)] * n)
        ba_S.extend(s_att.tolist())
        ba_E.extend(e_att.tolist())
        ba_F.extend(f_att.tolist())
        ba_FP.extend(fp_att.tolist())
    return np.array(ba_a), np.array(ba_S), np.array(ba_E), np.array(ba_F), np.array(ba_FP)


@st.cache_data(show_spinner=False)
def bf_time_series_ft(alpha_hold: float, ft_val: float, sim_time: int,
                      sys_params: tuple = ()) -> dict:
    p = DEFAULT_PARAMS.copy()
    if sys_params:
        p.update(dict(sys_params))
    p.update(_blast_params(alpha_hold))
    p['F_threshold'] = ft_val
    state = {k: np.float128(v) for k, v in DEFAULT_INIT_STATE.items()}
    sys = DynamicalSystem(p, state, "dimensionalized")
    ts = sys.time_series_plot(time=sim_time)
    return {k: v.astype(np.float64) for k, v in ts.items()}


@st.cache_data(show_spinner=False)
def bf_bifurcation_ft(alpha_hold: float, ft_min: float, ft_max: float,
                      resolution: int, bif_time: int, burn_frac: float,
                      sys_params: tuple = ()) -> tuple:
    ft_sweep = np.linspace(ft_min, ft_max, resolution)
    burn = int(bif_time * burn_frac)
    bf_f, bf_S, bf_E, bf_F, bf_FP = [], [], [], [], []
    for ft in ft_sweep:
        p = DEFAULT_PARAMS.copy()
        if sys_params:
            p.update(dict(sys_params))
        p.update(_blast_params(alpha_hold))
        p['F_threshold'] = float(ft)
        state = {k: np.float128(v) for k, v in DEFAULT_INIT_STATE.items()}
        sys = DynamicalSystem(p, state, "dimensionalized")
        ts = sys.time_series_plot(time=bif_time)
        s_att = ts['Seafood'][burn:].astype(np.float64)
        e_att = ts['Effort'][burn:].astype(np.float64)
        f_att = ts['Fraudsters'][burn:].astype(np.float64)
        fp_att = ts['Perception of Fraud'][burn:].astype(np.float64)
        n = len(s_att)
        bf_f.extend([float(ft)] * n)
        bf_S.extend(s_att.tolist())
        bf_E.extend(e_att.tolist())
        bf_F.extend(f_att.tolist())
        bf_FP.extend(fp_att.tolist())
    return np.array(bf_f), np.array(bf_S), np.array(bf_E), np.array(bf_F), np.array(bf_FP)


@st.cache_data(show_spinner=False)
def bf_baseline(sim_time: int, sys_params: tuple = ()) -> dict:
    """Baseline: no fraud (F=0, FP=0), standard parameters, no blast (alpha=0)."""
    p = DEFAULT_PARAMS.copy()
    if sys_params:
        p.update(dict(sys_params))
    p.update(_blast_params(0.0))
    p.update({'F_threshold': 0.5})
    state = {k: np.float128(v) for k, v in DEFAULT_INIT_STATE.items()}
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
def bf_baseline_ts(sim_time: int, sys_params: tuple = ()) -> dict:
    """Full time series at alpha=0, F=0, FP=0 for the fixed baseline column."""
    p = DEFAULT_PARAMS.copy()
    if sys_params:
        p.update(dict(sys_params))
    p.update(_blast_params(0.0))
    p.update({'F_threshold': 0.5})
    state = {k: np.float128(v) for k, v in DEFAULT_INIT_STATE.items()}
    state['F'] = np.float128(0.0)
    state['FP'] = np.float128(0.0)
    sys = DynamicalSystem(p, state, "dimensionalized")
    ts = sys.time_series_plot(time=sim_time)
    return {k: v.astype(np.float64) for k, v in ts.items()}


def scenario_bf():
    status_slot = scenario_header("Scenario 1 — Blast / Cyanide Fishing")
    st.caption(
        "Destructive methods: q₁↑  pw₁↓  c₁↓↓ (cost drops much more than price). "
        "A single destruction intensity α ∈ [0, 1] jointly scales all three."
    )

    with st.expander("Analysis Parameters", expanded=False):
        colA, colB = st.columns(2, gap="large")

        with colA:
            st.markdown("#### vs α")
            st.markdown("**Time Series & Poincare**")
            bf_simA = st.slider("Time period", 100, 1000, 400, 50, key="bf_simA")
            bf_a_vals = st.multiselect(
                "α values", _ALPHA_HOLD_OPTIONS,
                default=[0.15, 0.40, 0.70, 1.00], key="bf_av",
            )
            bf_ft_A = st.selectbox(
                "F_threshold", _FT_OPTIONS,
                index=_FT_OPTIONS.index(0.5), key="bf_ftA",
            )
            st.markdown("**Bifurcation**")
            bf_bifA_iter = st.slider(
                "Iteration length", 100, 1000, 300, 50, key="bf_bifA_iter",
            )
            bf_resA = st.slider(
                "Resolution", 50, 500, 200, 50, key="bf_resA",
            )
            bf_rng = st.slider(
                "α range", 0.0, 1.0, (0.0, 1.0), 0.05, key="bf_rng",
            )

        with colB:
            st.markdown("#### vs F_threshold")
            st.markdown("**Time Series & Poincare**")
            bf_simB = st.slider("Time period", 100, 1000, 400, 50, key="bf_simB")
            bf_ft_vals = st.multiselect(
                "F_threshold values", _FT_OPTIONS,
                default=[0.25, 0.5, 0.75, 0.95], key="bf_ftv",
            )
            bf_a_hold = st.selectbox(
                "α (held)", _ALPHA_HOLD_OPTIONS,
                index=_ALPHA_HOLD_OPTIONS.index(0.55), key="bf_ahold",
            )
            st.markdown("**Bifurcation**")
            bf_bifB_iter = st.slider(
                "Iteration length", 100, 1000, 300, 50, key="bf_bifB_iter",
            )
            bf_resB = st.slider(
                "Resolution", 50, 500, 200, 50, key="bf_resB",
            )
            bf_ft_rng = st.slider(
                "F_threshold range", 0.0, 1.0, (0.1, 1.0), 0.05, key="bf_ftrng",
            )

    sys_t = sys_params_ui("bf", exclude={'q1', 'pw1', 'c1'})

    if not bf_a_vals:
        st.warning("Select at least one *α* value.")
        return
    if not bf_ft_vals:
        st.warning("Select at least one *F_threshold* value.")
        return

    bf_a_vals = sorted(bf_a_vals)
    bf_ft_vals = sorted(bf_ft_vals)
    _burnA = int(bf_simA * 0.6)
    _burnB = int(bf_simB * 0.6)

    bp_labels = [
        f'α={a}  (q₁={_blast_params(a)["q1"]:.2f}, '
        f'pw₁={_blast_params(a)["pw1"]:.2f}, '
        f'c₁={_blast_params(a)["c1"]:.2f})'
        for a in bf_a_vals
    ]
    hold_tag = (
        f'α={bf_a_hold}  '
        f'(q₁={_blast_params(bf_a_hold)["q1"]:.2f}, '
        f'pw₁={_blast_params(bf_a_hold)["pw1"]:.2f}, '
        f'c₁={_blast_params(bf_a_hold)["c1"]:.2f})'
    )

    tab_ts, tab_bif, tab_rm = st.tabs(
        ["Time Series", "Bifurcation", "Poincare"]
    )

    with tab_ts:
        with status_indicator(status_slot, [
            "Running time-series simulations (α sweep)",
            "Running time-series simulations (F_threshold sweep)",
            "Computing baseline (no fraud, no blast)",
        ]):
            ts3 = {a: bf_time_series(float(a), float(bf_ft_A), bf_simA, sys_t) for a in bf_a_vals}
            ts_bl = bf_baseline_ts(bf_simA, sys_t)
            t3_A = np.arange(bf_simA + 1)
            ts3_ft = {
                ft: bf_time_series_ft(float(bf_a_hold), float(ft), bf_simB, sys_t)
                for ft in bf_ft_vals
            }
            t3_B = np.arange(bf_simB + 1)
            bf_baseline_vals = bf_baseline(bf_simA, sys_t)

        hm_metrics = st.pills(
            "Heatmap rows", HEATMAP_METRICS, default=HEATMAP_METRICS,
            selection_mode="multi", key="bf_hm_m",
        )
        st.caption("Heatmap: % change vs. baseline (no fraud, no fraud perception, no blast)")
        tsA, tsB = st.tabs(["vs α", "vs F_threshold"])
        with tsA:
            _ts_full = {'Baseline': ts_bl, **ts3}
            _vals_full = ['Baseline'] + bf_a_vals
            _all_labels = ['Baseline (α=0, F=FP=0)'] + bp_labels
            fig = plot_ts_with_economics(
                _ts_full, t3_A, _vals_full, 'α',
                f'Blast Fishing — Time Series by Destruction Intensity   '
                f'(F_threshold={bf_ft_A}  |  q₁↑  pw₁↓  c₁↓↓)',
            )
            for i, lbl in enumerate(_all_labels):
                fig.layout.annotations[i].text = lbl
            st.plotly_chart(fig, width='stretch')
            if hm_metrics:
                hm = plot_ts_heatmap(_ts_full, _vals_full, 'α', hm_metrics, baseline_dict=bf_baseline_vals)
                if hm:
                    for hm_fig in hm:
                        st.plotly_chart(hm_fig, width='stretch')
        with tsB:
            ts_bl_B = bf_baseline_ts(bf_simB, sys_t)
            _ts_full_ft = {'Baseline': ts_bl_B, **ts3_ft}
            _vals_full_ft = ['Baseline'] + bf_ft_vals
            _ft_labels = [f'Baseline (F=FP=0)'] + [str(ft) for ft in bf_ft_vals]
            fig = plot_ts_with_economics(
                _ts_full_ft, t3_B, _vals_full_ft, 'F_threshold',
                f'Blast Fishing — Time Series as F_threshold Increases   '
                f'(held {hold_tag})',
            )
            for i, lbl in enumerate(_ft_labels):
                fig.layout.annotations[i].text = lbl
            st.plotly_chart(fig, width='stretch')
            if hm_metrics:
                hm = plot_ts_heatmap(_ts_full_ft, _vals_full_ft, 'F_threshold', hm_metrics, baseline_dict=bf_baseline_vals)
                if hm:
                    for hm_fig in hm:
                        st.plotly_chart(hm_fig, width='stretch')

    with tab_bif:
        with status_indicator(status_slot, [
            "Computing bifurcation diagram (α sweep)",
            "Computing bifurcation diagram (F_threshold sweep)",
        ]):
            ba_a, ba_S, ba_E, ba_F, ba_FP = bf_bifurcation(
                float(bf_rng[0]), float(bf_rng[1]), bf_resA, bf_bifA_iter, 0.6,
                float(bf_ft_A), sys_t,
            )
            bf_f, bf_S, bf_E, bf_F, bf_FP = bf_bifurcation_ft(
                float(bf_a_hold), float(bf_ft_rng[0]), float(bf_ft_rng[1]),
                bf_resB, bf_bifB_iter, 0.6, sys_t,
            )

        bifA, bifB = st.tabs(["vs α", "vs F_threshold"])
        with bifA:
            fig = plot_bifurcation(
                ba_a, ba_S, ba_E, ba_F, ba_FP,
                xlabel='Destruction Intensity (α)',
                title='Bifurcation over α   '
                      f'(F_threshold={bf_ft_A}  |  α=0 → honest  |  α=1 → q₁=0.40, pw₁=0.60, c₁=0.10)',
            )
            st.plotly_chart(fig, width='stretch')
        with bifB:
            fig = plot_bifurcation(
                bf_f, bf_S, bf_E, bf_F, bf_FP,
                xlabel='F_threshold',
                title=f'Bifurcation over F_threshold   (held {hold_tag})',
            )
            st.plotly_chart(fig, width='stretch')

    with tab_rm:
        with status_indicator(status_slot, [
            "Running time-series simulations (α sweep)",
            "Running time-series simulations (F_threshold sweep)",
        ]):
            ts3 = {a: bf_time_series(float(a), float(bf_ft_A), bf_simA, sys_t) for a in bf_a_vals}
            ts_bl = bf_baseline_ts(bf_simA, sys_t)
            ts3_ft = {
                ft: bf_time_series_ft(float(bf_a_hold), float(ft), bf_simB, sys_t)
                for ft in bf_ft_vals
            }

        rmA, rmB = st.tabs(["vs α", "vs F_threshold"])
        with rmA:
            _ts_full = {'Baseline': ts_bl, **ts3}
            _vals_full = ['Baseline'] + bf_a_vals
            _all_labels = ['Baseline (α=0, F=FP=0)'] + bp_labels
            fig = plot_return_maps(_ts_full, _vals_full, 'α', _burnA)
            for i, lbl in enumerate(_all_labels):
                fig.layout.annotations[i].text = lbl
            st.plotly_chart(fig, width='stretch')
        with rmB:
            ts_bl_B = bf_baseline_ts(bf_simB, sys_t)
            _ts_full_ft = {'Baseline': ts_bl_B, **ts3_ft}
            _vals_full_ft = ['Baseline'] + bf_ft_vals
            _ft_labels = ['Baseline (F=FP=0)'] + [str(ft) for ft in bf_ft_vals]
            fig = plot_return_maps(_ts_full_ft, _vals_full_ft, 'F_threshold', _burnB)
            for i, lbl in enumerate(_ft_labels):
                fig.layout.annotations[i].text = lbl
            st.plotly_chart(fig, width='stretch')
