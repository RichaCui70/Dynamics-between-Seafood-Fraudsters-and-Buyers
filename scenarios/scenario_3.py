import streamlit as st
import numpy as np
from System import DynamicalSystem, DEFAULT_PARAMS

from .constants import _PW0, _C0, _Q0, FULL_INIT
from .plots import plot_bifurcation, plot_return_maps, plot_ts_with_economics, plot_ts_heatmap, HEATMAP_METRICS
from ._status import scenario_header, status_indicator
from ._sys_params import sys_params_ui


_FT_OPTIONS = [0.05, 0.25, 0.5, 0.75, 0.95]
_ALPHA_HOLD_OPTIONS = [0.0, 0.10, 0.15, 0.25, 0.40, 0.55, 0.70, 0.85, 1.00]


def _blast_params(alpha: float) -> dict:
    return {
        'q1': float(_Q0 + alpha * 0.33),
        'pw1': float(_PW0 - alpha * 0.40),
        'c1': float(_C0 - alpha * 0.80),
    }


@st.cache_data(show_spinner=False)
def s3_time_series(alpha_val: float, ft_val: float, sim_time: int,
                   sys_params: tuple = ()) -> dict:
    p = DEFAULT_PARAMS.copy()
    if sys_params:
        p.update(dict(sys_params))
    p.update(_blast_params(alpha_val))
    p['F_threshold'] = ft_val
    state = {k: np.float128(v) for k, v in FULL_INIT.items()}
    sys = DynamicalSystem(p, state, "dimensionalized")
    ts = sys.time_series_plot(time=sim_time)
    return {k: v.astype(np.float64) for k, v in ts.items()}


@st.cache_data(show_spinner=False)
def s3_bifurcation(a_min: float, a_max: float, resolution: int,
                   bif_time: int, burn_frac: float, ft_val: float,
                   sys_params: tuple = ()) -> tuple:
    a_sweep = np.linspace(a_min, a_max, resolution)
    burn = int(bif_time * burn_frac)
    ba_a, ba_S, ba_E = [], [], []
    for av in a_sweep:
        p = DEFAULT_PARAMS.copy()
        if sys_params:
            p.update(dict(sys_params))
        p.update(_blast_params(float(av)))
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
def s3_time_series_ft(alpha_hold: float, ft_val: float, sim_time: int,
                      sys_params: tuple = ()) -> dict:
    p = DEFAULT_PARAMS.copy()
    if sys_params:
        p.update(dict(sys_params))
    p.update(_blast_params(alpha_hold))
    p['F_threshold'] = ft_val
    state = {k: np.float128(v) for k, v in FULL_INIT.items()}
    sys = DynamicalSystem(p, state, "dimensionalized")
    ts = sys.time_series_plot(time=sim_time)
    return {k: v.astype(np.float64) for k, v in ts.items()}


@st.cache_data(show_spinner=False)
def s3_bifurcation_ft(alpha_hold: float, ft_min: float, ft_max: float,
                      resolution: int, bif_time: int, burn_frac: float,
                      sys_params: tuple = ()) -> tuple:
    ft_sweep = np.linspace(ft_min, ft_max, resolution)
    burn = int(bif_time * burn_frac)
    bf_f, bf_S, bf_E = [], [], []
    for ft in ft_sweep:
        p = DEFAULT_PARAMS.copy()
        if sys_params:
            p.update(dict(sys_params))
        p.update(_blast_params(alpha_hold))
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
def s3_baseline(sim_time: int, sys_params: tuple = ()) -> dict:
    """Baseline: no fraud (F=0, FP=0), standard parameters, no blast (alpha=0)."""
    p = DEFAULT_PARAMS.copy()
    if sys_params:
        p.update(dict(sys_params))
    p.update(_blast_params(0.0))
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


def scenario_3():
    status_slot = scenario_header("Scenario 3 — Blast / Cyanide Fishing")
    st.caption(
        "Destructive methods: q₁↑  pw₁↓  c₁↓↓ (cost drops much more than price). "
        "A single destruction intensity α ∈ [0, 1] jointly scales all three."
    )

    with st.expander("Analysis Parameters", expanded=False):
        colA, colB = st.columns(2, gap="large")

        with colA:
            st.markdown("#### vs α")
            st.markdown("**Time Series & Poincare**")
            s3_simA = st.slider("Time period", 100, 1000, 400, 50, key="s3_simA")
            s3_a_vals = st.multiselect(
                "α values", _ALPHA_HOLD_OPTIONS,
                default=[0.15, 0.40, 0.70, 1.00], key="s3_av",
            )
            s3_ft_A = st.selectbox(
                "F_threshold", _FT_OPTIONS,
                index=_FT_OPTIONS.index(0.5), key="s3_ftA",
            )
            st.markdown("**Bifurcation**")
            s3_bifA_iter = st.slider(
                "Iteration length", 100, 1000, 300, 50, key="s3_bifA_iter",
            )
            s3_resA = st.slider(
                "Resolution", 50, 500, 200, 50, key="s3_resA",
            )
            s3_rng = st.slider(
                "α range", 0.0, 1.0, (0.0, 1.0), 0.05, key="s3_rng",
            )

        with colB:
            st.markdown("#### vs F_threshold")
            st.markdown("**Time Series & Poincare**")
            s3_simB = st.slider("Time period", 100, 1000, 400, 50, key="s3_simB")
            s3_ft_vals = st.multiselect(
                "F_threshold values", _FT_OPTIONS,
                default=[0.25, 0.5, 0.75, 0.95], key="s3_ftv",
            )
            s3_a_hold = st.selectbox(
                "α (held)", _ALPHA_HOLD_OPTIONS,
                index=_ALPHA_HOLD_OPTIONS.index(0.55), key="s3_ahold",
            )
            st.markdown("**Bifurcation**")
            s3_bifB_iter = st.slider(
                "Iteration length", 100, 1000, 300, 50, key="s3_bifB_iter",
            )
            s3_resB = st.slider(
                "Resolution", 50, 500, 200, 50, key="s3_resB",
            )
            s3_ft_rng = st.slider(
                "F_threshold range", 0.0, 1.0, (0.1, 1.0), 0.05, key="s3_ftrng",
            )

    sys_t = sys_params_ui("s3")

    if not s3_a_vals:
        st.warning("Select at least one *α* value.")
        return
    if not s3_ft_vals:
        st.warning("Select at least one *F_threshold* value.")
        return

    s3_a_vals = sorted(s3_a_vals)
    s3_ft_vals = sorted(s3_ft_vals)
    _burnA = int(s3_simA * 0.6)
    _burnB = int(s3_simB * 0.6)

    bp_labels = [
        f'α={a}  (q₁={_blast_params(a)["q1"]:.2f}, '
        f'pw₁={_blast_params(a)["pw1"]:.2f}, '
        f'c₁={_blast_params(a)["c1"]:.2f})'
        for a in s3_a_vals
    ]
    hold_tag = (
        f'α={s3_a_hold}  '
        f'(q₁={_blast_params(s3_a_hold)["q1"]:.2f}, '
        f'pw₁={_blast_params(s3_a_hold)["pw1"]:.2f}, '
        f'c₁={_blast_params(s3_a_hold)["c1"]:.2f})'
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
            ts3 = {a: s3_time_series(float(a), float(s3_ft_A), s3_simA, sys_t) for a in s3_a_vals}
            t3_A = np.arange(s3_simA + 1)
            ts3_ft = {
                ft: s3_time_series_ft(float(s3_a_hold), float(ft), s3_simB, sys_t)
                for ft in s3_ft_vals
            }
            t3_B = np.arange(s3_simB + 1)
            s3_baseline_vals = s3_baseline(s3_simA, sys_t)

        hm_metrics = st.pills(
            "Heatmap rows", HEATMAP_METRICS, default=HEATMAP_METRICS,
            selection_mode="multi", key="s3_hm_m",
        )
        st.caption("Heatmap: % change vs. baseline (no fraud, no fraud perception, no blast)")
        tsA, tsB = st.tabs(["vs α", "vs F_threshold"])
        with tsA:
            fig = plot_ts_with_economics(
                ts3, t3_A, s3_a_vals, 'α',
                f'Blast Fishing — Time Series by Destruction Intensity   '
                f'(F_threshold={s3_ft_A}  |  q₁↑  pw₁↓  c₁↓↓)',
            )
            for i, lbl in enumerate(bp_labels):
                fig.layout.annotations[i].text = lbl
            st.plotly_chart(fig, width='stretch')
            if hm_metrics:
                hm = plot_ts_heatmap(ts3, s3_a_vals, 'α', hm_metrics, baseline_dict=s3_baseline_vals)
                if hm:
                    for hm_fig in hm:
                        st.plotly_chart(hm_fig, width='stretch')
        with tsB:
            fig = plot_ts_with_economics(
                ts3_ft, t3_B, s3_ft_vals, 'F_threshold',
                f'Blast Fishing — Time Series as F_threshold Increases   '
                f'(held {hold_tag})',
            )
            st.plotly_chart(fig, width='stretch')
            if hm_metrics:
                hm = plot_ts_heatmap(ts3_ft, s3_ft_vals, 'F_threshold', hm_metrics, baseline_dict=s3_baseline_vals)
                if hm:
                    for hm_fig in hm:
                        st.plotly_chart(hm_fig, width='stretch')

    with tab_bif:
        with status_indicator(status_slot, [
            "Computing bifurcation diagram (α sweep)",
            "Computing bifurcation diagram (F_threshold sweep)",
        ]):
            ba_a, ba_S, ba_E = s3_bifurcation(
                float(s3_rng[0]), float(s3_rng[1]), s3_resA, s3_bifA_iter, 0.6,
                float(s3_ft_A), sys_t,
            )
            bf_f, bf_S, bf_E = s3_bifurcation_ft(
                float(s3_a_hold), float(s3_ft_rng[0]), float(s3_ft_rng[1]),
                s3_resB, s3_bifB_iter, 0.6, sys_t,
            )

        bifA, bifB = st.tabs(["vs α", "vs F_threshold"])
        with bifA:
            fig = plot_bifurcation(
                ba_a, ba_S, ba_E,
                xlabel='Destruction Intensity (α)',
                title='Bifurcation over α   '
                      f'(F_threshold={s3_ft_A}  |  α=0 → honest  |  α=1 → q₁=0.40, pw₁=0.60, c₁=0.10)',
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
            ts3 = {a: s3_time_series(float(a), float(s3_ft_A), s3_simA, sys_t) for a in s3_a_vals}
            ts3_ft = {
                ft: s3_time_series_ft(float(s3_a_hold), float(ft), s3_simB, sys_t)
                for ft in s3_ft_vals
            }

        rmA, rmB = st.tabs(["vs α", "vs F_threshold"])
        with rmA:
            fig = plot_return_maps(ts3, s3_a_vals, 'α', _burnA)
            st.plotly_chart(fig, width='stretch')
        with rmB:
            fig = plot_return_maps(ts3_ft, s3_ft_vals, 'F_threshold', _burnB)
            st.plotly_chart(fig, width='stretch')
