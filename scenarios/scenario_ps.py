import streamlit as st
import numpy as np
import plotly.graph_objects as go
from core.System import DynamicalSystem

from core.constants import DEFAULT_INIT_STATE, DEFAULT_PARAMS
from core.plots import plot_4var_ts, plot_ts_with_economics, plot_bifurcation, plot_return_maps, plot_ts_heatmap, HEATMAP_METRICS
from ._status import scenario_header, status_indicator
from ._sys_params import sys_params_ui


_FT_OPTIONS = [0.05, 0.25, 0.5, 0.75, 0.95]
_ALPHA_HOLD_OPTIONS = [0.0, 0.10, 0.15, 0.25, 0.40, 0.55, 0.70, 0.85, 1.00]

_PS_SCALAR = 4.0


def _prized_params(alpha: float) -> dict:
    return {
        'pw1': float(DEFAULT_PARAMS['pw0'] + alpha * _PS_SCALAR),
        'c1': DEFAULT_PARAMS['c0'],
        'q1': DEFAULT_PARAMS['q0'],
    }


@st.cache_data(show_spinner=False)
def ps_time_series(alpha_val: float, ft_val: float, sim_time: int,
                   sys_params: tuple = ()) -> dict:
    p = DEFAULT_PARAMS.copy()
    if sys_params:
        p.update(dict(sys_params))
    p.update(_prized_params(alpha_val))
    p['F_threshold'] = ft_val
    state = {k: np.float128(v) for k, v in DEFAULT_INIT_STATE.items()}
    sys = DynamicalSystem(p, state, "dimensionalized")
    ts = sys.time_series_plot(time=sim_time)
    return {k: v.astype(np.float64) for k, v in ts.items()}


@st.cache_data(show_spinner=False)
def ps_bifurcation(a_min: float, a_max: float, resolution: int,
                   bif_time: int, burn_frac: float, ft_val: float,
                   sys_params: tuple = ()) -> tuple:
    a_sweep = np.linspace(a_min, a_max, resolution)
    burn = int(bif_time * burn_frac)
    bp_a, bp_S, bp_E, bp_F, bp_FP = [], [], [], [], []
    for av in a_sweep:
        p = DEFAULT_PARAMS.copy()
        if sys_params:
            p.update(dict(sys_params))
        p.update(_prized_params(float(av)))
        p['F_threshold'] = ft_val
        state = {k: np.float128(v) for k, v in DEFAULT_INIT_STATE.items()}
        sys = DynamicalSystem(p, state, "dimensionalized")
        ts = sys.time_series_plot(time=bif_time)
        s_att = ts['Seafood'][burn:].astype(np.float64)
        e_att = ts['Effort'][burn:].astype(np.float64)
        f_att = ts['Fraudsters'][burn:].astype(np.float64)
        fp_att = ts['Perception of Fraud'][burn:].astype(np.float64)
        n = len(s_att)
        bp_a.extend([float(av)] * n)
        bp_S.extend(s_att.tolist())
        bp_E.extend(e_att.tolist())
        bp_F.extend(f_att.tolist())
        bp_FP.extend(fp_att.tolist())
    return np.array(bp_a), np.array(bp_S), np.array(bp_E), np.array(bp_F), np.array(bp_FP)


@st.cache_data(show_spinner=False)
def ps_time_series_ft(alpha_hold: float, ft_val: float, sim_time: int,
                      sys_params: tuple = ()) -> dict:
    p = DEFAULT_PARAMS.copy()
    if sys_params:
        p.update(dict(sys_params))
    p.update(_prized_params(alpha_hold))
    p['F_threshold'] = ft_val
    state = {k: np.float128(v) for k, v in DEFAULT_INIT_STATE.items()}
    sys = DynamicalSystem(p, state, "dimensionalized")
    ts = sys.time_series_plot(time=sim_time)
    return {k: v.astype(np.float64) for k, v in ts.items()}


@st.cache_data(show_spinner=False)
def ps_bifurcation_ft(alpha_hold: float, ft_min: float, ft_max: float,
                      resolution: int, bif_time: int, burn_frac: float,
                      sys_params: tuple = ()) -> tuple:
    ft_sweep = np.linspace(ft_min, ft_max, resolution)
    burn = int(bif_time * burn_frac)
    bf_f, bf_S, bf_E, bf_F, bf_FP = [], [], [], [], []
    for ft in ft_sweep:
        p = DEFAULT_PARAMS.copy()
        if sys_params:
            p.update(dict(sys_params))
        p.update({
            **_prized_params(alpha_hold), 'F_threshold': float(ft),
        })
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
def ps_spectral_sweep(a_min: float, a_max: float, resolution: int,
                      sys_params: tuple = ()) -> tuple:
    a_vals = np.linspace(a_min, a_max, resolution)
    rho_vals = np.empty(resolution)
    for i, av in enumerate(a_vals):
        p = DEFAULT_PARAMS.copy()
        if sys_params:
            p.update(dict(sys_params))
        p.update(_prized_params(float(av)))
        state = {k: np.float128(v) for k, v in DEFAULT_INIT_STATE.items()}
        sys = DynamicalSystem(p, state, "dimensionalized")
        result = sys.stability_analysis()
        rho_vals[i] = result['spectral_radius']
    return a_vals.astype(np.float64), rho_vals.astype(np.float64)


@st.cache_data(show_spinner=False)
def ps_baseline(sim_time: int, sys_params: tuple = ()) -> dict:
    """Baseline: no fraud (F=0, FP=0), standard parameters, no premium (alpha=0)."""
    p = DEFAULT_PARAMS.copy()
    if sys_params:
        p.update(dict(sys_params))
    p.update(_prized_params(0.0))
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
def ps_baseline_ts(sim_time: int, sys_params: tuple = ()) -> dict:
    """Full time series at alpha=0, F=0, FP=0 for the fixed baseline column."""
    p = DEFAULT_PARAMS.copy()
    if sys_params:
        p.update(dict(sys_params))
    p.update(_prized_params(0.0))
    p.update({'F_threshold': 0.5})
    state = {k: np.float128(v) for k, v in DEFAULT_INIT_STATE.items()}
    state['F'] = np.float128(0.0)
    state['FP'] = np.float128(0.0)
    sys = DynamicalSystem(p, state, "dimensionalized")
    ts = sys.time_series_plot(time=sim_time)
    return {k: v.astype(np.float64) for k, v in ts.items()}


def scenario_ps():
    status_slot = scenario_header("Scenario 2 — Prized / Protected Seafood")
    st.caption(
        f"α drives a price premium for protected species: pw₁ = pw₀ + α·{_PS_SCALAR:.0f}. "
        f"Same gear: c₁ = c₀, q₁ = q₀. Focus parameter: α."
    )

    with st.expander("Analysis Parameters", expanded=False):
        colA, colB = st.columns(2, gap="large")

        with colA:
            st.markdown("#### vs α")
            st.markdown("**Time Series & Poincare**")
            ps_simA = st.slider("Time period", 100, 1000, 400, 50, key="ps_simA")
            ps_a_vals = st.multiselect(
                "α values", _ALPHA_HOLD_OPTIONS,
                default=[0.15, 0.40, 0.70, 1.00], key="ps_a",
            )
            ps_ft_A = st.selectbox(
                "F_threshold", _FT_OPTIONS,
                index=_FT_OPTIONS.index(0.5), key="ps_ftA",
            )
            st.markdown("**Bifurcation**")
            ps_bifA_iter = st.slider(
                "Iteration length", 100, 1000, 300, 50, key="ps_bifA_iter",
            )
            ps_resA = st.slider(
                "Resolution", 50, 500, 200, 50, key="ps_resA",
            )
            ps_rng = st.slider(
                "α range", 0.0, 1.0, (0.0, 1.0), 0.05,
                key="ps_rng",
            )

        with colB:
            st.markdown("#### vs F_threshold")
            st.markdown("**Time Series & Poincare**")
            ps_simB = st.slider("Time period", 100, 1000, 400, 50, key="ps_simB")
            ps_ft_vals = st.multiselect(
                "F_threshold values", _FT_OPTIONS,
                default=[0.25, 0.5, 0.75, 0.95], key="ps_ftv",
            )
            ps_a_hold = st.selectbox(
                "α (held)", _ALPHA_HOLD_OPTIONS,
                index=_ALPHA_HOLD_OPTIONS.index(0.40), key="ps_a_hold",
            )
            st.markdown("**Bifurcation**")
            ps_bifB_iter = st.slider(
                "Iteration length", 100, 1000, 300, 50, key="ps_bifB_iter",
            )
            ps_resB = st.slider(
                "Resolution", 50, 500, 200, 50, key="ps_resB",
            )
            ps_ft_rng = st.slider(
                "F_threshold range", 0.0, 1.0, (0.1, 1.0), 0.05, key="ps_ftrng",
            )

    sys_t = sys_params_ui("ps", exclude={'pw1'})

    if not ps_a_vals:
        st.warning("Select at least one *α* value.")
        return
    if not ps_ft_vals:
        st.warning("Select at least one *F_threshold* value.")
        return

    ps_a_vals = sorted(ps_a_vals)
    ps_ft_vals = sorted(ps_ft_vals)
    _burnA = int(ps_simA * 0.6)
    _burnB = int(ps_simB * 0.6)

    tab_ts, tab_bif, tab_rm, tab_stab = st.tabs(
        ["Time Series", "Bifurcation", "Poincare", "Stability"]
    )

    with tab_ts:
        with status_indicator(status_slot, [
            "Running time-series simulations (α sweep)",
            "Running time-series simulations (F_threshold sweep)",
            "Computing baseline (no fraud)",
        ]):
            ts2 = {a: ps_time_series(float(a), float(ps_ft_A), ps_simA, sys_t) for a in ps_a_vals}
            ts_bl = ps_baseline_ts(ps_simA, sys_t)
            t2_A = np.arange(ps_simA + 1)
            ts2_ft = {
                ft: ps_time_series_ft(float(ps_a_hold), float(ft), ps_simB, sys_t)
                for ft in ps_ft_vals
            }
            t2_B = np.arange(ps_simB + 1)
            ps_baseline_vals = ps_baseline(ps_simA, sys_t)

        # Build display labels: show derived pw₁ alongside α
        ps_col_labels = [
            f'α={a}  (pw₁={_prized_params(a)["pw1"]:.2f})'
            for a in ps_a_vals
        ]
        hold_tag = (
            f'α={ps_a_hold}  '
            f'(pw₁={_prized_params(ps_a_hold)["pw1"]:.2f})'
        )

        hm_metrics = st.pills(
            "Heatmap rows", HEATMAP_METRICS, default=HEATMAP_METRICS,
            selection_mode="multi", key="ps_hm_m",
        )
        st.caption("Heatmap: % change vs. baseline (no fraud, no fraud perception)")
        tsA, tsB = st.tabs(["vs α", "vs F_threshold"])
        with tsA:
            _ts_full = {'Baseline': ts_bl, **ts2}
            _vals_full = ['Baseline'] + ps_a_vals
            _all_labels = ['Baseline (α=0, F=FP=0)'] + ps_col_labels
            fig = plot_ts_with_economics(
                _ts_full, t2_A, _vals_full, 'α',
                f'Prized Seafood — Time Series as α Increases   '
                f'(c₁=c₀={DEFAULT_PARAMS['c0']},  q₁=q₀={DEFAULT_PARAMS['q0']},  F_threshold={ps_ft_A})',
            )
            for i, lbl in enumerate(_all_labels):
                fig.layout.annotations[i].text = lbl
            st.plotly_chart(fig, width='stretch')
            if hm_metrics:
                hm = plot_ts_heatmap(_ts_full, _vals_full, 'α', hm_metrics, baseline_dict=ps_baseline_vals)
                if hm:
                    for hm_fig in hm:
                        st.plotly_chart(hm_fig, width='stretch')
        with tsB:
            ts_bl_B = ps_baseline_ts(ps_simB, sys_t)
            _ts_full_ft = {'Baseline': ts_bl_B, **ts2_ft}
            _vals_full_ft = ['Baseline'] + ps_ft_vals
            _ft_labels = ['Baseline (F=FP=0)'] + [str(ft) for ft in ps_ft_vals]
            fig = plot_ts_with_economics(
                _ts_full_ft, t2_B, _vals_full_ft, 'F_threshold',
                f'Prized Seafood — Time Series as F_threshold Increases   '
                f'(held {hold_tag},  c₁=c₀={DEFAULT_PARAMS['c0']},  q₁=q₀={DEFAULT_PARAMS['q0']})',
            )
            for i, lbl in enumerate(_ft_labels):
                fig.layout.annotations[i].text = lbl
            st.plotly_chart(fig, width='stretch')
            if hm_metrics:
                hm = plot_ts_heatmap(_ts_full_ft, _vals_full_ft, 'F_threshold', hm_metrics, baseline_dict=ps_baseline_vals)
                if hm:
                    for hm_fig in hm:
                        st.plotly_chart(hm_fig, width='stretch')

    with tab_bif:
        with status_indicator(status_slot, [
            "Computing bifurcation diagram (α sweep)",
            "Computing bifurcation diagram (F_threshold sweep)",
        ]):
            bp_a, bp_S, bp_E, bp_F, bp_FP = ps_bifurcation(
                float(ps_rng[0]), float(ps_rng[1]), ps_resA, ps_bifA_iter, 0.6,
                float(ps_ft_A), sys_t,
            )
            bf_f, bf_S, bf_E, bf_F, bf_FP = ps_bifurcation_ft(
                float(ps_a_hold), float(ps_ft_rng[0]), float(ps_ft_rng[1]),
                ps_resB, ps_bifB_iter, 0.6, sys_t,
            )

        bifA, bifB = st.tabs(["vs α", "vs F_threshold"])
        with bifA:
            fig = plot_bifurcation(
                bp_a, bp_S, bp_E, bp_F, bp_FP,
                xlabel='α (price premium intensity)',
                title=f'Bifurcation Diagram over α   (F_threshold={ps_ft_A})',
                vline_x=0.0, vline_label='α = 0 (no premium)',
            )
            st.plotly_chart(fig, width='stretch')
        with bifB:
            fig = plot_bifurcation(
                bf_f, bf_S, bf_E, bf_F, bf_FP,
                xlabel='F_threshold',
                title=f'Bifurcation Diagram over F_threshold   (held {hold_tag})',
            )
            st.plotly_chart(fig, width='stretch')

    with tab_rm:
        with status_indicator(status_slot, [
            "Running time-series simulations (α sweep)",
            "Running time-series simulations (F_threshold sweep)",
        ]):
            ts2 = {a: ps_time_series(float(a), float(ps_ft_A), ps_simA, sys_t) for a in ps_a_vals}
            ts_bl = ps_baseline_ts(ps_simA, sys_t)
            ts2_ft = {
                ft: ps_time_series_ft(float(ps_a_hold), float(ft), ps_simB, sys_t)
                for ft in ps_ft_vals
            }

        rmA, rmB = st.tabs(["vs α", "vs F_threshold"])
        with rmA:
            _ts_full = {'Baseline': ts_bl, **ts2}
            _vals_full = ['Baseline'] + ps_a_vals
            _all_labels = ['Baseline (α=0, F=FP=0)'] + ps_col_labels
            fig = plot_return_maps(_ts_full, _vals_full, 'α', _burnA)
            for i, lbl in enumerate(_all_labels):
                fig.layout.annotations[i].text = lbl
            st.plotly_chart(fig, width='stretch')
        with rmB:
            ts_bl_B = ps_baseline_ts(ps_simB, sys_t)
            _ts_full_ft = {'Baseline': ts_bl_B, **ts2_ft}
            _vals_full_ft = ['Baseline'] + ps_ft_vals
            _ft_labels = ['Baseline (F=FP=0)'] + [str(ft) for ft in ps_ft_vals]
            fig = plot_return_maps(_ts_full_ft, _vals_full_ft, 'F_threshold', _burnB)
            for i, lbl in enumerate(_ft_labels):
                fig.layout.annotations[i].text = lbl
            st.plotly_chart(fig, width='stretch')

    with tab_stab:
        with status_indicator(status_slot, ["Computing stability sweep"]):
            ps_a_sweep, ps_rho = ps_spectral_sweep(0.0, 1.0, 100, sys_t)

        finite = np.isfinite(ps_rho)
        a_fin, rho_fin = ps_a_sweep[finite], ps_rho[finite]
        stable_mask = rho_fin < 1.0
        y_cap = max(float(np.max(rho_fin[rho_fin < 50])) * 1.1, 2.0) if np.any(rho_fin < 50) else 5.0
        rho_plot = np.clip(rho_fin, 0, y_cap)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=a_fin[stable_mask], y=rho_plot[stable_mask],
            mode='markers', marker=dict(color='#2E8B57', size=6),
            name='Stable (ρ < 1)',
        ))
        fig.add_trace(go.Scatter(
            x=a_fin[~stable_mask], y=rho_plot[~stable_mask],
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
                f'(c₁=c₀={DEFAULT_PARAMS['c0']},  q₁=q₀={DEFAULT_PARAMS['q0']},  F_threshold={DEFAULT_PARAMS["F_threshold"]})'
            ),
            xaxis_title='Destruction Intensity (α)',
            yaxis_title='Spectral Radius  ρ = max|λᵢ|',
            yaxis_range=[0, y_cap],
            margin=dict(t=60, b=40),
            legend=dict(yanchor='top', y=0.99, xanchor='right', x=0.99),
        )
        st.plotly_chart(fig, width='stretch')
