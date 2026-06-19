import streamlit as st
import numpy as np
import plotly.graph_objects as go

from System import DynamicalSystem, DEFAULT_PARAMS
from scenarios.constants import FULL_INIT, _PW0, _C0, _Q0, COLORS4, ECON_COLORS, HARVEST_COLOR

_BURN_FRAC = 0.6
_SIM_TIME  = 400

_NULL_F  = 0.0
_NULL_FP = 0.0

_SCENARIO_DESCS = {
    'A':   'Fraud exists (no IUU, no perception)',
    'B':   'Fraud with blast fishing (no perception)',
    'C.1': 'Fraud exists, no IUU, buyers aware — low εd',
    'C.2': 'Fraud exists, no IUU, buyers aware — high εd',
    'D.1': 'Fraud with blast fishing, buyers aware — low εd',
    'D.2': 'Fraud with blast fishing, buyers aware — high εd',
}

_METRIC_MAP = {
    'H̄': 'Harvest',
    'P̄ᵐ': 'Market Price',
    'S̄': 'Seafood',
}


def _blast_params(alpha: float) -> dict:
    return {
        'q1': float(_Q0 + alpha * 0.33),
        'pw1': float(_PW0 - alpha * 0.40),
        'c1': float(_C0 - alpha * 0.80),
    }


@st.cache_data(show_spinner=False)
def _run_scenario(
    F_init: float, FP_init: float, alpha: float,
    r: float, F_threshold: float, e_d: float,
) -> dict:
    p = DEFAULT_PARAMS.copy()
    p.update({'r': r, 'F_threshold': F_threshold, 'e_d': e_d})
    p.update(_blast_params(alpha))
    state = {
        'S':  np.float128(FULL_INIT['S']),
        'E':  np.float128(FULL_INIT['E']),
        'F':  np.float128(F_init),
        'FP': np.float128(FP_init),
    }
    sys = DynamicalSystem(p, state, "dimensionalized")
    ts = sys.time_series_plot(time=_SIM_TIME)
    burn = int(_SIM_TIME * _BURN_FRAC)
    return {
        'Harvest':      float(np.mean(ts['Harvest'][burn:])),
        'Market Price': float(np.mean(ts['Market Price'][burn:])),
        'Seafood':      float(np.mean(ts['Seafood'][burn:])),
    }


def _pct_change(sc_val: float, null_val: float) -> float:
    if null_val != 0:
        return (sc_val - null_val) / abs(null_val) * 100
    return 0.0


@st.cache_data(show_spinner=False)
def _run_scenario_ts(
    F_init: float, FP_init: float, alpha: float,
    r: float, F_threshold: float, e_d: float,
) -> dict:
    p = DEFAULT_PARAMS.copy()
    p.update({'r': r, 'F_threshold': F_threshold, 'e_d': e_d})
    p.update(_blast_params(alpha))
    state = {
        'S':  np.float128(FULL_INIT['S']),
        'E':  np.float128(FULL_INIT['E']),
        'F':  np.float128(F_init),
        'FP': np.float128(FP_init),
    }
    sys = DynamicalSystem(p, state, "dimensionalized")
    ts = sys.time_series_plot(time=_SIM_TIME)
    burn = int(_SIM_TIME * _BURN_FRAC)
    return {
        'S':  [float(v) for v in ts['Seafood'][burn:]],
        'H':  [float(v) for v in ts['Harvest'][burn:]],
        'Pm': [float(v) for v in ts['Market Price'][burn:]],
    }


def _ts_fig(label: str, ts: dict) -> go.Figure:
    t = list(range(len(ts['S'])))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=ts['S'],  name='S',  line=dict(color=COLORS4['S'],     width=1.5)))
    fig.add_trace(go.Scatter(x=t, y=ts['H'],  name='H',  line=dict(color=HARVEST_COLOR,     width=1.5)))
    fig.add_trace(go.Scatter(x=t, y=ts['Pm'], name='Pᵐ', line=dict(color=ECON_COLORS['Pm'], width=1.5)))
    fig.update_layout(
        title=dict(text=f"Scenario {label}", font=dict(size=13), x=0.5),
        height=240,
        margin=dict(l=40, r=10, t=40, b=30),
        legend=dict(
            orientation='h', yanchor='bottom', y=1.02,
            xanchor='left', x=0, font=dict(size=10),
        ),
        xaxis=dict(title='t', showgrid=False, tickfont=dict(size=10)),
        yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.08)', tickfont=dict(size=10)),
        plot_bgcolor='white',
        paper_bgcolor='white',
    )
    return fig


@st.fragment
def hypothesis_tab():
    st.markdown("### Hypothesis — Blast Fishing Scenarios vs. Null")
    st.caption(
        "**Null (0):** Pure bioeconomic baseline — no fraud (F = 0), no perception (FP = 0).  \n"
        "Green = scenario average **exceeds** null &nbsp;|&nbsp; "
        "Red = scenario average **falls below** null."
    )

    selected = st.pills(
        "Metrics",
        list(_METRIC_MAP.keys()),
        default=list(_METRIC_MAP.keys()),
        selection_mode="multi",
        key="hyp_metrics",
    )
    if not selected:
        st.warning("Select at least one metric.")
        return

    with st.expander("Scenario definitions & parameters", expanded=False):
        col_p, col_t = st.columns([1, 1.5], gap="large")

        with col_p:
            st.markdown("#### Parameters")
            r           = st.slider("r (growth rate)",           0.05, 0.50, float(DEFAULT_PARAMS['r']),           0.005, key="hyp_r")
            F_threshold = st.slider("F̂ (detection threshold)",   0.05, 0.95, float(DEFAULT_PARAMS['F_threshold']), 0.05,  key="hyp_ft")
            alpha_low   = st.slider("α low  (A, C.1, C.2)",      0.0,  1.0,  0.0,                                  0.05,  key="hyp_a_low")
            alpha_high  = st.slider("α high (B, D.1, D.2)",      0.0,  1.0,  0.5,                                  0.05,  key="hyp_a_high")
            e_d_low     = st.slider("εd low  (C.1, D.1)",        0.1,  2.0,  0.25,                                 0.05,  key="hyp_ed_low")
            e_d_high    = st.slider("εd high (C.2, D.2)",        0.1,  3.0,  1.25,                                 0.05,  key="hyp_ed_high")

        with col_t:
            st.markdown("#### Scenario definitions")
            _e_d_def = float(DEFAULT_PARAMS['e_d'])
            st.markdown(
                "| Scenario | F₀ | FP₀ | α | εd | Description |\n"
                "|---|---|---|---|---|---|\n"
                f"| 0 (Null) | 0.00 | 0.00 | — | {_e_d_def} | Pure bioeconomic baseline — no fraud |\n"
                f"| A   | {FULL_INIT['F']:.2f} | 0.00 | {alpha_low:.2f}  | {_e_d_def} | {_SCENARIO_DESCS['A']} |\n"
                f"| B   | {FULL_INIT['F']:.2f} | 0.00 | {alpha_high:.2f} | {_e_d_def} | {_SCENARIO_DESCS['B']} |\n"
                f"| C.1 | {FULL_INIT['F']:.2f} | {FULL_INIT['FP']:.2f} | {alpha_low:.2f}  | {e_d_low:.2f}  | {_SCENARIO_DESCS['C.1']} |\n"
                f"| C.2 | {FULL_INIT['F']:.2f} | {FULL_INIT['FP']:.2f} | {alpha_low:.2f}  | {e_d_high:.2f} | {_SCENARIO_DESCS['C.2']} |\n"
                f"| D.1 | {FULL_INIT['F']:.2f} | {FULL_INIT['FP']:.2f} | {alpha_high:.2f} | {e_d_low:.2f}  | {_SCENARIO_DESCS['D.1']} |\n"
                f"| D.2 | {FULL_INIT['F']:.2f} | {FULL_INIT['FP']:.2f} | {alpha_high:.2f} | {e_d_high:.2f} | {_SCENARIO_DESCS['D.2']} |\n"
            )

    _e_d_default = float(DEFAULT_PARAMS['e_d'])
    _SCENARIO_DEFS = {
        'A':   {'F': FULL_INIT['F'], 'FP': 0.0,             'alpha': alpha_low,  'e_d': _e_d_default},
        'B':   {'F': FULL_INIT['F'], 'FP': 0.0,             'alpha': alpha_high, 'e_d': _e_d_default},
        'C.1': {'F': FULL_INIT['F'], 'FP': FULL_INIT['FP'], 'alpha': alpha_low,  'e_d': e_d_low},
        'C.2': {'F': FULL_INIT['F'], 'FP': FULL_INIT['FP'], 'alpha': alpha_low,  'e_d': e_d_high},
        'D.1': {'F': FULL_INIT['F'], 'FP': FULL_INIT['FP'], 'alpha': alpha_high, 'e_d': e_d_low},
        'D.2': {'F': FULL_INIT['F'], 'FP': FULL_INIT['FP'], 'alpha': alpha_high, 'e_d': e_d_high},
    }

    with st.spinner("Running simulations…"):
        null_avgs = _run_scenario(_NULL_F, _NULL_FP, 0.0, r, F_threshold, _e_d_default)
        scenario_avgs = {
            sc: _run_scenario(
                float(defn['F']), float(defn['FP']), float(defn['alpha']),
                r, F_threshold, float(defn['e_d']),
            )
            for sc, defn in _SCENARIO_DEFS.items()
        }

    scenario_labels = list(_SCENARIO_DEFS.keys())
    ts_keys = [_METRIC_MAP[m] for m in selected]

    z, text = [], []
    for sc in scenario_labels:
        row_z, row_t = [], []
        for mkey in ts_keys:
            pct = _pct_change(scenario_avgs[sc][mkey], null_avgs[mkey])
            row_z.append(pct)
            row_t.append(f"{pct:+.1f}%")
        z.append(row_z)
        text.append(row_t)

    flat = [v for row in z for v in row]
    abs_max = max(abs(v) for v in flat) if flat else 1.0
    abs_max = abs_max or 1.0

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=selected,
        y=scenario_labels,
        colorscale=[[0.0, '#228B22'], [0.5, '#FFFFFF'], [1.0, '#DC143C']],
        zmin=-abs_max,
        zmax=abs_max,
        text=text,
        texttemplate="%{text}",
        textfont=dict(size=14),
        showscale=True,
        colorbar=dict(
            title=dict(text="% vs null", side="right"),
            ticksuffix="%",
        ),
        hovertemplate="Scenario %{y} | %{x}<br>%{text} vs null<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text="Average Outcomes — Scenario vs. Null Hypothesis", font=dict(size=16)),
        xaxis=dict(title="Metric", side="top"),
        yaxis=dict(title="Scenario", autorange="reversed"),
        height=460,
        margin=dict(l=80, r=80, t=100, b=40),
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Time Series Comparison")
    st.caption(
        "Post-burn-in dynamics for selected scenarios.  \n"
        "**S** = seafood biomass &nbsp;|&nbsp; **H** = harvest &nbsp;|&nbsp; **Pᵐ** = market price"
    )

    _TS_LABELS = ['Null (0)', 'A', 'B', 'C.1', 'C.2', 'D.1', 'D.2']
    selected_ts = st.pills(
        "Scenarios",
        _TS_LABELS,
        default=['Null (0)'],
        selection_mode="multi",
        key="hyp_ts_sc",
    )

    if not selected_ts:
        st.warning("Select at least one scenario.")
    else:
        _ts_params = {
            'Null (0)': {'F': _NULL_F, 'FP': _NULL_FP, 'alpha': 0.0, 'e_d': _e_d_default},
            **{
                sc: {
                    'F': float(d['F']), 'FP': float(d['FP']),
                    'alpha': float(d['alpha']), 'e_d': float(d['e_d']),
                }
                for sc, d in _SCENARIO_DEFS.items()
            },
        }

        _MAX_PER_ROW = 4
        for row_start in range(0, len(selected_ts), _MAX_PER_ROW):
            chunk = selected_ts[row_start:row_start + _MAX_PER_ROW]
            cols = st.columns(len(chunk))
            for col, sc_label in zip(cols, chunk):
                with col:
                    p = _ts_params[sc_label]
                    ts_data = _run_scenario_ts(
                        p['F'], p['FP'], p['alpha'],
                        r, F_threshold, p['e_d'],
                    )
                    st.plotly_chart(_ts_fig(sc_label, ts_data), use_container_width=True)

    with st.expander("Average values (long-term, for equilibrium or limit cycle)"):
        header   = "| Scenario | Avg Harvest | Avg Market Price | Avg Seafood |"
        sep      = "|---|---|---|---|"
        null_row = (
            f"| 0 (Null) | {null_avgs['Harvest']:.4f} "
            f"| {null_avgs['Market Price']:.4f} "
            f"| {null_avgs['Seafood']:.4f} |"
        )
        sc_rows = [
            f"| {sc} | {scenario_avgs[sc]['Harvest']:.4f} "
            f"| {scenario_avgs[sc]['Market Price']:.4f} "
            f"| {scenario_avgs[sc]['Seafood']:.4f} |"
            for sc in scenario_labels
        ]
        st.markdown("\n".join([header, sep, null_row] + sc_rows))
