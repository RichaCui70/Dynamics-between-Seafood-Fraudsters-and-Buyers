import streamlit as st
import numpy as np
import plotly.graph_objects as go

from System import DynamicalSystem, DEFAULT_PARAMS
from scenarios.constants import FULL_INIT, _PW0, _C0, _Q0

_BURN_FRAC = 0.6
_SIM_TIME  = 400

_NULL_F  = 0.0
_NULL_FP = 0.0

_SCENARIO_DESCS = {
    'A': 'Fraud exists (no destruction, no perception)',
    'B': 'Fraud with blast fishing (no perception)',
    'C': 'Fraud exists (no destruction, buyers aware)',
    'D': 'Fraud with blast fishing, buyers aware',
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
    r: float, F_threshold: float,
) -> dict:
    p = DEFAULT_PARAMS.copy()
    p.update({'r': r, 'F_threshold': F_threshold})
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
            r           = st.slider("r (growth rate)",          0.05, 0.50, float(DEFAULT_PARAMS['r']),           0.005, key="hyp_r")
            F_threshold = st.slider("F̂ (detection threshold)",  0.05, 0.95, float(DEFAULT_PARAMS['F_threshold']), 0.05,  key="hyp_ft")
            alpha_low   = st.slider("α_low  (scenarios A, C)",  0.0,  1.0,  0.0,                                  0.05,  key="hyp_a_low")
            alpha_high  = st.slider("α_high (scenarios B, D)",  0.0,  1.0,  0.5,                                  0.05,  key="hyp_a_high")

        with col_t:
            st.markdown("#### Scenario definitions")
            st.markdown(
                "| Scenario | F₀ | FP₀ | α | Description |\n"
                "|---|---|---|---|---|\n"
                f"| 0 (Null) | 0.00 | 0.00 | — | Pure bioeconomic baseline — no fraud |\n"
                f"| A | {FULL_INIT['F']:.2f} | 0.00 | {alpha_low:.2f} | {_SCENARIO_DESCS['A']} |\n"
                f"| B | {FULL_INIT['F']:.2f} | 0.00 | {alpha_high:.2f} | {_SCENARIO_DESCS['B']} |\n"
                f"| C | {FULL_INIT['F']:.2f} | {FULL_INIT['FP']:.2f} | {alpha_low:.2f} | {_SCENARIO_DESCS['C']} |\n"
                f"| D | {FULL_INIT['F']:.2f} | {FULL_INIT['FP']:.2f} | {alpha_high:.2f} | {_SCENARIO_DESCS['D']} |\n"
            )

    _SCENARIO_DEFS = {
        'A': {'F': FULL_INIT['F'], 'FP': 0.0,             'alpha': alpha_low},
        'B': {'F': FULL_INIT['F'], 'FP': 0.0,             'alpha': alpha_high},
        'C': {'F': FULL_INIT['F'], 'FP': FULL_INIT['FP'], 'alpha': alpha_low},
        'D': {'F': FULL_INIT['F'], 'FP': FULL_INIT['FP'], 'alpha': alpha_high},
    }

    with st.spinner("Running simulations…"):
        null_avgs = _run_scenario(_NULL_F, _NULL_FP, 0.0, r, F_threshold)
        scenario_avgs = {
            sc: _run_scenario(
                float(defn['F']), float(defn['FP']), float(defn['alpha']),
                r, F_threshold,
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
        colorscale=[[0.0, '#DC143C'], [0.5, '#FFFFFF'], [1.0, '#228B22']],
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
        height=380,
        margin=dict(l=80, r=80, t=100, b=40),
    )

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Average values (post burn-in)"):
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
