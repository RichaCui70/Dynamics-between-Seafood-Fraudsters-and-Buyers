import streamlit as st
from System import DEFAULT_PARAMS

_GROUPS = [
    ("Response speeds", [
        ('gamma_s',  'γS  (seafood response speed)',     0.1,  5.0,  0.05),
        ('gamma_e',  'γE  (effort response speed)',      0.01, 1.0,  0.005),
        ('gamma_f',  'γF  (fraudster response speed)',   0.1,  5.0,  0.05),
        ('gamma_fp', 'γFP (perception response speed)',  0.5,  20.0, 0.5),
        ('r',        'r   (intrinsic growth rate)',      0.01, 1.0,  0.005),
        ('K',        'K   (carrying capacity)',          0.1,  5.0,  0.1),
    ]),
    ("Market", [
        ('gamma_m', 'γM  (market price scaling)',       1.0,  50.0, 0.5),
        ('gamma_p', 'γP  (wholesale price scaling)',    0.1,  5.0,  0.05),
        ('e_d',     'εd  (demand elasticity)',          0.1,  3.0,  0.05),
        ('e_sw',    'εsw (wholesale supply elast.)',    0.1,  2.0,  0.05),
        ('e_sm',    'εsm (market supply elast.)',       0.1,  2.0,  0.05),
    ]),
    ("Fraud economics", [
        ('q0',          'q₀  (catchability, no fraud)',    0.01, 0.5,  0.01),
        ('q1',          'q₁  (catchability, full fraud)',  0.01, 0.5,  0.01),
        ('pw0',         'pw₀ (wholesale price, honest)',   0.1,  5.0,  0.05),
        ('pw1',         'pw₁ (wholesale price, fraud)',    0.1,  5.0,  0.05),
        ('c0',          'c₀  (fishing cost, honest)',      0.1,  3.0,  0.05),
        ('c1',          'c₁  (fishing cost, full fraud)',  0.01, 2.0,  0.01),
        ('F_threshold', 'F̂   (detection threshold)',       0.05, 0.95, 0.05),
    ]),
]


def sys_params_ui(prefix: str) -> tuple:
    """Render System Parameters expander; return sorted (key, val) tuple for cache keying."""
    vals = {}
    with st.expander("System Parameters", expanded=False):
        cols = st.columns(3, gap="large")
        for col, (group_name, params) in zip(cols, _GROUPS):
            with col:
                st.markdown(f"**{group_name}**")
                for key, label, lo, hi, step in params:
                    vals[key] = st.slider(
                        label, lo, hi, float(DEFAULT_PARAMS[key]), step,
                        key=f"{prefix}_{key}",
                    )
    return tuple(sorted(vals.items()))
