import streamlit as st
from core.constants import DEFAULT_INIT_STATE, DEFAULT_PARAMS

PARAM_GROUPS = [
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
        ('e_d',     'εd  (demand elasticity)',          0.0001,  3.0,  0.05),
        ('e_sw',    'εsw (wholesale supply elast.)',    0.1,  2.0,  0.05),
        ('e_sm',    'εsm (market supply elast.)',       0.1,  2.0,  0.05),
        ('FP_min',      'FP_min (min fraud perception)',       0.0,  1.0,  0.05),
        ('FP_max',      'FP_max (max fraud perception)',       0.0,  1.0,  0.05),
    ]),
    ("Fraud economics", [
        ('q0',          'q₀  (catchability, no fraud)',    0.01, 0.5,  0.01),
        ('q1',          'q₁  (catchability, full fraud)',  0.01, 0.5,  0.01),
        ('pw0',         'pw₀ (wholesale price, honest)',   0.1,  5.0,  0.05),
        ('pw1',         'pw₁ (wholesale price, fraud)',    0.1,  5.0,  0.05),
        ('c0',          'c₀  (fishing cost, honest)',      0.1,  3.0,  0.05),
        ('c1',          'c₁  (fishing cost, full fraud)',  0.01, 2.0,  0.01),
        ('F_threshold', 'F̂   (detection threshold)',       0.05, 0.95, 0.05),
        ('F_min',       'F_min (min fraudster share)',       0.0,  1.0,  0.05),
        ('F_max',       'F_max (max fraudster share)',       0.0,  1.0,  0.05),
    ]),
    ("Starting values", [
        ('S0',  'S₀  (starting stock)',             0.0, 1.0, 0.01),
        ('E0',  'E₀  (starting effort)',            0.0, 5.0, 0.01),
        ('F0',  'F₀  (starting fraudsters)',        0.0, 1.0, 0.01),
        ('FP0', 'FP₀ (starting fraud perception)',  0.0, 1.0, 0.01),
    ]),
]


def initial_state_from_overrides(
    system_param_overrides: tuple,
    defaults: dict | None = None,
) -> dict:
    """Build a simulation state from starting-value controls.

    Args:
        system_param_overrides: Cached key-value pairs returned by
            ``system_parameters_ui``.
        defaults: Scenario-specific starting values. Uses
            ``DEFAULT_INIT_STATE`` when omitted.

    Returns:
        Initial state keyed by ``S``, ``E``, ``F``, and ``FP``.
    """
    initial_state = dict(DEFAULT_INIT_STATE if defaults is None else defaults)
    override_values = dict(system_param_overrides)
    for state_key in initial_state:
        initial_state[state_key] = override_values.get(
            f"{state_key}0", initial_state[state_key],
        )
    return initial_state


def system_parameters_ui(
    prefix: str,
    exclude: set = frozenset(),
    initial_state_defaults: dict | None = None,
) -> tuple:
    """Render system controls and return cacheable override values.

    Args:
        prefix: Unique Streamlit widget-key prefix for the scenario.
        exclude: Parameter keys controlled elsewhere by the scenario.
        initial_state_defaults: Scenario-specific starting values. Uses
            ``DEFAULT_INIT_STATE`` when omitted.

    Returns:
        Sorted ``(key, value)`` pairs for parameter and starting-state
        overrides.
    """
    param_values = {}
    starting_values = dict(
        DEFAULT_INIT_STATE if initial_state_defaults is None
        else initial_state_defaults
    )
    default_values = {
        **DEFAULT_PARAMS,
        **{f"{state_key}0": value for state_key, value in starting_values.items()},
    }
    with st.expander("System Parameters", expanded=False):
        columns = st.columns(4, gap="large")
        for column, (group_name, params) in zip(columns, PARAM_GROUPS):
            with column:
                st.markdown(f"**{group_name}**")
                for key, label, slider_min, slider_max, slider_step in params:
                    if key in exclude:
                        continue
                    param_values[key] = st.slider(
                        label, slider_min, slider_max, float(default_values[key]), slider_step,
                        key=f"{prefix}_{key}",
                    )
    return tuple(sorted(param_values.items()))
