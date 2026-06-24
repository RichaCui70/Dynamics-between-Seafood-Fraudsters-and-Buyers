import streamlit as st
from text import INTRO
from scenarios import scenario_baseline, scenario_bf, scenario_ps, scenario_eez
from hypothesis import hypothesis_tab

st.set_page_config(
    layout="wide",
    page_title="Dynamics of Seafood, Fraudsters, and Buyers",
    page_icon=":fish:",
)

section = st.segmented_control(
    "Navigation",
    ["Introduction", "Scenarios", "Hypothesis"],
    default="Introduction",
    key="nav_section",
    label_visibility="collapsed",
)

if section == "Introduction":
    st.write(INTRO)

elif section == "Scenarios":
    scenario = st.segmented_control(
        "Scenario",
        [
            "Baseline",
            "1: Blast Fishing",
            "2: Prized Seafood",
            "3: EEZ"
        ],
        default="Baseline",
        key="nav_scenario",
        label_visibility="collapsed",
    )

    if scenario == "Baseline":
        scenario_baseline()
    elif scenario == "1: Blast Fishing":
        scenario_bf()
    elif scenario == "2: Prized Seafood":
        scenario_ps()
    elif scenario == "3: EEZ":
        scenario_eez()

elif section == "Hypothesis":
    hypothesis_tab()
