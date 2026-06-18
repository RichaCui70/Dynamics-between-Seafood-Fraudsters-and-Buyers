import streamlit as st
from text import INTRO
from scenarios import scenario_1, scenario_2, scenario_3, scenario_4
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
            "1: Baseline",
            "2: Prized Seafood",
            "3: Blast Fishing",
            "4: EEZ"
        ],
        default="2: Prized Seafood",
        key="nav_scenario",
        label_visibility="collapsed",
    )

    if scenario == "1: Baseline":
        scenario_1()
    elif scenario == "2: Prized Seafood":
        scenario_2()
    elif scenario == "3: Blast Fishing":
        scenario_3()
    elif scenario == "4: EEZ":
        scenario_4()

elif section == "Hypothesis":
    hypothesis_tab()
