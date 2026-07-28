from contextlib import contextmanager
from typing import Iterator, Sequence

import streamlit as st


def scenario_header(title: str):
    """Render a scenario header with an inline status slot to its right.

    Returns an `st.empty()` placeholder that `status_indicator` will fill
    with a spinner while computing and a green checkmark once done.
    """
    header_column, status_column = st.columns([20, 1], vertical_alignment="center")
    with header_column:
        st.header(title)
    return status_column.empty()


@contextmanager
def status_indicator(slot, steps: Sequence[str]) -> Iterator[None]:
    """Show a spinner in `slot` during the block, then a green checkmark.

    The checkmark has a browser tooltip listing the given computation steps
    (shown on hover).
    """
    with slot.container():
        with st.spinner(""):
            yield

    tooltip = "\n".join(f"• {step_label}" for step_label in steps)
    slot.markdown(
        "<div style='display:flex;justify-content:center;align-items:center;"
        "height:2rem;'>"
        f"<span title=\"{tooltip}\" "
        "style='font-size:1.4em;color:#2E8B57;cursor:help;'>&#10004;</span>"
        "</div>",
        unsafe_allow_html=True,
    )
