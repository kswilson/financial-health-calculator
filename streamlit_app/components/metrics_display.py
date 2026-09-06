"""Metrics display components for Streamlit."""

import streamlit as st

from fundedness.simulate import SimulationResult


def render_simulation_metrics(result: SimulationResult):
    """Render simulation metrics in a dashboard format.

    Args:
        result: Simulation result
    """
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        color = "off" if result.success_rate >= 0.9 else "normal" if result.success_rate >= 0.75 else "inverse"
        st.metric(
            label="Success Rate",
            value=f"{result.success_rate:.1%}",
            delta="Low Risk" if result.success_rate >= 0.9 else "Moderate Risk" if result.success_rate >= 0.75 else "High Risk",
            delta_color=color,
        )

    with col2:
        st.metric(
            label="Median Terminal Wealth",
            value=f"£{result.median_terminal_wealth:,.0f}",
        )

    with col3:
        st.metric(
            label="Floor Breach Rate",
            value=f"{result.floor_breach_rate:.1%}",
        )

    with col4:
        st.metric(
            label="Simulations",
            value=f"{result.n_simulations:,}",
        )
