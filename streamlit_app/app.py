"""Main Streamlit application entry point.

Pension planner — Time Runway Monte Carlo projections for a UK household.
"""

import streamlit as st

# Page config must be first Streamlit command
st.set_page_config(
    page_title="Pension Planner",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

import sys
from pathlib import Path

# Add parent directory to path for imports when running on Streamlit Cloud
sys.path.insert(0, str(Path(__file__).parent.parent))

from streamlit_app.utils.session_state import initialize_session_state

# Initialize session state
initialize_session_state()

# Main page content
st.title("📊 Pension Planner")

st.markdown("""
A personal retirement-runway model: will the portfolio last, given pensions, savings,
tax and uncertain markets?

## Pages

1. **📝 Inputs** — household members, assets, spending, pre-retirement savings, market assumptions
2. **📈 Time Runway** — Monte Carlo projection of the portfolio, spending and funding sources
3. **🎯 Sensitivity** — which inputs move the success rate, and how much you can spend

## Quick Overview
""")

# Show quick summary from session state
household = st.session_state.household
retirement_year = st.session_state.get("retirement_year", None)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Assets", f"£{household.total_assets:,.0f}")

with col2:
    st.metric("Annual Spending Target", f"£{household.total_spending:,.0f}")

with col3:
    st.metric("Retirement", str(retirement_year) if retirement_year else "Already retired")

with col4:
    st.metric("Planning Horizon", f"{household.planning_horizon} years")

st.markdown("---")

st.markdown("""
## How the Model Works

Everything is in **today's pounds** (real terms) — returns, spending and pensions are all
inflation-adjusted, so a fixed (non-linked) DB pension shrinks over time and inflation-linked
ones stay flat.

- **Before retirement**: salary covers spending. Savings streams (and any pension already in
  payment) flow into the portfolio. Rental income is a savings stream and stops at retirement —
  the assumption is that the properties are sold to buy a home.
- **At retirement**: any Thai severance is added to the portfolio.
- **After retirement**: State and DB pensions arrive first; the portfolio covers the rest,
  grossed up for UK income tax and CGT per person.
- **Markets**: thousands of possible return histories, either resampled from actual UK/World
  history or drawn from a log-normal model, give the range of outcomes (P10 / P50 / P90).

**Success rate** is the share of those histories where the portfolio never runs out.
**Floor breach rate** is the share where spending has to fall below the essential floor.
""")

# Footer
st.markdown("---")
st.markdown(
    "*Built with [Streamlit](https://streamlit.io) and "
    "[Plotly](https://plotly.com). "
    "This tool is for personal planning only and does not constitute financial advice.*"
)
