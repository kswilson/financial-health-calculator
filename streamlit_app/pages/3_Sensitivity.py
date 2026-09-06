"""Sensitivity page — how the runway success rate responds to each input."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from fundedness.runway import build_runway_plan, run_runway
from fundedness.viz.tornado import create_scenario_comparison_chart, create_tornado_chart
from streamlit_app.utils.session_state import (
    get_household,
    get_market_model,
    get_tax_model,
    initialize_session_state,
)

st.set_page_config(page_title="Sensitivity Analysis", page_icon="🎯", layout="wide")

initialize_session_state()

st.title("🎯 Sensitivity Analysis")

st.markdown("""
Which inputs move the **probability of the portfolio lasting** the most? Each bar re-runs the
Time Runway simulation with one input changed and everything else held at your current values.
""")

with st.sidebar:
    st.subheader("Sensitivity Settings")
    n_simulations = st.select_slider(
        "Simulations per scenario",
        options=[1000, 2500, 5000],
        value=2500,
        help="Fewer simulations = faster page, slightly noisier bars.",
    )
household = get_household()
_bs_stock_pct = int(round(household.balance_sheet.get_stock_allocation() * 100 / 5) * 5)

with st.sidebar:
    stock_allocation = st.slider(
        "Stock Allocation", 0, 100,
        st.session_state.get("stock_allocation", _bs_stock_pct), 5, format="%d%%",
        help=f"Shared with the Time Runway page. Balance sheet mix is {_bs_stock_pct}%.",
    )
    st.session_state.stock_allocation = stock_allocation

market_model = get_market_model()
tax_model = get_tax_model()
retirement_year = st.session_state.get("retirement_year", None)
return_model = st.session_state.get("return_model", "lognormal")
market_source = st.session_state.get("market_source", "uk")


def success_rate(
    *,
    spending=None,
    ret_year=retirement_year,
    wealth=None,
    n_years=None,
    stocks=stock_allocation / 100,
    return_shift=0.0,
) -> float:
    plan = build_runway_plan(
        household,
        ret_year,
        tax_model,
        annual_spending=spending,
        initial_wealth=wealth,
        n_years=n_years,
    )
    result = run_runway(
        plan,
        market_model,
        n_simulations=n_simulations,
        stock_allocation=stocks,
        return_model=return_model,
        market_source=market_source,
        return_shift=return_shift,
    )
    return result.success_rate


base_plan = build_runway_plan(household, retirement_year, tax_model)
base_spending = base_plan.annual_spending
base_wealth = base_plan.initial_wealth
base_years = base_plan.n_years

with st.spinner("Running sensitivity scenarios..."):
    base = success_rate()

    factors = []  # (label, low_value, high_value, low_label, high_label)

    spend_step = 10_000
    factors.append((
        f"Ongoing spending ±£{spend_step / 1000:.0f}K",
        success_rate(spending=base_spending + spend_step),
        success_rate(spending=max(0, base_spending - spend_step)),
        f"£{base_spending + spend_step:,.0f}", f"£{base_spending - spend_step:,.0f}",
    ))

    if retirement_year is not None:
        factors.append((
            "Retirement year ±2",
            success_rate(ret_year=retirement_year - 2),
            success_rate(ret_year=retirement_year + 2),
            str(retirement_year - 2), str(retirement_year + 2),
        ))

    factors.append((
        "Starting portfolio ±20%",
        success_rate(wealth=base_wealth * 0.8),
        success_rate(wealth=base_wealth * 1.2),
        f"£{base_wealth * 0.8:,.0f}", f"£{base_wealth * 1.2:,.0f}",
    ))

    factors.append((
        "Returns ±1%/yr",
        success_rate(return_shift=-0.01),
        success_rate(return_shift=+0.01),
        "−1%/yr", "+1%/yr",
    ))

    lo_stock = max(0.0, stock_allocation / 100 - 0.2)
    hi_stock = min(1.0, stock_allocation / 100 + 0.2)
    factors.append((
        "Stock allocation ±20pts",
        success_rate(stocks=lo_stock),
        success_rate(stocks=hi_stock),
        f"{lo_stock:.0%}", f"{hi_stock:.0%}",
    ))

    factors.append((
        "Planning horizon +5 years",
        success_rate(n_years=base_years + 5),
        base,
        f"{base_years + 5} yrs", f"{base_years} yrs (base)",
    ))

# --- Tornado ---
st.subheader("Tornado Chart — What Moves the Success Rate")

labels = [f[0] for f in factors]
tornado = create_tornado_chart(
    parameters=labels,
    low_values=[f[1] * 100 for f in factors],
    high_values=[f[2] * 100 for f in factors],
    base_value=base * 100,
    title=f"Success Rate Sensitivity (base {base:.1%})",
    value_label="Success rate (%)",
)
st.plotly_chart(tornado, use_container_width=True)

rows = []
for label, lo, hi, lo_lbl, hi_lbl in factors:
    rows.append({
        "Factor": label,
        "Worse case": f"{lo_lbl} → {lo:.1%}",
        "Better case": f"{hi_lbl} → {hi:.1%}",
        "Swing": f"{(hi - lo) * 100:+.1f} pts",
    })
st.dataframe(rows, use_container_width=True, hide_index=True)

st.divider()

# --- Spending sweep: the actionable one ---
st.subheader("How Much Can You Spend?")
st.markdown("Success rate as ongoing annual spending varies (dated one-off costs unchanged), with all other inputs fixed.")

col1, col2 = st.columns(2)
with col1:
    sweep_lo = st.number_input("From (£/yr)", value=int(max(10_000, base_spending - 30_000)), step=5_000)
with col2:
    sweep_hi = st.number_input("To (£/yr)", value=int(base_spending + 30_000), step=5_000)

if sweep_hi > sweep_lo:
    spend_grid = np.linspace(sweep_lo, sweep_hi, 13)
    with st.spinner("Sweeping spending levels..."):
        sweep_rates = [success_rate(spending=float(s)) for s in spend_grid]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=spend_grid, y=[r * 100 for r in sweep_rates],
        mode="lines+markers", name="Success rate",
        line=dict(color="#2980b9", width=3),
        hovertemplate="£%{x:,.0f}/yr → %{y:.1f}%<extra></extra>",
    ))
    for level, colour in ((90, "#27ae60"), (75, "#f39c12")):
        fig.add_hline(y=level, line_dash="dash", line_color=colour,
                      annotation_text=f"{level}%", annotation_position="right")
    fig.add_vline(x=base_spending, line_dash="dot", line_color="#7f8c8d",
                  annotation_text="Current", annotation_position="top")
    fig.update_layout(
        template="plotly_white",
        title="Success Rate vs Annual Spending",
        xaxis_title="Ongoing annual spending (£, today's money)",
        yaxis_title="Success rate (%)",
        yaxis=dict(range=[0, 102]),
        height=450,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Highest spending that still clears each threshold
    for level in (0.9, 0.75):
        ok = [s for s, r in zip(spend_grid, sweep_rates) if r >= level]
        if ok:
            st.metric(f"Max spending for ≥{level:.0%} success", f"£{max(ok):,.0f}/yr")
        else:
            st.metric(f"Max spending for ≥{level:.0%} success", f"below £{sweep_lo:,.0f}")

st.divider()

# --- Retirement year sweep ---
if retirement_year is not None:
    st.subheader("When Can You Retire?")
    years = list(range(retirement_year - 3, retirement_year + 4))
    with st.spinner("Sweeping retirement years..."):
        year_rates = [success_rate(ret_year=y) for y in years]
    ry_fig = create_scenario_comparison_chart(
        scenarios=[str(y) for y in years],
        values=[r * 100 for r in year_rates],
        base_scenario=str(retirement_year),
        title="Success Rate by Retirement Year",
        value_label="Success rate (%)",
    )
    st.plotly_chart(ry_fig, use_container_width=True)

with st.expander("How to read this"):
    st.markdown(f"""
    - **Success rate** = share of simulated market histories where the portfolio lasts to age
      {base_plan.starting_age + base_years}. Base case: **{base:.1%}**.
    - Each tornado bar changes one input and re-runs the full Time Runway model (pensions, savings,
      severance, per-person tax) — so the bars are directly comparable with the Time Runway page.
    - "Returns ±1%/yr" shifts every simulated year's return, for both the historical bootstrap and
      the log-normal model. It's the cleanest way to ask "what if the future is worse than the past?"
    - Bars use {n_simulations:,} simulations each, so differences under ~1 point are noise.
    """)
