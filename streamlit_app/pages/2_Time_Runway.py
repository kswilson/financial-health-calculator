"""Time Runway page with Monte Carlo projections."""

import sys
from collections import OrderedDict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import streamlit as st

from fundedness.runway import build_runway_plan, run_runway, total_spending_percentiles
from fundedness.viz.fan_chart import create_fan_chart, create_funding_sources_chart, create_spending_fan_chart
from fundedness.viz.histogram import create_time_distribution_histogram
from fundedness.viz.survival import create_dual_survival_chart, create_survival_curve
from streamlit_app.components.metrics_display import render_simulation_metrics
from streamlit_app.utils.session_state import (
    get_household,
    get_market_model,
    get_return_haircut,
    get_tax_model,
    initialize_session_state,
)

st.set_page_config(page_title="Time Runway", page_icon="📈", layout="wide")

initialize_session_state()

st.title("📈 Time Runway Analysis")

st.markdown("""
Monte Carlo simulation shows the range of possible outcomes based on market uncertainty.
""")

household = get_household()
# Default the slider to the balance sheet's actual equity share (by Asset Class on the Inputs page)
_bs_stock_pct = int(round(household.balance_sheet.get_stock_allocation() * 100 / 5) * 5)

# Sidebar controls
with st.sidebar:
    st.subheader("Simulation Settings")

    n_simulations = st.select_slider(
        "Number of Simulations",
        options=[1000, 2500, 5000, 10000],
        value=5000,
    )

    stock_allocation = st.slider(
        "Stock Allocation",
        min_value=0,
        max_value=100,
        value=st.session_state.get("stock_allocation", _bs_stock_pct),
        step=5,
        format="%d%%",
        help="Equity share of the portfolio for every simulated year; the remainder is gilts/bonds. "
             f"Defaults to your balance sheet's current mix ({_bs_stock_pct}%). Shared with the Sensitivity page.",
    )
    st.session_state.stock_allocation = stock_allocation
    if abs(stock_allocation - _bs_stock_pct) >= 5:
        st.caption(f"⚠️ Balance sheet is {_bs_stock_pct}% equities; simulating {stock_allocation}%.")

# Build the plan (schedules) and run it
market_model = get_market_model()
retirement_year = st.session_state.get("retirement_year", None)

plan = build_runway_plan(household, retirement_year, get_tax_model())
n_years = plan.n_years
starting_age = plan.starting_age
years_until_retired = plan.years_until_retired
annual_spending = plan.annual_spending
spending_floor = plan.spending_floor
initial_wealth = plan.initial_wealth
withdrawal_rate = (annual_spending / initial_wealth * 100) if initial_wealth > 0 else 0
return_haircut = get_return_haircut()

with st.spinner(f"Running {n_simulations:,} simulations..."):
    result = run_runway(
        plan,
        market_model,
        n_simulations=n_simulations,
        stock_allocation=stock_allocation / 100,
        return_model=st.session_state.get("return_model", "lognormal"),
        market_source=st.session_state.get("market_source", "uk"),
        return_shift=-return_haircut,
    )
    st.session_state.simulation_result = result

# Display metrics
render_simulation_metrics(result)
st.caption(
    f"Returns: {st.session_state.get('market_source', 'uk')} "
    f"{'historical bootstrap' if st.session_state.get('return_model') == 'bootstrap' else 'parametric'} "
    f"with a {return_haircut:.2%}/yr haircut (Inputs → Assumptions). "
    "P10 terminal wealth: 1 in 10 outcomes end below this."
)

st.divider()

# Age-based x-axis
ages = np.arange(starting_age + 1, starting_age + n_years + 1)

# Main visualizations
tab1, tab2, tab3 = st.tabs(["📈 Wealth Projection", "📊 Survival Analysis", "📉 Time to Events"])

with tab1:
    st.subheader("Portfolio Value Over Time")

    wealth_fig = create_fan_chart(
        years=ages,
        percentiles=result.wealth_percentiles,
        title="Portfolio Value Projection",
        y_label="Portfolio Value (£)",
        x_label="Age",
        show_floor=0,
    )
    st.plotly_chart(wealth_fig, use_container_width=True)

    # Spending projection if available
    if result.spending_paths is not None:
        st.subheader("Spending Over Time")
        spending_pct = total_spending_percentiles(plan, result)

        spending_fig = create_spending_fan_chart(
            years=ages,
            percentiles=spending_pct,
            floor_spending=spending_floor,
            target_spending=annual_spending,
            title="Total Household Spending (Portfolio + Pensions)",
            x_label="Age",
        )
        st.plotly_chart(spending_fig, use_container_width=True)

        # Funding sources breakdown (stacked area)
        st.subheader("Funded By")
        scenario_choice = st.radio(
            "Scenario",
            options=["Median (P50)", "Pessimistic (P10)"],
            index=0,
            horizontal=True,
            help="Median shows the typical outcome. Pessimistic shows what happens in a bad market.",
        )
        _pct_key = "P50" if scenario_choice == "Median (P50)" else "P10"
        p50_total = spending_pct.get(_pct_key, np.full(n_years, annual_spending))

        # Build ordered source dict (bottom to top on chart)
        funding: OrderedDict[str, np.ndarray] = OrderedDict()

        # State pensions first (bottom of stack), then DB pensions
        for prefix in ("State Pension", "DB Pension"):
            for src_name, src_arr in plan.pension_sources.items():
                if prefix in src_name:
                    clipped = src_arr.copy()
                    clipped[:years_until_retired] = 0
                    if np.any(clipped > 0):
                        funding[src_name] = clipped

        # Salary (pre-retirement) covers ongoing spending only; dated costs show as Portfolio
        if years_until_retired > 0:
            salary_arr = np.zeros(n_years)
            salary_arr[:years_until_retired] = annual_spending
            funding["Salary"] = salary_arr

        # Thai severance (one-off at retirement year)
        if plan.severance > 0 and 0 < years_until_retired < n_years:
            sev_arr = np.zeros(n_years)
            pensions_at_ret = sum(v[years_until_retired] for v in plan.pension_sources.values())
            remaining_need = max(0, p50_total[years_until_retired] - pensions_at_ret)
            sev_arr[years_until_retired] = min(plan.severance, remaining_need)
            funding["Thai Severance"] = sev_arr

        # Cap each source so the cumulative stack never exceeds p50_total.
        # When portfolio is depleted, total spending drops to pension-only,
        # so pension sources must be capped to actual spending that year.
        cumulative = np.zeros(n_years)
        for name in list(funding.keys()):
            capped = np.minimum(funding[name], np.maximum(0, p50_total - cumulative))
            funding[name] = capped
            cumulative += capped

        # Portfolio withdrawals (the remainder)
        funding["Portfolio"] = np.maximum(0, p50_total - cumulative)

        sources_fig = create_funding_sources_chart(
            years=ages,
            sources=funding,
            title=f"Spending Funded By ({scenario_choice})",
            x_label="Age",
        )
        st.plotly_chart(sources_fig, use_container_width=True)

with tab2:
    st.subheader("Survival Probability")

    survival_prob = result.get_survival_probability()
    floor_survival_prob = result.get_floor_survival_probability()

    col1, col2 = st.columns(2)

    # Threshold ages for markers
    threshold_ages = [age for age in [75, 85, 95] if age <= ages[-1]]

    with col1:
        survival_fig = create_survival_curve(
            years=ages,
            survival_prob=survival_prob,
            floor_survival_prob=floor_survival_prob,
            title="Probability of Portfolio Survival",
            threshold_years=threshold_ages,
            x_label="Age",
        )
        st.plotly_chart(survival_fig, use_container_width=True)

    with col2:
        risk_fig = create_dual_survival_chart(
            years=ages,
            ruin_prob=1 - survival_prob,
            floor_breach_prob=1 - floor_survival_prob,
            title="Cumulative Risk Over Time",
            x_label="Age",
        )
        st.plotly_chart(risk_fig, use_container_width=True)

    # Key survival metrics
    st.subheader("Key Survival Metrics")

    end_age = starting_age + n_years
    milestone_ages = sorted(set(a for a in [75, 85, 95, end_age] if a <= end_age))[:4]

    cols = st.columns(len(milestone_ages))
    for col, age in zip(cols, milestone_ages):
        idx = min(age - starting_age - 1, len(survival_prob) - 1)
        with col:
            st.metric(f"Survival to {age}", f"{survival_prob[idx]:.1%}")

with tab3:
    st.subheader("Time to Event Distributions")

    col1, col2 = st.columns(2)

    with col1:
        if result.time_to_ruin is not None:
            ruin_hist = create_time_distribution_histogram(
                time_to_event=result.time_to_ruin,
                event_name="Ruin",
                planning_horizon=n_years,
                percentiles_to_show=[10, 25, 50],
                title="Age at Portfolio Depletion",
                starting_age=starting_age,
            )
            st.plotly_chart(ruin_hist, use_container_width=True)

    with col2:
        if result.time_to_floor_breach is not None:
            floor_hist = create_time_distribution_histogram(
                time_to_event=result.time_to_floor_breach,
                event_name="Floor Breach",
                planning_horizon=n_years,
                percentiles_to_show=[10, 25, 50],
                title="Age at Floor Breach",
                starting_age=starting_age,
            )
            st.plotly_chart(floor_hist, use_container_width=True)

# --- Plan summary boxes ---
if years_until_retired > 0:
    total_annual_contrib = plan.savings_by_year[0] if n_years > 0 else 0
    savings_msg = f" Saving £{total_annual_contrib:,.0f}/year into portfolio." if total_annual_contrib > 0 else ""
    st.info(
        f"Retirement in {retirement_year} ({years_until_retired} years). "
        f"Salary covers spending until then — no portfolio draws.{savings_msg}"
    )
    if any("rent" in s.name.lower() for s in household.savings_contributions):
        st.warning(
            "**Reminder:** rental income is entered as a pre-retirement savings stream, so it "
            f"stops in {retirement_year}. That assumes the property is sold at retirement "
            "(e.g. to buy somewhere to live) — the sale proceeds and the new home are *not* in the portfolio. "
            "If you keep the property and the rent continues, add it as pension-style income instead."
        )

if plan.dated_liabilities:
    import datetime as _dt
    _cy = _dt.date.today().year
    _lines = []
    for l in plan.dated_liabilities:
        _first = _cy + l.start_year
        _last = _cy + l.end_year - 1 if l.end_year is not None else None
        _when = f"{_first}" if _last is None or _last == _first else f"{_first}–{_last}"
        _lines.append(f"{l.name} £{l.annual_amount:,.0f}/yr ({_when})")
    st.info(
        "One-off / dated costs drawn from the portfolio (grossed up for tax, money stays invested "
        "until paid): " + "; ".join(_lines) + "."
    )

if plan.pension_income_by_year.max() > 0:
    # First regular post-retirement year (skip severance year and any inflow years)
    _early_idx = min(years_until_retired + 1, n_years - 1)
    while _early_idx < n_years - 1 and plan.net_spending_by_year[_early_idx] <= 0:
        _early_idx += 1
    _early_draw = plan.net_spending_by_year[_early_idx]
    _late_draw = plan.net_spending_by_year[-1]
    _early_tax = plan.tax_by_year[_early_idx]
    _late_tax = plan.tax_by_year[-1]
    st.info(
        f"Spending £{annual_spending:,.0f}/yr net. "
        f"Early retirement: pensions £{plan.pension_income_by_year[_early_idx]:,.0f} + "
        f"portfolio £{_early_draw:,.0f} (incl. £{_early_tax:,.0f} tax). "
        f"After all pensions: pensions £{plan.pension_income_by_year[-1]:,.0f} + "
        f"portfolio £{_late_draw:,.0f} (incl. £{_late_tax:,.0f} tax)."
    )

mix = plan.draw_mix
st.caption(
    "Tax on portfolio draws is computed per person with UK bands (personal allowance, 20/40/45%) "
    "and CGT (£3,000 exemption, 18/24%), with the draw split equally between household members. "
    f"Each £1 withdrawn is assumed to come {mix.sipp_share:.0%} from SIPP (75% taxable), "
    f"{mix.deferred_share:.0%} from other tax-deferred accounts (fully taxable), "
    f"{mix.taxable_share:.0%} from GIA (CGT on {mix.gain_ratio:.0%} gain) and "
    f"{mix.exempt_share:.0%} from cash/ISA (tax-free), based on today's balance sheet."
)

# Interpretation
with st.expander("Understanding the Results"):
    member_lines = ""
    for member in household.members:
        sp = member.social_security_annual
        db = member.pension_annual
        sp_age = member.social_security_age or 67
        member_lines += f"\n    - **{member.name}**:"
        if sp > 0:
            member_lines += f" State Pension £{sp:,.0f}/yr from age {sp_age}"
        if db > 0:
            db_age = member.pension_start_age or 60
            linked = " (inflation-linked)" if member.pension_inflation_linked else " (fixed)"
            member_lines += f", DB Pension £{db:,.0f}/yr from age {db_age}{linked}"

    ret_line = f"**Retirement Year**: {retirement_year}" if retirement_year else "**Retirement**: already retired"
    horizon_member = max(household.members, key=lambda m: m.planning_horizon) if household.members else None
    horizon_note = (
        f" (until {horizon_member.name} reaches {horizon_member.life_expectancy})"
        if horizon_member and horizon_member is not household.primary_member
        else ""
    )

    st.markdown(f"""
    ### Simulation Summary

    - **Initial Portfolio**: £{initial_wealth:,.0f}
    - **Ongoing Spending Need**: £{annual_spending:,.0f}/yr ({withdrawal_rate:.1f}% of portfolio)
    - **Essential Spending Floor**: £{spending_floor:,.0f}
    - **Stock Allocation**: {stock_allocation}%
    - **Return Haircut**: {return_haircut:.2%}/yr
    - {ret_line}
    - **Current Age**: {starting_age}
    - **Planning to Age**: {starting_age + n_years}{horizon_note}
    {member_lines}

    Before retirement, salary covers ongoing spending; only dated one-off costs are drawn from the portfolio.
    After retirement, portfolio withdrawals are reduced by pension income.

    ### Interpretation

    - **Success Rate ({result.success_rate:.1%})**: The percentage of simulations where the portfolio lasted to age {starting_age + n_years}
    - **Floor Breach Rate ({result.floor_breach_rate:.1%})**: The percentage of simulations where spending had to be cut below essential needs
    - **P10/P50/P90**: The 10th, 50th (median), and 90th percentile outcomes

    ### Key Insights

    {"✅ Your success rate is above 90%, suggesting a sustainable withdrawal rate." if result.success_rate >= 0.9 else ""}
    {"⚠️ Your success rate is between 75-90%. Consider reducing spending or increasing savings." if 0.75 <= result.success_rate < 0.9 else ""}
    {"🚨 Your success rate is below 75%. Significant adjustments may be needed." if result.success_rate < 0.75 else ""}
    """)
