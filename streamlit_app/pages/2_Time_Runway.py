"""Time Runway page with Monte Carlo projections."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import streamlit as st

from fundedness.models.simulation import SimulationConfig
from fundedness.simulate import run_simulation
from fundedness.viz.fan_chart import create_fan_chart, create_funding_sources_chart, create_spending_fan_chart
from fundedness.viz.histogram import create_time_distribution_histogram
from fundedness.viz.survival import create_dual_survival_chart, create_survival_curve
from streamlit_app.components.metrics_display import render_simulation_metrics
from streamlit_app.utils.session_state import (
    get_household,
    get_market_model,
    get_simulation_config,
    initialize_session_state,
)

st.set_page_config(page_title="Time Runway", page_icon="📈", layout="wide")

initialize_session_state()

st.title("📈 Time Runway Analysis")

st.markdown("""
Monte Carlo simulation shows the range of possible outcomes based on market uncertainty.
""")

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
        value=60,
        step=5,
        format="%d%%",
    )

# Get data
household = get_household()
market_model = get_market_model()

# Calculate values from Inputs page
initial_wealth = household.total_assets
annual_spending = household.total_spending
spending_floor = household.essential_spending
n_years = household.planning_horizon

# Get starting age for x-axis (primary member)
starting_age = 65
if household.primary_member:
    starting_age = household.primary_member.age

# Build year-by-year net portfolio withdrawal schedule
# Joint model: one portfolio, one spending need.
# While still working, salary covers all spending (no portfolio draw).
# After retirement, spending comes from the portfolio minus pension income.

assumed_inflation = 0.025
import datetime as _dt
current_year = _dt.date.today().year

# Retirement year: None = already retired, otherwise a calendar year
retirement_year = st.session_state.get("retirement_year", None)
if retirement_year is not None:
    years_until_retired = max(0, retirement_year - current_year)
else:
    years_until_retired = 0

# Pension income schedule — per source and combined
pension_income_by_year = np.zeros(n_years)
pension_sources: dict[str, np.ndarray] = {}  # name -> per-year array

for member in household.members:
    label = member.name
    member_age = member.age
    sp_age = getattr(member, "social_security_age", None) or 67
    sp_annual = member.social_security_annual
    db_annual = member.pension_annual
    db_start_age = getattr(member, "pension_start_age", None) or 60
    db_linked = getattr(member, "pension_inflation_linked", False)

    if sp_annual > 0:
        key = f"State Pension ({label})"
        arr = np.zeros(n_years)
        for yr in range(n_years):
            member_age_in_year = member_age + yr + 1
            if member_age_in_year >= sp_age:
                arr[yr] = sp_annual
        pension_sources[key] = arr
        pension_income_by_year += arr

    if db_annual > 0:
        key = f"DB Pension ({label})"
        arr = np.zeros(n_years)
        for yr in range(n_years):
            member_age_in_year = member_age + yr + 1
            if member_age_in_year >= db_start_age:
                if db_linked:
                    arr[yr] = db_annual
                else:
                    years_paying = member_age_in_year - db_start_age
                    arr[yr] = db_annual / (1 + assumed_inflation) ** years_paying
        pension_sources[key] = arr
        pension_income_by_year += arr

# Build savings schedule from household savings contributions
savings_by_year = np.zeros(n_years)
for s in household.savings_contributions:
    if s.annual_amount > 0:
        for yr in range(min(years_until_retired, n_years)):
            savings_by_year[yr] += s.annual_amount * (1 + s.growth_rate) ** yr

# Tax gross-up: linear blended rate based on desired gross income.
# Calibrated to UK tax system for retirees drawing from mixed account types:
#   £40K gross → 6% blended rate (mostly 0%/20% bands, GIA covers gap)
#   £80K gross → 20% blended rate (significant 40% exposure)
# Linear interpolation between these anchors, clamped to [0%, 30%].

def _blended_tax_rate(gross_income: float) -> float:
    """Compute blended effective tax rate for a given gross income level."""
    if gross_income <= 0:
        return 0.0
    # Linear: 6% at £40K, 20% at £80K → slope = 0.14 / 40000
    rate = 0.06 + (gross_income - 40_000) * (0.20 - 0.06) / (80_000 - 40_000)
    return max(0.0, min(0.30, rate))

def _tax_gross_up(gross_income: float) -> float:
    """Gross-up factor for a given income level."""
    rate = _blended_tax_rate(gross_income)
    return 1 / (1 - rate) if rate < 1 else 1.0

# Before retirement: no spending draw, but savings are added (negative spending).
# After retirement: tax applies to ALL income (pensions + portfolio), not just
# portfolio withdrawals. So we gross up the total spending target, then subtract
# the gross pension income, and the portfolio covers the remainder.
#   total_gross_needed = annual_spending / (1 - tax_rate(annual_spending))
#   portfolio_draw = total_gross_needed - pension_income
net_spending_by_year = np.zeros(n_years)
tax_rate_by_year = np.zeros(n_years)
for yr in range(n_years):
    if yr < years_until_retired:
        # Pre-retirement: savings flow in (negative = contribution)
        net_spending_by_year[yr] = -savings_by_year[yr]
    else:
        # Tax rate based on total spending level (all income is taxable)
        rate = _blended_tax_rate(annual_spending)
        tax_rate_by_year[yr] = rate
        total_gross_needed = annual_spending / (1 - rate) if rate < 1 else annual_spending
        # Pensions contribute gross, portfolio covers the rest
        portfolio_draw = max(0, total_gross_needed - pension_income_by_year[yr])
        net_spending_by_year[yr] = portfolio_draw

# Representative values for display and chart use
tax_gross_up = 1 / (1 - _blended_tax_rate(annual_spending)) if _blended_tax_rate(annual_spending) < 1 else 1.0
tax_gross_up_by_year = np.where(tax_rate_by_year > 0, 1 / (1 - tax_rate_by_year), 1.0)
# Thai severance: inject as negative spending (cash inflow) at retirement year
total_severance_gbp = sum(
    m.thai_severance_gbp(retirement_year) for m in household.members
)
if total_severance_gbp > 0 and years_until_retired > 0 and years_until_retired < n_years:
    net_spending_by_year[years_until_retired] -= total_severance_gbp

withdrawal_rate = (annual_spending / initial_wealth * 100) if initial_wealth > 0 else 0

# Run simulation
config = SimulationConfig(
    n_simulations=n_simulations,
    n_years=n_years,
    market_model=market_model,
    random_seed=42,
    return_model=st.session_state.get("return_model", "lognormal"),
    market_source=st.session_state.get("market_source", "uk"),
)

with st.spinner(f"Running {n_simulations:,} simulations..."):
    # Floor for portfolio withdrawals = essential spending minus pension income,
    # grossed up for tax (since the simulation tracks gross withdrawals).
    # Varies by year as pension income changes.
    floor_by_year = np.maximum(0, spending_floor - pension_income_by_year) * tax_gross_up_by_year
    # Pre-retirement years: no floor (salary covers spending)
    floor_by_year[:years_until_retired] = 0

    result = run_simulation(
        initial_wealth=initial_wealth,
        annual_spending=net_spending_by_year,
        config=config,
        stock_weight=stock_allocation / 100,
        spending_floor=floor_by_year,
    )
    st.session_state.simulation_result = result

# Display metrics
render_simulation_metrics(result)

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
        # The simulation tracks gross portfolio withdrawals (including tax).
        # Convert back to total household consumption:
        # - Pre-retirement: salary covers spending, so consumption = annual_spending
        # - Post-retirement: net portfolio withdrawal + pension income
        total_spending_percentiles = {}
        for key, vals in result.spending_percentiles.items():
            total = np.zeros_like(vals)
            for yr in range(len(vals)):
                if yr < years_until_retired:
                    # Pre-retirement: salary covers full spending
                    total[yr] = annual_spending
                else:
                    pension = pension_income_by_year[yr] if yr < len(pension_income_by_year) else 0
                    # The target gross withdrawal for this year
                    target_withdrawal = net_spending_by_year[yr]
                    actual_withdrawal = vals[yr]

                    if target_withdrawal < 0:
                        # One-off inflow year (e.g. severance) — still consuming fully
                        total[yr] = annual_spending
                    elif actual_withdrawal <= 0:
                        # Portfolio depleted — only pension income available
                        total[yr] = pension
                    elif actual_withdrawal >= target_withdrawal * 0.99:
                        # Portfolio fully funded the withdrawal — full spending
                        total[yr] = annual_spending
                    else:
                        # Partial funding — portfolio couldn't cover full withdrawal
                        # De-gross what was actually withdrawn + pension
                        net_withdrawal = actual_withdrawal / tax_gross_up_by_year[yr]
                        total[yr] = net_withdrawal + pension
            total_spending_percentiles[key] = total

        spending_fig = create_spending_fan_chart(
            years=ages,
            percentiles=total_spending_percentiles,
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
        p50_total = total_spending_percentiles.get(_pct_key, np.full(n_years, annual_spending))

        # Build ordered source dict (bottom to top on chart)
        from collections import OrderedDict
        funding: OrderedDict[str, np.ndarray] = OrderedDict()

        # State pensions first (bottom of stack)
        for src_name, src_arr in pension_sources.items():
            if "State Pension" in src_name:
                clipped = src_arr.copy()
                clipped[:years_until_retired] = 0
                if np.any(clipped > 0):
                    funding[src_name] = clipped

        # DB pensions next
        for src_name, src_arr in pension_sources.items():
            if "DB Pension" in src_name:
                clipped = src_arr.copy()
                clipped[:years_until_retired] = 0
                if np.any(clipped > 0):
                    funding[src_name] = clipped

        # Salary (pre-retirement)
        salary_arr = np.zeros(n_years)
        salary_arr[:years_until_retired] = annual_spending
        if years_until_retired > 0:
            funding["Salary"] = salary_arr

        # Thai severance (one-off at retirement year)
        if total_severance_gbp > 0 and years_until_retired > 0 and years_until_retired < n_years:
            sev_arr = np.zeros(n_years)
            # Severance funds spending in retirement year up to the remaining need
            pensions_at_ret = sum(
                v[years_until_retired] for v in pension_sources.values()
            )
            remaining_need = max(0, p50_total[years_until_retired] - pensions_at_ret)
            sev_arr[years_until_retired] = min(total_severance_gbp, remaining_need)
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
        portfolio_arr = np.maximum(0, p50_total - cumulative)
        funding["Portfolio"] = portfolio_arr

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
        # Calculate cumulative risk
        ruin_prob = 1 - survival_prob
        floor_breach_prob = 1 - floor_survival_prob

        risk_fig = create_dual_survival_chart(
            years=ages,
            ruin_prob=ruin_prob,
            floor_breach_prob=floor_breach_prob,
            title="Cumulative Risk Over Time",
            x_label="Age",
        )
        st.plotly_chart(risk_fig, use_container_width=True)

    # Key survival metrics
    st.subheader("Key Survival Metrics")

    end_age = starting_age + n_years
    milestone_ages = [75, 85, 95, end_age]
    # Filter to ages within range and deduplicate
    milestone_ages = sorted(set(a for a in milestone_ages if a <= end_age))[:4]

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

# Income/draw summary
initial_draw = net_spending_by_year[0] if len(net_spending_by_year) > 0 else annual_spending
max_pension = pension_income_by_year.max() if len(pension_income_by_year) > 0 else 0
final_draw = net_spending_by_year[-1] if len(net_spending_by_year) > 0 else annual_spending

# Show retirement & pension info
if years_until_retired > 0:
    total_annual_contrib = savings_by_year[0] if len(savings_by_year) > 0 else 0
    savings_msg = ""
    if total_annual_contrib > 0:
        savings_msg = f" Saving £{total_annual_contrib:,.0f}/year into portfolio."
    st.info(
        f"Retirement in {retirement_year} ({years_until_retired} years). "
        f"Salary covers spending until then — no portfolio draws.{savings_msg}"
    )

if max_pension > 0:
    # Use first regular post-retirement year (skip severance year)
    _early_idx = min(years_until_retired + 1, n_years - 1)
    # Find a year where net_spending is positive (not distorted by one-off inflows)
    while _early_idx < n_years - 1 and net_spending_by_year[_early_idx] <= 0:
        _early_idx += 1
    _early_draw = net_spending_by_year[_early_idx]
    _late_draw = net_spending_by_year[-1]
    _early_pension = pension_income_by_year[_early_idx]
    _late_pension = pension_income_by_year[-1]
    st.info(
        f"Gross income needed: £{annual_spending * tax_gross_up:,.0f}/yr "
        f"(£{annual_spending:,.0f} spending + {_blended_tax_rate(annual_spending):.0%} tax). "
        f"Early retirement: pensions £{_early_pension:,.0f} + portfolio £{_early_draw:,.0f}. "
        f"After all pensions: pensions £{_late_pension:,.0f} + portfolio £{_late_draw:,.0f}."
    )

_tax_rate = _blended_tax_rate(annual_spending)
if _tax_rate > 0:
    _total_gross = annual_spending / (1 - _tax_rate)
    st.info(
        f"Tax on all income (pensions + portfolio): {_tax_rate:.0%} blended rate on £{annual_spending:,.0f}/yr spending. "
        f"Total gross income needed: £{_total_gross:,.0f}/yr. "
        f"Pensions contribute gross, portfolio covers the remainder."
    )

# Interpretation
with st.expander("Understanding the Results"):
    member_lines = ""
    for member in household.members:
        sp = member.social_security_annual
        db = member.pension_annual
        sp_age = getattr(member, "social_security_age", None) or 67
        member_lines += f"\n    - **{member.name}**:"
        if sp > 0:
            member_lines += f" State Pension £{sp:,.0f}/yr from age {sp_age}"
        if db > 0:
            db_age = getattr(member, "pension_start_age", None) or 60
            linked = " (inflation-linked)" if getattr(member, "pension_inflation_linked", False) else " (fixed)"
            member_lines += f", DB Pension £{db:,.0f}/yr from age {db_age}{linked}"

    ret_line = f"**Retirement Year**: {retirement_year}" if retirement_year else "**Retirement**: already retired"

    st.markdown(f"""
    ### Simulation Summary

    - **Initial Portfolio**: £{initial_wealth:,.0f}
    - **Annual Spending Need**: £{annual_spending:,.0f} ({withdrawal_rate:.1f}% of portfolio)
    - **Essential Spending Floor**: £{spending_floor:,.0f}
    - **Stock Allocation**: {stock_allocation}%
    - {ret_line}
    - **Current Age**: {starting_age}
    - **Planning to Age**: {starting_age + n_years}
    {member_lines}

    Before retirement, salary covers all household spending — no portfolio draws.
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
