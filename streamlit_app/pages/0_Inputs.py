# UK version — first customisation
"""Inputs page for asset, liability, and assumption entry."""

import datetime
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
from pydantic import TypeAdapter

from fundedness.models.household import Household, Person
from fundedness.models.market import MarketModel
from fundedness.models.tax import TaxModel
from streamlit_app.components.asset_editor import render_asset_editor
from streamlit_app.components.liability_editor import render_liability_editor
from streamlit_app.utils.session_state import (
    get_household,
    get_market_model,
    initialize_session_state,
    update_household,
    update_market_model,
)

st.set_page_config(page_title="Inputs", page_icon="📝", layout="wide")

initialize_session_state()

st.title("📝 Inputs")

st.markdown("""
Enter your financial information below. Changes are automatically saved.
""")

# Tabs for different input sections
tab1, tab2, tab_savings, tab3, tab4 = st.tabs([
    "👤 Personal Info",
    "💰 Assets",
    "💷 Pre-Retirement Savings",
    "📋 Spending & One-offs",
    "⚙️ Assumptions",
])

household = get_household()

# UK State Pension age based on date of birth
def _uk_state_pension_age(dob: datetime.date) -> int:
    if dob < datetime.date(1954, 10, 6):
        return 66
    elif dob <= datetime.date(1960, 4, 5):
        return 66
    elif dob <= datetime.date(1961, 3, 5):
        # Transitional: rises from 66 to 67
        months_after = (
            (dob.year - 1960) * 12 + dob.month
            - (4 * 12 + 4)  # months after Apr 2060
        )
        return 66 if months_after < 6 else 67
    elif dob <= datetime.date(1977, 4, 5):
        return 67
    else:
        return 68  # Legislated rise to 68 (2044-2046)


def _render_member_editor(member: Person, idx: int) -> Person:
    """Render editor fields for a single household member."""
    col1, col2 = st.columns(2)

    with col1:
        name = st.text_input("Name", value=member.name, key=f"member_name_{idx}")
        default_dob = getattr(member, "date_of_birth", None) or datetime.date(
            datetime.date.today().year - member.age, 1, 1
        )
        dob = st.date_input(
            "Date of Birth",
            value=default_dob,
            min_value=datetime.date(1920, 1, 1),
            max_value=datetime.date.today(),
            key=f"member_dob_{idx}",
        )
        today = datetime.date.today()
        age = (
            today.year - dob.year
            - ((today.month, today.day) < (dob.month, dob.day))
        )
        st.metric("Current Age", f"{age}")
        life_expectancy = st.number_input(
            "Planning Life Expectancy",
            value=member.life_expectancy,
            min_value=age + 1,
            max_value=120,
            key=f"member_le_{idx}",
        )

    with col2:
        spa = _uk_state_pension_age(dob)
        st.metric("State Pension Age", f"{spa}")
        ss_annual = st.number_input(
            "Expected Annual State Pension (£)",
            value=int(member.social_security_annual),
            min_value=0,
            step=1000,
            help="Full new State Pension is £11,973/year (2025/26)",
            key=f"member_sp_{idx}",
        )
        pension = st.number_input(
            "Defined Benefit Private Pension (£)",
            value=int(member.pension_annual),
            min_value=0,
            step=1000,
            key=f"member_dbp_{idx}",
        )
        pension_start_age = st.number_input(
            "DB Pension Start Age",
            value=getattr(member, "pension_start_age", None) or 60,
            min_value=50,
            max_value=75,
            key=f"member_dbp_age_{idx}",
        )
        pension_inflation_linked = st.checkbox(
            "Inflation-linked",
            value=getattr(member, "pension_inflation_linked", False),
            help="Tick if your DB pension increases with CPI/RPI. "
                 "If not, its real value will erode over time.",
            key=f"member_dbp_linked_{idx}",
        )

    # Thai Severance Pay
    has_severance = st.checkbox(
        "Include Thai severance pay",
        value=getattr(member, "thai_severance_enabled", False),
        help="Thai Labour Protection Act mandatory severance (lump sum at retirement).",
        key=f"member_thai_sev_{idx}",
    )

    thai_wage = 0
    thai_start = datetime.date.today().year
    thai_fx = 44.0
    if has_severance:
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            thai_wage = st.number_input(
                "Monthly wage (THB)",
                value=int(getattr(member, "thai_monthly_wage_thb", 0) or 100000),
                min_value=0,
                step=5000,
                key=f"member_thai_wage_{idx}",
            )
        with col_s2:
            thai_start = st.number_input(
                "Employment start year",
                value=getattr(member, "thai_employment_start_year", None) or 2015,
                min_value=1970,
                max_value=datetime.date.today().year,
                key=f"member_thai_start_{idx}",
            )
        with col_s3:
            thai_fx = st.number_input(
                "THB per £1",
                value=getattr(member, "thai_thb_per_gbp", 44.0),
                min_value=1.0,
                step=0.5,
                format="%.1f",
                key=f"member_thai_fx_{idx}",
            )

    person = Person(
        name=name,
        date_of_birth=dob,
        age=age,
        life_expectancy=life_expectancy,
        social_security_age=spa,
        social_security_annual=float(ss_annual),
        pension_annual=float(pension),
        pension_start_age=pension_start_age,
        pension_inflation_linked=pension_inflation_linked,
        is_primary=(idx == 0),
        thai_severance_enabled=has_severance,
        thai_monthly_wage_thb=float(thai_wage),
        thai_employment_start_year=thai_start if has_severance else None,
        thai_thb_per_gbp=thai_fx,
    )

    if has_severance:
        retirement_yr = st.session_state.get("retirement_year", None)
        sev_gbp = person.thai_severance_gbp(retirement_yr)
        daily_wage = thai_wage / 30
        sev_thb = sev_gbp * thai_fx
        end_yr = retirement_yr or datetime.date.today().year
        yos = end_yr - thai_start
        col_r1, col_r2, col_r3 = st.columns(3)
        col_r1.metric("Years of Service", f"{yos}")
        col_r2.metric("Severance (THB)", f"฿{sev_thb:,.0f}")
        col_r3.metric("Severance (GBP)", f"£{sev_gbp:,.0f}")

    return person


# Personal Info Tab
with tab1:
    st.subheader("Personal Information")

    # Household retirement year
    current_year = datetime.date.today().year
    stored_ret_year = st.session_state.get("retirement_year", current_year + 5)
    already_retired = st.checkbox(
        "Already retired",
        value=(stored_ret_year is None),
        key="household_retired",
    )
    if already_retired:
        st.session_state.retirement_year = None
    else:
        st.session_state.retirement_year = st.number_input(
            "Planned Retirement Year",
            value=stored_ret_year if stored_ret_year is not None else current_year + 5,
            min_value=current_year,
            max_value=current_year + 40,
            key="household_ret_year",
        )


    st.divider()

    if household.members:
        members_changed = False

        for idx, member in enumerate(household.members):
            label = "You" if idx == 0 else "Partner"
            st.markdown(f"#### {label}")
            updated = _render_member_editor(member, idx)
            if updated != member:
                household.members[idx] = updated
                members_changed = True

            # Remove partner button
            if idx > 0:
                if st.button("Remove Partner", key=f"remove_member_{idx}"):
                    household.members.pop(idx)
                    # Clear owner on assets that referenced this partner
                    for asset in household.balance_sheet.assets:
                        if asset.owner == member.name:
                            asset.owner = household.members[0].name
                    update_household(household)
                    st.rerun()

            if idx < len(household.members) - 1:
                st.divider()

        # Add partner button (max 2 members)
        if len(household.members) < 2:
            if st.button("Add Partner"):
                household.members.append(
                    Person(
                        name="Partner",
                        age=55,
                        retirement_age=None,
                        life_expectancy=95,
                        social_security_age=67,
                        social_security_annual=11973,
                        pension_annual=0,
                        pension_start_age=60,
                        is_primary=False,
                    )
                )
                update_household(household)
                st.rerun()

        if members_changed:
            update_household(household)

# Assets Tab
with tab2:
    member_names = [m.name for m in household.members]
    new_balance_sheet = render_asset_editor(household.balance_sheet, member_names=member_names)

    if new_balance_sheet != household.balance_sheet:
        household.balance_sheet = new_balance_sheet
        update_household(household)

# Pre-Retirement Savings Tab
with tab_savings:
    from fundedness.models.household import SavingsContribution

    st.subheader("Pre-Retirement Savings")
    retirement_yr = st.session_state.get("retirement_year", None)
    if retirement_yr is None:
        st.info("You are marked as already retired — savings contributions don't apply.")
    else:
        st.caption(
            f"Annual savings that flow into your portfolio until retirement ({retirement_yr}). "
            "These stop automatically at the retirement date."
        )
        if any("rent" in s.name.lower() for s in household.savings_contributions):
            st.warning(
                "**Rental income is modelled as a savings stream and stops at retirement.** "
                "That's right if the property will be sold at retirement (e.g. to buy somewhere to live) — "
                "neither the sale proceeds nor the new home enter the portfolio. "
                "If the property is kept and the rent continues into retirement, enter it "
                "as pension-style income for the member instead."
            )

    member_names = [m.name for m in household.members]
    savings = list(household.savings_contributions)
    savings_changed = False

    if savings:
        for i, s in enumerate(savings):
            owner_label = f" ({s.member_name})" if s.member_name and len(member_names) > 1 else ""
            with st.expander(f"{s.name}{owner_label}", expanded=False):
                st.caption(f"£{s.annual_amount:,.0f}/yr")
                col1, col2 = st.columns(2)
                with col1:
                    new_name = st.text_input("Description", value=s.name, key=f"sav_name_{i}")
                    new_amount = st.number_input(
                        "Annual Amount (£)", value=int(s.annual_amount),
                        min_value=0, step=1000, key=f"sav_amount_{i}",
                    )
                with col2:
                    new_growth = st.slider(
                        "Real Growth Rate",
                        min_value=0.0, max_value=5.0,
                        value=s.growth_rate * 100,
                        step=0.5, format="%.1f%%",
                        help="Annual real increase (e.g. salary growth above inflation)",
                        key=f"sav_growth_{i}",
                    ) / 100
                    new_member = s.member_name
                    if len(member_names) > 1:
                        cur_idx = member_names.index(s.member_name) if s.member_name in member_names else 0
                        new_member = st.selectbox(
                            "Member", options=member_names, index=cur_idx, key=f"sav_member_{i}",
                        )

                updated_s = SavingsContribution(
                    name=new_name, annual_amount=float(new_amount),
                    growth_rate=new_growth, member_name=new_member,
                )
                if updated_s != s:
                    savings[i] = updated_s
                    savings_changed = True

                if st.button("Delete", key=f"sav_del_{i}"):
                    savings.pop(i)
                    household.savings_contributions = savings
                    update_household(household)
                    st.rerun()

    st.divider()
    if st.button("Add Savings Stream"):
        savings.append(SavingsContribution(
            name="New Savings",
            annual_amount=10000,
            growth_rate=0.0,
            member_name=member_names[0] if member_names else None,
        ))
        household.savings_contributions = savings
        update_household(household)
        st.rerun()

    if savings_changed:
        household.savings_contributions = savings
        update_household(household)

    total_annual = sum(s.annual_amount for s in savings)
    st.metric("Total Annual Savings", f"£{total_annual:,.0f}")

# Spending Tab
with tab3:
    new_liabilities = render_liability_editor(household.liabilities)

    if new_liabilities != household.liabilities:
        household.liabilities = new_liabilities
        update_household(household)

# Assumptions Tab
with tab4:
    st.subheader("Market Assumptions")

    from fundedness.data.registry import (
        MARKET_LABELS,
        calibrate_market_model as calibrate_from_source,
        get_bond_stats,
        get_equity_stats,
    )

    # Market data source selector
    EQUITY_LABELS = {
        "uk": "UK Equities (FTSE All-Share)",
        "us": "US Equities (S&P 500)",
        "world": "World Equities (MSCI World)",
        "50/50": "50/50 UK + World Equities",
    }
    BOND_LABELS = {
        "uk": "UK Gilts (Government Bonds)",
        "us": "US Treasuries (10-Year)",
        "world": "Global Government Bonds",
        "50/50": "50/50 UK Gilts + Global Bonds",
    }

    _market_keys = list(MARKET_LABELS.keys())
    _saved_source = st.session_state.get("market_source", "uk")
    _source_idx = _market_keys.index(_saved_source) if _saved_source in _market_keys else 0

    market_source = st.radio(
        "Market data",
        options=_market_keys,
        format_func=lambda k: MARKET_LABELS[k],
        index=_source_idx,
        horizontal=True,
        help="Choose which historical market dataset to use for calibration and bootstrap sampling.",
    )

    _return_options = ["Historical Bootstrap", "Parametric"]
    _saved_return = st.session_state.get("return_model", "bootstrap")
    _return_idx = 0 if _saved_return == "bootstrap" else 1

    return_model = st.radio(
        "Return model",
        options=_return_options,
        index=_return_idx,
        horizontal=True,
        help="**Historical Bootstrap**: samples actual return years with replacement "
             "(preserves real-world fat tails and equity/bond correlation). "
             "**Parametric**: generates returns from a statistical distribution with "
             "the parameters below.",
    )

    # Show historical context for selected market
    eq_stats = get_equity_stats(market_source)
    bond_stats = get_bond_stats(market_source)
    calibrated = calibrate_from_source(market_source)

    st.markdown(
        f"Based on **{eq_stats['n_years']} years** of historical data (1970–2024):"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**{EQUITY_LABELS[market_source]}**")
        st.metric("Real Return (geometric)", f"{eq_stats['geometric_mean']:.1%}")
        st.metric("Volatility", f"{eq_stats['volatility']:.1%}")
        st.metric("Worst Year", f"{eq_stats['min']:.0%}")
        st.metric("Best Year", f"{eq_stats['max']:.0%}")

    with col2:
        st.markdown(f"**{BOND_LABELS[market_source]}**")
        st.metric("Real Return (geometric)", f"{bond_stats['geometric_mean']:.1%}")
        st.metric("Volatility", f"{bond_stats['volatility']:.1%}")
        st.metric("Worst Year", f"{bond_stats['min']:.0%}")
        st.metric("Best Year", f"{bond_stats['max']:.0%}")

    market_model = get_market_model()

    if return_model == "Historical Bootstrap":
        new_market_model = calibrated

        st.info(f"Simulations will sample from actual historical {MARKET_LABELS[market_source]} return years. "
                "No return/volatility assumptions needed.")
    else:
        st.markdown("---")
        st.markdown(f"Adjust parameters below, or leave at {MARKET_LABELS[market_source]} historical defaults:")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Expected Returns (Real)**")
            stock_return = st.slider(
                "Equities",
                min_value=0.0,
                max_value=0.15,
                value=calibrated.stock_return,
                step=0.005,
                format="%.1f%%",
            )
            bond_return = st.slider(
                "Bonds",
                min_value=-0.02,
                max_value=0.05,
                value=calibrated.bond_return,
                step=0.005,
                format="%.1f%%",
            )

            st.markdown("**Volatility**")
            stock_vol = st.slider(
                "Equity Volatility",
                min_value=0.10,
                max_value=0.30,
                value=calibrated.stock_volatility,
                step=0.01,
                format="%.0f%%",
            )

        with col2:
            st.markdown("**Inflation**")
            inflation = st.slider(
                "Expected Inflation",
                min_value=0.01,
                max_value=0.05,
                value=calibrated.inflation_mean,
                step=0.005,
                format="%.1f%%",
            )

            st.markdown("**Discount Rate**")
            discount = st.slider(
                "Real Discount Rate",
                min_value=0.0,
                max_value=0.05,
                value=calibrated.real_discount_rate,
                step=0.005,
                format="%.1f%%",
            )

            use_fat_tails = st.checkbox(
                "Use Fat Tails (t-distribution)",
                value=False,
            )

        new_market_model = MarketModel(
            stock_return=stock_return,
            bond_return=bond_return,
            stock_volatility=stock_vol,
            inflation_mean=inflation,
            real_discount_rate=discount,
            use_fat_tails=use_fat_tails,
        )

    # Store return model and market source in session state
    st.session_state.return_model = "bootstrap" if return_model == "Historical Bootstrap" else "lognormal"
    st.session_state.market_source = market_source

    if new_market_model != market_model:
        update_market_model(new_market_model)

    st.divider()
    st.subheader("Return Haircut")
    from streamlit_app.utils.session_state import get_return_haircut, set_return_haircut
    _haircut_pct = st.number_input(
        "Subtract from every simulated year's return (%/yr)",
        min_value=0.0,
        max_value=5.0,
        value=round(get_return_haircut() * 100, 2),
        step=0.25,
        format="%.2f",
        help="Applied to both the historical bootstrap and the parametric model.",
    )
    set_return_haircut(_haircut_pct / 100)
    st.caption(
        "The historical series are index returns with no costs, from a period (1970–2024) when "
        "equity valuations rose and interest rates fell — conditions that can't simply repeat. "
        "The haircut is a crude allowance for **fees and platform costs (~0.3–0.6%/yr)** plus "
        "**lower forward expected returns than history (~1–2%/yr)**. It shifts the centre of the "
        "return distribution only; it does not add crashes, bad decades or sequence risk. "
        "1% ≈ honest base case; 2% ≈ mainstream cautious assumptions; 0% ≈ the future is 1970–2024 with free investing."
    )

    st.subheader("Tax Assumptions")
    st.markdown("UK income tax is calculated using progressive bands. "
                "Each person's SIPP withdrawals are stacked on top of their own income.")

    tax_model = st.session_state.tax_model

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Income Tax Bands**")
        personal_allowance = st.number_input(
            "Personal Allowance (£)",
            value=int(getattr(tax_model, "personal_allowance", 12_570)),
            min_value=0,
            step=100,
            help="Tapers £1 for every £2 above £100k income",
        )

        # Show per-person other income (auto-computed from pensions)
        st.markdown("**Other Income (per person)**")
        for m in household.members:
            other_inc = m.social_security_annual + m.pension_annual
            st.metric(f"{m.name}", f"£{other_inc:,.0f}/year")

    with col2:
        st.markdown("**Capital Gains Tax**")
        cgt_annual_exempt = st.number_input(
            "CGT Annual Exempt Amount (£)",
            value=int(getattr(tax_model, "cgt_annual_exempt", 3_000)),
            min_value=0,
            step=500,
            help="Each person gets their own annual exempt amount",
        )
        st.caption("Basic rate: 18% / Higher rate: 24% (2025/26)")

    # Store base tax model (per-person stacking is done in fundedness.runway)
    st.session_state.tax_model = TaxModel(
        personal_allowance=float(personal_allowance),
        cgt_annual_exempt=float(cgt_annual_exempt),
    )

# Save / Load scenario
st.divider()
st.subheader("💾 Save / Load Scenario")

col_save, col_load = st.columns(2)

with col_save:
    import json as _json
    _full_scenario = {
        "household": _json.loads(household.model_dump_json()),
        "market_model": _json.loads(st.session_state.market_model.model_dump_json()),
        "tax_model": _json.loads(st.session_state.tax_model.model_dump_json()),
        "retirement_year": st.session_state.get("retirement_year", None),
        "market_source": st.session_state.get("market_source", "uk"),
        "return_model": st.session_state.get("return_model", "lognormal"),
        "return_haircut": st.session_state.get("return_haircut", 0.01),
    }
    st.download_button(
        label="💾 Download my full situation as JSON",
        data=_json.dumps(_full_scenario, indent=2),
        file_name="my_uk_retirement_scenario.json",
        mime="application/json",
    )

with col_load:
    import json as _json
    uploaded = st.file_uploader(
        "📤 Load previous scenario",
        type=["json"],
        key="scenario_upload",
    )
    if uploaded is not None:
        try:
            data = _json.loads(uploaded.read())
            # Support both old format (just Household) and new format (full scenario)
            if "household" in data:
                loaded = TypeAdapter(Household).validate_python(data["household"])
                update_household(loaded)
                if "market_model" in data:
                    st.session_state.market_model = TypeAdapter(MarketModel).validate_python(data["market_model"])
                if "tax_model" in data:
                    from fundedness.models.tax import TaxModel as _TaxModel
                    st.session_state.tax_model = TypeAdapter(_TaxModel).validate_python(data["tax_model"])
                if "retirement_year" in data:
                    st.session_state.retirement_year = data["retirement_year"]
                if "market_source" in data:
                    st.session_state.market_source = data["market_source"]
                if "return_model" in data:
                    st.session_state.return_model = data["return_model"]
                if "return_haircut" in data:
                    st.session_state.return_haircut = data["return_haircut"]
            else:
                # Old format: entire JSON is a Household
                loaded = TypeAdapter(Household).validate_python(data)
                update_household(loaded)
            st.success("Scenario loaded successfully!")
            st.rerun()
        except Exception as e:
            st.error(f"Invalid scenario file: {e}")

# Summary
st.divider()
st.subheader("Summary")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Assets", f"£{household.total_assets:,.0f}")
col2.metric("Annual Spending", f"£{household.total_spending:,.0f}")
col3.metric("Planning Horizon", f"{household.planning_horizon} years")
col4.metric("Withdrawal Rate", f"{household.total_spending / household.total_assets * 100:.1f}%" if household.total_assets > 0 else "N/A")

# Auto-save all inputs (captures tax_model, retirement_year, etc.)
from streamlit_app.utils.session_state import _save_state
_save_state()
