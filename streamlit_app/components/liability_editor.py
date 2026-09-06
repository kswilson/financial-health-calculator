"""Spending & one-off cost editor component for Streamlit."""

import datetime

import streamlit as st

from fundedness.models.liabilities import InflationLinkage, Liability, LiabilityType
from fundedness.runway import is_ongoing

_INFLATION_HELP = (
    "CPI: keeps its value in today's pounds (the normal choice). "
    "NONE: a fixed cash amount that inflation erodes over time."
)


def _save_liabilities(liabilities: list[Liability]):
    """Persist liabilities to session state before a rerun (rerun raises, so the
    caller never gets the returned list)."""
    from streamlit_app.utils.session_state import update_household
    household = st.session_state.household
    household.liabilities = liabilities
    update_household(household)


def _describe(liability: Liability, current_year: int) -> str:
    """One-line summary of when a liability applies."""
    amount = f"£{liability.annual_amount:,.0f}/year"
    if is_ongoing(liability):
        return f"{amount}, ongoing for life"
    first = current_year + liability.start_year
    if liability.end_year is None:
        return f"{amount} from {first} for life"
    n = liability.end_year - liability.start_year
    last = current_year + liability.end_year - 1
    if n <= 1:
        return f"{amount}, one-off in {first}"
    return f"{amount} from {first} to {last} ({n} years, £{liability.annual_amount * n:,.0f} total)"


def render_liability_editor(liabilities: list[Liability]) -> list[Liability]:
    """Render an editable list of spending items and dated one-off costs.

    Args:
        liabilities: Current list of liabilities

    Returns:
        Updated list of liabilities
    """
    current_year = datetime.date.today().year

    st.subheader("Spending & One-off Costs")
    st.caption(
        "Ongoing items are your regular living costs — salary covers them until retirement, "
        "then pensions and the portfolio do. Dated items (university fees, a car, a wedding) "
        "are drawn from the portfolio in the years they fall, before or after retirement."
    )

    updated_liabilities = liabilities.copy()

    if updated_liabilities:
        for i, liability in enumerate(updated_liabilities):
            with st.expander(liability.name, expanded=False):
                st.caption(_describe(liability, current_year))
                col1, col2 = st.columns(2)

                with col1:
                    new_name = st.text_input(
                        "Name",
                        value=liability.name,
                        key=f"liability_name_{i}",
                    )
                    new_amount = st.number_input(
                        "Annual Amount (£)",
                        value=int(liability.annual_amount),
                        min_value=0,
                        step=1000,
                        key=f"liability_amount_{i}",
                    )
                    new_type = st.selectbox(
                        "Type",
                        options=list(LiabilityType),
                        index=list(LiabilityType).index(liability.liability_type),
                        format_func=lambda x: x.value.replace("_", " ").title(),
                        key=f"liability_type_{i}",
                    )

                with col2:
                    ongoing = st.checkbox(
                        "Ongoing (starts now, runs for life)",
                        value=is_ongoing(liability),
                        key=f"liability_ongoing_{i}",
                    )
                    if ongoing:
                        new_start, new_end = 0, None
                    else:
                        first_year = st.number_input(
                            "First year",
                            value=current_year + liability.start_year,
                            min_value=current_year,
                            max_value=current_year + 60,
                            key=f"liability_first_{i}",
                        )
                        default_last = (
                            current_year + liability.end_year - 1
                            if liability.end_year is not None
                            else first_year
                        )
                        last_year = st.number_input(
                            "Last year (inclusive)",
                            value=max(default_last, first_year),
                            min_value=first_year,
                            max_value=current_year + 80,
                            key=f"liability_last_{i}",
                            help="Same as first year for a single one-off payment.",
                        )
                        new_start = first_year - current_year
                        new_end = last_year - current_year + 1
                    new_inflation = st.selectbox(
                        "Inflation Linkage",
                        options=list(InflationLinkage),
                        index=list(InflationLinkage).index(liability.inflation_linkage),
                        format_func=lambda x: x.value.upper(),
                        key=f"liability_inflation_{i}",
                        help=_INFLATION_HELP,
                    )

                new_essential = st.checkbox(
                    "Essential (Floor) Spending",
                    value=liability.is_essential,
                    key=f"liability_essential_{i}",
                    help="Essential items define the spending floor used for the floor-breach rate.",
                )

                # Update liability
                updated_liabilities[i] = Liability(
                    name=new_name,
                    liability_type=new_type,
                    annual_amount=float(new_amount),
                    start_year=new_start,
                    end_year=new_end,
                    inflation_linkage=new_inflation,
                    is_essential=new_essential,
                )

                # Delete button
                if st.button("Delete", key=f"delete_liability_{i}"):
                    updated_liabilities.pop(i)
                    _save_liabilities(updated_liabilities)
                    st.rerun()

    # Add buttons
    st.divider()
    col_a, col_b = st.columns(2)
    if col_a.button("Add Ongoing Spending"):
        updated_liabilities.append(
            Liability(
                name="New Expense",
                liability_type=LiabilityType.ESSENTIAL_SPENDING,
                annual_amount=10000,
                is_essential=True,
            )
        )
        _save_liabilities(updated_liabilities)
        st.rerun()
    if col_b.button("Add One-off / Dated Cost"):
        updated_liabilities.append(
            Liability(
                name="New One-off Cost",
                liability_type=LiabilityType.DISCRETIONARY_SPENDING,
                annual_amount=10000,
                start_year=1,
                end_year=2,
                is_essential=True,
            )
        )
        _save_liabilities(updated_liabilities)
        st.rerun()

    # Summary
    ongoing_items = [l for l in updated_liabilities if is_ongoing(l)]
    dated_items = [l for l in updated_liabilities if not is_ongoing(l)]
    essential = sum(l.annual_amount for l in ongoing_items if l.is_essential)
    discretionary = sum(l.annual_amount for l in ongoing_items if not l.is_essential)

    col1, col2, col3 = st.columns(3)
    col1.metric("Essential Spending", f"£{essential:,.0f}/year")
    col2.metric("Discretionary", f"£{discretionary:,.0f}/year")
    col3.metric("Ongoing Total", f"£{essential + discretionary:,.0f}/year")

    if dated_items:
        st.markdown("**Dated / one-off costs** (drawn from the portfolio in the years shown):")
        for l in dated_items:
            st.markdown(f"- **{l.name}**: {_describe(l, current_year)}")

    return updated_liabilities
