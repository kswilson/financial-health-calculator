"""Session state management for Streamlit app."""

import datetime
import json
from pathlib import Path

import streamlit as st
from pydantic import TypeAdapter

from fundedness.models.assets import (
    AccountType,
    Asset,
    AssetClass,
    BalanceSheet,
    ConcentrationLevel,
    LiquidityClass,
)
from fundedness.models.household import Household, Person
from fundedness.models.liabilities import InflationLinkage, Liability, LiabilityType
from fundedness.models.market import MarketModel
from fundedness.models.simulation import SimulationConfig
from fundedness.models.tax import TaxModel

_SAVE_DIR = Path(__file__).parent.parent / ".user_data"
_SAVE_FILE = _SAVE_DIR / "session_state.json"


def _save_state():
    """Persist current inputs to a local JSON file."""
    _SAVE_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        "household": json.loads(st.session_state.household.model_dump_json()),
        "market_model": json.loads(st.session_state.market_model.model_dump_json()),
        "tax_model": json.loads(st.session_state.tax_model.model_dump_json()),
        "simulation_config": json.loads(
            st.session_state.simulation_config.model_dump_json()
        ),
        "retirement_year": st.session_state.get("retirement_year", 2033),
        "market_source": st.session_state.get("market_source", "uk"),
        "return_model": st.session_state.get("return_model", "lognormal"),
    }
    _SAVE_FILE.write_text(json.dumps(data, indent=2))


def _load_saved_state() -> bool:
    """Load persisted state if available. Returns True if loaded."""
    if not _SAVE_FILE.exists():
        return False
    try:
        data = json.loads(_SAVE_FILE.read_text())
        st.session_state.household = TypeAdapter(Household).validate_python(
            data["household"]
        )
        st.session_state.market_model = TypeAdapter(MarketModel).validate_python(
            data["market_model"]
        )
        st.session_state.tax_model = TypeAdapter(TaxModel).validate_python(
            data["tax_model"]
        )
        st.session_state.simulation_config = TypeAdapter(
            SimulationConfig
        ).validate_python(data["simulation_config"])
        st.session_state.retirement_year = data.get("retirement_year", 2033)
        st.session_state.market_source = data.get("market_source", "uk")
        st.session_state.return_model = data.get("return_model", "lognormal")
        return True
    except Exception:
        return False


def initialize_session_state():
    """Initialize all session state variables with defaults."""
    if "initialized" in st.session_state:
        return

    # Try loading from saved state first
    if _load_saved_state():
        # Caches
        st.session_state.cefr_result = None
        st.session_state.simulation_result = None
        st.session_state.comparison_result = None
        st.session_state.initialized = True
        return

    # Default person
    if "household" not in st.session_state:
        st.session_state.household = Household(
            name="My Household",
            members=[
                Person(
                    name="Primary",
                    date_of_birth=datetime.date(1972, 2, 18),
                    age=54,
                    retirement_age=None,  # Already retired
                    life_expectancy=95,
                    social_security_age=67,
                    social_security_annual=11973,
                    pension_annual=9500,
                    pension_start_age=60,
                    is_primary=True,
                )
            ],
            balance_sheet=BalanceSheet(
                assets=[
                    Asset(
                        name="SIPP",
                        value=220000,
                        account_type=AccountType.SIPP,
                        asset_class=AssetClass.STOCKS,
                        liquidity_class=LiquidityClass.RETIREMENT,
                        concentration_level=ConcentrationLevel.DIVERSIFIED,
                    ),
                    Asset(
                        name="General Investment Account",
                        value=250000,
                        account_type=AccountType.GENERAL,
                        asset_class=AssetClass.STOCKS,
                        liquidity_class=LiquidityClass.TAXABLE_INDEX,
                        concentration_level=ConcentrationLevel.DIVERSIFIED,
                        cost_basis=150000,
                    ),
                    Asset(
                        name="Cash Savings",
                        value=20000,
                        account_type=AccountType.TAXABLE,
                        asset_class=AssetClass.CASH,
                        liquidity_class=LiquidityClass.CASH,
                        concentration_level=ConcentrationLevel.DIVERSIFIED,
                    ),
                ]
            ),
            liabilities=[
                Liability(
                    name="Essential Living Expenses",
                    liability_type=LiabilityType.ESSENTIAL_SPENDING,
                    annual_amount=50000,
                    is_essential=True,
                    inflation_linkage=InflationLinkage.CPI,
                ),
                Liability(
                    name="Discretionary Spending",
                    liability_type=LiabilityType.DISCRETIONARY_SPENDING,
                    annual_amount=20000,
                    is_essential=False,
                    inflation_linkage=InflationLinkage.CPI,
                ),
            ],
        )

    # Market assumptions
    if "market_model" not in st.session_state:
        st.session_state.market_model = MarketModel()

    # Tax assumptions
    if "tax_model" not in st.session_state:
        st.session_state.tax_model = TaxModel()

    # Simulation config
    if "simulation_config" not in st.session_state:
        st.session_state.simulation_config = SimulationConfig(
            n_simulations=5000,
            n_years=40,
        )

    # CEFR result cache
    if "cefr_result" not in st.session_state:
        st.session_state.cefr_result = None

    # Simulation result cache
    if "simulation_result" not in st.session_state:
        st.session_state.simulation_result = None

    # Strategy comparison cache
    if "comparison_result" not in st.session_state:
        st.session_state.comparison_result = None

    if "retirement_year" not in st.session_state:
        st.session_state.retirement_year = 2033

    st.session_state.initialized = True


def get_household() -> Household:
    """Get the current household from session state."""
    initialize_session_state()
    return st.session_state.household


def update_household(household: Household):
    """Update the household in session state and clear caches."""
    st.session_state.household = household
    st.session_state.cefr_result = None
    st.session_state.simulation_result = None
    _save_state()


def get_market_model() -> MarketModel:
    """Get the current market model from session state."""
    initialize_session_state()
    return st.session_state.market_model


def update_market_model(market_model: MarketModel):
    """Update market model and clear caches."""
    st.session_state.market_model = market_model
    st.session_state.simulation_result = None
    _save_state()


def get_simulation_config() -> SimulationConfig:
    """Get the current simulation config from session state."""
    initialize_session_state()
    return st.session_state.simulation_config


def clear_all_caches():
    """Clear all cached results."""
    st.session_state.cefr_result = None
    st.session_state.simulation_result = None
    st.session_state.comparison_result = None
