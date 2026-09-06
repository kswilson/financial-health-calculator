"""Session state management for Streamlit app."""

import datetime
import json
import os
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
from fundedness.models.household import Household, Person, SavingsContribution
from fundedness.models.liabilities import InflationLinkage, Liability, LiabilityType
from fundedness.models.market import MarketModel
from fundedness.models.simulation import SimulationConfig
from fundedness.models.tax import TaxModel

# Annual amount subtracted from every simulated return: costs + lower forward
# expected returns than the 1970-2024 history. Stored as a positive fraction.
DEFAULT_RETURN_HAIRCUT = 0.01

_SAVE_DIR = Path(__file__).parent.parent / ".user_data"
_SAVE_FILE = _SAVE_DIR / "session_state.json"


def is_shared_deployment() -> bool:
    """True when running somewhere other people use (e.g. Streamlit Community Cloud).

    In that case inputs are kept in the browser session only and never written
    to disk — the server filesystem would be shared between every visitor.
    Detected via the PENSION_PLANNER_SHARED env var / secret, or the
    /mount/src path Streamlit Cloud checks the repo out to.
    """
    if os.environ.get("PENSION_PLANNER_SHARED", "").lower() in ("1", "true", "yes"):
        return True
    try:
        if str(st.secrets.get("PENSION_PLANNER_SHARED", "")).lower() in ("1", "true", "yes"):
            return True
    except Exception:
        pass
    return str(Path(__file__).resolve()).startswith("/mount/src")


def default_retirement_year() -> int:
    return datetime.date.today().year + 5


def _save_state():
    """Persist current inputs to a local JSON file (local runs only)."""
    if is_shared_deployment():
        return
    _SAVE_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        "household": json.loads(st.session_state.household.model_dump_json()),
        "market_model": json.loads(st.session_state.market_model.model_dump_json()),
        "tax_model": json.loads(st.session_state.tax_model.model_dump_json()),
        "simulation_config": json.loads(
            st.session_state.simulation_config.model_dump_json()
        ),
        "retirement_year": st.session_state.get("retirement_year", default_retirement_year()),
        "market_source": st.session_state.get("market_source", "uk"),
        "return_model": st.session_state.get("return_model", "lognormal"),
        "return_haircut": st.session_state.get("return_haircut", DEFAULT_RETURN_HAIRCUT),
    }
    _SAVE_FILE.write_text(json.dumps(data, indent=2))


def _load_saved_state() -> bool:
    """Load persisted state if available. Returns True if loaded."""
    if is_shared_deployment() or not _SAVE_FILE.exists():
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
        st.session_state.retirement_year = data.get("retirement_year", default_retirement_year())
        st.session_state.market_source = data.get("market_source", "uk")
        st.session_state.return_model = data.get("return_model", "lognormal")
        st.session_state.return_haircut = data.get("return_haircut", DEFAULT_RETURN_HAIRCUT)
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
        st.session_state.simulation_result = None
        st.session_state.comparison_result = None
        st.session_state.initialized = True
        return

    # Illustrative default household — a generic UK example, deliberately not anyone's real numbers
    if "household" not in st.session_state:
        st.session_state.household = Household(
            name="Example Household",
            members=[
                Person(
                    name="You",
                    age=55,
                    retirement_age=None,
                    life_expectancy=95,
                    social_security_age=67,
                    social_security_annual=11973,  # full new State Pension, 2025/26
                    pension_annual=0,
                    pension_start_age=60,
                    is_primary=True,
                )
            ],
            balance_sheet=BalanceSheet(
                assets=[
                    Asset(
                        name="SIPP",
                        value=300000,
                        account_type=AccountType.SIPP,
                        asset_class=AssetClass.STOCKS,
                        liquidity_class=LiquidityClass.RETIREMENT,
                        concentration_level=ConcentrationLevel.DIVERSIFIED,
                    ),
                    Asset(
                        name="Stocks & Shares ISA",
                        value=150000,
                        account_type=AccountType.ISA,
                        asset_class=AssetClass.STOCKS,
                        liquidity_class=LiquidityClass.TAXABLE_INDEX,
                        concentration_level=ConcentrationLevel.DIVERSIFIED,
                    ),
                    Asset(
                        name="General Investment Account",
                        value=100000,
                        account_type=AccountType.GENERAL,
                        asset_class=AssetClass.STOCKS,
                        liquidity_class=LiquidityClass.TAXABLE_INDEX,
                        concentration_level=ConcentrationLevel.DIVERSIFIED,
                        cost_basis=70000,
                    ),
                    Asset(
                        name="Cash Savings",
                        value=25000,
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
                    annual_amount=26000,
                    is_essential=True,
                    inflation_linkage=InflationLinkage.CPI,
                ),
                Liability(
                    name="Discretionary Spending",
                    liability_type=LiabilityType.DISCRETIONARY_SPENDING,
                    annual_amount=8000,
                    is_essential=False,
                    inflation_linkage=InflationLinkage.CPI,
                ),
            ],
            savings_contributions=[
                SavingsContribution(name="Pension & ISA contributions", annual_amount=25000, growth_rate=0.0),
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

    # Simulation result cache
    if "simulation_result" not in st.session_state:
        st.session_state.simulation_result = None

    # Strategy comparison cache
    if "comparison_result" not in st.session_state:
        st.session_state.comparison_result = None

    if "retirement_year" not in st.session_state:
        st.session_state.retirement_year = default_retirement_year()

    if "return_haircut" not in st.session_state:
        st.session_state.return_haircut = DEFAULT_RETURN_HAIRCUT

    st.session_state.initialized = True


def get_household() -> Household:
    """Get the current household from session state."""
    initialize_session_state()
    return st.session_state.household


def update_household(household: Household):
    """Update the household in session state and clear caches."""
    st.session_state.household = household
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


def get_tax_model() -> TaxModel:
    """Get the current UK tax model from session state."""
    initialize_session_state()
    return st.session_state.tax_model


def get_return_haircut() -> float:
    """Annual return haircut as a positive fraction (0.01 = 1%/yr)."""
    initialize_session_state()
    return float(st.session_state.get("return_haircut", DEFAULT_RETURN_HAIRCUT))


def set_return_haircut(value: float):
    """Store the return haircut and persist it."""
    if st.session_state.get("return_haircut") != value:
        st.session_state.return_haircut = value
        st.session_state.simulation_result = None
        _save_state()


def get_simulation_config() -> SimulationConfig:
    """Get the current simulation config from session state."""
    initialize_session_state()
    return st.session_state.simulation_config


def clear_all_caches():
    """Clear all cached results."""
    st.session_state.simulation_result = None
    st.session_state.comparison_result = None
