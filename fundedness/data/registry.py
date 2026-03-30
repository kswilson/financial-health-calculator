"""Market data registry — maps market source names to data modules.

Supported sources:
    - "uk": FTSE All-Share + UK Gilts (1970–2024)
    - "us": S&P 500 + US 10-Year Treasury (1970–2024)
    - "world": MSCI World + Global Government Bonds (1970–2024)
    - "50/50": Equal-weighted blend of UK and World data
"""

from typing import Literal

import numpy as np

from fundedness.models.market import MarketModel

MarketSource = Literal["uk", "us", "world", "50/50"]

MARKET_LABELS: dict[str, str] = {
    "uk": "UK (FTSE All-Share + Gilts)",
    "us": "US (S&P 500 + Treasuries)",
    "world": "World (MSCI World + Global Bonds)",
    "50/50": "50/50 UK + World",
}


def _get_module(source: MarketSource):
    """Lazily import and return the correct data module."""
    if source == "uk":
        from fundedness.data import uk_market
        return uk_market
    elif source == "us":
        from fundedness.data import us_market
        return us_market
    elif source == "world":
        from fundedness.data import world_market
        return world_market
    elif source == "50/50":
        return None  # Handled specially via blending
    else:
        raise ValueError(f"Unknown market source: {source!r}")


def get_equity_stats(source: MarketSource) -> dict:
    """Get summary statistics for equity returns from the given market."""
    if source == "50/50":
        from fundedness.data import uk_market, world_market
        blended = 0.5 * uk_market.UK_EQUITY_REAL_RETURNS + 0.5 * world_market.WORLD_EQUITY_REAL_RETURNS
        return uk_market.compute_statistics(blended)
    mod = _get_module(source)
    equity_arr = _get_equity_array(source)
    return mod.compute_statistics(equity_arr)


def get_bond_stats(source: MarketSource) -> dict:
    """Get summary statistics for bond returns from the given market."""
    if source == "50/50":
        from fundedness.data import uk_market, world_market
        blended = 0.5 * uk_market.UK_GILT_REAL_RETURNS + 0.5 * world_market.WORLD_BOND_REAL_RETURNS
        return uk_market.compute_statistics(blended)
    mod = _get_module(source)
    bond_arr = _get_bond_array(source)
    return mod.compute_statistics(bond_arr)


def _get_equity_array(source: MarketSource) -> np.ndarray:
    if source == "uk":
        from fundedness.data.uk_market import UK_EQUITY_REAL_RETURNS
        return UK_EQUITY_REAL_RETURNS
    elif source == "us":
        from fundedness.data.us_market import US_EQUITY_REAL_RETURNS
        return US_EQUITY_REAL_RETURNS
    elif source == "world":
        from fundedness.data.world_market import WORLD_EQUITY_REAL_RETURNS
        return WORLD_EQUITY_REAL_RETURNS
    elif source == "50/50":
        from fundedness.data.uk_market import UK_EQUITY_REAL_RETURNS
        from fundedness.data.world_market import WORLD_EQUITY_REAL_RETURNS
        return 0.5 * UK_EQUITY_REAL_RETURNS + 0.5 * WORLD_EQUITY_REAL_RETURNS
    raise ValueError(f"Unknown source: {source!r}")


def _get_bond_array(source: MarketSource) -> np.ndarray:
    if source == "uk":
        from fundedness.data.uk_market import UK_GILT_REAL_RETURNS
        return UK_GILT_REAL_RETURNS
    elif source == "us":
        from fundedness.data.us_market import US_BOND_REAL_RETURNS
        return US_BOND_REAL_RETURNS
    elif source == "world":
        from fundedness.data.world_market import WORLD_BOND_REAL_RETURNS
        return WORLD_BOND_REAL_RETURNS
    elif source == "50/50":
        from fundedness.data.uk_market import UK_GILT_REAL_RETURNS
        from fundedness.data.world_market import WORLD_BOND_REAL_RETURNS
        return 0.5 * UK_GILT_REAL_RETURNS + 0.5 * WORLD_BOND_REAL_RETURNS
    raise ValueError(f"Unknown source: {source!r}")


def calibrate_market_model(source: MarketSource) -> MarketModel:
    """Create a MarketModel calibrated from the chosen historical dataset."""
    eq = get_equity_stats(source)
    bond = get_bond_stats(source)
    equity_arr = _get_equity_array(source)
    bond_arr = _get_bond_array(source)
    correlation = float(np.corrcoef(equity_arr, bond_arr)[0, 1])

    return MarketModel(
        stock_return=round(eq["geometric_mean"], 4),
        bond_return=round(bond["geometric_mean"], 4),
        stock_volatility=round(eq["volatility"], 4),
        bond_volatility=round(bond["volatility"], 4),
        stock_bond_correlation=round(correlation, 2),
        inflation_mean=0.025,
        real_discount_rate=round(bond["geometric_mean"], 4),
    )


def get_bootstrap_returns(
    source: MarketSource,
    n_simulations: int,
    n_years: int,
    random_seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap equity and bond returns from the chosen market dataset.

    Returns:
        Tuple of (equity_returns, bond_returns), each shape (n_simulations, n_years)
    """
    equity_arr = _get_equity_array(source)
    bond_arr = _get_bond_array(source)

    rng = np.random.default_rng(random_seed)
    n_history = len(equity_arr)
    indices = rng.integers(0, n_history, size=(n_simulations, n_years))

    return equity_arr[indices], bond_arr[indices]


def get_bootstrap_portfolio_returns(
    source: MarketSource,
    n_simulations: int,
    n_years: int,
    stock_weight: float,
    random_seed: int | None = None,
) -> np.ndarray:
    """Generate blended portfolio returns by bootstrapping the chosen market."""
    equity, bonds = get_bootstrap_returns(source, n_simulations, n_years, random_seed)
    return stock_weight * equity + (1 - stock_weight) * bonds
