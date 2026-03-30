"""Historical global market return data.

Sources:
    - Credit Suisse / UBS Global Investment Returns Yearbook (Dimson, Marsh, Staunton)
    - MSCI World Index (from 1970, developed markets)
    - Barclays Global Aggregate Bond Index
    - World Bank / IMF global inflation data

All returns are REAL (inflation-adjusted, USD-based) total returns, annual.
Equity = MSCI World (developed markets, net of local inflation).
Bonds = Global government bond composite.
"""

import numpy as np
from fundedness.models.market import MarketModel


# Annual real total returns: World equities and bonds, 1970–2024
# Based on MSCI World and global bond composites (DMS Yearbook data)
YEARS = list(range(1970, 2025))

# MSCI World real total returns (USD-based, inflation-adjusted)
WORLD_EQUITY_REAL_RETURNS = np.array([
    -0.050,  # 1970
    0.190,   # 1971
    0.170,   # 1972
    -0.200,  # 1973
    -0.310,  # 1974
    0.230,   # 1975
    0.070,   # 1976
    0.050,   # 1977
    0.130,   # 1978
    0.060,   # 1979
    0.200,   # 1980
    -0.050,  # 1981
    0.110,   # 1982
    0.190,   # 1983
    0.020,   # 1984
    0.340,   # 1985
    0.330,   # 1986
    0.130,   # 1987
    0.170,   # 1988
    0.130,   # 1989
    -0.190,  # 1990
    0.140,   # 1991
    -0.070,  # 1992
    0.190,   # 1993
    0.030,   # 1994
    0.170,   # 1995
    0.115,   # 1996
    0.130,   # 1997
    0.215,   # 1998
    0.225,   # 1999
    -0.150,  # 2000
    -0.185,  # 2001
    -0.210,  # 2002
    0.290,   # 2003
    0.120,   # 2004
    0.070,   # 2005
    0.170,   # 2006
    0.060,   # 2007
    -0.400,  # 2008
    0.260,   # 2009
    0.090,   # 2010
    -0.070,  # 2011
    0.130,   # 2012
    0.230,   # 2013
    0.030,   # 2014
    -0.020,  # 2015
    0.055,   # 2016
    0.200,   # 2017
    -0.110,  # 2018
    0.250,   # 2019
    0.140,   # 2020
    0.190,   # 2021
    -0.195,  # 2022
    0.205,   # 2023
    0.170,   # 2024
])

# Global government bond real total returns
WORLD_BOND_REAL_RETURNS = np.array([
    0.010,   # 1970
    0.060,   # 1971
    -0.010,  # 1972
    -0.050,  # 1973
    -0.080,  # 1974
    0.020,   # 1975
    0.030,   # 1976
    -0.010,  # 1977
    -0.030,  # 1978
    -0.050,  # 1979
    0.020,   # 1980
    -0.020,  # 1981
    0.300,   # 1982
    0.050,   # 1983
    0.070,   # 1984
    0.180,   # 1985
    0.130,   # 1986
    0.030,   # 1987
    0.030,   # 1988
    0.060,   # 1989
    0.010,   # 1990
    0.120,   # 1991
    0.060,   # 1992
    0.130,   # 1993
    -0.070,  # 1994
    0.170,   # 1995
    0.020,   # 1996
    0.070,   # 1997
    0.120,   # 1998
    -0.040,  # 1999
    0.080,   # 2000
    0.010,   # 2001
    0.080,   # 2002
    0.010,   # 2003
    0.030,   # 2004
    0.020,   # 2005
    -0.010,  # 2006
    0.050,   # 2007
    0.100,   # 2008
    -0.020,  # 2009
    0.040,   # 2010
    0.070,   # 2011
    0.020,   # 2012
    -0.040,  # 2013
    0.080,   # 2014
    -0.010,  # 2015
    0.030,   # 2016
    -0.010,  # 2017
    -0.010,  # 2018
    0.050,   # 2019
    0.060,   # 2020
    -0.070,  # 2021
    -0.200,  # 2022
    0.000,   # 2023
    -0.010,  # 2024
])


def compute_statistics(returns: np.ndarray) -> dict:
    """Compute key statistics from a return series."""
    geo_mean = np.exp(np.mean(np.log(1 + returns))) - 1
    arith_mean = np.mean(returns)
    volatility = np.std(returns, ddof=1)
    return {
        "geometric_mean": geo_mean,
        "arithmetic_mean": arith_mean,
        "volatility": volatility,
        "min": np.min(returns),
        "max": np.max(returns),
        "n_years": len(returns),
    }


def get_world_equity_stats() -> dict:
    return compute_statistics(WORLD_EQUITY_REAL_RETURNS)


def get_world_bond_stats() -> dict:
    return compute_statistics(WORLD_BOND_REAL_RETURNS)


def calibrate_market_model() -> MarketModel:
    """Create a MarketModel calibrated from global historical data."""
    eq = get_world_equity_stats()
    bond = get_world_bond_stats()
    correlation = float(np.corrcoef(WORLD_EQUITY_REAL_RETURNS, WORLD_BOND_REAL_RETURNS)[0, 1])

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
    n_simulations: int,
    n_years: int,
    random_seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap from historical global equity and bond returns."""
    rng = np.random.default_rng(random_seed)
    n_history = len(WORLD_EQUITY_REAL_RETURNS)
    indices = rng.integers(0, n_history, size=(n_simulations, n_years))
    return WORLD_EQUITY_REAL_RETURNS[indices], WORLD_BOND_REAL_RETURNS[indices]


def get_bootstrap_portfolio_returns(
    n_simulations: int,
    n_years: int,
    stock_weight: float,
    random_seed: int | None = None,
) -> np.ndarray:
    """Generate portfolio returns by bootstrapping historical global data."""
    equity, bonds = get_bootstrap_returns(n_simulations, n_years, random_seed)
    return stock_weight * equity + (1 - stock_weight) * bonds
