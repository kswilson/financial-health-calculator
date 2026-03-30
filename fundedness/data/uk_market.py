"""Historical UK market return data.

Sources:
    - Barclays Equity Gilt Study (annual editions)
    - Credit Suisse Global Investment Returns Yearbook (Dimson, Marsh, Staunton)
    - Bank of England historical data
    - ONS CPI/RPI series

All returns are REAL (inflation-adjusted) total returns, annual.
Equity = FTSE All-Share (or FT All-Share predecessor index).
Gilts = UK Government bonds (medium-dated gilt index).
"""

import numpy as np
from fundedness.models.market import MarketModel


# Annual real total returns: UK equities and gilts, 1970–2024
# These are representative figures based on widely published UK market data.
# Returns are decimal (e.g. 0.10 = 10%).
YEARS = list(range(1970, 2025))

# FTSE All-Share real total returns
UK_EQUITY_REAL_RETURNS = np.array([
    -0.080,  # 1970
    0.370,   # 1971
    0.100,   # 1972
    -0.290,  # 1973
    -0.520,  # 1974
    0.130,   # 1975
    -0.020,  # 1976
    0.400,   # 1977
    0.050,   # 1978
    0.070,   # 1979
    0.230,   # 1980
    0.050,   # 1981
    0.230,   # 1982
    0.210,   # 1983
    0.270,   # 1984
    0.150,   # 1985
    0.200,   # 1986
    0.040,   # 1987
    0.060,   # 1988
    0.300,   # 1989
    -0.130,  # 1990
    0.160,   # 1991
    0.170,   # 1992
    0.240,   # 1993
    -0.100,  # 1994
    0.200,   # 1995
    0.120,   # 1996
    0.210,   # 1997
    0.120,   # 1998
    0.210,   # 1999
    -0.080,  # 2000
    -0.140,  # 2001
    -0.250,  # 2002
    0.170,   # 2003
    0.090,   # 2004
    0.190,   # 2005
    0.140,   # 2006
    0.020,   # 2007
    -0.310,  # 2008
    0.280,   # 2009
    0.120,   # 2010
    -0.060,  # 2011
    0.100,   # 2012
    0.170,   # 2013
    -0.010,  # 2014
    -0.020,  # 2015
    0.140,   # 2016
    0.090,   # 2017
    -0.120,  # 2018
    0.160,   # 2019
    -0.120,  # 2020
    0.140,   # 2021
    -0.040,  # 2022
    0.040,   # 2023
    0.060,   # 2024
])

# UK Gilt real total returns (medium-dated government bonds)
UK_GILT_REAL_RETURNS = np.array([
    -0.120,  # 1970
    0.050,   # 1971
    -0.050,  # 1972
    -0.140,  # 1973
    -0.240,  # 1974
    0.030,   # 1975
    -0.100,  # 1976
    0.320,   # 1977
    -0.040,  # 1978
    -0.020,  # 1979
    0.110,   # 1980
    0.000,   # 1981
    0.420,   # 1982
    0.090,   # 1983
    0.040,   # 1984
    0.080,   # 1985
    0.060,   # 1986
    0.110,   # 1987
    0.030,   # 1988
    0.000,   # 1989
    -0.020,  # 1990
    0.120,   # 1991
    0.130,   # 1992
    0.180,   # 1993
    -0.100,  # 1994
    0.130,   # 1995
    0.050,   # 1996
    0.120,   # 1997
    0.140,   # 1998
    -0.030,  # 1999
    0.060,   # 2000
    -0.010,  # 2001
    0.060,   # 2002
    0.010,   # 2003
    0.030,   # 2004
    0.050,   # 2005
    -0.010,  # 2006
    0.030,   # 2007
    0.090,   # 2008
    0.010,   # 2009
    0.050,   # 2010
    0.090,   # 2011
    0.020,   # 2012
    -0.050,  # 2013
    0.120,   # 2014
    -0.010,  # 2015
    0.070,   # 2016
    -0.020,  # 2017
    -0.010,  # 2018
    0.040,   # 2019
    0.040,   # 2020
    -0.050,  # 2021
    -0.250,  # 2022
    0.010,   # 2023
    -0.020,  # 2024
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


def get_uk_equity_stats() -> dict:
    """Get summary statistics for UK equity returns."""
    return compute_statistics(UK_EQUITY_REAL_RETURNS)


def get_uk_gilt_stats() -> dict:
    """Get summary statistics for UK gilt returns."""
    return compute_statistics(UK_GILT_REAL_RETURNS)


def calibrate_market_model() -> MarketModel:
    """Create a MarketModel calibrated from UK historical data.

    Uses geometric mean returns (more appropriate for long-term projections)
    and historical volatilities.
    """
    eq = get_uk_equity_stats()
    gilt = get_uk_gilt_stats()

    correlation = float(np.corrcoef(UK_EQUITY_REAL_RETURNS, UK_GILT_REAL_RETURNS)[0, 1])

    return MarketModel(
        stock_return=round(eq["geometric_mean"], 4),
        bond_return=round(gilt["geometric_mean"], 4),
        stock_volatility=round(eq["volatility"], 4),
        bond_volatility=round(gilt["volatility"], 4),
        stock_bond_correlation=round(correlation, 2),
        inflation_mean=0.025,  # BoE 2% target + margin
        real_discount_rate=round(gilt["geometric_mean"], 4),
    )


def get_bootstrap_returns(
    n_simulations: int,
    n_years: int,
    random_seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate returns by sampling from historical UK data with replacement.

    Preserves the joint distribution of equity and gilt returns
    (same year is sampled for both, maintaining correlation).

    Args:
        n_simulations: Number of simulation paths
        n_years: Number of years per path
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (equity_returns, gilt_returns), each shape (n_simulations, n_years)
    """
    rng = np.random.default_rng(random_seed)
    n_history = len(UK_EQUITY_REAL_RETURNS)

    # Sample year indices with replacement
    indices = rng.integers(0, n_history, size=(n_simulations, n_years))

    equity_returns = UK_EQUITY_REAL_RETURNS[indices]
    gilt_returns = UK_GILT_REAL_RETURNS[indices]

    return equity_returns, gilt_returns


def get_bootstrap_portfolio_returns(
    n_simulations: int,
    n_years: int,
    stock_weight: float,
    random_seed: int | None = None,
) -> np.ndarray:
    """Generate portfolio returns by bootstrapping historical UK data.

    Args:
        n_simulations: Number of simulation paths
        n_years: Number of years per path
        stock_weight: Allocation to equities (rest in gilts)
        random_seed: Random seed for reproducibility

    Returns:
        Array of shape (n_simulations, n_years) with portfolio returns
    """
    equity, gilts = get_bootstrap_returns(n_simulations, n_years, random_seed)
    bond_weight = 1 - stock_weight
    return stock_weight * equity + bond_weight * gilts
