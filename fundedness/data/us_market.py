"""Historical US market return data.

Sources:
    - Shiller CAPE / S&P 500 historical data
    - Ibbotson / Morningstar SBBI
    - Damodaran (NYU) annual return datasets
    - Federal Reserve (10-year Treasury)

All returns are REAL (inflation-adjusted) total returns, annual.
Equity = S&P 500 total return index.
Bonds = 10-year US Treasury total return.
"""

import numpy as np
from fundedness.models.market import MarketModel


# Annual real total returns: US equities and bonds, 1970–2024
YEARS = list(range(1970, 2025))

# S&P 500 real total returns
US_EQUITY_REAL_RETURNS = np.array([
    0.001,   # 1970
    0.107,   # 1971
    0.152,   # 1972
    -0.261,  # 1973
    -0.370,  # 1974
    0.234,   # 1975
    0.168,   # 1976
    -0.113,  # 1977
    -0.011,  # 1978
    0.053,   # 1979
    0.201,   # 1980
    -0.098,  # 1981
    0.146,   # 1982
    0.166,   # 1983
    -0.024,  # 1984
    0.258,   # 1985
    0.142,   # 1986
    0.022,   # 1987
    0.118,   # 1988
    0.262,   # 1989
    -0.093,  # 1990
    0.264,   # 1991
    0.044,   # 1992
    0.070,   # 1993
    -0.013,  # 1994
    0.342,   # 1995
    0.193,   # 1996
    0.295,   # 1997
    0.267,   # 1998
    0.176,   # 1999
    -0.123,  # 2000
    -0.155,  # 2001
    -0.237,  # 2002
    0.258,   # 2003
    0.080,   # 2004
    0.017,   # 2005
    0.119,   # 2006
    0.023,   # 2007
    -0.384,  # 2008
    0.233,   # 2009
    0.128,   # 2010
    -0.013,  # 2011
    0.133,   # 2012
    0.298,   # 2013
    0.115,   # 2014
    -0.007,  # 2015
    0.097,   # 2016
    0.193,   # 2017
    -0.063,  # 2018
    0.287,   # 2019
    0.156,   # 2020
    0.216,   # 2021
    -0.194,  # 2022
    0.222,   # 2023
    0.211,   # 2024
])

# US 10-Year Treasury real total returns
US_BOND_REAL_RETURNS = np.array([
    0.052,   # 1970
    0.074,   # 1971
    -0.007,  # 1972
    -0.043,  # 1973
    -0.058,  # 1974
    0.007,   # 1975
    0.060,   # 1976
    -0.023,  # 1977
    -0.050,  # 1978
    -0.064,  # 1979
    -0.039,  # 1980
    -0.014,  # 1981
    0.342,   # 1982
    0.006,   # 1983
    0.100,   # 1984
    0.222,   # 1985
    0.191,   # 1986
    -0.037,  # 1987
    0.044,   # 1988
    0.134,   # 1989
    0.031,   # 1990
    0.131,   # 1991
    0.042,   # 1992
    0.111,   # 1993
    -0.084,  # 1994
    0.225,   # 1995
    -0.009,  # 1996
    0.098,   # 1997
    0.116,   # 1998
    -0.087,  # 1999
    0.143,   # 2000
    0.034,   # 2001
    0.130,   # 2002
    -0.006,  # 2003
    0.022,   # 2004
    -0.002,  # 2005
    -0.005,  # 2006
    0.068,   # 2007
    0.163,   # 2008
    -0.094,  # 2009
    0.058,   # 2010
    0.131,   # 2011
    0.020,   # 2012
    -0.088,  # 2013
    0.087,   # 2014
    -0.004,  # 2015
    -0.005,  # 2016
    0.005,   # 2017
    -0.020,  # 2018
    0.068,   # 2019
    0.086,   # 2020
    -0.104,  # 2021
    -0.175,  # 2022
    -0.005,  # 2023
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


def get_us_equity_stats() -> dict:
    return compute_statistics(US_EQUITY_REAL_RETURNS)


def get_us_bond_stats() -> dict:
    return compute_statistics(US_BOND_REAL_RETURNS)


def calibrate_market_model() -> MarketModel:
    """Create a MarketModel calibrated from US historical data."""
    eq = get_us_equity_stats()
    bond = get_us_bond_stats()
    correlation = float(np.corrcoef(US_EQUITY_REAL_RETURNS, US_BOND_REAL_RETURNS)[0, 1])

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
    """Bootstrap from historical US equity and bond returns."""
    rng = np.random.default_rng(random_seed)
    n_history = len(US_EQUITY_REAL_RETURNS)
    indices = rng.integers(0, n_history, size=(n_simulations, n_years))
    return US_EQUITY_REAL_RETURNS[indices], US_BOND_REAL_RETURNS[indices]


def get_bootstrap_portfolio_returns(
    n_simulations: int,
    n_years: int,
    stock_weight: float,
    random_seed: int | None = None,
) -> np.ndarray:
    """Generate portfolio returns by bootstrapping historical US data."""
    equity, bonds = get_bootstrap_returns(n_simulations, n_years, random_seed)
    return stock_weight * equity + (1 - stock_weight) * bonds
