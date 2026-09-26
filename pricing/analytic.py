from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from math import log, sqrt, exp
from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize
from scipy.stats import norm

Array = npt.NDArray[np.float64]
from models.stochastic_processes import GeometricBrownianMotion
from pricing.payoffs import Payoff

def blackScholesPrice(process: GeometricBrownianMotion, strike: float,
                       isCall: bool = True) -> float:
    """Closed-form European price under GBM. Present value (already discounted)."""
    d1 = ((log(process.spotPrice / strike) 
           + (process.riskFreeRate - process.dividendRate + 0.5 * process.volatility**2) * process.maxTime) 
           / (process.volatility * sqrt(process.maxTime)))
    d2 = d1 - process.volatility * sqrt(process.maxTime)
    if isCall:
        return (process.spotPrice * exp(-process.dividendRate * process.maxTime) * norm.cdf(d1) 
                - strike * exp(- process.riskFreeRate * process.maxTime) * norm.cdf(d2))
    return strike * exp(-process.riskFreeRate * process.maxTime) * norm.cdf(-d2) - process.spotPrice * exp(-process.dividendRate * process.maxTime) * norm.cdf(-d1)

def geometricAsianPrice(process: GeometricBrownianMotion, strike: float,
                         n_steps: int, isCall: bool = True) -> float:
    """Closed-form price of a discretely-monitored geometric-average Asian option under GBM
    (Kemna-Vorst). Used as a control variate for the (path-dependent, no closed form)
    arithmetic-average AsianOption -- the two are highly correlated since the geometric and
    arithmetic means of a set of positive numbers are close whenever the spread is not huge."""
    time_step = process.maxTime / n_steps
    mean = log(process.spotPrice) + (process.riskFreeRate - process.dividendRate - 0.5 * process.volatility**2) * time_step * (n_steps + 1) / 2
    var = process.volatility**2 * time_step * (n_steps + 1) * (2 * n_steps + 1) / (6 * n_steps)
    d1 = (mean - log(strike) + var) / sqrt(var)
    d2 = d1 - sqrt(var)
    if isCall:
        undiscounted = exp(mean + 0.5 * var) * norm.cdf(d1) - strike * norm.cdf(d2)
    else:
        undiscounted = strike * norm.cdf(-d2) - exp(mean + 0.5 * var) * norm.cdf(-d1)
    return exp(-process.riskFreeRate * process.maxTime) * undiscounted

@dataclass(frozen=True)
class ControlVariate:
    """A payoff whose true present value is known in closed form, paired with that value.
    Correlated with the target payoff -> reduces variance without introducing bias."""
    payoff: Payoff
    presentValue: float
