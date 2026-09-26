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
class Payoff(ABC):
    @abstractmethod
    def __call__(self, paths: Array) -> Array:
        """Undiscounted payoff per path."""

@dataclass(frozen=True)
class EuropeanOption(Payoff):
    strikePrice: float
    isCall: bool = True

    def __call__(self, paths):
        s = paths[:, -1]
        return np.maximum(s - self.strikePrice, 0.0) if self.isCall else np.maximum(self.strikePrice - s, 0.0)
    
@dataclass(frozen=True)
class Underlying(Payoff):
    """Terminal asset price itself. E[S_T]e^{-rT} = S_0 e^{-qT} in closed form for any of the
    processes above (it's the martingale property), so this is a free, universal control variate."""

    def __call__(self, paths):
        return paths[:, -1]

@dataclass(frozen=True)
class AsianOption(Payoff):
    """Arithmetic-average price option (average over monitoring dates, excluding t=0)."""
    strikePrice: float
    isCall: bool = True

    def __call__(self, paths):
        avg = paths[:, 1:].mean(axis=1)
        return np.maximum(avg - self.strikePrice, 0.0) if self.isCall else np.maximum(self.strikePrice - avg, 0.0)

@dataclass(frozen=True)
class BarrierOption(Payoff):
    """Knock-in/knock-out barrier option, monitored at every simulated time step (t=0 excluded).
    direction: 'up' or 'down' -- which side of spot the barrier sits on.
    knock: 'in' (payoff only if the barrier IS touched) or 'out' (only if it is NOT touched)."""
    strike: float
    barrier: float
    direction: Literal["up", "down"]
    knock: Literal["in", "out"]
    isCall: bool = True
    rebate: float = 0.0  # paid on the "wrong side" of the knock condition

    def __call__(self, paths):
        monitored = paths[:, 1:]
        touched = ((monitored >= self.barrier).any(axis=1)
                   if self.direction == "up" 
                   else (monitored <= self.barrier).any(axis=1))
        final_value = paths[:, -1]
        vanilla = (np.maximum(final_value - self.strike, 0.0) 
                   if self.isCall 
                   else np.maximum(self.strike - final_value, 0.0))
        active = touched if self.knock == "in" else ~touched
        return np.where(active, vanilla, self.rebate)


def barrierContinuityCorrection(barrier: float, direction: Literal["up", "down"],
                                volatility: float, n_steps: int, max_time: float) -> float:
    """Broadie-Glasserman-Kou adjustment: discretely-monitored simulation systematically
    underestimates the probability of touching a barrier relative to continuous monitoring,
    because the path can cross and return between monitoring dates. Shifting the barrier
    outward by this amount before simulating brings BarrierOption's discrete-monitoring price
    close to the continuous-barrier analytic value. Apply it to the `barrier` argument, not
    to the payoff after the fact."""
    time_step = max_time / n_steps
    shift = 0.5826 * volatility * sqrt(time_step)   # 0.5826 ~= -zeta(1/2)/sqrt(2*pi)
    return barrier * exp(shift) if direction == "up" else barrier * exp(-shift)

class ExercisePayoff(ABC):
    """Unlike Payoff, this is evaluated at an arbitrary exercise time t, not just maturity --
    it takes the asset price *at that time* (shape (n_sim,)) rather than the full path."""
    @abstractmethod
    def __call__(self, assetPrice: Array) -> Array:
        """Value of exercising immediately, given the asset price at the current time."""

@dataclass(frozen=True)
class AmericanOption(ExercisePayoff):
    strike: float
    isCall: bool = True

    def __call__(self, assetPrice):
        return np.maximum(assetPrice - self.strike, 0.0) if self.isCall \
            else np.maximum(self.strike - assetPrice, 0.0)
