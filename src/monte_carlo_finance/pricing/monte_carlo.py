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
from ..models.stochastic_processes import StochasticProcess
from .payoffs import Payoff, ExercisePayoff
from .analytic import ControlVariate

@dataclass(frozen=True)
class PriceEstimate:
    price: float
    standardError: float
    nSimulation: int

    def __post_init__(self):
        if self.standardError<=0.0:
            raise ValueError("Standard error should be a positive number.")
        if self.nSimulation<=0:
            raise ValueError("The number of simulations should be a positive integer.")
        
    def confidenceInterval(self, z_value: float = 1.96) -> tuple[float, float]:
        return (self.price - z_value * self.standardError, self.price + z_value * self.standardError)

class MonteCarloPricer:
    def __init__(self, process: StochasticProcess):
        self.process = process

    def price(self, payoff: Payoff, n_simulations: int = 100_000,
              n_steps: int = 252, seed: int | None = None,
              antithetic: bool = False) -> PriceEstimate:
        rng = np.random.default_rng(seed)
        paths = self.process.simulatePath(n_simulations, n_steps, rng, antithetic)
        payoffs = payoff(paths)
        if antithetic:  # average the *payoffs* of each mirrored pair -> n_simulations i.i.d. samples
            payoffs = 0.5 * (payoffs[:n_simulations] + payoffs[n_simulations:])
        discount_payoffs = np.exp(-self.process.riskFreeRate * self.process.maxTime) * payoffs
        return PriceEstimate(discount_payoffs.mean(), discount_payoffs.std(ddof=1) / np.sqrt(len(discount_payoffs)), n_simulations)

    def priceWithControlVariate(self, payoff: Payoff, control: ControlVariate,
                                 n_simulations: int = 100_000, n_steps: int = 252,
                                 seed: int | None = None, antithetic: bool = False) -> PriceEstimate:
        """Standard control-variate estimator: Y_adj = Y - c*(X - E[X]), c chosen to minimise
        Var(Y_adj) and estimated from the same sample (introduces a slight, O(1/n), higher-order
        bias -- negligible at typical simulation sizes)."""
        rng = np.random.default_rng(seed)
        paths = self.process.simulatePath(n_simulations, n_steps, rng, antithetic)
        y, x = payoff(paths), control.payoff(paths)
        if antithetic:
            y = 0.5 * (y[:n_simulations] + y[n_simulations:])
            x = 0.5 * (x[:n_simulations] + x[n_simulations:])
        discount = np.exp(-self.process.riskFreeRate * self.process.maxTime)
        y *= discount
        x *= discount

        adjusted = y - (np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1)) * (x - control.presentValue)
        return PriceEstimate(adjusted.mean(), adjusted.std(ddof=1) / np.sqrt(len(adjusted)), n_simulations)

    def priceWithImportanceSampling(self, payoff: Payoff, theta: float,
                                     n_simulations: int = 100_000, n_steps: int = 252,
                                     seed: int | None = None, factorIndex: int = 0,
                                     antithetic: bool = False) -> PriceEstimate:
        """Shifts factor `factorIndex`'s shocks by a constant mean `theta` (sampling from
        N(theta, 1) instead of N(0, 1) at every step) and reweights by the resulting
        Radon-Nikodym derivative, so the estimator stays unbiased. Useful for rare-event
        payoffs -- deep OTM options, barrier hits -- where plain Monte Carlo wastes almost
        all its paths on a zero payoff. theta > 0 pushes paths up (good for OTM calls / up
        barriers), theta < 0 pushes paths down (OTM puts / down barriers).

        Choosing theta: a common rule of thumb is the drift that moves the median terminal
        log-price to the strike, i.e. solve for the theta that centres the distribution near
        the payoff's trigger; in the absence of that, try a few values and compare standard
        errors -- a good theta typically cuts the standard error several-fold."""
        rng = np.random.default_rng(seed)
        time_step = self.process.maxTime / n_steps
        normal_random_draws = rng.standard_normal((n_simulations, n_steps, self.process.nFactors))
        shifted_random_draws = normal_random_draws.copy()
        shifted_random_draws[..., factorIndex] += theta
        if antithetic:
            z = np.concatenate([normal_random_draws, -normal_random_draws], axis=0)
            zShifted = np.concatenate([shifted_random_draws, -shifted_random_draws + 2 * theta], axis=0)  # mirror around theta

        paths = self.process.simulateFromShocks(shifted_random_draws, time_step, rng)
        logWeight = -theta * normal_random_draws[..., factorIndex].sum(axis=1) - 0.5 * theta**2 * n_steps
        weighted = payoff(paths) * np.exp(logWeight)
        if antithetic:
            weighted = 0.5 * (weighted[:n_simulations] + weighted[n_simulations:])

        discounted = np.exp(-self.process.riskFreeRate * self.process.maxTime) * weighted
        return PriceEstimate(discounted.mean(), discounted.std(ddof=1) / np.sqrt(len(discounted)), n_simulations)

    def priceAmerican(self, exercise: ExercisePayoff,
                       n_simulations: int = 100_000, n_steps: int = 252,
                       seed: int | None = None, basisDegree: int = 2,
                       antithetic: bool = False) -> PriceEstimate:
        """Longstaff-Schwartz least-squares Monte Carlo. At each step, walking backward from
        maturity, regresses the (discounted) realised future cashflow on a degree-`basisDegree`
        polynomial in the in-the-money paths' asset price to estimate the continuation value,
        then exercises wherever immediate exercise beats that estimated continuation.

        Note: this is a regression-based estimator, so it carries a small look-ahead bias and
        is known to be a slight *underestimate* of the true American price (a tighter bound
        needs a separate dual/upper-bound pass, e.g. Andersen-Broadie -- not implemented here).
        Increasing n_simulations and n_steps shrinks the bias as well as the standard error."""
        rng = np.random.default_rng(seed)
        paths = self.process.simulatePath(n_simulations, n_steps, rng, antithetic)
        time_step = self.process.maxTime / n_steps
        discount = np.exp(-self.process.riskFreeRate * time_step)

        cashflow = exercise(paths[:, -1])
        for t in range(n_steps - 1, 0, -1):
            cashflow = cashflow * discount
            spot_t = paths[:, t]
            immediate = exercise(spot_t)
            itm = immediate > 0
            if itm.sum() >= basisDegree + 1:            # need enough points to fit the basis
                basis = np.vander(spot_t[itm], basisDegree + 1, increasing=True)
                coeffs, *_ = np.linalg.lstsq(basis, cashflow[itm], rcond=None)
                continuation = basis @ coeffs
                exerciseNow = immediate[itm] > continuation
                cashflow[np.flatnonzero(itm)[exerciseNow]] = immediate[itm][exerciseNow]
        cashflow = cashflow * discount                       # discount time-1 value back to time 0

        if antithetic:
            cashflow = 0.5 * (cashflow[:n_simulations] + cashflow[n_simulations:])

        immediateToday = exercise(np.array([self.process.spotPrice]))[0]
        price = max(cashflow.mean(), immediateToday)     # can always exercise now
        return PriceEstimate(price, cashflow.std(ddof=1) / np.sqrt(len(cashflow)), n_simulations)
