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
from .monte_carlo import MonteCarloPricer, PriceEstimate

@dataclass(frozen=True)
class MarketQuote:
    """One calibration target: an instrument, its maturity, and the market price it should
    match. `instrument` is a Payoff (priced via MonteCarloPricer.price) unless `american=True`,
    in which case it must be an ExercisePayoff (priced via MonteCarloPricer.priceAmerican)."""
    instrument: Payoff | ExercisePayoff
    maxTime: float
    marketPrice: float
    weight: float = 1.0
    american: bool = False

@dataclass(frozen=True)
class CalibrationResult:
    parameters: dict[str, float]
    objectiveValue: float # sum of weighted squared pricing errors at the optimum
    converged: bool
    nIterations: int

class Calibrator:
    """Fits a StochasticProcess's free parameters to a set of market option quotes by
    least squares. A Monte Carlo price is a random variable, so re-simulating fresh paths on
    every optimizer iteration makes the objective noisy and optimizers unstable or unable to
    converge. The standard fix, used here, is *common random numbers*: each quote gets its own
    seed, fixed for the life of the calibration, so every evaluation of the objective re-prices
    with exactly the same underlying shocks and only the parameters change. That makes the
    objective a smooth(ish) deterministic function of the parameters, at the cost of the fitted
    parameters being (very slightly) tied to that particular seed -- rerun with a different
    seed and check the fit doesn't move much if that matters.

    `processFactory` builds a StochasticProcess from the free parameters by name, e.g.:
        factory = lambda volatility: GeometricBrownianMotion(S0, r, q, volatility)
        Calibrator(factory, ['volatility'], quotes)
    Any process argument not in `paramNames` (spot, rate, dividend, ...) is simply closed over
    by the factory and held fixed -- the calibrator never needs to know about it.
    """

    def __init__(self, processFactory: Callable[..., StochasticProcess], paramNames: list[str],
                 quotes: list[MarketQuote], n_simulations: int = 50_000, n_steps: int = 100,
                 seed: int = 0):
        self.processFactory = processFactory
        self.paramNames = paramNames
        self.quotes = quotes
        self.n_simulations = n_simulations
        self.n_steps = n_steps
        self.seed = seed

    def _priceQuote(self, pricer: MonteCarloPricer, quote: MarketQuote, index: int) -> PriceEstimate:
        seed = self.seed + index  # fixed per quote, identical on every call -> common random numbers
        if quote.american:
            return pricer.priceAmerican(quote.instrument, quote.maxTime,
                                         self.n_simulations, self.n_steps, seed=seed)
        return pricer.price(quote.instrument, quote.maxTime,
                             self.n_simulations, self.n_steps, seed=seed)

    def _objective(self, x: Array) -> float:
        process = self.processFactory(**dict(zip(self.paramNames, x)))
        pricer = MonteCarloPricer(process)
        error = 0.0
        for i, q in enumerate(self.quotes):
            modelPrice = self._priceQuote(pricer, q, i).price
            error += q.weight * (modelPrice - q.marketPrice) ** 2
        return error

    def calibrate(self, initialGuess: dict[str, float],
                  bounds: dict[str, tuple[float, float]] | None = None,
                  method: str | None = None) -> CalibrationResult:
        x0 = np.array([initialGuess[p] for p in self.paramNames])
        scipyBounds = [bounds[p] for p in self.paramNames] if bounds is not None else None
        if method is None:
            # Nelder-Mead needs no gradient (robust to residual MC noise); L-BFGS-B is faster
            # once bounds are given but relies on finite-difference gradients of a somewhat
            # noisy objective -- fine in practice with CRN and a reasonable n_simulations.
            method = "L-BFGS-B" if scipyBounds is not None else "Nelder-Mead"

        result = minimize(self._objective, x0, method=method, bounds=scipyBounds)
        fitted = dict(zip(self.paramNames, result.x.tolist()))
        return CalibrationResult(fitted, float(result.fun), bool(result.success), int(result.nit))

    def residuals(self, parameters: dict[str, float]) -> list[dict[str, float]]:
        """Model vs. market price at a given parameter set, one row per quote -- use this to
        inspect the fit (e.g. after calibrate()) with the same CRN seeds used to obtain it."""
        process = self.processFactory(**parameters)
        pricer = MonteCarloPricer(process)
        rows = []
        for i, q in enumerate(self.quotes):
            est = self._priceQuote(pricer, q, i)
            rows.append({
                "marketPrice": q.marketPrice,
                "modelPrice": est.price,
                "error": est.price - q.marketPrice,
                "standardError": est.standardError,
            })
        return rows
