from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from math import exp, sqrt

import numpy as np
import numpy.typing as npt

from ..models.stochastic_processes import GeometricBrownianMotion
from .payoffs import AmericanOption, EuropeanOption

Array = npt.NDArray[np.float64]


def _validate_steps(n_steps: int) -> int:
    """Return ``n_steps`` as an int after validating the tree size."""
    if isinstance(n_steps, bool) or not isinstance(n_steps, Integral):
        raise TypeError("The number of binomial steps must be an integer.")
    if n_steps <= 0:
        raise ValueError("The number of binomial steps must be positive.")
    return int(n_steps)


@dataclass(frozen=True)
class BinomialTree:
    """Cox-Ross-Rubinstein stock-price tree calibrated to a GBM process."""

    process: GeometricBrownianMotion
    n_steps: int

    def __post_init__(self) -> None:
        if not isinstance(self.process, GeometricBrownianMotion):
            raise TypeError("`process` must be a `GeometricBrownianMotion` instance.")
        _validate_steps(self.n_steps)
        if not np.isfinite(self.process.volatility) or self.process.volatility <= 0.0:
            raise ValueError("The process volatility must be positive and finite.")

    @property
    def time_step(self) -> float:
        """Length of one tree period in years."""
        return self.process.maxTime / self.n_steps

    @property
    def up_factor(self) -> float:
        """Multiplicative factor for an upward movement."""
        return exp(self.process.volatility * sqrt(self.time_step))

    @property
    def down_factor(self) -> float:
        """Multiplicative factor for a downward movement."""
        return 1.0 / self.up_factor

    @property
    def risk_neutral_up_probability(self) -> float:
        """Risk-neutral probability assigned to an upward movement."""
        growth = exp(
            (self.process.riskFreeRate - self.process.dividendRate)
            * self.time_step
        )
        probability = (growth - self.down_factor) / (
            self.up_factor - self.down_factor
        )
        if not 0.0 < probability < 1.0:
            raise ValueError(
                "The process parameters do not produce a valid risk-neutral probability."
            )
        return probability

    @property
    def discount_factor(self) -> float:
        """One-period risk-free discount factor."""
        return exp(-self.process.riskFreeRate * self.time_step)

    @property
    def stock_prices(self) -> Array:
        """Stock values with rows for down moves and columns for time steps."""
        prices = np.zeros((self.n_steps + 1, self.n_steps + 1), dtype=float)
        for period in range(self.n_steps + 1):
            down_moves = np.arange(period + 1)
            prices[: period + 1, period] = (
                self.process.spotPrice
                * self.up_factor ** (period - down_moves)
                * self.down_factor ** down_moves
            )
        return prices


@dataclass(frozen=True)
class BinomialPricer:
    """Price vanilla European and American options with a binomial tree.

    The underlying process is risk-neutral GBM, matching the process used by
    :class:`MonteCarloPricer`. European options are valued by discounting the
    expected terminal payoff. American options additionally compare the
    continuation value with immediate exercise at every node.
    """

    process: GeometricBrownianMotion

    def __post_init__(self) -> None:
        if not isinstance(self.process, GeometricBrownianMotion):
            raise TypeError("`process` must be a `GeometricBrownianMotion` instance.")
        if not np.isfinite(self.process.volatility) or self.process.volatility <= 0.0:
            raise ValueError("The process volatility must be positive and finite.")

    def price(self, payoff: EuropeanOption, n_steps: int = 100) -> float:
        """Return the binomial price of a European vanilla option."""
        if not isinstance(payoff, EuropeanOption):
            raise TypeError("`payoff` must be a `EuropeanOption` instance.")
        return self._price(
            strike=payoff.strikePrice,
            is_call=payoff.isCall,
            n_steps=n_steps,
            american=False,
        )

    def priceAmerican(self, payoff: AmericanOption, n_steps: int = 100) -> float:
        """Return the binomial price of an American vanilla option."""
        if not isinstance(payoff, AmericanOption):
            raise TypeError("`payoff` must be an `AmericanOption` instance.")
        return self._price(
            strike=payoff.strike,
            is_call=payoff.isCall,
            n_steps=n_steps,
            american=True,
        )

    def _price(
        self,
        strike: float,
        is_call: bool,
        n_steps: int,
        american: bool,
    ) -> float:
        if not np.isfinite(strike) or strike < 0.0:
            raise ValueError("The option strike must be non-negative and finite.")

        tree = BinomialTree(self.process, _validate_steps(n_steps))
        prices = tree.stock_prices
        probability = tree.risk_neutral_up_probability
        discount = tree.discount_factor

        def exercise_value(stock_prices: Array) -> Array:
            if is_call:
                return np.maximum(stock_prices - strike, 0.0)
            return np.maximum(strike - stock_prices, 0.0)

        values = exercise_value(prices[: tree.n_steps + 1, tree.n_steps])
        for period in range(tree.n_steps - 1, -1, -1):
            values = discount * (
                probability * values[:-1] + (1.0 - probability) * values[1:]
            )
            if american:
                values = np.maximum(
                    values,
                    exercise_value(prices[: period + 1, period]),
                )

        return float(values[0])


__all__ = ["BinomialTree", "BinomialPricer"]
