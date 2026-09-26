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
@dataclass(frozen=True)
class StochasticProcess(ABC):
    """Risk-neutral asset dynamics. Paths have shape (n_sim, n_steps + 1), column 0 = spot."""
    spotPrice: float
    riskFreeRate: float
    dividendRate: float
    maxTime: float

    nFactors = 1  # number of independent N(0,1) shocks per step

    def __post_init__(self):
        if self.maxTime<=0.0:
            raise ValueError("Maximum time must be a positive number.")
        if self.spotPrice<0.0:
            raise ValueError("Spot price must be a non-negative number.")

    def simulatePath(self, n_simulations: int, n_steps: int,
                     rng: np.random.Generator, antithetic: bool = False) -> Array:
        """If antithetic, returns 2*n_simulations paths: the first half and its mirror image.
        Average payoffs of the two halves (not the paths themselves) -- see MonteCarloPricer."""
        time_step = self.maxTime / n_steps
        normal_random_draws = rng.standard_normal((n_simulations, n_steps, self.nFactors))
        if antithetic:
            normal_random_draws = np.concatenate([normal_random_draws, -normal_random_draws], axis=0)
        return self._evolveSimulation(normal_random_draws, time_step, rng)

    def simulateFromShocks(self, random_draws: Array, time_step: float, rng: np.random.Generator) -> Array:
        """Evolve the process from caller-supplied shocks z, shape (n_sim, n_steps, nFactors).
        Exposed publicly (rather than just _evolve) so importance sampling can pass in
        shocks drawn from a shifted proposal distribution."""
        return self._evolveSimulation(random_draws, time_step, rng)

    @abstractmethod
    def _evolveSimulation(self, random_draws: Array, time_step: float, rng: np.random.Generator) -> Array:
        pass

    def _calcAssetPaths(self, logReturns: Array) -> Array:
        """Cumulative log-returns (n_sim, n_steps) -> price paths including t=0."""
        logPaths = np.cumsum(logReturns, axis=1)
        logPaths = np.concatenate([np.zeros((logPaths.shape[0], 1)), logPaths], axis=1)
        return self.spotPrice * np.exp(logPaths)

@dataclass(frozen=True)
class GeometricBrownianMotion(StochasticProcess):
    volatility: float = 0.2

    def _evolveSimulation(self, random_draws, time_step, rng):
        drift = ((self.riskFreeRate - self.dividendRate - 0.5 * self.volatility**2) * time_step
                 + self.volatility * np.sqrt(time_step) * random_draws[..., 0])
        return self._calcAssetPaths(drift)

@dataclass(frozen=True)
class HestonModel(StochasticProcess):
    initialVariance: float = 0.04
    longTermVariance: float = 0.04
    varianceReversion: float = 1.5
    varianceVolatility: float = 0.3
    wienerCorr: float = -0.7

    nFactors = 2

    def _evolveSimulation(self, random_draws, time_step, rng):
        n_sim, n_steps, _ = random_draws.shape
        asset_rand = random_draws[:,:, 0]
        vol_rand = self.wienerCorr * asset_rand + np.sqrt(1 - self.wienerCorr**2) * random_draws[:,:, 1] # correlate via Cholesky

        logReturns = np.empty((n_sim, n_steps))
        vol = np.full(n_sim, self.initialVariance)
        for t in range(n_steps):
            vp = np.maximum(vol, 0.0) # truncate volatility to be positive
            logReturns[:, t] = ((self.riskFreeRate - self.dividendRate - 0.5 * vp) * time_step
                                + np.sqrt(vp * time_step) * asset_rand[:, t])
            vol = (vol + self.varianceReversion * (self.longTermVariance - vp) * time_step
                   + self.varianceVolatility * np.sqrt(vp * time_step) * vol_rand[:, t])
            
        return self._calcAssetPaths(logReturns)


@dataclass(frozen=True)
class JumpDiffusionModel(StochasticProcess):
    """Merton jump diffusion. jumpMean / jumpVariance describe log jump size."""
    volatility: float = 0.2
    jumpRate: float = 0.5
    jumpMean: float = -0.1
    jumpVariance: float = 0.04

    nFactors = 2

    def _evolveSimulation(self, random_draws, time_step, rng):
        n_sim, n_steps, _ = random_draws.shape
        jump_stdDev = np.sqrt(self.jumpVariance)
        # compensator so that E[S_T] = S_0 exp((r - q) T)
        kappa = np.exp(self.jumpMean + 0.5 * self.jumpVariance) - 1.0
        drift = (self.riskFreeRate - self.dividendRate
                 - 0.5 * self.volatility**2 - self.jumpRate * kappa) * time_step

        counts = rng.poisson(self.jumpRate * time_step, size=(n_sim, n_steps))    # per step, same rng
        jumps = counts * self.jumpMean + np.sqrt(counts) * jump_stdDev * random_draws[..., 1]  # sum of N normals
        diffusion = self.volatility * np.sqrt(time_step) * random_draws[..., 0]
        return self._calcAssetPaths(drift + diffusion + jumps)
