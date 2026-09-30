# Monte Carlo Finance

Installable Python tools for Monte Carlo simulation and option pricing, including geometric Brownian motion, Heston and jump-diffusion processes, analytic prices, Monte Carlo estimators, and calibration.

## Install from GitHub

```bash
python -m pip install "git+https://github.com/jordanbchilds/monte-carlo-pricing.git"
```

To install a particular branch or tag, append `@branch-name` or `@tag` before `.git`.

For local development, clone the repository and install it in editable mode:

```bash
git clone https://github.com/jordanbchilds/monte-carlo-pricing.git
cd monte-carlo-pricing
python -m pip install -e .
```

## Quick start

```python
import numpy as np

from monte_carlo_finance import (
	EuropeanOption,
	GeometricBrownianMotion,
	MonteCarloPricer,
)

process = GeometricBrownianMotion(
	spotPrice=100.0,
	riskFreeRate=0.02,
	dividendRate=0.0,
	maxTime=1.0,
	volatility=0.20,
)

paths = process.simulatePath(10_000, 252, np.random.default_rng(42))
estimate = MonteCarloPricer(process).price(
	EuropeanOption(100.0),
	n_simulations=10_000,
	n_steps=252,
	seed=42,
)

print(paths.shape)
print(estimate.price, estimate.standardError)
```

## Binomial pricing

The same `GeometricBrownianMotion` process can be priced with a
Cox-Ross-Rubinstein binomial tree. European options use `price`, while
American vanilla options use `priceAmerican`.

```python
from monte_carlo_finance import (
    AmericanOption,
    BinomialPricer,
    EuropeanOption,
    GeometricBrownianMotion,
)

process = GeometricBrownianMotion(
    spotPrice=100.0,
    riskFreeRate=0.05,
    dividendRate=0.0,
    maxTime=1.0,
    volatility=0.20,
)
pricer = BinomialPricer(process)

european_price = pricer.price(EuropeanOption(100.0), n_steps=100)
american_price = pricer.priceAmerican(AmericanOption(100.0), n_steps=100)
```

The underlying tree is available for inspection with
`BinomialTree(process, n_steps=100).stock_prices`.

The full API is also available from `monte_carlo_finance.models` and `monte_carlo_finance.pricing`.
