# Monte Carlo Finance

Python package structure for the Monte Carlo option-pricing framework. The numerical implementation is split from the Streamlit UI so the same models and pricers can be used in notebooks, tests, and the website.

## Structure

```text
monte_carlo_finance/
├── models/
│   ├── __init__.py
│   └── stochastic_processes.py
├── pricing/
│   ├── __init__.py
│   ├── payoffs.py
│   ├── analytic.py
│   ├── monte_carlo.py
│   └── calibration.py
├── requirements.txt
└── README.md
```

## Example

```python
import numpy as np
from models import GeometricBrownianMotion, HestonModel, JumpDiffusionModel
from pricing import MonteCarloPricer, EuropeanOption

process = GeometricBrownianMotion(100.0, 0.02, 0.0, 1.0, 0.20)
paths = process.simulatePath(10_000, 252, np.random.default_rng(42))

pricer = MonteCarloPricer(process)
price = pricer.price(EuropeanOption(100.0), n_simulations=10_000, n_steps=252, seed=42)
print(price.price, price.standardError)
```
