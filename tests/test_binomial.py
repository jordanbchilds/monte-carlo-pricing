import unittest

from monte_carlo_finance import (
    AmericanOption,
    BinomialPricer,
    BinomialTree,
    EuropeanOption,
    GeometricBrownianMotion,
)


class BinomialPricerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.process = GeometricBrownianMotion(
            spotPrice=100.0,
            riskFreeRate=0.05,
            dividendRate=0.0,
            maxTime=1.0,
            volatility=0.20,
        )
        self.pricer = BinomialPricer(self.process)

    def test_tree_shape_and_root(self) -> None:
        tree = BinomialTree(self.process, n_steps=100)

        self.assertEqual(tree.stock_prices.shape, (101, 101))
        self.assertAlmostEqual(tree.stock_prices[0, 0], 100.0)

    def test_european_call_matches_crr_benchmark(self) -> None:
        price = self.pricer.price(EuropeanOption(100.0), n_steps=100)

        self.assertAlmostEqual(price, 10.4306, places=3)

    def test_american_put_is_at_least_european_put(self) -> None:
        european = self.pricer.price(EuropeanOption(100.0, isCall=False), n_steps=100)
        american = self.pricer.priceAmerican(AmericanOption(100.0, isCall=False), n_steps=100)

        self.assertGreaterEqual(american, european)


if __name__ == "__main__":
    unittest.main()
