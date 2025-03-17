import unittest
from iron_condor_app import IronCondorFinder

class TestIronCondorFinder(unittest.TestCase):
    def test_black_scholes_greeks(self):
        S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2
        delta, gamma, theta, vega = IronCondorFinder.black_scholes_greeks(S, K, T, r, sigma, 'call')
        self.assertAlmostEqual(delta, 0.595, delta=0.01)
        self.assertGreater(gamma, 0)
        self.assertLess(theta, 0)
        self.assertGreater(vega, 0)

if __name__ == '__main__':
    unittest.main()