import unittest
from itertools import count
from unittest.mock import patch
from microgrid_sim.components.price_responsive import PriceResponsiveLoad


class TestPriceResponsive(unittest.TestCase):
    def setUp(self) -> None:
        self.price_resp = PriceResponsiveLoad(0.0, 0.5, 3)

    def test_execute_load(self):
        cases = [
            {"case": "zero prob", "rand_out": 0.0, "load": 2.0, "l_time": 1, "c_time": 3, "price": 2, "res": False},
            {"case": "one prob", "rand_out": 0.99, "load": 2.0, "l_time": 1, "c_time": 3, "price": -2, "res": True},
            {"case": "long time", "rand_out": 0.5, "load": 2.0, "l_time": 1, "c_time": 11, "price": 2, "res": True},
            {"case": "short time", "rand_out": 0.8, "load": 2.0, "l_time": 1, "c_time": 2, "price": 0, "res": False},
            {"case": "due", "rand_out": 0.8, "load": 2.0, "l_time": 1, "c_time": 4, "price": 0, "res": True},
        ]
        for case in cases:
            with self.subTest(case["case"]):
                with patch("microgrid_sim.components.price_responsive.random", return_value=case["rand_out"]):
                    res = self.price_resp._execute_load(case["load"], case["l_time"], case["c_time"], case["price"])
                self.assertEqual(res, case["res"])

    def test_get_load(self):
        """Just one simple case for this one."""
        self.price_resp._timestep_counter = count(4)
        self.price_resp._shifted_loads = {
            1: -1.0,
            2: 1.0,
            3: -1.0,
        }
        with patch("microgrid_sim.components.price_responsive.random", return_value=0.5):
            load = self.price_resp.get_load(3.0, -2)
        self.assertEqual(7.0, load)
        self.assertIn(1, self.price_resp._shifted_loads)
        self.assertNotIn(2, self.price_resp._shifted_loads)
        self.assertIn(3, self.price_resp._shifted_loads)
        self.assertEqual(-3.0, self.price_resp._shifted_loads[4])


if __name__ == '__main__':
    unittest.main()
