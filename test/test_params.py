import unittest
import numpy as np
from microgrid_sim.components.tcl_aggregator import TCLParams


class TestTCLParams(unittest.TestCase):
    def test_get_params_from_dict(self):
        params_dict = {
            "num_tcls": 10,
            "out_temps": np.array([0.0, 1.0, 2.0]),
            "max_temp": 22.0
        }
        params = TCLParams.from_dict(params_dict)
        self.assertEqual(params.num_tcls, 10)
        self.assertEqual(params.out_temperatures[0], 0.0)
        self.assertEqual(params.min_temp, 19.0)
        self.assertEqual(params.max_temp, 22.0)


if __name__ == '__main__':
    unittest.main()
