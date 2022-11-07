import os
import unittest
from microgrid_sim.components.der import DER


class TestDER(unittest.TestCase):
    def test_der(self):
        curr_path = os.getcwd()
        parent_folder = os.path.dirname(curr_path)
        path = os.path.join(parent_folder, "data", "wind_generation.csv")
        der = DER(path)
        for i, _ in enumerate(der._data):
            self.assertIsInstance(der.get_generated_energy(i), float)


if __name__ == '__main__':
    unittest.main()
