import os
import unittest
import numpy as np

from microgrid_sim.environment import Environment


class TestMicrogridEnvironment(unittest.TestCase):
    def test_init_and_step(self):
        """Testing with 'default' params"""
        curr_path = os.getcwd()
        parent_folder = os.path.dirname(curr_path)
        data_folder = os.path.join(parent_folder, "data")

        tcl_params = {
            "num_tcls": 100,  # REQUIRED
            "thermal_mass_air": (0.004, 0.0008),
            "thermal_mass_building": (0.3, 0.004),
            "internal_heating": (0.0, 0.01),
            "nominal_power": (1.5, 0.01),
            "min_temp": 19.0,
            "max_temp": 25.0,
        }
        ess_params = {
            "charge_efficiency": 0.9,
            "discharge_efficiency": 0.9,
            "max_charge": 250.0,
            "max_discharge": 250.0,
            "max_energy": 500.0,
        }
        main_grid_params = {
            "up_prices_file_path": os.path.join(data_folder, "up_regulation.csv"),  # REQUIRED
            "down_prices_file_path": os.path.join(data_folder, "down_regulation.csv"),  # REQUIRED
            "import_transmission_price": 9.7,
            "export_transmission_price": 0.9,
        }
        der_params = {
            "hourly_generated_energies_file_path": os.path.join(data_folder, "wind_generation.csv"),  # REQUIRED
            "generation_cost": 32.0,
        }
        residential_params = {
            "num_households": 150,  # REQUIRED
            "patience": (10, 6),
            "sensitivity": (0.4, 0.3),
            "price_interval": 1.5,
            "over_pricing_threshold": 4,
        }

        params = {
            "tcl_params": tcl_params,
            "ess_params": ess_params,
            "main_grid_params": main_grid_params,
            "der_params": der_params,
            "residential_params": residential_params,
        }

        prices_and_temps_path = os.path.join(data_folder, "default_price_and_temperatures.npy")

        env = Environment(params, prices_and_temps_path, 25)

        action = (3, 4, 1, 1)
        state, reward = env.step(action)
        print(f"state vector: {state}")
        print(f"    TCL SoC:               {state[0]}")
        print(f"    ESS SoC:               {state[1]}")
        print(f"    Pricing counter:       {state[2]}")
        print(f"    Outdoor temperature:   {state[3]}")
        print(f"    Generated energy:      {state[4]}")
        print(f"    Up price:              {state[5]}")
        print(f"    Base residential load: {state[6]}")
        print(f"    Hour of day:           {state[7]}")
        print(f"reward: {reward}")


if __name__ == '__main__':
    unittest.main()
