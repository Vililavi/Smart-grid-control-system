import os
from itertools import count
from typing import Any

import numpy as np

from microgrid_sim.components.components import get_components_by_param_dicts


def get_default_microgrid_params(path_to_data: str) -> dict[str, dict[str, Any]]:
    """
    Get default parameters for the microgrid.

    :param path_to_data: Path to the folder containing the simulation data.
    :return: Parameters as a dictionary
    """
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
        "up_prices_file_path": os.path.join(path_to_data, "up_regulation.csv"),  # REQUIRED
        "down_prices_file_path": os.path.join(path_to_data, "down_regulation.csv"),  # REQUIRED
        "import_transmission_price": 0.0097,
        "export_transmission_price": 0.0009,
    }
    der_params = {
        "hourly_generated_energies_file_path": os.path.join(path_to_data, "wind_generation.csv"),  # REQUIRED
        "generation_cost": 0.032,
    }
    residential_params = {
        "num_households": 150,  # REQUIRED
        "patience": (10, 6),
        "sensitivity": (0.4, 0.3),
        "price_interval": 0.0015,
        "over_pricing_threshold": 4,
    }

    params = {
        "tcl_params": tcl_params,
        "ess_params": ess_params,
        "main_grid_params": main_grid_params,
        "der_params": der_params,
        "residential_params": residential_params,
    }
    return params


class Environment:
    """Environment that the EMS agent interacts with, combining the components together."""

    __slots__ = ("components", "_timestep_counter", "_idx")

    def __init__(self, params_dict: dict[str, dict[str, Any]], prices_and_temps_path: str, start_time_idx: int):
        tcl_params = params_dict["tcl_params"]
        ess_params = params_dict["ess_params"]
        main_grid_params = params_dict["main_grid_params"]
        der_params = params_dict["der_params"]
        residential_params = params_dict["residential_params"]

        prices_and_temps = np.load(prices_and_temps_path)
        residential_params["hourly_base_prices"] = prices_and_temps[:, 0]
        tcl_params["out_temps"] = prices_and_temps[:, 1]

        self.components = get_components_by_param_dicts(
            tcl_params, ess_params, main_grid_params, der_params, residential_params
        )
        self._timestep_counter = count(start_time_idx)
        self._idx = start_time_idx

    def step(
        self, action: tuple[int, int, int, int]
    ) -> tuple[tuple[float, float, float, float, float, float, int, int], float]:
        """
        Simulate one timestep with the given control actions.

        Returns state of the environment and reward (generated profit).
        """
        self._idx = next(self._timestep_counter)
        tcl_action, price_level = action[0], action[1]
        def_prio = "ESS" if action[2] == 1 else "BUY"
        excess_prio = "ESS" if action[3] == 1 else "SELL"
        reward = self._apply_action(tcl_action, price_level, def_prio, excess_prio)
        state = self.get_state()
        return state, reward

    def _get_tcl_energy(self, tcl_action: int) -> float:
        """Returns energy amount from options {0%, 33%, 67%, 100%} of the max consumption."""
        max_cons = self.components.tcl_aggregator.get_number_of_tcls() * 1.5
        return max_cons * tcl_action / 3

    def _apply_action(self, tcl_action: int, price_level: int, deficiency_prio: str, excess_prio: str) -> float:
        """Apply the choices of the agent and return reward."""
        tcl_cons = self.components.tcl_aggregator.allocate_energy(self._get_tcl_energy(tcl_action), self._idx)
        res_cons, res_profit = self.components.households_manager.get_consumption_and_profit(
            self.components.get_hour_of_day(self._idx), price_level, self._idx)
        generated_energy = self.components.der.get_generated_energy(self._idx)
        excess = generated_energy - tcl_cons - res_cons
        if excess > 0:
            main_grid_returns = self._handle_excess_energy(excess, excess_prio)
        else:
            main_grid_returns = - self._cover_energy_deficiency(-excess, deficiency_prio)
        return self._compute_reward(tcl_cons, res_profit, main_grid_returns)

    def _cover_energy_deficiency(self, energy: float, priority: str) -> float:
        """Cover energy deficiency from ESS and/or MainGrid. Returns cost."""
        if priority == "BUY":
            return self.components.main_grid.get_bought_cost(energy, self._idx)
        ess_energy = self.components.ess.discharge(energy)
        return self.components.main_grid.get_bought_cost(energy - ess_energy, self._idx)

    def _handle_excess_energy(self, energy: float, priority: str) -> float:
        """Store excess energy to the ESS or sell it to the MainGrid. Returns profit."""
        if priority == "SELL":
            return self.components.main_grid.get_sold_profit(energy, self._idx)
        ess_excess = self.components.ess.charge(energy)
        return self.components.main_grid.get_sold_profit(energy - ess_excess, self._idx)

    def _compute_reward(self, tcl_consumption: float, residential_profit: float, main_grid_profit: float) -> float:
        gen_cost = self.components.der.generation_cost
        return tcl_consumption * gen_cost + residential_profit + main_grid_profit

    def get_state(self) -> tuple[float, float, float, float, float, float, int, int]:
        """Collect and return new environment state for the agent."""
        tcl_soc = self.components.tcl_aggregator.get_state_of_charge()
        ess_soc = self.components.ess.soc
        pricing_counter = self.components.households_manager.get_pricing_counter()
        out_temp = self.components.get_outdoor_temperature(self._idx)
        generated_energy = self.components.der.get_generated_energy(self._idx)
        up_price = self.components.main_grid.get_up_price(self._idx)
        hour_of_day = self.components.get_hour_of_day(self._idx)
        base_res_load = self.components.households_manager.get_base_residential_load(hour_of_day)

        state_vector = (
            min(1.0, max(0.0, tcl_soc)),
            min(1.0, max(0.0, ess_soc)),
            out_temp,
            generated_energy,
            up_price,
            base_res_load,
            pricing_counter,
            hour_of_day,
        )
        return state_vector


def get_default_microgrid_env(path_to_data: str, start_idx: int) -> Environment:
    params = get_default_microgrid_params(path_to_data)
    prices_and_temps_path = os.path.join(path_to_data, "default_price_and_temperatures.npy")
    return Environment(params, prices_and_temps_path, start_idx)
