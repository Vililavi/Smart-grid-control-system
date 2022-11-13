from itertools import count
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from microgrid_sim.components.components import get_components_by_param_dicts


class Environment:
    """Environment that the EMS agent interacts with, combining the environment together."""

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

    def step(self, action: ArrayLike) -> tuple[ArrayLike, float]:
        """
        Simulate one timestep with the given control actions.

        Returns state of the environment and reward (generated profit).
        """
        # TODO: assert validity of the action
        idx = next(self._timestep_counter)
        tcl_action, price_level = action[0], action[1] - 2
        def_prio = "ESS" if action[2] == 1 else "BUY"
        excess_prio = "ESS" if action[3] == 1 else "SELL"
        reward = self._apply_action(tcl_action, price_level, def_prio, excess_prio, idx)
        state = self._get_state(idx)
        return state, reward

    def _get_tcl_energy(self, tcl_action: int) -> float:
        """Returns energy amount from options {20%, 40%, 60%, 80%} of the max consumption."""
        max_cons = self.components.tcl_aggregator.get_number_of_tcls() * 1.5
        return max_cons * (tcl_action + 1) / 5

    def _apply_action(
        self, tcl_action: int, price_level: int, deficiency_prio: str, excess_prio: str, idx: int
    ) -> float:
        """Apply the choices of the agent and return reward."""
        tcl_cons = self.components.tcl_aggregator.allocate_energy(self._get_tcl_energy(tcl_action), idx)
        res_cons, res_profit = self.components.households_manager.get_consumption_and_profit(
            self.components.get_hour_of_day(idx), price_level, idx)
        generated_energy = self.components.der.get_generated_energy(idx)
        excess = generated_energy - tcl_cons - res_cons
        if excess > 0:
            main_grid_returns = self._handle_excess_energy(excess, excess_prio, idx)
        else:
            main_grid_returns = - self._cover_energy_deficiency(-excess, deficiency_prio, idx)
        return self._compute_reward(tcl_cons, res_profit, main_grid_returns)

    def _cover_energy_deficiency(self, energy: float, priority: str, idx: int) -> float:
        """Cover energy deficiency from ESS and/or MainGrid. Returns cost."""
        if priority == "BUY":
            return self.components.main_grid.get_bought_cost(energy, idx)
        ess_energy = self.components.ess.discharge(energy)
        return self.components.main_grid.get_bought_cost(energy - ess_energy, idx)

    def _handle_excess_energy(self, energy: float, priority: str, idx: int) -> float:
        """Store excess energy to the ESS or sell it to the MainGrid. Returns profit."""
        if priority == "SELL":
            return self.components.main_grid.get_sold_profit(energy, idx)
        ess_excess = self.components.ess.charge(energy)
        return self.components.main_grid.get_sold_profit(energy - ess_excess, idx)

    def _compute_reward(self, tcl_consumption: float, residential_profit: float, main_grid_profit: float) -> float:
        gen_cost = self.components.der.generation_cost
        return tcl_consumption * gen_cost + residential_profit + main_grid_profit

    def _get_state(self, idx: int) -> ArrayLike:
        """Collect and return new environment state for the agent."""
        tcl_soc = self.components.tcl_aggregator.get_state_of_charge()
        ess_soc = self.components.ess.soc
        pricing_counter = self.components.households_manager.get_pricing_counter()
        out_temp = self.components.get_outdoor_temperature(idx)
        generated_energy = self.components.der.get_generated_energy(idx)
        up_price = self.components.main_grid.get_up_price(idx)
        hour_of_day = self.components.get_hour_of_day(idx)
        base_res_load = self.components.households_manager.get_base_residential_load(hour_of_day)

        state_vector = [
            tcl_soc,
            ess_soc,
            float(pricing_counter),
            out_temp,
            generated_energy,
            up_price,
            base_res_load,
            float(hour_of_day)
        ]
        return np.array(state_vector, dtype=np.float32)
