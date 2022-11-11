from dataclasses import dataclass, InitVar
from itertools import count
import numpy as np
from numpy.typing import ArrayLike

from microgrid_sim.components.main_grid import MainGrid
from microgrid_sim.components.der import DER
from microgrid_sim.components.ess import ESS
from microgrid_sim.components.households import HouseholdsManager
from microgrid_sim.components.tcl_aggregator import TCLAggregator
from microgrid_sim.action import Action


@dataclass
class Components:
    """Helper class for handling the components of the environment."""
    main_grid: MainGrid
    der: DER
    ess: ESS
    households_manager: HouseholdsManager
    tcl_aggregator: TCLAggregator

    def get_hour_of_day(self, idx: int) -> int:
        """Utility wrapper to simplify getting the hour of day."""
        return self.der.get_hour_of_day(idx)


@dataclass
class Environment:
    """Environment that the EMS agent interacts with, combining the environment together."""
    prices_and_temps: ArrayLike
    components: Components
    _timestep_counter: count
    start_time_idx: InitVar[int]

    def __post_init__(self, start_time_idx: int):
        self._timestep_counter = count(start_time_idx)

    def tick(self, action: Action) -> tuple[float, ArrayLike]:
        """
        Simulate one (next) timestep with the given control actions.
        Returns generated profit (reward) and the new state of the environment.
        """

        idx = next(self._timestep_counter)
        reward = self._apply_action(
            # TODO: figure out the signature (dataclass object or separate values - ints vs floats and strings etc)
        )
        state = self._get_state(idx)
        return reward, state

    def _get_tcl_energy(self, tcl_action: int) -> float:
        """Returns energy amount from options {20%, 40%, 60%, 80%} of the max consumption."""
        max_cons = self.components.tcl_aggregator.get_number_of_tcls() * 1.5
        return max_cons * (tcl_action + 1) / 5

    def _apply_action(
        self, tcl_action: int, price_level: int, energy_deficiency_prio: str, energy_excess_prio: str, idx: int
    ) -> float:
        """Apply the choices of the agent and return reward."""
        base_price, out_temp = self.prices_and_temps[idx]
        tcl_cons = self.components.tcl_aggregator.allocate_energy(self._get_tcl_energy(tcl_action), out_temp)
        res_cons, res_profit = self.components.households_manager.get_consumption_and_profit(
            self.components.get_hour_of_day(idx), price_level, idx)
        generated_energy = self.components.der.get_generated_energy(idx)
        excess = generated_energy - tcl_cons - res_cons
        if excess > 0:
            main_grid_returns = self._handle_excess_energy(excess, energy_excess_prio, idx)
        else:
            main_grid_returns = - self._cover_energy_deficiency(-excess, energy_deficiency_prio, idx)
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
        _, out_temp = self.prices_and_temps[idx]
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
        return np.array(state_vector)
