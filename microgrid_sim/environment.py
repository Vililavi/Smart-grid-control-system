from dataclasses import dataclass, field, InitVar
from itertools import count
from numpy.typing import ArrayLike

from microgrid_sim.components.main_grid import MainGrid
from microgrid_sim.components.der import DER
from microgrid_sim.components.ess import ESS
from microgrid_sim.components.price_responsive import PriceResponsiveLoad
from microgrid_sim.components.tcl import TCL
from microgrid_sim.components.tcl_aggregator import TCLAggregator
from microgrid_sim.action import Action
from microgrid_sim.state import State


# Based on Figure 13 in https://doi.org/10.1016/j.segan.2020.100413
base_hourly_residential_loads = [
    0.4, 0.3, 0.2, 0.2, 0.2, 0.2,
    0.3, 0.5, 0.6, 0.6, 0.5, 0.5,
    0.5, 0.4, 0.4, 0.6, 0.8, 1.4,
    1.2, 0.9, 0.8, 0.6, 0.5, 0.4
]


@dataclass
class Components:
    """Helper class for handling the components of the environment."""
    main_grid: MainGrid
    der: DER
    ess: ESS
    res_loads: list[PriceResponsiveLoad]
    tcl_aggregator: TCLAggregator = field(init=False)
    tcls: InitVar[list[TCL]]

    def __post_init__(self, tcls: list[TCL]):
        self.tcl_aggregator = TCLAggregator(tcls)

    def get_residential_consumption(self, hour_of_day: int, price_level: int) -> float:
        """Get accumulated energy consumption of all households in the microgrid."""
        consumption = 0.0
        for pr_load in self.res_loads:
            consumption += pr_load.get_load(base_hourly_residential_loads[hour_of_day], price_level)
        return consumption


@dataclass
class Environment:
    """Environment that the EMS agent interacts with, combining the environment together."""
    prices_and_temps: ArrayLike  # TODO: Make a class for price counter tracking?
    price_interval: float
    components: Components
    _timestep_counter: count
    start_time_idx: InitVar[int]

    def __post_init__(self, start_time_idx: int):
        self._timestep_counter = count(start_time_idx)

    def tick(self, action: Action) -> tuple[float, State]:
        """
        Simulate one (next) timestep with the given control actions.
        Returns generated profit (reward) and the new state of the environment.
        """
        # TODO:
        #  - Apply action:
        #    * control TCLs & get spent energy
        #    * apply price level & get energy spent by residential loads
        #    * compute energy excess/deficiency from generated and used energy amounts
        #    * handle energy excess or deficiency
        #  - Compute reward based on sold/purchased energy etc.
        #  - Get next state:
        #    * TCL & ESS states
        #    * New temperature and base price level
        #    * Generated energy & buying price
        #    * Time of day and base residential load

        idx = next(self._timestep_counter)
        reward = self._apply_action(
            # TODO: figure out the signature (dataclass object or separate values - ints vs floats and strings etc)
        )
        state = self._get_next_state()
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
        res_cons = self.components.get_residential_consumption(self.components.der.get_hour_of_day(idx), price_level)
        generated_energy = self.components.der.get_generated_energy(idx)
        excess = generated_energy - tcl_cons - res_cons
        if excess > 0:
            main_grid_returns = self._handle_excess_energy(excess, energy_excess_prio)
        else:
            main_grid_returns = - self._cover_energy_deficiency(-excess, energy_deficiency_prio)
        return self._compute_reward(
            tcl_cons, res_cons, base_price + price_level * self.price_interval, main_grid_returns
        )

    def _cover_energy_deficiency(self, energy: float, priority: str) -> float:
        """Cover energy deficiency from ESS and/or MainGrid. Returns cost."""
        # TODO: Could be essentially what is now in energy_deficiency.py

    def _handle_excess_energy(self, energy: float, priority: str) -> float:
        """Store excess energy to the ESS or sell it to the MainGrid. Returns profit."""
        # TODO: Could be essentially what is now in energy_excess.py

    def _compute_reward(
        self, tcl_consumption: float, residential_consumption: float, price: float, main_grid_profit: float
    ) -> float:
        gen_cost = self.components.der.generation_cost
        return tcl_consumption * gen_cost + residential_consumption * price + main_grid_profit

    def _get_next_state(self) -> State:
        """Collect and return new environment state for the agent."""
        # TODO
