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

    def get_component_states(self) -> tuple[float, float, float]:
        """
        Get states of the environment components.

        :return: (tcl_soc, ess_soc, generated_energy)
        """

    def control(self, tcl_energy: float, price_level: int, deficiency_pro: str, excess_prio: str) -> float:
        pass


@dataclass
class Environment:
    """Environment that the EMS agent interacts with, combining the environment together."""
    prices_and_temps: ArrayLike  # TODO: Make a class for price counter tracking?
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
        reward = self._apply_action(
            # TODO: figure out the signature (dataclass object or separate values - ints vs floats and strings etc)
        )
        state = self._get_next_state()
        return reward, state

    def _apply_action(self, tcl_action, price_level, energy_deficiency: str, energy_excess: str) -> float:
        """Apply the choices of the agent and return reward."""
        # TODO: split computing reward to a separate method or do it here?

    def _cover_energy_deficiency(self, energy: float, priority: str) -> float:
        """Cover energy deficiency from ESS and/or MainGrid. Returns cost."""
        # TODO: Could be essentially what is now in energy_deficiency.py

    def _handle_excess_energy(self, energy: float, priority: str) -> float:
        """Store excess energy to the ESS or sell it to the MainGrid. Returns profit."""
        # TODO: Could be essentially what is now in energy_excess.py

    def _get_next_state(self) -> State:
        """Collect and return new environment state for the agent."""
        # TODO
