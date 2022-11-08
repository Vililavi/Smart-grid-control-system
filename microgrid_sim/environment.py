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
class Environment:
    """Environment that the EMS agent interacts with, combining the environment together."""
    prices_and_temps: ArrayLike  # TODO: Make a class for price counter tracking?
    main_grid: MainGrid
    der: DER
    ess: ESS
    res_loads: list[PriceResponsiveLoad]
    tcl_aggregator: TCLAggregator = field(init=False)
    tcls: InitVar[list[TCL]]
    _timestep_counter: count
    start_time_idx: InitVar[int]

    def __post_init__(self, tcls: list[TCL], start_time_idx: int):
        self.tcl_aggregator = TCLAggregator(tcls)
        self._timestep_counter = count(start_time_idx)

    def tick(self, action: Action) -> tuple[float, State]:
        """
        Simulate one (next) timestep with the given control actions.
        Returns generated profit (reward) and the new state of the environment.
        """
        # TODO

    def _cover_energy_deficiency(self, energy: float, priority: str) -> float:
        """Cover energy deficiency from ESS and/or MainGrid. Returns cost."""
        # TODO: Could be essentially what is now in energy_deficiency.py

    def _handle_excess_energy(self, energy: float, priority: str) -> float:
        """Store excess energy to the ESS or sell it to the MainGrid. Returns profit."""
        # TODO: Could be essentially what is now in energy_excess.py
