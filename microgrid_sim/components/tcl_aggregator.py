from dataclasses import dataclass
from microgrid_sim.components.tcl import TCL


@dataclass
class TCLAggregator:
    """TCL-aggregator agent that controls division of power amongst a cluster of TCLs."""
    _tcls: list[TCL]

    def get_state_of_charge(self) -> float:
        """Returns the average state of charge (SoC) of the TCL cluster."""
        return sum(tcl.soc for tcl in self._tcls) / len(self._tcls)

    def allocate_energy(self, energy: float, out_temperature: float) -> float:
        """Allocate energy to be used by the TCL cluster. Returns the amount of energy actually spent."""
        consumed_energy = 0.0
        self._tcls.sort(key=lambda x: x.soc)
        for tcl in self._tcls:
            action = self._get_desired_tcl_action(tcl, energy)
            tcl_energy_consumption = tcl.update(out_temperature, action)
            consumed_energy += tcl_energy_consumption
            energy -= tcl_energy_consumption
        return consumed_energy

    def get_number_of_tcls(self) -> int:
        return len(self._tcls)

    @staticmethod
    def _get_desired_tcl_action(tcl: TCL, energy_left: float) -> int:
        if tcl.nominal_power < energy_left:
            return 1
        return 0
