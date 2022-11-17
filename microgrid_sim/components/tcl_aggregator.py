from dataclasses import dataclass
from typing import Union
from random import gauss

from numpy.typing import ArrayLike

from microgrid_sim.components.tcl import TCL, TCLTemperatureModel, BackupController


@dataclass(slots=True)
class TCLParams:
    num_tcls: int
    out_temperatures: ArrayLike
    thermal_mass_air: tuple[float, float] = (0.004, 0.0008)  # mean, standard deviation
    thermal_mass_building: tuple[float, float] = (0.3, 0.004)  # mean, standard deviation
    internal_heating: tuple[float, float] = (0.0, 0.01)  # mean, standard deviation
    nominal_power: tuple[float, float] = (1.5, 0.01)  # mean, standard deviation
    min_temp: float = 19.0
    max_temp: float = 25.0

    @classmethod
    def from_dict(cls, tcl_params_dict: dict[str, Union[int, float, ArrayLike, tuple[float, float]]]) -> "TCLParams":
        num_tcls = tcl_params_dict.pop("num_tcls")
        out_temps = tcl_params_dict.pop("out_temps")
        return TCLParams(num_tcls, out_temps, **tcl_params_dict)


@dataclass(slots=True)
class TCLAggregator:
    """TCL-aggregator agent that controls division of power amongst a cluster of TCLs."""
    _tcls: list[TCL]
    _out_temps: ArrayLike

    @classmethod
    def from_params(cls, params: TCLParams) -> "TCLAggregator":
        tcls = []
        for _ in range(params.num_tcls):
            backup_controller = BackupController(params.min_temp, params.max_temp)
            temp_model = cls._get_temp_model_from_params(params)

            mean, std_dev = params.nominal_power
            power = gauss(mean, std_dev)
            tcls.append(TCL(power, backup_controller, temp_model))
        return TCLAggregator(tcls, params.out_temperatures)

    @classmethod
    def _get_temp_model_from_params(cls, params: TCLParams) -> TCLTemperatureModel:
        in_temp = min(params.max_temp, max(params.min_temp, gauss((params.max_temp + params.min_temp) / 2, 1.5)))
        mean, std_dev = params.thermal_mass_air
        tm_air = max(0.001, gauss(mean, std_dev))
        mean, std_dev = params.thermal_mass_building
        tm_building = max(0.01, gauss(mean, std_dev))
        mean, std_dev = params.internal_heating
        heating = gauss(mean, std_dev)
        temp_model = TCLTemperatureModel(
            in_temp,
            params.out_temperatures[0],
            min(params.max_temp, max(params.min_temp, gauss((params.max_temp + params.min_temp) / 2, 3.5))),
            tm_air,
            tm_building,
            heating
        )
        return temp_model

    def get_outdoor_temperature(self, idx: int) -> float:
        return self._out_temps[idx]

    def get_state_of_charge(self) -> float:
        """Returns the average state of charge (SoC) of the TCL cluster."""
        return sum(tcl.soc for tcl in self._tcls) / len(self._tcls)

    def allocate_energy(self, energy: float, idx: int) -> float:
        """Allocate energy to be used by the TCL cluster. Returns the amount of energy actually spent."""
        consumed_energy = 0.0
        self._tcls.sort(key=lambda x: x.soc)
        for tcl in self._tcls:
            action = self._get_desired_tcl_action(tcl, energy)
            tcl_energy_consumption = tcl.update(self._out_temps[idx], action)
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
