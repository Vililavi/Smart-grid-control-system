from dataclasses import dataclass, field
from random import gauss


@dataclass(slots=True)
class ESSParams:
    charge_efficiency: float = 0.9
    discharge_efficiency: float = 0.9
    max_charge: float = 250.0
    max_discharge: float = 250.0
    max_energy: float = 500.0

    @classmethod
    def from_dict(cls, ess_params_dict: dict[str, float]) -> "ESSParams":
        return ESSParams(**ess_params_dict)


@dataclass(slots=True)
class ESS:
    """Model for an Energy Storage System (ESS) e.g. a battery"""
    energy: float
    _max_energy: float
    soc: float = field(init=False)
    _max_charge_power: float
    _max_discharge_power: float
    _charge_efficiency: float
    _discharge_efficiency: float

    def __post_init__(self):
        assert self._max_charge_power > 0
        assert self._max_discharge_power > 0
        assert 0 < self._charge_efficiency <= 1
        assert 0 < self._discharge_efficiency <= 1
        self._update_state_of_charge()

    @classmethod
    def from_params(cls, params: ESSParams) -> "ESS":
        energy = min(params.max_energy, max(100.0, gauss(250.0, 100.0)))
        return ESS(
            energy,
            params.max_energy,
            params.max_charge,
            params.max_discharge,
            params.charge_efficiency,
            params.discharge_efficiency
        )

    def _update_state_of_charge(self) -> None:
        assert self._max_energy > 0
        self.soc = self.energy / self._max_energy

    def charge(self, energy: float) -> float:
        """
        Charge the ESS with given power.

        :param energy: Charging power.
        :return: Excess energy.
        """
        return self._update(energy, 0.0)

    def discharge(self, energy: float) -> float:
        """
        Draw power from the ESS.

        :param energy: Requested power.
        :return: Provided energy.
        """
        return self._update(0.0, energy)

    def _update(self, charge_power: float, discharge_power: float) -> float:
        """
        Update the state of the ESS.

        :param charge_power: Desired charge power.
        :param discharge_power: Desired discharge power.
        :return: Energy output (provided or unused energy).
        """
        charging = self._get_limited_charge_power(charge_power)
        discharging = self._get_limited_discharge_power(discharge_power)

        self.energy += self._charge_efficiency * charging - discharging / self._discharge_efficiency
        self._update_state_of_charge()
        return discharging + charge_power - charging

    def _get_limited_charge_power(self, charge_power: float) -> float:
        max_intake = (self._max_energy - self.energy) / self._charge_efficiency
        charging = min(max(charge_power, 0.0), self._max_charge_power, max_intake)
        return charging

    def _get_limited_discharge_power(self, discharge_power: float) -> float:
        max_output = self.energy * self._discharge_efficiency
        discharging = min(max(discharge_power, 0.0), self._max_discharge_power, max_output)
        return discharging
