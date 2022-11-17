from dataclasses import dataclass, field


@dataclass(slots=True)
class BackupController:
    """Model for a backup controller making sure that indoor temperature stays acceptable."""
    min_temp: float
    max_temp: float

    def get_action(self, tcl_action: int, in_temp: float) -> int:
        """
        Limits the given TCL action based on set temperature limits.

        :param tcl_action: Desired TCL action given by the control agent: ON = 1, OFF = 0.
        :param in_temp: Current indoor temperature.
        :return: Modified/limited action: : ON = 1, OFF = 0.
        """
        if in_temp > self.max_temp:
            return 0
        if in_temp < self.min_temp:
            return 1
        return tcl_action

    def get_state_of_charge(self, in_temp: float) -> float:
        """
        Computes and return the state of charge (SoC) of the TCL.

        :param in_temp: Current indoor temperature.
        :return: State of charge.
        """
        assert self.max_temp > self.min_temp
        return (in_temp - self.min_temp) / (self.max_temp - self.min_temp)


@dataclass(slots=True)
class TCLTemperatureModel:
    """Class for storing and updating temperature information."""
    in_temp: float
    _out_temp: float
    _building_temp: float
    _therm_mass_air: float
    _therm_mass_building: float
    _building_heating: float

    def update(self, out_temp: float, tcl_heating: float) -> float:
        """
        Update the model according to the current outdoor temperature and heating/cooling provided by the TCL.

        :param out_temp: Outdoor temperature.
        :param tcl_heating: Heating/cooling provided by the TCL.
        :return: New indoor temperature.
        """
        self._out_temp = out_temp
        new_in_temp = self._get_new_in_temp(tcl_heating)
        new_building_temp = self._get_new_building_temp()

        self.in_temp = new_in_temp
        self._building_temp = new_building_temp
        return new_in_temp

    def _get_new_in_temp(self, tcl_heating: float) -> float:
        assert self._therm_mass_air > 0
        air_comp = (self._out_temp - self.in_temp) * self._therm_mass_air
        building_comp = - self._get_building_temp_change()
        return self.in_temp + air_comp + building_comp + tcl_heating + self._building_heating

    def _get_new_building_temp(self) -> float:
        return self._building_temp + self._get_building_temp_change()

    def _get_building_temp_change(self) -> float:
        assert self._therm_mass_building > 0
        return (self.in_temp - self._building_temp) * self._therm_mass_building


@dataclass(slots=True)
class TCL:
    """Model for a Thermostatically Controlled Load (TCL), e.g. an air conditioner or a water heater etc."""
    soc: float = field(init=False)
    nominal_power: float
    _backup_controller: BackupController
    _temp_model: TCLTemperatureModel

    def __post_init__(self):
        self.soc = self._backup_controller.get_state_of_charge(self._temp_model.in_temp)

    def update(self, out_temp: float, tcl_action: int) -> float:
        """
        Update the state of the TCL.

        :param out_temp: Outside temperature.
        :param tcl_action: Desired TCL action (ON/OFF).
        :return: Energy consumed.
        """
        action = self._backup_controller.get_action(tcl_action, self._temp_model.in_temp)
        tcl_heating = self.nominal_power * action
        in_temp = self._temp_model.update(out_temp, tcl_heating)
        self.soc = self._backup_controller.get_state_of_charge(in_temp)
        return tcl_heating
