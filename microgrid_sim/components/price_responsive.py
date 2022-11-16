from dataclasses import dataclass, field
from itertools import count
from random import random
from math import copysign


@dataclass(slots=True)
class PriceResponsiveLoad:
    """Model for a price responsive load."""
    sensitivity: float
    patience: int
    _shifted_loads: dict[int, float] = field(init=False, default_factory=lambda: {})
    _timestep_counter: count = field(init=False, default=count(0))

    def get_load(self, base_load: float, price_level: int) -> float:
        """
        Update the model and get the load to execute on this timestep.

        :param base_load: Base load.
        :param price_level: Current price level in {-2, -1, 0, 1, 2}.
        :return: Final load.
        """
        timestep = next(self._timestep_counter)
        shifted_load_to_execute = self._get_shifted_load_to_execute(price_level, timestep)
        load_to_shift = base_load * self.sensitivity * price_level
        self._add_new_shifted_load(load_to_shift, timestep)
        return base_load - load_to_shift + shifted_load_to_execute

    def _get_shifted_load_to_execute(self, current_price_level: int, current_timestep: int) -> float:
        """Returns the shifted load to be executed in this time step."""
        load = 0.0
        for timestep, shifted_load in list(self._shifted_loads.items()):
            if self._execute_load(shifted_load, timestep, current_timestep, current_price_level):
                load += self._shifted_loads.pop(timestep)
        return load

    def _execute_load(self, load: float, load_timestep: int, current_timestep: int, current_price_level: int) -> bool:
        price_term = - current_price_level * copysign(1.0, load) / 2
        time_term = (current_timestep - load_timestep) / self.patience
        exec_prob = min(1.0, max(0.0, price_term + time_term))
        return random() < exec_prob

    def _add_new_shifted_load(self, load: float, timestep: int) -> None:
        """Adds a new load to be executed later."""
        self._shifted_loads[timestep] = load
