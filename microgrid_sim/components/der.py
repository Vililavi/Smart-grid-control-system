from dataclasses import dataclass
from typing import Union

import pandas as pd


@dataclass(slots=True)
class DERParams:
    hourly_generated_energies_file_path: str
    generation_cost: float = 0.032

    @classmethod
    def from_dict(cls, der_params_dict: dict[str, Union[float, str]]) -> "DERParams":
        generated_energies_file_path = der_params_dict.pop("hourly_generated_energies_file_path")
        return DERParams(generated_energies_file_path, **der_params_dict)


class DER:
    """
    Simulation for Distributed Energy Resource (DER), e.g. a set of wind turbines.
    This implementation simply reads data from the provided csv file and return it one value at a time
    when requested.
    """
    __slots__ = ("_data", "generation_cost")

    def __init__(self, energy_generation_data: pd.DataFrame, generation_cost: float):
        self._data = energy_generation_data
        self.generation_cost = generation_cost

    @classmethod
    def from_params(cls, params: DERParams) -> "DER":
        data = pd.read_csv(params.hourly_generated_energies_file_path, delimiter=",")
        return DER(data, params.generation_cost)

    def get_generated_energy(self, idx: int) -> float:
        return float(self._data.iloc[idx][-1])

    def get_data_size(self) -> int:
        return len(self._data)

    def get_hour_of_day(self, idx: int) -> int:
        """
        Get the hour of day for the given row in data.
        We use this data set for this purpose because it is the one with the least amount of entries.
        """
        date_str: str = self._data.iloc[idx][2]
        hour_str = date_str.split(" ")[1].split(":")[0]
        return int(hour_str)
