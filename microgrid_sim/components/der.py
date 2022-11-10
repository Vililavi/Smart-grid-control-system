
import pandas as pd


class DER:
    """
    Simulation for Distributed Energy Resource (DER), e.g. a set of wind turbines.
    This implementation simply reads data from the provided csv file and return it one value at a time
    when requested.
    """
    def __init__(self, data_file: str, generation_cost: float):
        self._data = pd.read_csv(data_file, delimiter=",")
        self.generation_cost = generation_cost

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

