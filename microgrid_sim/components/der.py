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
