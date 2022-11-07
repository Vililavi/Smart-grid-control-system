from microgrid_sim.components.csv_data_parser import CSVDataParser


class DER(CSVDataParser):
    """
    Simulation for Distributed Energy Resource (DER), e.g. a set of wind turbines.
    This implementation simply reads data from the provided csv file and return it one value at a time
    when requested.
    """
    def __init__(self, data_file: str):
        super().__init__(data_file)

    def get_generated_energy(self, idx: int) -> float:
        return self.get_data(idx)
