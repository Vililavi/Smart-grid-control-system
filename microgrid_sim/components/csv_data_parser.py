import csv


class CSVDataParser:
    """Parses timestamped data from a csv file."""

    def __init__(self, data_file: str):
        """
        :param data_file: Path to the csv file containing the data.
        """
        self._rows = []
        self._data: list[float] = []
        with open(data_file, newline="", encoding="utf-8") as csvfile:
            reader = csv.reader(csvfile, delimiter=",")
            for row in reader:
                self._rows.append(row)
                self._data.append(float(row[-1]))

    def get_data(self, idx: int) -> float:
        return self._data[idx]
