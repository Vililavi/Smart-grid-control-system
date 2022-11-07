import csv


class CSVDataParser:
    """Parses timestamped data from a csv file."""

    def __init__(self, data_file: str):
        self._rows = []

