import pandas as pd


class MainGrid:
    """Model for the main electricity grid."""

    def __init__(self, up_prices_file: str, down_prices_file: str):
        self._up_prices = pd.read_csv(up_prices_file, delimiter=",")
        self._down_prices = pd.read_csv(down_prices_file, delimiter=",")

    def get_prices(self, idx: int) -> tuple[float, float]:
        """
        Returns up and down prices.

        :param idx: The desired price index.
        :return: (up price, down price)
        """
        return self.get_up_price(idx), self.get_down_price(idx)

    def get_up_price(self, idx: int) -> float:
        return float(self._up_prices.iloc[idx][-1])

    def get_down_price(self, idx: int) -> float:
        return float(self._down_prices.iloc[idx][-1])

    def get_bought_cost(self, bought_energy: float, price_idx: int) -> float:
        """Returns the cost of energy bought from the main electricity grid at a given hour."""
        return bought_energy * self.get_up_price(price_idx)

    def get_sold_cost(self, sold_energy: float, price_idx: int) -> float:
        """Returns the cost of energy sold to the main electricity grid at a given hour."""
        return sold_energy * self.get_down_price(price_idx)
