import pandas as pd


class MainGrid:
    """Model for the main electricity grid."""

    def __init__(self, up_prices_file: str, down_prices_file: str, imp_trans_cost: float, exp_trans_cost: float):
        self._up_prices = pd.read_csv(up_prices_file, delimiter=",")
        self._down_prices = pd.read_csv(down_prices_file, delimiter=",")
        self.imp_transmission_cost = imp_trans_cost
        self.exp_transmission_cost = exp_trans_cost

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
        """
        Returns the cost of energy bought from the main electricity grid at a given hour,
        including transmission cost.
        """
        return bought_energy * (self.get_up_price(price_idx) + self.imp_transmission_cost)

    def get_sold_profit(self, sold_energy: float, price_idx: int) -> float:
        """
        Returns the profit gained by selling energy to the main electricity grid at a given hour.
        Accounts for the transmission cost.
        """
        return sold_energy * (self.get_down_price(price_idx) - self.exp_transmission_cost)
