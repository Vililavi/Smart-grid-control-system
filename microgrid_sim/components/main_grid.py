from dataclasses import dataclass
from typing import Union

import pandas as pd
from pandas import DataFrame


@dataclass(slots=True)
class MainGridParams:
    up_prices_file_path: str
    down_prices_file_path: str
    import_transmission_price: float = 0.0097
    export_transmission_price: float = 0.0009

    @classmethod
    def from_dict(cls, main_grid_params_dict: dict[str, Union[float, str]]) -> "MainGridParams":
        up_prices_file_path = main_grid_params_dict.pop("up_prices_file_path")
        down_prices_file_path = main_grid_params_dict.pop("down_prices_file_path")
        return MainGridParams(up_prices_file_path, down_prices_file_path, **main_grid_params_dict)


class MainGrid:
    """Model for the main electricity grid."""

    __slots__ = ("_up_prices", "_down_prices", "imp_transmission_cost", "exp_transmission_cost")

    def __init__(self, up_prices: DataFrame, down_prices: DataFrame, imp_trans_cost: float, exp_trans_cost: float):
        self._up_prices = up_prices
        self._down_prices = down_prices
        self.imp_transmission_cost = imp_trans_cost
        self.exp_transmission_cost = exp_trans_cost

    @classmethod
    def from_params(cls, params: MainGridParams) -> "MainGrid":
        up_prices = pd.read_csv(params.up_prices_file_path, delimiter=",")
        down_prices = pd.read_csv(params.down_prices_file_path, delimiter=",")
        return MainGrid(up_prices, down_prices, params.import_transmission_price, params.export_transmission_price)

    def get_prices(self, idx: int) -> tuple[float, float]:
        """
        Returns up and down prices.

        :param idx: The desired price index.
        :return: (up price, down price)
        """
        return self.get_up_price(idx), self.get_down_price(idx)

    def get_up_price(self, idx: int) -> float:
        return float(self._up_prices.iloc[idx][-1]) / 1000

    def get_down_price(self, idx: int) -> float:
        return float(self._down_prices.iloc[idx][-1]) / 1000

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
