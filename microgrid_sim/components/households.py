from numpy.typing import ArrayLike
from microgrid_sim.components.price_responsive import PriceResponsiveLoad


# Based on Figure 13 in https://doi.org/10.1016/j.segan.2020.100413
base_hourly_residential_loads = [
    0.4, 0.3, 0.2, 0.2, 0.2, 0.2,
    0.3, 0.5, 0.6, 0.6, 0.5, 0.5,
    0.5, 0.4, 0.4, 0.6, 0.8, 1.4,
    1.2, 0.9, 0.8, 0.6, 0.5, 0.4
]


class HouseholdsManager:
    """
    Helper class for handling households (PriceResponsiveLoads) in the microgrid.
    Also handles the prices of energy for households.
    """

    def __init__(self, pr_loads: list[PriceResponsiveLoad], prices: ArrayLike, price_interval: float):
        self._pr_loads = pr_loads
        self._prices = prices
        self._price_interval = price_interval

    def get_consumption_and_profit(self, hour_of_day: int, price_level: int, price_idx: int) -> tuple[float, float]:
        """
        Returns the energy consumption of the households and the profit gained by selling said energy.

        :param hour_of_day: Hour of the day (by starting hour).
        :param price_level: Price level.
        :param price_idx: Data index.
        :return: (consumed_energy, profit)
        """
        consumption = self._get_residential_consumption(hour_of_day, price_level)
        price = self._prices[price_idx] + price_level * self._price_interval
        return consumption, price * consumption

    def _get_residential_consumption(self, hour_of_day: int, price_level: int) -> float:
        """Get accumulated energy consumption of all households in the microgrid."""
        consumption = 0.0
        for pr_load in self._pr_loads:
            consumption += pr_load.get_load(base_hourly_residential_loads[hour_of_day], price_level)
        return consumption
