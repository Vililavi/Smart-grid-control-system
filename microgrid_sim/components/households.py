from numpy.typing import ArrayLike
from microgrid_sim.components.price_responsive import PriceResponsiveLoad


# Based on Figure 13 in https://doi.org/10.1016/j.segan.2020.100413
base_hourly_residential_loads = [
    0.4, 0.3, 0.2, 0.2, 0.2, 0.2,
    0.3, 0.5, 0.6, 0.6, 0.5, 0.5,
    0.5, 0.4, 0.4, 0.6, 0.8, 1.4,
    1.2, 0.9, 0.8, 0.6, 0.5, 0.4
]


class PricingManager:
    """Keeps track of energy prices and validates the agent's price-level decisions."""

    def __init__(self, over_pricing_threshold: int = 4):
        self._over_pricing_threshold = over_pricing_threshold
        self.price_levels_sum = 0

    def validate_price_level(self, price_level: int) -> int:
        """Validates the price level given by the agent. Returns the effective (potentially modified) price level."""
        # NOTE: This is according to formula (12) in https://doi.org/10.1016/j.segan.2020.100413
        # That is, this is not exactly the percentage threshold but rather a simplification of it.
        # With the parameters given in Table 1, this leads to a maximum daily over-price percentage of
        # about 4.56 %: Threshold = 4, cst = 1.5 and P_market = 5.48, so maximum average price of the
        # day is P_avg = (4 * 1.5 + 24 * 5.48) / 24 = 5.73. Thus, the percentage over the market price
        # is (5.73 - 5.48) / 5.48 ~ 0.04562 ~ 4.56 %.
        #
        # Implementation of the actual percentage threshold could be done e.g. by
        # - keeping track of market prices and price levels over the last 24 (or 23) hours in a deque
        # - for each price level given by the agent, check if using that would make the daily average to be more
        #   than 2.9 % over the market price.
        # - if yes, use price level 0 instead, otherwise use the suggested price level.
        level = price_level
        if self.price_levels_sum > self._over_pricing_threshold:
            level = 0
        self.price_levels_sum += level
        return level


class HouseholdsManager:
    """
    Helper class for handling households (PriceResponsiveLoads) in the microgrid.
    Also handles the prices of energy for households.
    """

    def __init__(
        self,
        pr_loads: list[PriceResponsiveLoad],
        prices: ArrayLike,
        price_interval: float,
        pricing_manager: PricingManager
     ):
        self._pr_loads = pr_loads
        self._prices = prices
        self._price_interval = price_interval
        self._pricing_manager = pricing_manager

    def get_pricing_counter(self) -> int:
        return self._pricing_manager.price_levels_sum

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
