from dataclasses import dataclass, field
from typing import Union
from random import gauss

from numpy.typing import ArrayLike
from microgrid_sim.components.price_responsive import PriceResponsiveLoad


def _get_default_base_hourly_loads() -> list[float]:
    # Based on Figure 13 in https://doi.org/10.1016/j.segan.2020.100413
    base_hourly_residential_loads = [
        0.4, 0.3, 0.2, 0.2, 0.2, 0.2,
        0.3, 0.5, 0.6, 0.6, 0.5, 0.5,
        0.5, 0.4, 0.4, 0.6, 0.8, 1.4,
        1.2, 0.9, 0.8, 0.6, 0.5, 0.4
    ]
    return base_hourly_residential_loads


@dataclass(slots=True)
class ResidentialLoadParams:
    num_households: int
    hourly_base_prices: ArrayLike
    base_hourly_loads: list[float] = field(default_factory=lambda: _get_default_base_hourly_loads())
    patience: tuple[int, int] = (10, 6)  # mean, standard deviation
    sensitivity: tuple[float, float] = (0.4, 0.3)  # mean, standard deviation
    price_interval: float = 0.0015
    over_pricing_threshold: int = 4

    @classmethod
    def from_dict(
            cls, res_load_params_dict: dict[str, Union[int, float, list[float], tuple[int, int], tuple[float, float]]]
    ) -> "ResidentialLoadParams":
        num_households = res_load_params_dict.pop("num_households")
        hourly_base_prices = res_load_params_dict.pop("hourly_base_prices")
        return ResidentialLoadParams(num_households, hourly_base_prices, **res_load_params_dict)


class PricingManager:
    """Keeps track of energy prices and validates the agent's price-level decisions."""

    __slots__ = ("_over_pricing_threshold", "price_levels_sum")

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

    __slots__ = ("_pr_loads", "_prices", "_price_interval", "_pricing_manager", "_base_loads")

    def __init__(
        self,
        pr_loads: list[PriceResponsiveLoad],
        prices: ArrayLike,
        price_interval: float,
        pricing_manager: PricingManager,
        base_hourly_loads: list[float]
     ):
        self._pr_loads = pr_loads
        self._prices = prices
        self._price_interval = price_interval
        self._pricing_manager = pricing_manager
        self._base_loads = base_hourly_loads

    @classmethod
    def from_params(cls, params: ResidentialLoadParams) -> "HouseholdsManager":
        price_resp_loads = []
        for _ in range(params.num_households):
            mean, std_dev = params.patience
            patience = round(gauss(mean, std_dev))  # not quite exactly correct but shouldn't matter here
            patience = max(1, patience)
            mean, std_dev = params.sensitivity
            sensitivity = gauss(mean, std_dev)
            price_resp_loads.append(PriceResponsiveLoad(sensitivity, patience))
        pricing_manager = PricingManager(params.over_pricing_threshold)
        return HouseholdsManager(
            price_resp_loads,
            params.hourly_base_prices,
            params.price_interval,
            pricing_manager,
            params.base_hourly_loads,
        )

    def get_pricing_counter(self) -> int:
        return self._pricing_manager.price_levels_sum

    def get_base_residential_load(self, hour_of_day: int) -> float:
        return self._base_loads[hour_of_day]

    def get_consumption_and_profit(self, hour_of_day: int, price_level: int, price_idx: int) -> tuple[float, float]:
        """
        Returns the energy consumption of the households and the profit gained by selling said energy.

        :param hour_of_day: Hour of the day (by starting hour).
        :param price_level: Price level.
        :param price_idx: Data index.
        :return: (consumed_energy, profit)
        """
        price_level = self._pricing_manager.validate_price_level(price_level)
        consumption = self._get_residential_consumption(hour_of_day, price_level)
        price = self._prices[price_idx] / 100 + price_level * self._price_interval
        return consumption, price * consumption

    def _get_residential_consumption(self, hour_of_day: int, price_level: int) -> float:
        """Get accumulated energy consumption of all households in the microgrid."""
        consumption = 0.0
        for pr_load in self._pr_loads:
            consumption += pr_load.get_load(self._base_loads[hour_of_day], price_level)
        return consumption
