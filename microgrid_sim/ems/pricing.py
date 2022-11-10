
class PricingManager:
    """Keeps track of energy prices and validates the agent's price-level decisions."""

    def __init__(self, over_pricing_threshold: int = 4):
        self._over_pricing_threshold = over_pricing_threshold
        self._price_levels_sum = 0

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
        if self._price_levels_sum > self._over_pricing_threshold:
            level = 0
        self._price_levels_sum += level
        return level
