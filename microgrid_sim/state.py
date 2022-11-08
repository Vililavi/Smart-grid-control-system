from dataclasses import dataclass


@dataclass
class State:
    """Model for the microgrid state"""
    tcl_soc: float
    ess_soc: float
    pricing_counter: float  # TODO: yet to be implemented
    out_temperature: float
    generated_energy: float
    up_price: float
    base_residential_load: float
    hour_of_the_day: int  # TODO: does this need to be float for the nn?
