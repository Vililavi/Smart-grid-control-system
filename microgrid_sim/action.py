from dataclasses import dataclass


@dataclass
class Action:
    """Model for the actions available for controlling the microgrid."""
    # TODO: Is this packaging necessary.
    tcl_action: int  # {0, 1, 2, 3}
    price_level: int  # {0, 1, 2, 3, 4}
    energy_deficiency: int  # {0, 1}
    energy_excess: int  # {0, 1}
