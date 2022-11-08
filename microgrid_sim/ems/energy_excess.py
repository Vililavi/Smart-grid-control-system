from dataclasses import dataclass
from components.ess import ESS
from components.main_grid import MainGrid

@dataclass
class EnergyExcessAction:
    """Model for the energy excess action"""
    energy: float
    priority: str

    def excess_action(self):
        if self.priority == "ESS":
            excess = ESS.charge(self.energy)
            if excess != 0:
                return MainGrid.get_sold_cost(excess)

            return 0
        else:
            return MainGrid.get_sold_cost(self.energy)