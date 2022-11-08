from microgrid_sim.components.der import DER
from microgrid_sim.components.ess import ESS
from microgrid_sim.components.main_grid import MainGrid

class EnergyDeficiency:
    """Model for energy deficiency action"""
    demand: float
    priority: str

    def energy_deficiency_action(self):
        generated = DER.get_generated_energy
        if generated < self.demand:
            if self.priority == "ESS":
                energy = ESS.discharge(generated - self.demand)
                if energy + generated < self.demand:
                    return MainGrid.get_bought_cost(self.demand - energy + generated)
                else:
                    return 0
            else:
                return MainGrid.get_bought_cost(self.demand - generated)
        else:
            return 0