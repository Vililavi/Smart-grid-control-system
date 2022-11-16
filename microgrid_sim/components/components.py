from dataclasses import dataclass

from microgrid_sim.components.der import DER
from microgrid_sim.components.ess import ESS
from microgrid_sim.components.households import HouseholdsManager
from microgrid_sim.components.main_grid import MainGrid
from microgrid_sim.components.tcl_aggregator import TCLAggregator
from microgrid_sim.components.from_dict_factories import (
    get_tcl_aggregator_from_params_dict,
    get_ess_from_params_dict,
    get_main_grid_from_params_dict,
    get_der_from_params_dict,
    get_household_manager_from_params_dict
)


@dataclass(slots=True)
class Components:
    """Helper class for handling the components of the environment."""
    main_grid: MainGrid
    der: DER
    ess: ESS
    households_manager: HouseholdsManager
    tcl_aggregator: TCLAggregator

    def get_hour_of_day(self, idx: int) -> int:
        """Utility wrapper to simplify getting the hour of day."""
        return self.der.get_hour_of_day(idx)

    def get_outdoor_temperature(self, idx: int) -> float:
        """Utility wrapper."""
        return self.tcl_aggregator.get_outdoor_temperature(idx)


def get_components_by_param_dicts(
    tcl_params_dict: dict,
    ess_params_dict: dict,
    main_grid_params_dict: dict,
    der_params_dict: dict,
    residential_load_params_dict: dict
) -> Components:
    tcl_aggr = get_tcl_aggregator_from_params_dict(tcl_params_dict)
    ess = get_ess_from_params_dict(ess_params_dict)
    main_grid = get_main_grid_from_params_dict(main_grid_params_dict)
    der = get_der_from_params_dict(der_params_dict)
    household_manager = get_household_manager_from_params_dict(residential_load_params_dict)
    return Components(main_grid, der, ess, household_manager, tcl_aggr)
