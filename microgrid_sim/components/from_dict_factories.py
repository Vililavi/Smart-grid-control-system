from microgrid_sim.components.main_grid import MainGrid, MainGridParams
from microgrid_sim.components.der import DER, DERParams
from microgrid_sim.components.ess import ESS, ESSParams
from microgrid_sim.components.households import HouseholdsManager, ResidentialLoadParams
from microgrid_sim.components.tcl_aggregator import TCLAggregator, TCLParams


def get_tcl_aggregator_from_params_dict(params_dict: dict) -> TCLAggregator:
    params = TCLParams.from_dict(params_dict)
    return TCLAggregator.from_params(params)


def get_ess_from_params_dict(params_dict: dict) -> ESS:
    params = ESSParams.from_dict(params_dict)
    return ESS.from_params(params)


def get_main_grid_from_params_dict(params_dict: dict) -> MainGrid:
    params = MainGridParams.from_dict(params_dict)
    return MainGrid.from_params(params)


def get_der_from_params_dict(params_dict: dict) -> DER:
    params = DERParams.from_dict(params_dict)
    return DER.from_params(params)


def get_household_manager_from_params_dict(params_dict: dict) -> HouseholdsManager:
    params = ResidentialLoadParams.from_dict(params_dict)
    return HouseholdsManager.from_params(params)
