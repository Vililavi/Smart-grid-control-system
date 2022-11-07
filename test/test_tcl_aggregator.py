import unittest
from microgrid_sim.components.tcl import TCL, BackupController, TCLTemperatureModel
from microgrid_sim.components.tcl_aggregator import TCLAggregator


class TestTCLAggregator(unittest.TestCase):
    def setUp(self) -> None:
        self.tcl_1 = self.get_tcl()
        self.tcl_2 = self.get_tcl()
        self.tcl_3 = self.get_tcl()
        self.tcl_2._temp_model.in_temp = 30.0
        self.tcl_2.soc = 1.5
        self.tcl_3._temp_model.in_temp = 10.0
        self.tcl_3.soc = -0.5

        self.aggregator = TCLAggregator([self.tcl_1, self.tcl_2, self.tcl_3])

    @staticmethod
    def get_tcl() -> TCL:
        backup = BackupController(min_temp=15.0, max_temp=25.0)
        model = TCLTemperatureModel(
            in_temp=20.0,
            _building_temp=20.0,
            _out_temp=10.0,
            _therm_mass_air=0.004,
            _therm_mass_building=0.3,
            _building_heating=0.0,
        )
        return TCL(
            nominal_power=1.0,
            _backup_controller=backup,
            _temp_model=model,
        )

    def test_get_soc(self):
        soc = self.aggregator.get_state_of_charge()
        self.assertEqual(0.5, soc)

    def test_allocate_energy(self):
        cases = [
            {"case": "too much energy", "energy": 3.0, "consumed": 2.0},
            {"case": "too little energy", "energy": 0.5, "consumed": 1.0},
            {"case": "enough for one", "energy": 1.5, "consumed": 1.0},
            {"case": "enough for two", "energy": 2.5, "consumed": 2.0},
        ]
        for case in cases:
            with self.subTest(case["case"]):
                consumed_energy = self.aggregator.allocate_energy(case["energy"], 10.0)
                self.assertEqual(case["consumed"], consumed_energy)
                self.assertEqual(self.tcl_3, self.aggregator._tcls[0])
                self.assertEqual(self.tcl_1, self.aggregator._tcls[1])
                self.assertEqual(self.tcl_2, self.aggregator._tcls[2])


if __name__ == '__main__':
    unittest.main()
