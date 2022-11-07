import unittest

from microgrid_sim.components.tcl import TCL, BackupController, TCLTemperatureModel


class TestTCL(unittest.TestCase):
    def setUp(self) -> None:
        self.backup = BackupController(min_temp=15.0, max_temp=25.0)
        self.model = TCLTemperatureModel(
            in_temp=20.0,
            _building_temp=18.0,
            _out_temp=10.0,
            _therm_mass_air=0.004,
            _therm_mass_building=0.3,
            _building_heating=0.0,
        )
        self.tcl = TCL(
            nominal_power=1.5,
            _backup_controller=self.backup,
            _temp_model=self.model,
        )

    def test_backup_get_action(self):
        cases = [
            {"case": "temp too low 0", "action": 0, "temp": 14.0, "res": 1},
            {"case": "temp too low 1", "action": 1, "temp": 14.0, "res": 1},
            {"case": "temp too high 0", "action": 0, "temp": 26.0, "res": 0},
            {"case": "temp too high 1", "action": 1, "temp": 26.0, "res": 0},
            {"case": "temp okay 0", "action": 0, "temp": 20.0, "res": 0},
            {"case": "temp okay 1", "action": 1, "temp": 20.0, "res": 1},
        ]
        for case in cases:
            with self.subTest(case["case"]):
                final_action = self.backup.get_action(case["action"], case["temp"])
                self.assertEqual(final_action, case["res"])

    def test_get_soc(self):
        cases = [
            {"case": "under min", "temp": 10.0, "res": -0.5},
            {"case": "over max", "temp": 30.0, "res": 1.5},
            {"case": "between 1", "temp": 20.0, "res": 0.5},
            {"case": "between 2", "temp": 22.0, "res": 0.7},
        ]
        for case in cases:
            with self.subTest(case["case"]):
                soc = self.backup.get_state_of_charge(case["temp"])
                self.assertEqual(soc, case["res"])

    def test_model_update(self):
        """Test that directions of indoor temperature change are correct with a few crude cases."""
        cases = [
            {"case": "cold out, no heating", "out_temp": -5.0, "tcl_heating": 0.0, "temp_incr": False},
            {"case": "cold out, heating", "out_temp": -5.0, "tcl_heating": 1.5, "temp_incr": True},
            {"case": "warm out, no heating", "out_temp": 25.0, "tcl_heating": 0.0, "temp_incr": False},
            {"case": "warm out, heating", "out_temp": 25.0, "tcl_heating": 1.5, "temp_incr": True},
            {"case": "hot out, no heating", "out_temp": 35.0, "tcl_heating": 0.0, "temp_incr": True},
            {"case": "hot out, heating", "out_temp": 35.0, "tcl_heating": 1.5, "temp_incr": True},
        ]
        for case in cases:
            with self.subTest(case["case"]):
                self.model.in_temp = 20.0
                self.model._building_temp = 18.0
                if "hot out" in case["case"]:
                    self.model._building_temp = 20.0

                in_temp = self.model.in_temp
                new_in_temp = self.model.update(case["out_temp"], case["tcl_heating"])
                if case["temp_incr"]:
                    self.assertTrue(new_in_temp > in_temp, f"{case['case']}: {new_in_temp} not more than {in_temp}")
                else:
                    self.assertTrue(new_in_temp < in_temp, f"{case['case']}: {new_in_temp} not less than {in_temp}")


if __name__ == '__main__':
    unittest.main()
