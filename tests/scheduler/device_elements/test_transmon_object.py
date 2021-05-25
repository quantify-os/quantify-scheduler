# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

from quantify.scheduler.compilation import validate_config
from quantify.scheduler.device_elements.transmon_element import TransmonElement

# import qitt_experiments.init.mock_setup_initialization as mi
# cls.mi = mi


class TestTransmonElement:
    @classmethod
    def setup_class(cls):
        cls.q0 = TransmonElement("q0")

    @classmethod
    def teardown_class(cls):
        for inststr in list(cls.q0._all_instruments):
            try:
                inst = cls.q0.find_instrument(inststr)
                inst.close()
            except KeyError:
                pass

    def test_qubit_name(self):
        assert self.q0.name == "q0"

    def test_generate_qubit_config(self):

        # set some values
        self.q0.mw_port("port")
        self.q0.ro_pulse_type("square")
        self.q0.ro_pulse_duration(400e-9)

        q_cfg = self.q0.generate_qubit_config()

        # assert values in right place in config.
        assert q_cfg["q0"]["resources"]["port_mw"] == "port"
        assert q_cfg["q0"]["params"]["ro_pulse_type"] == "square"
        assert q_cfg["q0"]["params"]["ro_pulse_duration"] == 400e-9

    def test_generate_device_config(self):
        dev_cfg = self.q0.generate_device_config()
        assert validate_config(dev_cfg, scheme_fn="transmon_cfg.json")
