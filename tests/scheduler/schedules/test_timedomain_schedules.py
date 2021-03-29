# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

import inspect
import os
import json
import tempfile
import quantify.scheduler.schemas.examples as es
from quantify.scheduler.schedules import timedomain_schedules as ts
from quantify.scheduler.compilation import determine_absolute_timing, qcompile
from quantify.data.handling import set_datadir

# TODO to be replaced with fixture in tests/fixtures/schedule from !49
tmp_dir = tempfile.TemporaryDirectory()
esp = inspect.getfile(es)
cfg_f = os.path.abspath(os.path.join(esp, "..", "transmon_test_config.json"))
with open(cfg_f, "r") as f:
    DEVICE_CFG = json.load(f)

map_f = os.path.abspath(os.path.join(esp, "..", "qblox_test_mapping.json"))
with open(map_f, "r") as f:
    HARDWARE_MAPPING = json.load(f)


class TestRabiPulse:
    @classmethod
    def setup_class(cls):
        set_datadir(tmp_dir.name)
        cls.sched_kwargs = {
            "reset_duration": 200e-6,
            "mw_G_amp": 0.5,
            "mw_D_amp": 0,
            "mw_frequency": 5.4e9,
            "mw_clock": "q0.01",
            "mw_port": "q0:mw",
            "mw_pulse_duration": 20e-9,
            "ro_pulse_amp": 0.1,
            "ro_pulse_duration": 1e-6,
            "ro_pulse_delay": 200e-9,
            "ro_pulse_port": "q0:res",
            "ro_pulse_clock": "q0.ro",
            "ro_pulse_frequency": 8e9,
            "ro_integration_time": 400e-9,
            "ro_acquisition_delay": 120e-9,
        }

        cls.sched = ts.rabi_pulse_sched(**cls.sched_kwargs)

    def test_timing(self):
        sched = determine_absolute_timing(self.sched)
        # test that the right operations are added and timing is as expected.
        labels = ["qubit reset", "Rabi_pulse", "readout_pulse", "acquisition"]
        t2 = (
            self.sched_kwargs["reset_duration"]
            + self.sched_kwargs["mw_pulse_duration"]
            + self.sched_kwargs["ro_pulse_delay"]
        )
        t3 = t2 + self.sched_kwargs["ro_acquisition_delay"]
        abs_times = [0, self.sched_kwargs["reset_duration"], t2, t3]

        for i, constr in enumerate(sched.timing_constraints):
            assert constr["label"] == labels[i]
            assert constr["abs_time"] == abs_times[i]

    def test_compiles_device_cfg_only(self):
        # assert that files properly compile
        qcompile(self.sched, DEVICE_CFG)

    def test_compiles_qblox_backend(self):
        # assert that files properly compile
        qcompile(self.sched, DEVICE_CFG, HARDWARE_MAPPING)

    def test_compiles_zi_backend(self):
        pass


class TestRabiSched:
    @classmethod
    def setup_class(cls):
        set_datadir(tmp_dir.name)
        cls.sched_kwargs = {
            "pulse_amplitude": 0.5,
            "pulse_duration": 20e-9,
            "frequency": 5.442e9,
            "qubit": "q0",
            "port": None,
            "clock": None,
        }

        cls.sched = ts.rabi_sched(**cls.sched_kwargs)
        cls.sched = qcompile(cls.sched, DEVICE_CFG)

    def test_timing(self):
        # test that the right operations are added and timing is as expected.
        labels = ["Reset", "Rabi_pulse", "Measurement"]
        abs_times = [0, 200e-6, 200e-6 + 20e-9]

        for i, constr in enumerate(self.sched.timing_constraints):
            assert constr["label"] == labels[i]
            assert constr["abs_time"] == abs_times[i]

    def test_correct_inference_of_port_clock(self):
        # operation 1 is tested in test_timing to be the Rabi pulse
        op_name = self.sched.timing_constraints[1]["operation_hash"]
        op = self.sched.operations[op_name]
        assert "q0:mw" == op["pulse_info"][0]["port"]
        assert "q0.01" == op["pulse_info"][0]["clock"]

    def test_compiles_qblox_backend(self):
        # assert that files properly compile
        qcompile(self.sched, DEVICE_CFG, HARDWARE_MAPPING)

    def test_compiles_zi_backend(self):
        pass
