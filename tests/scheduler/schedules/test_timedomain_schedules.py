# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=no-self-use

from pathlib import Path
import json
import tempfile
import pytest
import numpy as np
import quantify.scheduler.schemas.examples as es
from quantify.scheduler.schedules import timedomain_schedules as ts
from quantify.scheduler.compilation import determine_absolute_timing, qcompile
from quantify.data.handling import set_datadir

# FIXME to be replaced with fixture in tests/fixtures/schedule from !49 # pylint: disable=fixme
tmp_dir = tempfile.TemporaryDirectory()

path = Path(es.__file__).parent.joinpath("transmon_test_config.json")
DEVICE_CFG = json.loads(path.read_text())

path = Path(es.__file__).parent.joinpath("qblox_test_mapping.json")
HARDWARE_MAPPING = json.loads(path.read_text())


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
            "pulse_amplitude": 0.2,
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
        labels = ["Reset 0", "Rabi_pulse 0", "Measurement 0"]
        abs_times = [0, 200e-6, 200e-6 + 20e-9]

        assert len(self.sched.timing_constraints) == len(labels)
        for i, constr in enumerate(self.sched.timing_constraints):
            assert constr["label"] == labels[i]
            assert constr["abs_time"] == abs_times[i]

    def test_rabi_pulse_ops(self):
        rabi_op_hash = self.sched.timing_constraints[1]["operation_hash"]
        rabi_pulse = self.sched.operations[rabi_op_hash]["pulse_info"][0]
        assert rabi_pulse["G_amp"] == 0.2
        assert rabi_pulse["D_amp"] == 0
        assert rabi_pulse["duration"] == 20e-9
        assert self.sched.resources["q0.01"]["freq"] == 5.442e9

    def test_batched_variant_single_val(self):
        sched = ts.rabi_sched(
            pulse_amplitude=[0.5],
            pulse_duration=20e-9,
            frequency=5.442e9,
            qubit="q0",
            port=None,
            clock=None,
        )
        sched = qcompile(sched, DEVICE_CFG)

        # test that the right operations are added and timing is as expected.
        labels = ["Reset 0", "Rabi_pulse 0", "Measurement 0"]
        assert len(sched.timing_constraints) == len(labels)
        for i, constr in enumerate(sched.timing_constraints):
            assert constr["label"] == labels[i]

        rabi_op_hash = sched.timing_constraints[1]["operation_hash"]
        rabi_pulse = sched.operations[rabi_op_hash]["pulse_info"][0]
        assert rabi_pulse["G_amp"] == 0.5
        assert rabi_pulse["D_amp"] == 0
        assert rabi_pulse["duration"] == 20e-9

    def test_batched_variant_amps(self):

        amps = np.linspace(-0.5, 0.5, 5)
        sched = ts.rabi_sched(
            pulse_amplitude=amps,
            pulse_duration=20e-9,
            frequency=5.442e9,
            qubit="q0",
            port=None,
            clock=None,
        )
        sched = qcompile(sched, DEVICE_CFG)

        # test that the right operations are added and timing is as expected.
        labels = []
        for j in range(5):
            labels += [f"Reset {j}", f"Rabi_pulse {j}", f"Measurement {j}"]
        assert len(sched.timing_constraints) == len(labels)
        for i, constr in enumerate(sched.timing_constraints):
            assert constr["label"] == labels[i]

        for i, exp_amp in enumerate(amps):
            rabi_op_hash = sched.timing_constraints[3 * i + 1]["operation_hash"]
            rabi_pulse = sched.operations[rabi_op_hash]["pulse_info"][0]
            assert rabi_pulse["G_amp"] == exp_amp
            assert rabi_pulse["D_amp"] == 0
            assert rabi_pulse["duration"] == 20e-9

    def test_batched_variant_durations(self):

        durations = np.linspace(3e-9, 30e-9, 6)
        sched = ts.rabi_sched(
            pulse_amplitude=0.5,
            pulse_duration=durations,
            frequency=5.442e9,
            qubit="q0",
            port=None,
            clock=None,
        )
        sched = qcompile(sched, DEVICE_CFG)

        # test that the right operations are added and timing is as expected.
        labels = []
        for j in range(6):
            labels += [f"Reset {j}", f"Rabi_pulse {j}", f"Measurement {j}"]

        assert len(sched.timing_constraints) == len(labels)
        for i, constr in enumerate(sched.timing_constraints):
            assert constr["label"] == labels[i]

    def test_batched_variant_incompatible(self):
        with pytest.raises(ValueError):
            _ = ts.rabi_sched(
                pulse_amplitude=np.linspace(-0.3, 0.5, 3),
                pulse_duration=np.linspace(5e-9, 19e-9, 8),
                frequency=5.442e9,
                qubit="q0",
                port=None,
                clock=None,
            )

    def test_correct_inference_of_port_clock(self):
        # operation 1 is tested in test_timing to be the Rabi pulse
        op_name = self.sched.timing_constraints[1]["operation_hash"]
        rabi_op = self.sched.operations[op_name]
        assert rabi_op["pulse_info"][0]["port"] == "q0:mw"
        assert rabi_op["pulse_info"][0]["clock"] == "q0.01"

    def test_compiles_qblox_backend(self):
        # assert that files properly compile
        qcompile(self.sched, DEVICE_CFG, HARDWARE_MAPPING)

    def test_compiles_zi_backend(self):
        pass


class TestT1Sched:
    @classmethod
    def setup_class(cls):
        set_datadir(tmp_dir.name)
        cls.sched_kwargs = {
            "times": np.linspace(0, 80e-6, 21),
            "qubit": "q0",
        }

        cls.sched = ts.t1_sched(**cls.sched_kwargs)
        cls.sched = qcompile(cls.sched, DEVICE_CFG)

    def test_timing(self):
        # test that the right operations are added and timing is as expected.
        labels = []
        label_tmpl = ["Reset {}", "pi {}", "Measurement {}"]
        for i in range(len(self.sched_kwargs["times"])):
            labels += [l.format(i) for l in label_tmpl]

        for i, constr in enumerate(self.sched.timing_constraints):
            assert constr["label"] == labels[i]
            if (i - 2) % 3 == 0:  # every measurement operation
                assert constr["rel_time"] == self.sched_kwargs["times"][i // 3]

    # pylint: disable=no-self-use
    def test_sched_float_times(self):
        sched_kwargs = {
            "times": 3e-6,  # a floating point time
            "qubit": "q0",
        }

        sched = ts.t1_sched(**sched_kwargs)
        sched = qcompile(sched, DEVICE_CFG)

    def test_operations(self):
        assert len(self.sched.operations) == 3  # init, pi and measure

    def test_compiles_qblox_backend(self):
        # assert that files properly compile
        qcompile(self.sched, DEVICE_CFG, HARDWARE_MAPPING)

    def test_compiles_zi_backend(self):
        pass


class TestRamseySched:
    @classmethod
    def setup_class(cls):
        set_datadir(tmp_dir.name)
        cls.sched_kwargs = {
            "times": np.linspace(4.0e-6, 80e-6, 20),
            "qubit": "q0",
        }

        cls.sched = ts.ramsey_sched(**cls.sched_kwargs)
        cls.sched = qcompile(cls.sched, DEVICE_CFG)

    def test_timing(self):
        # test that the right operations are added and timing is as expected.
        for i, constr in enumerate(self.sched.timing_constraints):
            if i % 4 == 0:
                assert constr["label"][:5] == "Reset"
            if (i - 2) % 4 == 0:  # every second pi/2 operation
                assert constr["rel_time"] == self.sched_kwargs["times"][i // 4]
            if (i - 3) % 4 == 0:
                assert constr["label"][:11] == "Measurement"

    # pylint: disable=no-self-use
    def test_sched_float_times(self):
        sched_kwargs = {
            "times": 3e-6,  # a floating point time
            "qubit": "q0",
        }

        sched = ts.ramsey_sched(**sched_kwargs)
        sched = qcompile(sched, DEVICE_CFG)

    def test_operations(self):
        # 4 for a regular Ramsey, more with artificial detuning
        assert len(self.sched.operations) == 4  # init, x90, Rxy(90,0) and measure

    def test_compiles_qblox_backend(self):
        # assert that files properly compile
        qcompile(self.sched, DEVICE_CFG, HARDWARE_MAPPING)

    def test_compiles_zi_backend(self):
        pass


class TestEchoSched:
    @classmethod
    def setup_class(cls):
        set_datadir(tmp_dir.name)
        cls.sched_kwargs = {
            "times": np.linspace(4.0e-6, 80e-6, 20),
            "qubit": "q0",
        }

        cls.sched = ts.echo_sched(**cls.sched_kwargs)
        cls.sched = qcompile(cls.sched, DEVICE_CFG)

    # pylint: disable=no-self-use
    def test_sched_float_times(self):
        sched_kwargs = {
            "times": 3e-6,  # a floating point time
            "qubit": "q0",
        }

        sched = ts.echo_sched(**sched_kwargs)
        sched = qcompile(sched, DEVICE_CFG)

    def test_timing(self):
        # test that the right operations are added and timing is as expected.
        for i, constr in enumerate(self.sched.timing_constraints):
            if i % 5 == 0:
                assert constr["label"][:5] == "Reset"
            if (i - 2) % 5 == 0:  # every second pi/2 operation
                assert constr["rel_time"] == self.sched_kwargs["times"][i // 5] / 2
            if (i - 3) % 5 == 0:  # every second pi/2 operation
                assert constr["rel_time"] == self.sched_kwargs["times"][i // 5] / 2
            if (i - 4) % 5 == 0:
                assert constr["label"][:11] == "Measurement"

    def test_operations(self):
        # 4 for an echo
        assert len(self.sched.operations) == 4  # init, x90, X and measure

    def test_compiles_qblox_backend(self):
        # assert that files properly compile
        qcompile(self.sched, DEVICE_CFG, HARDWARE_MAPPING)

    def test_compiles_zi_backend(self):
        pass


class TestAllXYSched:
    @classmethod
    def setup_class(cls):
        set_datadir(tmp_dir.name)
        cls.sched_kwargs = {
            "qubit": "q0",
        }

        cls.sched = ts.allxy_sched(**cls.sched_kwargs)
        cls.sched = qcompile(cls.sched, DEVICE_CFG)

    def test_timing(self):
        # test that the right operations are added and timing is as expected.
        for i, constr in enumerate(self.sched.timing_constraints):
            if i % 4 == 0:
                assert constr["label"][:5] == "Reset"
            if (i - 3) % 4 == 0:
                assert constr["label"][:11] == "Measurement"

    def test_operations(self):
        # 7 operations (x90, y90, X180, Y180, idle, reset measurement)
        assert len(self.sched.operations) == 7

    @pytest.mark.xfail  # see #89
    def test_compiles_qblox_backend(self):
        # assert that files properly compile
        qcompile(self.sched, DEVICE_CFG, HARDWARE_MAPPING)

    def test_compiles_zi_backend(self):
        pass
