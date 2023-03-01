# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=no-self-use

import numpy as np
import pytest

from quantify_scheduler.backends import SerialCompiler
from quantify_scheduler.schedules import timedomain_schedules as ts

from .compiles_all_backends import _CompilesAllBackends


# FIXME classmethods cannot use fixtures, these test are mixing testing style # pylint: disable=fixme
class TestRabiPulse(_CompilesAllBackends):
    @classmethod
    def setup_class(cls):
        # Clock frequency should match the one defined in the device_cfg
        # to avoid conflicts
        cls.sched_kwargs = {
            "init_duration": 200e-6,
            "mw_G_amp": 0.5,
            "mw_D_amp": 0,
            "mw_frequency": 6.02e9,
            "mw_clock": "q0.01",
            "mw_port": "q0:mw",
            "mw_pulse_duration": 20e-9,
            "ro_pulse_amp": 0.1,
            "ro_pulse_duration": 1e-6,
            "ro_pulse_delay": 200e-9,
            "ro_pulse_port": "q0:res",
            "ro_pulse_clock": "q0.ro",
            "ro_pulse_frequency": 7.04e9,
            "ro_integration_time": 400e-9,
            "ro_acquisition_delay": 120e-9,
            "repetitions": 10,
        }
        cls.uncomp_sched = ts.rabi_pulse_sched(**cls.sched_kwargs)

    def test_repetitions(self):
        assert self.uncomp_sched.repetitions == self.sched_kwargs["repetitions"]

    def test_timing(self, device_compile_config_basic_transmon):
        # This will determine the timing
        compiler = SerialCompiler(name="compiler")
        sched = compiler.compile(
            schedule=self.uncomp_sched, config=device_compile_config_basic_transmon
        )

        # test that the right operations are added and timing is as expected.
        labels = ["qubit reset", "Rabi_pulse", "readout_pulse", "acquisition"]
        t2 = (
            self.sched_kwargs["init_duration"]
            + self.sched_kwargs["mw_pulse_duration"]
            + self.sched_kwargs["ro_pulse_delay"]
        )
        t3 = t2 + self.sched_kwargs["ro_acquisition_delay"]
        abs_times = [0, self.sched_kwargs["init_duration"], t2, t3]

        for i, schedulable in enumerate(sched.schedulables.values()):
            assert schedulable["label"] == labels[i]
            assert schedulable["abs_time"] == abs_times[i]

    def test_compiles_device_cfg_only(self, device_compile_config_basic_transmon):
        # assert that files properly compile
        compiler = SerialCompiler(name="compiler")
        compiler.compile(
            schedule=self.uncomp_sched, config=device_compile_config_basic_transmon
        )


class TestRabiSched(_CompilesAllBackends):
    @classmethod
    def setup_class(cls):
        # Clock frequency should match the one defined in the device_cfg
        # to avoid conflicts
        cls.sched_kwargs = {
            "pulse_amp": 0.2,
            "pulse_duration": 20e-9,
            "frequency": 6.02e9,
            "qubit": "q0",
            "port": None,
            "clock": None,
            "repetitions": 10,
        }
        cls.uncomp_sched = ts.rabi_sched(**cls.sched_kwargs)

    def test_repetitions(self):
        assert self.uncomp_sched.repetitions == self.sched_kwargs["repetitions"]

    def test_timing(self, device_compile_config_basic_transmon):
        # This will determine the timing
        compiler = SerialCompiler(name="compiler")
        sched = compiler.compile(
            schedule=self.uncomp_sched, config=device_compile_config_basic_transmon
        )

        # test that the right operations are added and timing is as expected.
        labels = ["Reset 0", "Rabi_pulse 0", "Measurement 0"]
        abs_times = [0, 200e-6, 200e-6 + 20e-9]

        assert len(sched.schedulables) == len(labels)
        for i, schedulable in enumerate(sched.schedulables.values()):
            assert schedulable["label"] == labels[i]
            assert schedulable["abs_time"] == abs_times[i]

    def test_rabi_pulse_ops(self):
        rabi_op_hash = list(self.uncomp_sched.schedulables.values())[1][
            "operation_repr"
        ]
        rabi_pulse = self.uncomp_sched.operations[rabi_op_hash]["pulse_info"][0]
        assert rabi_pulse["G_amp"] == 0.2
        assert rabi_pulse["D_amp"] == 0
        assert rabi_pulse["duration"] == 20e-9
        assert self.uncomp_sched.resources["q0.01"]["freq"] == 6.02e9

    def test_batched_variant_single_val(self, device_compile_config_basic_transmon):
        sched = ts.rabi_sched(
            pulse_amp=[0.5],
            pulse_duration=20e-9,
            frequency=6.02e9,
            qubit="q0",
            port=None,
            clock=None,
        )
        compiler = SerialCompiler(name="compiler")
        _ = compiler.compile(
            schedule=sched, config=device_compile_config_basic_transmon
        )

        # test that the right operations are added and timing is as expected.
        labels = ["Reset 0", "Rabi_pulse 0", "Measurement 0"]
        assert len(sched.schedulables) == len(labels)
        for i, schedulable in enumerate(sched.schedulables.values()):
            assert schedulable["label"] == labels[i]

        rabi_op_hash = list(sched.schedulables.values())[1]["operation_repr"]
        rabi_pulse = sched.operations[rabi_op_hash]["pulse_info"][0]
        assert rabi_pulse["G_amp"] == 0.5
        assert rabi_pulse["D_amp"] == 0
        assert rabi_pulse["duration"] == 20e-9

    def test_batched_variant_amps(self, device_compile_config_basic_transmon):
        amps = np.linspace(-0.5, 0.5, 5)
        # Clock frequency should match the one defined in the device_cfg
        # to avoid conflicts
        sched = ts.rabi_sched(
            pulse_amp=amps,
            pulse_duration=20e-9,
            frequency=6.02e9,
            qubit="q0",
            port=None,
            clock=None,
        )
        compiler = SerialCompiler(name="compiler")
        _ = compiler.compile(
            schedule=sched, config=device_compile_config_basic_transmon
        )

        # test that the right operations are added and timing is as expected.
        labels = []
        for j in range(5):
            labels += [f"Reset {j}", f"Rabi_pulse {j}", f"Measurement {j}"]
        assert len(sched.schedulables) == len(labels)
        for i, schedulable in enumerate(sched.schedulables.values()):
            assert schedulable["label"] == labels[i]

        for i, exp_amp in enumerate(amps):
            rabi_op_hash = list(sched.schedulables.values())[3 * i + 1][
                "operation_repr"
            ]
            rabi_pulse = sched.operations[rabi_op_hash]["pulse_info"][0]
            assert rabi_pulse["G_amp"] == exp_amp
            assert rabi_pulse["D_amp"] == 0
            assert rabi_pulse["duration"] == 20e-9

    def test_batched_variant_durations(self, device_compile_config_basic_transmon):
        durations = np.linspace(3e-9, 30e-9, 6)
        # Clock frequency should match the one defined in the device_cfg
        # to avoid conflicts
        sched = ts.rabi_sched(
            pulse_amp=0.5,
            pulse_duration=durations,
            frequency=6.02e9,
            qubit="q0",
            port=None,
            clock=None,
        )
        compiler = SerialCompiler(name="compiler")
        _ = compiler.compile(
            schedule=sched, config=device_compile_config_basic_transmon
        )

        # test that the right operations are added and timing is as expected.
        labels = []
        for j in range(6):
            labels += [f"Reset {j}", f"Rabi_pulse {j}", f"Measurement {j}"]

        assert len(sched.schedulables) == len(labels)
        for i, schedulable in enumerate(sched.schedulables.values()):
            assert schedulable["label"] == labels[i]

    def test_batched_variant_incompatible(self):
        with pytest.raises(ValueError):
            _ = ts.rabi_sched(
                pulse_amp=np.linspace(-0.3, 0.5, 3),
                pulse_duration=np.linspace(5e-9, 19e-9, 8),
                frequency=6.02e9,
                qubit="q0",
                port=None,
                clock=None,
            )

    def test_correct_inference_of_port_clock(self):
        # operation 1 is tested in test_timing to be the Rabi pulse
        op_name = list(self.uncomp_sched.schedulables.values())[1]["operation_repr"]
        rabi_op = self.uncomp_sched.operations[op_name]
        assert rabi_op["pulse_info"][0]["port"] == "q0:mw"
        assert rabi_op["pulse_info"][0]["clock"] == "q0.01"


class TestT1Sched(_CompilesAllBackends):
    @classmethod
    def setup_class(cls):
        cls.sched_kwargs = {
            "times": np.linspace(0, 80e-6, 21),
            "qubit": "q0",
            "repetitions": 10,
        }
        cls.uncomp_sched = ts.t1_sched(**cls.sched_kwargs)

    def test_repetitions(self):
        assert self.uncomp_sched.repetitions == self.sched_kwargs["repetitions"]

    def test_timing(self):
        # test that the right operations are added and timing is as expected.
        labels = []
        label_tmpl = ["Reset {}", "pi {}", "Measurement {}"]
        for i in range(len(self.sched_kwargs["times"])):
            labels += [l.format(i) for l in label_tmpl]

        for i, schedulable in enumerate(self.uncomp_sched.schedulables.values()):
            assert schedulable["label"] == labels[i]
            if (i - 2) % 3 == 0:  # every measurement operation
                assert (
                    schedulable["timing_constraints"][0]["rel_time"]
                    == self.sched_kwargs["times"][i // 3]
                )

    # pylint: disable=no-self-use
    def test_sched_float_times(self, device_compile_config_basic_transmon):
        sched_kwargs = {
            "times": 3e-6,  # a floating point time
            "qubit": "q0",
        }
        sched = ts.t1_sched(**sched_kwargs)
        compiler = SerialCompiler(name="compiler")
        _ = compiler.compile(
            schedule=sched, config=device_compile_config_basic_transmon
        )

    def test_operations(self):
        assert len(self.uncomp_sched.operations) == 2 + 21  # init, pi and 21*measure


class TestRamseySchedDetuning(_CompilesAllBackends):
    @classmethod
    def setup_class(cls):
        times = np.linspace(4.0e-6, 80e-6, 20)
        cls.sched_kwargs = {
            "times": times,
            "qubit": "q0",
            "artificial_detuning": 8 / times[-1],
            "repetitions": 10,
        }
        cls.uncomp_sched = ts.ramsey_sched(**cls.sched_kwargs)

    def test_repetitions(self):
        assert self.uncomp_sched.repetitions == self.sched_kwargs["repetitions"]

    def test_timing(self):
        # test that the right operations are added and timing is as expected.
        for i, schedulable in enumerate(self.uncomp_sched.schedulables.values()):
            if i % 4 == 0:
                assert schedulable["label"][:5] == "Reset"
            if (i - 2) % 4 == 0:  # every second pi/2 operation
                assert (
                    schedulable["timing_constraints"][0]["rel_time"]
                    == self.sched_kwargs["times"][i // 4]
                )
            if (i - 3) % 4 == 0:
                assert schedulable["label"][:11] == "Measurement"

    # pylint: disable=no-self-use
    def test_sched_float_times(self, device_compile_config_basic_transmon):
        sched_kwargs = {
            "times": 3e-6,  # a floating point time
            "qubit": "q0",
            "artificial_detuning": 250e3,
        }
        sched = ts.ramsey_sched(**sched_kwargs)
        compiler = SerialCompiler(name="compiler")
        _ = compiler.compile(
            schedule=sched, config=device_compile_config_basic_transmon
        )
        assert any(
            op["timing_constraints"][0]["rel_time"] == 3e-6
            for op in sched.schedulables.values()
        )

    def test_operations(self):
        # 2 initial pi/2, 20 acquisitions + 6 unique rotation angles for 2nd pi/2
        assert len(self.uncomp_sched.operations) == 2 + 20 + 6


class TestRamseySched(_CompilesAllBackends):
    @classmethod
    def setup_class(cls):
        cls.sched_kwargs = {
            "times": np.linspace(4.0e-6, 80e-6, 20),
            "qubit": "q0",
            "repetitions": 10,
        }
        cls.uncomp_sched = ts.ramsey_sched(**cls.sched_kwargs)

    def test_repetitions(self):
        assert self.uncomp_sched.repetitions == self.sched_kwargs["repetitions"]

    def test_timing(self):
        # test that the right operations are added and timing is as expected.
        for i, schedulable in enumerate(self.uncomp_sched.schedulables.values()):
            if i % 4 == 0:
                assert schedulable["label"][:5] == "Reset"
            if (i - 2) % 4 == 0:  # every second pi/2 operation
                assert (
                    schedulable["timing_constraints"][0]["rel_time"]
                    == self.sched_kwargs["times"][i // 4]
                )
            if (i - 3) % 4 == 0:
                assert schedulable["label"][:11] == "Measurement"

    # pylint: disable=no-self-use
    def test_sched_float_times(self, device_compile_config_basic_transmon):
        sched_kwargs = {
            "times": 3e-6,  # a floating point time
            "qubit": "q0",
        }
        sched = ts.ramsey_sched(**sched_kwargs)
        compiler = SerialCompiler(name="compiler")
        _ = compiler.compile(
            schedule=sched, config=device_compile_config_basic_transmon
        )

    def test_operations(self):
        assert (
            len(self.uncomp_sched.operations) == 3 + 20
        )  # init, x90, Rxy(90,0) and 20 * measure


class TestEchoSched(_CompilesAllBackends):
    @classmethod
    def setup_class(cls):
        cls.sched_kwargs = {
            "times": np.linspace(4.0e-6, 80e-6, 20),
            "qubit": "q0",
            "repetitions": 10,
        }
        cls.uncomp_sched = ts.echo_sched(**cls.sched_kwargs)

    def test_repetitions(self):
        assert self.uncomp_sched.repetitions == self.sched_kwargs["repetitions"]

    # pylint: disable=no-self-use
    def test_sched_float_times(self, device_compile_config_basic_transmon):
        sched_kwargs = {
            "times": 3e-6,  # a floating point time
            "qubit": "q0",
        }
        sched = ts.echo_sched(**sched_kwargs)
        compiler = SerialCompiler(name="compiler")
        _ = compiler.compile(
            schedule=sched, config=device_compile_config_basic_transmon
        )

    def test_timing(self):
        # test that the right operations are added and timing is as expected.
        for i, schedulable in enumerate(self.uncomp_sched.schedulables.values()):
            if i % 5 == 0:
                assert schedulable["label"][:5] == "Reset"
            if (i - 2) % 5 == 0:  # every second pi/2 operation
                assert (
                    schedulable["timing_constraints"][0]["rel_time"]
                    == self.sched_kwargs["times"][i // 5] / 2
                )
            if (i - 3) % 5 == 0:  # every second pi/2 operation
                assert (
                    schedulable["timing_constraints"][0]["rel_time"]
                    == self.sched_kwargs["times"][i // 5] / 2
                )
            if (i - 4) % 5 == 0:
                assert schedulable["label"][:11] == "Measurement"

    def test_operations(self):
        # 4 for an echo
        assert len(self.uncomp_sched.operations) == 23  # init, x90, X and 20x measure


class TestAllXYSched(_CompilesAllBackends):
    @classmethod
    def setup_class(cls):
        cls.sched_kwargs = {"qubit": "q0", "repetitions": 10}
        cls.uncomp_sched = ts.allxy_sched(**cls.sched_kwargs)

    def test_repetitions(self):
        assert self.uncomp_sched.repetitions == self.sched_kwargs["repetitions"]

    def test_timing(self):
        # test that the right operations are added and timing is as expected.
        for i, schedulable in enumerate(self.uncomp_sched.schedulables.values()):
            if i % 4 == 0:
                assert schedulable["label"][:5] == "Reset"
            if (i - 3) % 4 == 0:
                assert schedulable["label"][:11] == "Measurement"

    def test_operations(self):
        # 6 +21 operations (x90, y90, X180, Y180, idle, reset, 21*measurement)
        assert len(self.uncomp_sched.operations) == 6 + 21


class TestAllXYSchedElement(_CompilesAllBackends):
    @classmethod
    def setup_class(cls):
        cls.sched_kwargs = {
            "qubit": "q0",
            "element_select_idx": 4,
        }
        cls.uncomp_sched = ts.allxy_sched(**cls.sched_kwargs)

    def test_timing(self):
        # test that the right operations are added and timing is as expected.
        for i, schedulable in enumerate(self.uncomp_sched.schedulables.values()):
            if i % 4 == 0:
                assert schedulable["label"][:5] == "Reset"
            if (i - 3) % 4 == 0:
                assert schedulable["label"][:11] == "Measurement"

    def test_operations(self):
        # 4 operations (X180, Y180, reset, measurement)
        assert len(self.uncomp_sched.operations) == 4
