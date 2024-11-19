# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Tests for Qblox backend crosstalk compensation."""
import numpy as np

from quantify_scheduler import ClockResource, Schedule, SerialCompiler
from quantify_scheduler.backends.qblox import constants, helpers
from quantify_scheduler.operations import DRAGPulse, GaussPulse, RampPulse, SquarePulse


class TestCrosstalkCompensation:
    def test_crosstalk_comp_single_pulse(self, compile_config_basic_transmon_qblox_hardware):

        sched = Schedule("crosstalk comp")
        sched.add(
            SquarePulse(
                amp=0.4,
                port="q0:mw",
                duration=20e-9,
                clock="q0.01",
                t0=4e-9,
            )
        )
        compiler = SerialCompiler(name="compiler")
        (
            compile_config_basic_transmon_qblox_hardware.hardware_compilation_config.hardware_options
        ).crosstalk = {
            "q0:mw-q0.01": {"q4:mw-q4.01": 0.5},
            "q4:mw-q4.01": {"q0:mw-q0.01": 0.5},
        }
        compiled = compiler.compile(sched, config=compile_config_basic_transmon_qblox_hardware)
        assert len(compiled.schedulables) == 2
        schedulables = list(compiled.schedulables.values())
        operations = [compiled.operations[schedulables[i].data["operation_id"]] for i in range(2)]
        assert operations[0].data["pulse_info"][0]["amp"] == 0.5333333333333333
        assert operations[1].data["pulse_info"][0]["amp"] == -0.26666666666666666

    def test_crosstalk_comp_two_pulses(self, compile_config_basic_transmon_qblox_hardware):

        sched = Schedule("crosstalk comp")
        sched.add(SquarePulse(amp=0.4, duration=20e-9, port="q0:mw", clock="q0.01"))
        sched.add(SquarePulse(amp=0.5, duration=20e-9, port="q0:mw", clock="q0.01"))
        compiler = SerialCompiler(name="compiler")
        (
            compile_config_basic_transmon_qblox_hardware.hardware_compilation_config.hardware_options
        ).crosstalk = {
            "q0:mw-q0.01": {"q4:mw-q4.01": 0.5},
            "q4:mw-q4.01": {"q0:mw-q0.01": 0.5},
        }
        compiled = compiler.compile(sched, config=compile_config_basic_transmon_qblox_hardware)
        assert len(compiled.schedulables) == 4
        expected_amplitudes = [
            0.5333333333333333,
            0.6666666666666666,
            -0.26666666666666666,
            -0.3333333333333333,
        ]
        operations = list(compiled.operations.values())
        for i, expected_amp in enumerate(expected_amplitudes):
            assert operations[i].data["pulse_info"][0]["amp"] == expected_amp

    def test_crosstalk_comp_two_overlapping_pulses(
        self, compile_config_basic_transmon_qblox_hardware
    ):
        sched = Schedule("crosstalk comp")
        ref = sched.add(SquarePulse(amp=0.3, duration=20e-9, port="q0:mw", clock="q0.01"))
        sched.add(
            SquarePulse(amp=0.4, duration=20e-9, port="q4:mw", clock="q4.01", t0=12e-9),
            ref_op=ref,
            ref_pt="start",
        )
        compiler = SerialCompiler(name="compiler")
        (
            compile_config_basic_transmon_qblox_hardware.hardware_compilation_config.hardware_options
        ).crosstalk = {
            "q0:mw-q0.01": {"q4:mw-q4.01": 0.5},
            "q4:mw-q4.01": {"q0:mw-q0.01": 0.5},
        }
        compiled = compiler.compile(sched, config=compile_config_basic_transmon_qblox_hardware)

        assert len(compiled.schedulables) == 6
        schedulables = list(compiled.schedulables.values())
        expected_amplitudes = [0.4, 0.133, -0.267, -0.2, 0.333, 0.533]
        for i, expected_amp in enumerate(expected_amplitudes):
            operation = compiled.operations[schedulables[i].data["operation_id"]]
            assert np.round(operation.data["pulse_info"][0]["amp"], 3) == expected_amp

    def test_crosstalk_comp_gauss(self, compile_config_basic_transmon_qblox_hardware):
        sched = Schedule("crosstalk comp")
        gauss = GaussPulse(
            G_amp=0.5,
            phase=90,
            port="q4:mw",
            duration=40e-9,
            clock="q4.01",
            t0=4e-9,
        )
        sched.add(gauss)
        compiler = SerialCompiler(name="compiler")
        (
            compile_config_basic_transmon_qblox_hardware.hardware_compilation_config.hardware_options
        ).crosstalk = {
            "q0:mw-q0.01": {"q4:mw-q4.01": 0.5},
            "q4:mw-q4.01": {"q0:mw-q0.01": 0.5},
        }
        compiled = compiler.compile(sched, config=compile_config_basic_transmon_qblox_hardware)
        assert len(compiled.schedulables) == 2
        schedulables = list(compiled.schedulables.values())
        expected_amplitudes = [0.667, -0.333]
        for i, expected_amp in enumerate(expected_amplitudes):
            operation = compiled.operations[schedulables[i].data["operation_id"]]
            assert np.round(operation.data["pulse_info"][0]["G_amp"], 3) == expected_amp
