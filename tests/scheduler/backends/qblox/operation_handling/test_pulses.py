# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Tests for pulses module."""
# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=redefined-outer-name
# pylint: disable=unused-argument
# pylint: disable=no-self-use
# pylint: disable=too-many-locals
# pylint: disable=too-many-arguments

import pytest
import numpy as np
import re

from quantify_scheduler import waveforms, Schedule
from quantify_scheduler.backends import SerialCompiler
from quantify_scheduler import waveforms
from quantify_scheduler.helpers.waveforms import normalize_waveform_data
from quantify_scheduler.backends.types import qblox as types
from quantify_scheduler.backends.qblox import constants
from quantify_scheduler.backends.qblox.operation_handling import pulses
from quantify_scheduler.operations.gate_library import Measure
from quantify_scheduler.operations.pulse_library import MarkerPulse, SquarePulse
from quantify_scheduler.resources import ClockResource

from tests.scheduler.instrument_coordinator.components.test_qblox import (  # pylint: disable=unused-import
    make_cluster_component,
)

from .empty_qasm_program import (  # pylint: disable=unused-import
    fixture_empty_qasm_program,
)


class TestGenericPulseStrategy:
    def test_constructor(self):
        pulses.GenericPulseStrategy(
            types.OpInfo(name="", data={}, timing=0), io_mode="real"
        )

    def test_operation_info_property(self):
        # arrange
        op_info = types.OpInfo(name="", data={}, timing=0)
        strategy = pulses.GenericPulseStrategy(op_info, io_mode="real")

        # act
        from_property = strategy.operation_info

        # assert
        assert op_info == from_property

    @pytest.mark.parametrize(
        "wf_func, wf_func_path, wf_kwargs",
        [
            (
                waveforms.square,
                "quantify_scheduler.waveforms.square",
                {"amp": 1},
            ),
            (
                waveforms.ramp,
                "quantify_scheduler.waveforms.ramp",
                {"amp": 0.1234},
            ),
            (
                waveforms.soft_square,
                "quantify_scheduler.waveforms.soft_square",
                {"amp": -0.1234},
            ),
        ],
    )
    def test_generate_data_real(self, wf_func, wf_func_path, wf_kwargs):
        # arrange
        duration = 24e-9
        data = {"wf_func": wf_func_path, "duration": duration, **wf_kwargs}

        op_info = types.OpInfo(name="", data=data, timing=0)
        strategy = pulses.GenericPulseStrategy(op_info, io_mode="real")
        wf_dict = {}
        t_test = np.linspace(0, duration, int(duration * constants.SAMPLING_RATE))

        # act
        strategy.generate_data(wf_dict=wf_dict)

        # assert
        waveforms_generated = list(wf_dict.values())
        waveform0_data = waveforms_generated[0]["data"]
        waveform1_data = waveforms_generated[1]["data"]
        normalized_data, amp_real, amp_imag = normalize_waveform_data(
            wf_func(t=t_test, **wf_kwargs)
        )
        assert waveform0_data == normalized_data.real.tolist()
        assert waveform1_data == normalized_data.imag.tolist()
        assert strategy.amplitude_path0 == amp_real
        assert strategy.amplitude_path1 == amp_imag

    def test_generate_data_complex(self):
        # arrange
        duration = 24e-9
        data = {
            "wf_func": "quantify_scheduler.waveforms.drag",
            "duration": duration,
            "G_amp": 0.1234,
            "D_amp": 1,
            "nr_sigma": 3,
            "phase": 0,
        }

        op_info = types.OpInfo(name="", data=data, timing=0)
        strategy = pulses.GenericPulseStrategy(op_info, io_mode="complex")
        wf_dict = {}
        t_test = (
            np.arange(0, int(round(duration * constants.SAMPLING_RATE)), 1)
            / constants.SAMPLING_RATE
        )

        # act
        strategy.generate_data(wf_dict=wf_dict)

        # assert
        waveforms_generated = list(wf_dict.values())
        waveform0_data = waveforms_generated[0]["data"]
        waveform1_data = waveforms_generated[1]["data"]
        del data["wf_func"]
        # pylint: disable=unexpected-keyword-arg
        # pylint doesn't understand the del so it thinks we are passing wf_func
        normalized_data, amp_real, amp_imag = normalize_waveform_data(
            waveforms.drag(t=t_test, **data)
        )
        assert waveform0_data == normalized_data.real.tolist()
        assert waveform1_data == normalized_data.imag.tolist()
        assert strategy.amplitude_path0 == amp_real
        assert strategy.amplitude_path1 == amp_imag

    @pytest.mark.parametrize(
        "wf_func, wf_func_path, wf_kwargs",
        [
            (
                waveforms.square,
                "quantify_scheduler.waveforms.square",
                {"amp": 1},
            ),
            (
                waveforms.ramp,
                "quantify_scheduler.waveforms.ramp",
                {"amp": 0.1234},
            ),
            (
                waveforms.soft_square,
                "quantify_scheduler.waveforms.soft_square",
                {"amp": -0.1234},
            ),
        ],
    )
    def test_generate_data_imag(self, wf_func, wf_func_path, wf_kwargs):
        # arrange
        duration = 24e-9
        data = {"wf_func": wf_func_path, "duration": duration, **wf_kwargs}

        op_info = types.OpInfo(name="", data=data, timing=0)
        strategy = pulses.GenericPulseStrategy(op_info, io_mode="imag")
        wf_dict = {}
        t_test = np.arange(0, duration, step=1e-9)

        # act
        strategy.generate_data(wf_dict=wf_dict)

        # assert
        waveforms_generated = list(wf_dict.values())
        waveform0_data = waveforms_generated[0]["data"]
        waveform1_data = waveforms_generated[1]["data"]
        normalized_data, amp_real, amp_imag = normalize_waveform_data(
            wf_func(t=t_test, **wf_kwargs)
        )
        assert waveform0_data == normalized_data.real.tolist()
        assert waveform1_data == normalized_data.imag.tolist()
        assert strategy.amplitude_path0 == amp_imag
        assert strategy.amplitude_path1 == amp_real

    @pytest.mark.parametrize(
        "io_mode",
        ["real", "imag"],
    )
    def test_exception_wrong_mode(self, io_mode):
        # arrange
        duration = 24e-9
        data = {
            "wf_func": "quantify_scheduler.waveforms.drag",
            "duration": duration,
            "G_amp": 0.1234,
            "D_amp": 1,
            "nr_sigma": 3,
            "phase": 0,
        }

        op_info = types.OpInfo(name="test_pulse_name", data=data, timing=0)
        strategy = pulses.GenericPulseStrategy(op_info, io_mode=io_mode)
        wf_dict = {}

        # act
        with pytest.raises(ValueError) as error:
            strategy.generate_data(wf_dict=wf_dict)

        # assert
        assert (
            error.value.args[0]
            == 'Complex valued Pulse "test_pulse_name" (t0=0, duration=2.4e-08) '
            "detected but the sequencer is not expecting complex input. This "
            "can be caused by attempting to play complex valued waveforms on "
            "an output marked as real.\n\nException caused by Pulse "
            "test_pulse_name (t=0 to 2.4e-08)\ndata={'wf_func': "
            "'quantify_scheduler.waveforms.drag', 'duration': 2.4e-08, '"
            "G_amp': 0.1234, 'D_amp': 1, 'nr_sigma': 3, 'phase': 0}."
        )

    def test_insert_qasm(self, empty_qasm_program_qcm):
        # arrange
        qasm = empty_qasm_program_qcm
        duration = 24e-9
        wf_func_path, wf_kwargs = ("quantify_scheduler.waveforms.square", {"amp": 1})
        data = {"wf_func": wf_func_path, "duration": duration, **wf_kwargs}

        op_info = types.OpInfo(name="test_pulse", data=data, timing=0)
        strategy = pulses.GenericPulseStrategy(op_info, io_mode="real")
        strategy.generate_data(wf_dict={})

        # act
        strategy.insert_qasm(qasm)

        # assert
        line0 = ["", "set_awg_gain", "32767,0", "# setting gain for test_pulse"]
        line1 = ["", "play", "0,1,4", "# play test_pulse (24 ns)"]
        assert qasm.instructions[0] == line0
        assert qasm.instructions[1] == line1


class TestStitchedSquarePulseStrategy:
    @pytest.mark.filterwarnings("ignore::FutureWarning")
    def test_constructor(self):
        pulses.StitchedSquarePulseStrategy(
            types.OpInfo(name="", data={}, timing=0), io_mode="real"
        )

    @pytest.mark.filterwarnings("ignore::FutureWarning")
    def test_operation_info_property(self):
        # arrange
        op_info = types.OpInfo(name="", data={}, timing=0)
        strategy = pulses.StitchedSquarePulseStrategy(op_info, io_mode="real")

        # act
        from_property = strategy.operation_info

        # assert
        assert op_info == from_property

    @pytest.mark.filterwarnings("ignore::FutureWarning")
    @pytest.mark.parametrize("duration", [400e-9, 1e-6, 1e-3])
    def test_generate_data(self, duration):
        # arrange
        num_samples = int(constants.PULSE_STITCHING_DURATION * constants.SAMPLING_RATE)
        op_info = types.OpInfo(name="", data={"amp": 0.4}, timing=0)
        strategy = pulses.StitchedSquarePulseStrategy(op_info, io_mode="complex")

        wf_dict = {}

        # act
        strategy.generate_data(wf_dict)

        # assert
        waveforms_generated = list(wf_dict.values())
        waveform0_data = waveforms_generated[0]["data"]
        waveform1_data = waveforms_generated[1]["data"]

        answer_path0 = np.ones(num_samples).tolist()
        answer_path1 = np.zeros(num_samples).tolist()
        assert waveform0_data == answer_path0
        assert waveform1_data == answer_path1

    @pytest.mark.filterwarnings("ignore::FutureWarning")
    @pytest.mark.parametrize(
        "duration, io_mode",
        [
            (400e-9, "real"),
            (1e-6, "real"),
            (1e-3, "real"),
            (400e-9, "imag"),
            (1e-6, "imag"),
            (1e-3, "imag"),
        ],
    )
    def test_generate_data_real_or_imag(self, duration, io_mode):
        # arrange
        num_samples = int(constants.PULSE_STITCHING_DURATION * constants.SAMPLING_RATE)
        op_info = types.OpInfo(name="", data={"amp": 0.4}, timing=0)
        strategy = pulses.StitchedSquarePulseStrategy(op_info, io_mode=io_mode)

        wf_dict = {}

        # act
        strategy.generate_data(wf_dict)

        # assert
        waveforms_generated = list(wf_dict.values())
        waveform0_data = waveforms_generated[0]["data"]

        answer_path0 = np.ones(num_samples).tolist()
        assert waveform0_data == answer_path0
        assert len(waveforms_generated) == 1

    @pytest.mark.filterwarnings("ignore::FutureWarning")
    @pytest.mark.parametrize(
        "duration, answer",
        [
            (
                400e-9,
                [
                    ["", "set_awg_gain", "32767,0", "# setting gain for test_pulse"],
                    ["", "play", "0,1,400", ""],
                    ["", "set_awg_gain", "0,0", "# set to 0 at end of pulse"],
                ],
            ),
            (
                1e-6,
                [
                    ["", "set_awg_gain", "32767,0", "# setting gain for test_pulse"],
                    ["", "play", "0,1,1000", ""],
                    ["", "set_awg_gain", "0,0", "# set to 0 at end of pulse"],
                ],
            ),
            (
                1.2e-6,
                [
                    ["", "set_awg_gain", "32767,0", "# setting gain for test_pulse"],
                    ["", "play", "0,1,1000", ""],
                    ["", "play", "0,1,200", ""],
                    ["", "set_awg_gain", "0,0", "# set to 0 at end of pulse"],
                ],
            ),
            (
                2e-6,
                [
                    ["", "set_awg_gain", "32767,0", "# setting gain for test_pulse"],
                    ["", "move", "2,R0", "# iterator for loop with label stitch1"],
                    ["stitch1:", "", "", ""],
                    ["", "play", "0,1,1000", ""],
                    ["", "loop", "R0,@stitch1", ""],
                ],
            ),
            (
                2.4e-6,
                [
                    ["", "set_awg_gain", "32767,0", "# setting gain for test_pulse"],
                    ["", "move", "2,R0", "# iterator for loop with label stitch1"],
                    ["stitch1:", "", "", ""],
                    ["", "play", "0,1,1000", ""],
                    ["", "loop", "R0,@stitch1", ""],
                    ["", "play", "0,1,400", ""],
                    ["", "set_awg_gain", "0,0", "# set to 0 at end of pulse"],
                ],
            ),
            (
                1e-3,
                [
                    ["", "set_awg_gain", "32767,0", "# setting gain for test_pulse"],
                    ["", "move", "1000,R0", "# iterator for loop with label stitch1"],
                    ["stitch1:", "", "", ""],
                    ["", "play", "0,1,1000", ""],
                    ["", "loop", "R0,@stitch1", ""],
                ],
            ),
        ],
    )
    def test_insert_qasm(self, empty_qasm_program_qcm, duration, answer):
        # arrange
        qasm = empty_qasm_program_qcm
        wf_func_path, wf_kwargs = ("quantify_scheduler.waveforms.square", {"amp": 1})
        data = {"wf_func": wf_func_path, "duration": duration, **wf_kwargs}

        op_info = types.OpInfo(name="test_pulse", data=data, timing=0)
        strategy = pulses.StitchedSquarePulseStrategy(op_info, io_mode="complex")
        strategy.generate_data(wf_dict={})

        # act
        strategy.insert_qasm(qasm)

        # assert
        for row_idx, instruction in enumerate(qasm.instructions):
            assert instruction == answer[row_idx]


class TestStaircasePulseStrategy:
    @pytest.mark.filterwarnings("ignore::FutureWarning")
    def test_constructor(self):
        pulses.StaircasePulseStrategy(
            types.OpInfo(name="", data={}, timing=0), io_mode="complex"
        )

    @pytest.mark.filterwarnings("ignore::FutureWarning")
    def test_operation_info_property(self):
        # arrange
        op_info = types.OpInfo(name="", data={}, timing=0)
        strategy = pulses.StaircasePulseStrategy(op_info, io_mode="real")

        # act
        from_property = strategy.operation_info

        # assert
        assert op_info == from_property

    @pytest.mark.filterwarnings("ignore::FutureWarning")
    def test_generate_data(self):
        # arrange
        op_info = types.OpInfo(name="", data={}, timing=0)
        strategy = pulses.StaircasePulseStrategy(op_info, io_mode="real")

        # act
        # pylint: disable=assignment-from-none
        # this is what we want to verify
        data = strategy.generate_data({})

        # assert
        assert data is None

    @pytest.mark.filterwarnings("ignore::FutureWarning")
    @pytest.mark.parametrize(
        "start_amp, final_amp, num_steps, io_mode, answer",
        [
            (
                0,
                1,
                10,
                "real",
                [
                    ["", "set_awg_gain", "32767,32767", "# set gain to known value"],
                    ["", "move", "0,R0", "# keeps track of the offsets"],
                    ["", "move", "0,R1", "# zero for unused output path"],
                    ["", "", "", ""],
                    ["", "move", "10,R10", "# iterator for loop with label ramp4"],
                    ["ramp4:", "", "", ""],
                    ["", "set_awg_offs", "R0,R1", ""],
                    ["", "upd_param", "4", ""],
                    ["", "add", "R0,3640,R0", "# next incr offs by 3640"],
                    ["", "wait", "116", "# auto generated wait (116 ns)"],
                    ["", "loop", "R10,@ramp4", ""],
                    ["", "set_awg_offs", "0,0", "# return offset to 0 after staircase"],
                    ["", "", "", ""],
                ],
            ),
            (
                0,
                1,
                10,
                "imag",
                [
                    ["", "set_awg_gain", "32767,32767", "# set gain to known value"],
                    ["", "move", "0,R0", "# keeps track of the offsets"],
                    ["", "move", "0,R1", "# zero for unused output path"],
                    ["", "", "", ""],
                    ["", "move", "10,R10", "# iterator for loop with label ramp4"],
                    ["ramp4:", "", "", ""],
                    ["", "set_awg_offs", "R1,R0", ""],
                    ["", "upd_param", "4", ""],
                    ["", "add", "R0,3640,R0", "# next incr offs by 3640"],
                    ["", "wait", "116", "# auto generated wait (116 ns)"],
                    ["", "loop", "R10,@ramp4", ""],
                    ["", "set_awg_offs", "0,0", "# return offset to 0 after staircase"],
                    ["", "", "", ""],
                ],
            ),
            (
                0,
                1,
                10,
                "complex",
                [
                    ["", "set_awg_gain", "32767,32767", "# set gain to known value"],
                    ["", "move", "0,R0", "# keeps track of the offsets"],
                    ["", "move", "0,R1", "# zero for unused output path"],
                    ["", "", "", ""],
                    ["", "move", "10,R10", "# iterator for loop with label ramp4"],
                    ["ramp4:", "", "", ""],
                    ["", "set_awg_offs", "R0,R1", ""],
                    ["", "upd_param", "4", ""],
                    ["", "add", "R0,3640,R0", "# next incr offs by 3640"],
                    ["", "wait", "116", "# auto generated wait (116 ns)"],
                    ["", "loop", "R10,@ramp4", ""],
                    ["", "set_awg_offs", "0,0", "# return offset to 0 after staircase"],
                    ["", "", "", ""],
                ],
            ),
            (
                -1,
                2,
                12,
                "real",
                [
                    ["", "set_awg_gain", "32767,32767", "# set gain to known value"],
                    ["", "move", "4294934527,R0", "# keeps track of the offsets"],
                    ["", "move", "0,R1", "# zero for unused output path"],
                    ["", "", "", ""],
                    ["", "move", "12,R10", "# iterator for loop with label ramp4"],
                    ["ramp4:", "", "", ""],
                    ["", "set_awg_offs", "R0,R1", ""],
                    ["", "upd_param", "4", ""],
                    ["", "add", "R0,8936,R0", "# next incr offs by 8936"],
                    ["", "wait", "96", "# auto generated wait (96 ns)"],
                    ["", "loop", "R10,@ramp4", ""],
                    ["", "set_awg_offs", "0,0", "# return offset to 0 after staircase"],
                    ["", "", "", ""],
                ],
            ),
            (
                1,
                -2,
                12,
                "imag",
                [
                    ["", "set_awg_gain", "32767,32767", "# set gain to known value"],
                    ["", "move", "32767,R0", "# keeps track of the offsets"],
                    ["", "move", "0,R1", "# zero for unused output path"],
                    ["", "", "", ""],
                    ["", "move", "12,R10", "# iterator for loop with label ramp4"],
                    ["ramp4:", "", "", ""],
                    ["", "set_awg_offs", "R1,R0", ""],
                    ["", "upd_param", "4", ""],
                    ["", "sub", "R0,8937,R0", "# next decr offs by 8937"],
                    ["", "wait", "96", "# auto generated wait (96 ns)"],
                    ["", "loop", "R10,@ramp4", ""],
                    ["", "set_awg_offs", "0,0", "# return offset to 0 after staircase"],
                    ["", "", "", ""],
                ],
            ),
        ],
    )
    def test_insert_qasm(
        self,
        empty_qasm_program_qcm,
        start_amp,
        final_amp,
        num_steps,
        io_mode,
        answer,
    ):
        # arrange
        qasm = empty_qasm_program_qcm
        wf_func_path, wf_kwargs = (
            "quantify_scheduler.waveforms.staircase",
            {"start_amp": start_amp, "final_amp": final_amp, "num_steps": num_steps},
        )

        data = {"wf_func": wf_func_path, "duration": 1.2e-6, **wf_kwargs}
        op_info = types.OpInfo(name="", data=data, timing=0)
        strategy = pulses.StaircasePulseStrategy(op_info, io_mode=io_mode)

        strategy.generate_data({})

        # act
        strategy.insert_qasm(qasm)

        # assert
        for row_idx, instruction in enumerate(qasm.instructions):
            assert instruction == answer[row_idx]


class TestMarkerPulseStrategy:
    def test_constructor(self):
        pulses.MarkerPulseStrategy(
            types.OpInfo(name="", data={}, timing=0), io_mode="real"
        )

    def test_operation_info_property(self):
        # arrange
        op_info = types.OpInfo(name="", data={}, timing=0)
        strategy = pulses.MarkerPulseStrategy(op_info, io_mode="real")

        # act
        from_property = strategy.operation_info

        # assert
        assert op_info == from_property

    def test_generate_data(self):
        # arrange
        op_info = types.OpInfo(name="", data={}, timing=0)
        strategy = pulses.MarkerPulseStrategy(op_info, io_mode="real")

        # act
        # pylint: disable=assignment-from-none
        # this is what we want to verify
        data = strategy.generate_data({})

        # assert
        assert data is None

    def test_marker_pulse_compilation_qrm(
        self, mock_setup_basic_transmon_with_standard_params, make_cluster_component
    ):
        hardware_cfg = {
            "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
            "cluster0": {
                "ref": "internal",
                "instrument_type": "Cluster",
                "cluster0_module1": {
                    "instrument_type": "QRM",
                    "complex_input_0": {
                        "portclock_configs": [
                            {"port": "q0:res", "clock": "q0.ro"},
                        ],
                    },
                    "digital_output_1": {
                        "portclock_configs": [
                            {"port": "q0:switch"},
                        ],
                    },
                },
            },
        }

        # Setup objects needed for experiment
        mock_setup = mock_setup_basic_transmon_with_standard_params
        quantum_device = mock_setup["quantum_device"]
        quantum_device.hardware_config(hardware_cfg)

        # Define experiment schedule
        schedule = Schedule("test MarkerPulse compilation")
        schedule.add(
            MarkerPulse(
                duration=500e-9,
                port="q0:switch",
            ),
        )
        schedule.add(
            Measure("q0", acq_protocol="SSBIntegrationComplex"),
            rel_time=300e-9,
            ref_pt="start",
        )
        schedule.add_resource(ClockResource(name="q0.res", freq=50e6))

        # Generate compiled schedule
        compiler = SerialCompiler(name="compiler")
        compiled_sched = compiler.compile(
            schedule=schedule, config=quantum_device.generate_compilation_config()
        )

        # # Assert markers were set correctly, and wait time is correct for QRM
        seq0_analog = compiled_sched.compiled_instructions["cluster0"][
            "cluster0_module1"
        ]["sequencers"]["seq0"]["sequence"]["program"].splitlines()
        seq1_digital = compiled_sched.compiled_instructions["cluster0"][
            "cluster0_module1"
        ]["sequencers"]["seq1"]["sequence"]["program"].splitlines()
        idx = 0
        for i, string in enumerate(seq0_analog):
            if re.search(r"^\s*reset_ph\s+", string):
                idx = i
                break
        assert re.search(r"^\s*wait\s+300\s*($|#)", seq0_analog[idx + 2])
        idx = 0
        for i, string in enumerate(seq1_digital):
            if re.search(r"^\s*set_mrk\s+2\s*($|#)", string):
                idx = i
                break
        assert re.search(r"^\s*upd_param\s+4\s*($|#)", seq1_digital[idx + 1])
        assert re.search(r"^\s*wait\s+496\s*($|#)", seq1_digital[idx + 2])
        assert re.search(r"^\s*set_mrk\s+0\s*($|#)", seq1_digital[idx + 3])

    def test_marker_pulse_compilation_qcm_rf(
        self, mock_setup_basic_transmon_with_standard_params, make_cluster_component
    ):
        hardware_cfg = {
            "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
            "cluster0": {
                "ref": "internal",
                "instrument_type": "Cluster",
                "cluster0_module1": {
                    "instrument_type": "QCM_RF",
                    "complex_output_0": {
                        "portclock_configs": [
                            {"port": "q0:res", "clock": "q0.ro", "interm_freq": 0},
                        ],
                    },
                    "digital_output_1": {
                        "portclock_configs": [
                            {
                                "port": "q0:switch",
                            },
                        ],
                    },
                },
            },
        }

        # Setup objects needed for experiment
        mock_setup = mock_setup_basic_transmon_with_standard_params
        quantum_device = mock_setup["quantum_device"]
        quantum_device.hardware_config(hardware_cfg)

        # Define experiment schedule
        schedule = Schedule("test MarkerPulse compilation")
        schedule.add(
            MarkerPulse(
                duration=500e-9,
                port="q0:switch",
            ),
        )
        schedule.add(
            SquarePulse(amp=0.2, duration=300e-9, port="q0:res", clock="q0.ro"),
            rel_time=300e-9,
            ref_pt="start",
        )
        schedule.add_resource(ClockResource(name="q0.res", freq=50e6))
        # Generate compiled schedule for QRM
        compiler = SerialCompiler(name="compiler")
        compiled_sched = compiler.compile(
            schedule=schedule, config=quantum_device.generate_compilation_config()
        )

        # Assert markers were set correctly, and wait time is correct for QRM
        seq0_analog = compiled_sched.compiled_instructions["cluster0"][
            "cluster0_module1"
        ]["sequencers"]["seq0"]["sequence"]["program"].splitlines()
        seq1_digital = compiled_sched.compiled_instructions["cluster0"][
            "cluster0_module1"
        ]["sequencers"]["seq1"]["sequence"]["program"].splitlines()
        idx = 0
        for i, string in enumerate(seq0_analog):
            if re.search(r"^\s*reset_ph\s+", string):
                idx = i
                break
        assert re.search(r"^\s*wait\s+300\s*($|#)", seq0_analog[idx + 2])
        idx = 0
        for i, string in enumerate(seq1_digital):
            if re.search(r"^\s*set_mrk\s+11\s*($|#)", string):
                idx = i
                break
        assert re.search(r"^\s*upd_param\s+4\s*($|#)", seq1_digital[idx + 1])
        assert re.search(r"^\s*wait\s+496\s*($|#)", seq1_digital[idx + 2])
        assert re.search(r"^\s*set_mrk\s+3\s*($|#)", seq1_digital[idx + 3])
