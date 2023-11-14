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
        normalized_data, amp_real, amp_imag = normalize_waveform_data(
            wf_func(t=t_test, **wf_kwargs)
        )
        assert waveform0_data == normalized_data.real.tolist()
        assert strategy._amplitude_path0 == amp_real
        assert strategy._amplitude_path1 == amp_imag
        assert strategy._waveform_index0 == 0
        assert strategy._waveform_index1 == None

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
        assert strategy._amplitude_path0 == amp_real
        assert strategy._amplitude_path1 == amp_imag
        assert strategy._waveform_index0 == 0
        assert strategy._waveform_index1 == 1

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
        normalized_data, amp_real, amp_imag = normalize_waveform_data(
            wf_func(t=t_test, **wf_kwargs)
        )
        assert waveform0_data == normalized_data.real.tolist()
        assert strategy._amplitude_path0 == amp_imag
        assert strategy._amplitude_path1 == amp_real
        assert strategy._waveform_index0 == None
        assert strategy._waveform_index1 == 0

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
        wf_func_path = "quantify_scheduler.waveforms.drag"
        wf_kwargs = {
            "G_amp": 1.0,
            "D_amp": 1.0,
            "duration": 24e-9,
            "nr_sigma": 3,
            "phase": 0,
        }
        data = {"wf_func": wf_func_path, "duration": duration, **wf_kwargs}

        op_info = types.OpInfo(name="test_pulse", data=data, timing=0)
        strategy = pulses.GenericPulseStrategy(op_info, io_mode="complex")
        strategy.generate_data(wf_dict={})

        # act
        strategy.insert_qasm(qasm)

        # assert
        line0 = ["", "set_awg_gain", "32212,20355", "# setting gain for test_pulse"]
        line1 = ["", "play", "0,1,4", "# play test_pulse (24 ns)"]
        assert qasm.instructions[0] == line0
        assert qasm.instructions[1] == line1


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
        # Generate compiled schedule for QCM
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
            if re.search(r"^\s*set_mrk\s+7\s*($|#)", string):
                idx = i
                break
        assert re.search(r"^\s*upd_param\s+4\s*($|#)", seq1_digital[idx + 1])
        assert re.search(r"^\s*wait\s+496\s*($|#)", seq1_digital[idx + 2])
        assert re.search(r"^\s*set_mrk\s+3\s*($|#)", seq1_digital[idx + 3])
