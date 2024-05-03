# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Tests for pulses module."""


import re

import numpy as np
import pytest

from quantify_scheduler import Schedule, waveforms
from quantify_scheduler.backends import SerialCompiler
from quantify_scheduler.device_under_test.quantum_device import QuantumDevice
from quantify_scheduler.backends.qblox import constants
from quantify_scheduler.backends.qblox.operation_handling import pulses
from quantify_scheduler.backends.types import qblox as types
from quantify_scheduler.helpers.waveforms import normalize_waveform_data
from quantify_scheduler.operations.gate_library import Measure
from quantify_scheduler.operations.pulse_library import (
    IdlePulse,
    MarkerPulse,
    SquarePulse,
)
from quantify_scheduler.resources import ClockResource
from tests.scheduler.instrument_coordinator.components.test_qblox import (
    make_cluster_component,
)

from .empty_qasm_program import (
    fixture_empty_qasm_program,
)


class TestGenericPulseStrategy:
    def test_constructor(self):
        pulses.GenericPulseStrategy(
            operation_info=types.OpInfo(name="", data={}, timing=0),
            channel_name="real_output_0",
        )

    def test_operation_info_property(self):
        # arrange
        operation_info = types.OpInfo(name="", data={}, timing=0)
        strategy = pulses.GenericPulseStrategy(
            operation_info=operation_info,
            channel_name="real_output_0",
        )

        # act
        from_property = strategy.operation_info

        # assert
        assert operation_info == from_property

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

        operation_info = types.OpInfo(name="", data=data, timing=0)
        strategy = pulses.GenericPulseStrategy(
            operation_info=operation_info,
            channel_name="real_output_0",
        )
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
        assert strategy._amplitude_path_I == amp_real
        assert strategy._amplitude_path_Q == amp_imag
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
            "nr_sigma": 4,
            "sigma": None,
            "phase": 0,
        }

        strategy = pulses.GenericPulseStrategy(
            operation_info=types.OpInfo(name="", data=data, timing=0),
            channel_name="complex_output_0",
        )
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

        normalized_data, amp_real, amp_imag = normalize_waveform_data(
            waveforms.drag(t=t_test, **data)
        )
        assert waveform0_data == normalized_data.real.tolist()
        assert waveform1_data == normalized_data.imag.tolist()
        assert strategy._amplitude_path_I == amp_real
        assert strategy._amplitude_path_Q == amp_imag
        assert strategy._waveform_index0 == 0
        assert strategy._waveform_index1 == 1

    def test_exception_wrong_mode(self):
        # arrange
        duration = 24e-9
        data = {
            "wf_func": "quantify_scheduler.waveforms.drag",
            "duration": duration,
            "G_amp": 0.1234,
            "D_amp": 1,
            "nr_sigma": 4,
            "sigma": None,
            "phase": 0,
        }

        strategy = pulses.GenericPulseStrategy(
            operation_info=types.OpInfo(name="test_pulse_name", data=data, timing=0),
            channel_name="real_output_0",
        )
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
            "G_amp': 0.1234, 'D_amp': 1, 'nr_sigma': 4, 'sigma': None, 'phase': 0}."
        )

    def test_insert_qasm(self, empty_qasm_program_qcm):
        # arrange
        duration = 24e-9
        wf_func_path = "quantify_scheduler.waveforms.drag"
        wf_kwargs = {
            "G_amp": 1.0,
            "D_amp": 1.0,
            "duration": 24e-9,
            "nr_sigma": 3,
            "sigma": None,
            "phase": 0,
        }
        data = {"wf_func": wf_func_path, "duration": duration, **wf_kwargs}

        strategy = pulses.GenericPulseStrategy(
            operation_info=types.OpInfo(name="test_pulse", data=data, timing=0),
            channel_name="complex_output_0",
        )
        strategy.generate_data(wf_dict={})

        # act
        strategy.insert_qasm(empty_qasm_program_qcm)

        # assert
        line0 = ["", "set_awg_gain", "32213,20356", "# setting gain for test_pulse"]
        line1 = ["", "play", "0,1,4", "# play test_pulse (24 ns)"]
        assert empty_qasm_program_qcm.instructions[0] == line0
        assert empty_qasm_program_qcm.instructions[1] == line1


class TestMarkerPulseStrategy:
    def test_constructor(self):
        pulses.MarkerPulseStrategy(
            operation_info=types.OpInfo(name="", data={}, timing=0),
            channel_name="digital_output_0",
        )

    def test_insert_qasm_exception(self, empty_qasm_program_qcm):
        duration = 24e-9
        wf_func_path = "quantify_scheduler.waveforms.drag"
        wf_kwargs = {
            "G_amp": 1.0,
            "D_amp": 1.0,
            "duration": 24e-9,
            "nr_sigma": 4,
            "phase": 0,
        }
        data = {"wf_func": wf_func_path, "duration": duration, **wf_kwargs}

        strategy = pulses.MarkerPulseStrategy(
            operation_info=types.OpInfo(name="test_pulse", data=data, timing=0),
            channel_name="complex_output_0",
        )
        strategy.generate_data(wf_dict={})

        with pytest.raises(
            ValueError,
            match=re.escape(
                "MarkerPulseStrategy can only be used with a digital channel. "
                "Please make sure that 'digital' keyword is included "
                "in the channel_name in the hardware configuration for port-clock combination"
                " 'None-None' (current channel_name is 'complex_output_0').Operation causing exception: Pulse \"test_pulse\" (t0=0, duration=2.4e-08)"
            ),
        ):
            strategy.insert_qasm(empty_qasm_program_qcm)

    def test_operation_info_property(self):
        # arrange
        operation_info = types.OpInfo(name="", data={}, timing=0)
        strategy = pulses.MarkerPulseStrategy(
            operation_info=operation_info,
            channel_name="digital_output_0",
        )

        # act
        from_property = strategy.operation_info

        # assert
        assert operation_info == from_property

    def test_generate_data(self):
        # arrange
        strategy = pulses.MarkerPulseStrategy(
            operation_info=types.OpInfo(name="", data={}, timing=0),
            channel_name="digital_output_0",
        )

        # act

        # this is what we want to verify
        data = strategy.generate_data({})

        # assert
        assert data is None

    def test_marker_pulse_compilation_qrm(
        self, mock_setup_basic_transmon_with_standard_params, make_cluster_component
    ):
        hardware_cfg = {
            "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
            "hardware_description": {
                "cluster0": {
                    "instrument_type": "Cluster",
                    "modules": {
                        "1": {"instrument_type": "QRM", "digital_output_1": {}}
                    },
                    "ref": "internal",
                }
            },
            "hardware_options": {},
            "connectivity": {
                "graph": [
                    ["cluster0.module1.complex_input_0", "q0:res"],
                    ["cluster0.module1.digital_output_1", "q0:switch"],
                ]
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
            "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
            "hardware_description": {
                "cluster0": {
                    "instrument_type": "Cluster",
                    "modules": {
                        "1": {"instrument_type": "QCM_RF", "digital_output_1": {}}
                    },
                    "ref": "internal",
                }
            },
            "hardware_options": {
                "modulation_frequencies": {"q0:res-q0.ro": {"interm_freq": 0}}
            },
            "connectivity": {
                "graph": [
                    ["cluster0.module1.complex_output_0", "q0:res"],
                    ["cluster0.module1.digital_output_1", "q0:switch"],
                ]
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

    def test_marker_pulse_added_to_operation(self):
        hw_config = {
            "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
            "hardware_description": {
                "cluster0": {
                    "instrument_type": "Cluster",
                    "modules": {
                        "1": {"instrument_type": "QCM_RF", "digital_output_1": {}}
                    },
                    "ref": "internal",
                }
            },
            "hardware_options": {
                "modulation_frequencies": {"q0:mw-q0.01": {"interm_freq": 100000000.0}}
            },
            "connectivity": {
                "graph": [
                    ["cluster0.module1.complex_output_0", "q0:mw"],
                    ["cluster0.module1.digital_output_1", "q0:switch"],
                ]
            },
        }

        quantum_device = QuantumDevice("marker_test_device")
        quantum_device.hardware_config(hw_config)

        # Define experiment schedule
        schedule = Schedule("test MarkerPulse add to Operation")
        schedule.add_resource(ClockResource(name="q0.01", freq=5.1e9))
        square_pulse_op = SquarePulse(
            amp=0.5, duration=1e-9, port="q0:mw", clock="q0.01"
        )
        square_pulse_op.add_pulse(
            MarkerPulse(duration=100e-9, port="q0:switch", t0=40e-9)
        )
        schedule.add(square_pulse_op)
        schedule.add(IdlePulse(4e-9))

        # Generate compiled schedule
        compiler = SerialCompiler(name="compiler")
        _ = compiler.compile(
            schedule=schedule, config=quantum_device.generate_compilation_config()
        )
