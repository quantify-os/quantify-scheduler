# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Tests for pulses module."""

import re

import numpy as np
import pytest

from quantify_scheduler import waveforms
from quantify_scheduler.backends.qblox import constants, helpers
from quantify_scheduler.backends.qblox.operation_handling import pulses
from quantify_scheduler.backends.qblox.operations.pulse_library import (
    SimpleNumericalPulse,
)
from quantify_scheduler.backends.types import qblox as types
from quantify_scheduler.helpers.waveforms import normalize_waveform_data


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
                {"amp": 0.1234, "duration": 24e-9},
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
        t_test = np.linspace(0, duration, int(duration * constants.SAMPLING_RATE), endpoint=False)

        # act
        strategy.generate_data(wf_dict=wf_dict)

        # assert
        waveforms_generated = list(wf_dict.values())
        waveform0_data = waveforms_generated[0]["data"]
        normalized_data, amp_real, amp_imag = normalize_waveform_data(
            wf_func(t=t_test, **wf_kwargs)
        )
        assert np.allclose(waveform0_data, normalized_data.real.tolist())
        assert strategy._amplitude_path_I == pytest.approx(amp_real)
        assert strategy._amplitude_path_Q == pytest.approx(amp_imag)
        assert strategy._waveform_index0 == 0
        assert strategy._waveform_index1 is None

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
            np.arange(0, round(duration * constants.SAMPLING_RATE), 1) / constants.SAMPLING_RATE
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

    def test_insert_qasm_no_known_indices_because_low_amplitude(self, empty_qasm_program_qcm):
        # Test that an update_param is inserted
        # so that previously defined params are actually updated even if no play is played
        data = {
            "wf_func": "quantify_scheduler.waveforms.drag",
            "G_amp": 2 / constants.IMMEDIATE_SZ_GAIN,  # Right on the border
            "D_amp": 2 / constants.IMMEDIATE_SZ_GAIN,
            "duration": 24e-9,
            "nr_sigma": 3,
            "sigma": None,
            "phase": 0,
        }

        strategy = pulses.GenericPulseStrategy(
            operation_info=types.OpInfo(name="test_pulse", data=data, timing=0),
            channel_name="complex_output_0",
        )
        strategy.generate_data(wf_dict={})
        strategy.insert_qasm(empty_qasm_program_qcm)

        assert len(empty_qasm_program_qcm.instructions) == 1
        assert empty_qasm_program_qcm.instructions[0][0] == ""
        assert empty_qasm_program_qcm.instructions[0][1] == "upd_param"
        assert empty_qasm_program_qcm.instructions[0][2] == "4"
        assert empty_qasm_program_qcm.elapsed_time == 4


class TestDigitalPulseStrategy:
    def test_constructor(self):
        pulses.DigitalPulseStrategy(
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

        strategy = pulses.DigitalPulseStrategy(
            operation_info=types.OpInfo(name="test_pulse", data=data, timing=0),
            channel_name="complex_output_0",
        )
        strategy.generate_data(wf_dict={})

        with pytest.raises(
            ValueError,
            match=re.escape(
                "DigitalPulseStrategy can only be used with a digital channel. "
                "Please make sure that 'digital' keyword is included "
                "in the channel_name in the hardware configuration for port-clock combination"
                " 'None-None' "
                "(current channel_name is 'complex_output_0').Operation causing exception: "
                'Pulse "test_pulse" (t0=0, duration=2.4e-08)'
            ),
        ):
            strategy.insert_qasm(empty_qasm_program_qcm)

    def test_operation_info_property(self):
        # arrange
        operation_info = types.OpInfo(name="", data={}, timing=0)
        strategy = pulses.DigitalPulseStrategy(
            operation_info=operation_info,
            channel_name="digital_output_0",
        )

        # act
        from_property = strategy.operation_info

        # assert
        assert operation_info == from_property

    def test_generate_data(self):
        strategy = pulses.DigitalPulseStrategy(
            operation_info=types.OpInfo(name="", data={}, timing=0),
            channel_name="digital_output_0",
        )

        assert strategy.generate_data({}) is None


def test_simple_numerical_pulse():
    values = [0.2, 0.3, 0.4, 0.5]
    num_pulse = SimpleNumericalPulse(samples=values, port="q0:mw", clock="q0.01", t0=4e-9)
    waveform = helpers.generate_waveform_data(
        num_pulse.data["pulse_info"][0], sampling_rate=constants.SAMPLING_RATE
    )
    np.testing.assert_array_equal(values, waveform)


def test_simple_numerical_pulse_empty():
    values = []
    num_pulse = SimpleNumericalPulse(samples=values, port="q0:mw", clock="q0.01", t0=4e-9)
    with pytest.raises(IndexError) as error:
        helpers.generate_waveform_data(
            num_pulse.data["pulse_info"][0], sampling_rate=constants.SAMPLING_RATE
        )

    assert error.value.args[0] == "list index out of range"
