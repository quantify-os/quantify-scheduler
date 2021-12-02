import pytest
import numpy as np

from quantify_scheduler import waveforms

from quantify_scheduler.helpers.waveforms import normalize_waveform_data
from quantify_scheduler.backends.types import qblox as types
from quantify_scheduler.backends.qblox import constants
from quantify_scheduler.backends.qblox.qasm_program import QASMProgram
from quantify_scheduler.backends.qblox.register_manager import RegisterManager
from quantify_scheduler.backends.qblox.operation_handling import pulses


@pytest.fixture(name="empty_qasm_program")
def fixture_empty_qasm_program():
    static_hw_properties = types.StaticHardwareProperties(
        instrument_type="QCM",
        max_sequencers=constants.NUMBER_OF_SEQUENCERS_QCM,
        max_awg_output_voltage=2.5,
        marker_configuration=types.MarkerConfiguration(start=0b1111, end=0b0000),
        mixer_dc_offset_range=types.BoundedParameter(
            min_val=-2.5, max_val=2.5, units="V"
        ),
    )
    yield QASMProgram(static_hw_properties, RegisterManager())




class TestGenericPulseStrategy:
    def test_constructor(self):
        pulses.GenericPulseStrategy(
            types.OpInfo(name="", data={}, timing=0), output_mode="real"
        )

    def test_operation_info_property(self):
        # arrange
        op_info = types.OpInfo(name="", data={}, timing=0)
        strategy = pulses.GenericPulseStrategy(op_info, output_mode="real")

        # act
        from_property = strategy.operation_info

        # assert
        assert op_info == from_property

    @pytest.mark.parametrize(
        "wf_func, wf_func_path, wf_kwargs",
        [
            (waveforms.square, "quantify_scheduler.waveforms.square", {"amp": 1}),
            (waveforms.ramp, "quantify_scheduler.waveforms.ramp", {"amp": 0.1234}),
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
        strategy = pulses.GenericPulseStrategy(op_info, output_mode="real")
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
        assert strategy.amplitude_path0 == amp_real
        assert strategy.amplitude_path1 == amp_imag

    def test_insert_qasm(self, empty_qasm_program):
        # arrange
        qasm = empty_qasm_program
        duration = 24e-9
        wf_func_path, wf_kwargs = ("quantify_scheduler.waveforms.square", {"amp": 1})
        data = {"wf_func": wf_func_path, "duration": duration, **wf_kwargs}

        op_info = types.OpInfo(name="test_pulse", data=data, timing=0)
        strategy = pulses.GenericPulseStrategy(op_info, output_mode="real")
        strategy.generate_data(wf_dict={})

        # act
        strategy.insert_qasm(qasm)

        # assert
        line0 = ['', 'set_awg_gain', '13107,0', '# setting gain for test_pulse']
        line1 = ['', 'play', '0,1,4', '# play test_pulse (24 ns)']
        assert qasm.instructions[0] == line0
        assert qasm.instructions[1] == line1
