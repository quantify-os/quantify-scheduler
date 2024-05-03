# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Tests for acquisitions module."""
import pprint
import re
from typing import Any, Dict

import numpy as np
import pytest
import xarray as xr
from qblox_instruments import (
    ClusterType,
    DummyBinnedAcquisitionData,
    DummyScopeAcquisitionData,
)
from qcodes.instrument.parameter import ManualParameter
from xarray import DataArray, Dataset

from quantify_scheduler import Schedule, waveforms
from quantify_scheduler.backends import SerialCompiler
from quantify_scheduler.backends.qblox import constants
from quantify_scheduler.backends.qblox.instrument_compilers import QRMCompiler
from quantify_scheduler.backends.qblox.operation_handling import acquisitions
from quantify_scheduler.backends.qblox.qasm_program import QASMProgram
from quantify_scheduler.backends.qblox.register_manager import RegisterManager
from quantify_scheduler.backends.types import qblox as types
from quantify_scheduler.enums import BinMode
from quantify_scheduler.gettables import ScheduleGettable
from quantify_scheduler.helpers.mock_instruments import MockLocalOscillator
from quantify_scheduler.instrument_coordinator.components.generic import (
    GenericInstrumentCoordinatorComponent,
)
from quantify_scheduler.instrument_coordinator.components.qblox import (
    _ModuleComponentBase,
)
from quantify_scheduler.operations.acquisition_library import SSBIntegrationComplex
from quantify_scheduler.operations.control_flow_library import Loop
from quantify_scheduler.operations.gate_library import Measure
from quantify_scheduler.operations.pulse_library import SquarePulse
from quantify_scheduler.resources import ClockResource
from quantify_scheduler.schedules.trace_schedules import (
    long_time_trace_with_qubit,
    trace_schedule_circuit_layer,
)
from tests.fixtures.mock_setup import close_instruments
from tests.scheduler.instrument_coordinator.components.test_qblox import (
    make_cluster_component,
)


@pytest.fixture(name="empty_qasm_program_qrm")
def fixture_empty_qasm_program():
    yield QASMProgram(
        static_hw_properties=QRMCompiler.static_hw_properties,
        register_manager=RegisterManager(),
        align_fields=True,
        acq_metadata=None,
    )


class MockAcquisition(acquisitions.AcquisitionStrategyPartial):
    """Used for TestAcquisitionStrategyPartial."""

    def generate_data(self, wf_dict: Dict[str, Any]):
        pass

    def _acquire_append(self, qasm_program: QASMProgram):
        pass

    def _acquire_average(self, qasm_program: QASMProgram):
        pass


class TestAcquisitionStrategyPartial:
    """
    There is some logic in the AcquisitionStrategyPartial class that deserves
    testing.
    """

    def test_operation_info_property(self):
        # arrange
        data = {"bin_mode": None, "acq_channel": 0, "acq_index": 0}
        op_info = types.OpInfo(name="", data=data, timing=0)
        strategy = MockAcquisition(op_info)

        # act
        from_property = strategy.operation_info

        # assert
        assert op_info == from_property

    @pytest.mark.parametrize("bin_mode", [BinMode.AVERAGE, BinMode.APPEND])
    def test_bin_mode(self, empty_qasm_program_qrm, bin_mode, mocker):
        # arrange
        data = {"bin_mode": bin_mode, "acq_channel": 0, "acq_index": 0}
        op_info = types.OpInfo(name="", data=data, timing=0)
        strategy = MockAcquisition(op_info)
        append_mock = mocker.patch.object(strategy, "_acquire_append")
        average_mock = mocker.patch.object(strategy, "_acquire_average")

        strategy.bin_idx_register = "R0" if bin_mode == BinMode.APPEND else None

        # act
        strategy.insert_qasm(empty_qasm_program_qrm)

        # assert
        if bin_mode == BinMode.AVERAGE:
            average_mock.assert_called_once()
            append_mock.assert_not_called()
        else:
            average_mock.assert_not_called()
            append_mock.assert_called_once()

    def test_invalid_bin_mode(self, empty_qasm_program_qrm):
        # arrange
        data = {"bin_mode": "nonsense", "acq_channel": 0, "acq_index": 0}
        op_info = types.OpInfo(name="", data=data, timing=0)
        strategy = MockAcquisition(op_info)

        # act
        with pytest.raises(RuntimeError) as exc:
            strategy.insert_qasm(empty_qasm_program_qrm)

        # assert
        assert (
            exc.value.args[0]
            == "Attempting to process an acquisition with unknown bin mode nonsense."
        )

    def test_start_acq_too_soon(self, empty_qasm_program_qrm):
        # arrange
        qasm = empty_qasm_program_qrm
        qasm.time_last_acquisition_triggered = 0
        data = {
            "bin_mode": "nonsense",
            "acq_channel": 0,
            "acq_index": 0,
            "duration": 1e-6,
        }
        op_info = types.OpInfo(name="", data=data, timing=0)
        strategy = MockAcquisition(op_info)

        # act
        with pytest.raises(ValueError) as exc:
            strategy.insert_qasm(qasm)

        # assert
        assert (
            exc.value.args[0]
            == "Attempting to start an acquisition at t=0 ns, while the last "
            "acquisition was started at t=0 ns. Please ensure a minimum interval of "
            "300 ns between acquisitions.\n\nError caused by acquisition:\n"
            "Acquisition  (t=0 to 1e-06)\ndata={'bin_mode': 'nonsense', "
            "'acq_channel': 0, 'acq_index': 0, 'duration': 1e-06}."
        )

    @pytest.mark.parametrize("bin_mode", [BinMode.AVERAGE, BinMode.APPEND])
    def test_bin_index_register_invalid(self, empty_qasm_program_qrm, bin_mode):
        # arrange
        data = {"bin_mode": bin_mode, "acq_channel": 0, "acq_index": 0}
        op_info = types.OpInfo(name="", data=data, timing=0)
        strategy = MockAcquisition(op_info)
        strategy.bin_idx_register = None if bin_mode == BinMode.APPEND else "R0"

        # act
        with pytest.raises(ValueError) as exc:
            strategy.insert_qasm(empty_qasm_program_qrm)

        # assert
        assert (
            exc.value.args[0] == f"Attempting to add acquisition with "
            f"{'append' if bin_mode == BinMode.APPEND else 'average'} binmode. "
            f"bin_idx_register {'cannot' if bin_mode == BinMode.APPEND else 'must'} "
            f"be None."
        )


class TestSquareAcquisitionStrategy:
    @pytest.mark.parametrize("bin_mode", [BinMode.AVERAGE, BinMode.APPEND])
    def test_constructor(self, bin_mode):
        data = {"bin_mode": bin_mode, "acq_channel": 0, "acq_index": 0}
        acquisitions.SquareAcquisitionStrategy(
            types.OpInfo(name="", data=data, timing=0)
        )

    def test_generate_data(self):
        # arrange
        data = {"bin_mode": None, "acq_channel": 0, "acq_index": 0}
        strategy = acquisitions.SquareAcquisitionStrategy(
            types.OpInfo(name="", data=data, timing=0)
        )
        wf_dict = {}

        # act
        strategy.generate_data(wf_dict)

        # assert
        assert len(wf_dict) == 0

    def test_acquire_average(self, empty_qasm_program_qrm):
        # arrange
        qasm = empty_qasm_program_qrm
        data = {
            "bin_mode": None,
            "acq_channel": 0,
            "acq_index": 0,
            "duration": 1e-6,
        }
        strategy = acquisitions.SquareAcquisitionStrategy(
            types.OpInfo(name="", data=data, timing=0)
        )
        strategy.generate_data({})

        # act
        strategy._acquire_average(qasm)

        # assert
        assert qasm.instructions == [["", "acquire", "0,0,4", ""]]

    def test_acquire_append(self, empty_qasm_program_qrm):
        # arrange
        qasm = empty_qasm_program_qrm
        data = {
            "bin_mode": None,
            "acq_channel": 0,
            "acq_index": 1,
            "duration": 1e-6,
        }
        strategy = acquisitions.SquareAcquisitionStrategy(
            types.OpInfo(name="", data=data, timing=0)
        )
        strategy.bin_idx_register = qasm.register_manager.allocate_register()
        strategy.generate_data({})

        # act
        strategy._acquire_append(qasm)

        # assert
        assert qasm.instructions == [
            ["", "", "", ""],
            ["", "acquire", "0,R0,4", ""],
            ["", "add", "R0,1,R0", "# Increment bin_idx for ch0"],
            ["", "", "", ""],
        ]


class TestWeightedAcquisitionStrategy:
    @pytest.mark.parametrize("bin_mode", [BinMode.AVERAGE, BinMode.APPEND])
    def test_constructor(self, bin_mode):
        data = {"bin_mode": bin_mode, "acq_channel": 0, "acq_index": 0}
        acquisitions.WeightedAcquisitionStrategy(
            types.OpInfo(name="", data=data, timing=0)
        )

    def test_generate_data(self):
        # arrange
        duration = 1e-6
        t_test = np.arange(0, duration, step=1e-9)
        weights = [
            {
                "wf_func": "quantify_scheduler.waveforms.square",
                "amp": 1,
                "duration": duration,
            },
            {
                "wf_func": "quantify_scheduler.waveforms.square",
                "amp": 0,
                "duration": duration,
            },
        ]
        data = {
            "bin_mode": None,
            "acq_channel": 0,
            "acq_index": 0,
            "waveforms": weights,
        }
        strategy = acquisitions.WeightedAcquisitionStrategy(
            types.OpInfo(name="", data=data, timing=0)
        )
        wf_dict = {}

        # act
        strategy.generate_data(wf_dict)

        # assert
        answers = [
            waveforms.square(t_test, amp=1).tolist(),
            waveforms.square(t_test, amp=0).tolist(),
        ]
        for idx, waveform in enumerate(wf_dict.values()):
            assert waveform["data"] == answers[idx]

    def test_acquire_average(self, empty_qasm_program_qrm):
        # arrange
        qasm = empty_qasm_program_qrm
        weights = [
            {
                "wf_func": "quantify_scheduler.waveforms.square",
                "amp": 1,
                "duration": 1e-6,
            },
            {
                "wf_func": "quantify_scheduler.waveforms.square",
                "amp": 0,
                "duration": 1e-6,
            },
        ]
        data = {
            "bin_mode": None,
            "acq_channel": 2,
            "acq_index": 12,
            "waveforms": weights,
        }
        strategy = acquisitions.WeightedAcquisitionStrategy(
            types.OpInfo(name="", data=data, timing=0)
        )
        strategy.generate_data({})

        # act
        strategy._acquire_average(qasm)

        # assert
        assert qasm.instructions == [
            [
                "",
                "acquire_weighed",
                "2,12,0,1,4",
                "# Store acq in acq_channel:2, bin_idx:12",
            ]
        ]

    def test_duration_must_be_present(self, empty_qasm_program_qrm):
        # arrange
        qasm = empty_qasm_program_qrm
        weights = [
            {
                "wf_func": "quantify_scheduler.waveforms.square",
                "amp": 1,
            },
            {
                "wf_func": "quantify_scheduler.waveforms.square",
                "amp": 0,
            },
        ]
        data = {
            "bin_mode": None,
            "acq_channel": 2,
            "acq_index": 12,
            "waveforms": weights,
        }
        strategy = acquisitions.WeightedAcquisitionStrategy(
            types.OpInfo(name="", data=data, timing=0)
        )
        strategy.bin_idx_register = qasm.register_manager.allocate_register()
        with pytest.raises(KeyError):
            strategy.generate_data({})

    def test_acquire_append(self, empty_qasm_program_qrm):
        # arrange
        qasm = empty_qasm_program_qrm
        weights = [
            {
                "wf_func": "quantify_scheduler.waveforms.square",
                "amp": 1,
                "duration": 1e-6,
            },
            {
                "wf_func": "quantify_scheduler.waveforms.square",
                "amp": 0,
                "duration": 1e-6,
            },
        ]
        data = {
            "bin_mode": None,
            "acq_channel": 2,
            "acq_index": 12,
            "waveforms": weights,
        }
        strategy = acquisitions.WeightedAcquisitionStrategy(
            types.OpInfo(name="", data=data, timing=0)
        )
        strategy.bin_idx_register = qasm.register_manager.allocate_register()
        strategy.generate_data({})

        # act
        strategy._acquire_append(qasm)

        assert qasm.instructions == [
            ["", "", "", ""],
            ["", "move", "0,R1", "# Store idx of acq I wave in R1"],
            ["", "move", "1,R10", "# Store idx of acq Q wave in R10."],
            [
                "",
                "acquire_weighed",
                "2,R0,R1,R10,4",
                "# Store acq in acq_channel:2, bin_idx:R0",
            ],
            ["", "add", "R0,1,R0", "# Increment bin_idx for ch2"],
            ["", "", "", ""],
        ]

    def test_bad_weights(self, empty_qasm_program_qrm):
        # arrange
        qasm = empty_qasm_program_qrm
        weights = [
            {
                "wf_func": "quantify_scheduler.waveforms.square",
                "amp": 1.2,
                "duration": 1e-6,
            },
            {
                "wf_func": "quantify_scheduler.waveforms.square",
                "amp": 0,
                "duration": 1e-6,
            },
        ]
        data = {
            "bin_mode": None,
            "acq_channel": 2,
            "acq_index": 12,
            "waveforms": weights,
        }
        strategy = acquisitions.WeightedAcquisitionStrategy(
            types.OpInfo(name="", data=data, timing=0)
        )
        strategy.bin_idx_register = qasm.register_manager.allocate_register()
        with pytest.raises(ValueError):
            strategy.generate_data({})


class TestTriggerCountStrategy:
    @pytest.mark.parametrize("bin_mode", [BinMode.AVERAGE, BinMode.APPEND])
    def test_constructor(self, bin_mode):
        data = {"bin_mode": bin_mode, "acq_channel": 0, "acq_index": 0}
        acquisitions.TriggerCountAcquisitionStrategy(
            types.OpInfo(name="", data=data, timing=0)
        )

    def test_generate_data(self):
        # arrange
        data = {"bin_mode": None, "acq_channel": 0, "acq_index": 0}
        strategy = acquisitions.TriggerCountAcquisitionStrategy(
            types.OpInfo(name="", data=data, timing=0)
        )
        wf_dict = {}

        # act
        strategy.generate_data(wf_dict)

        # assert
        assert len(wf_dict) == 0

    def test_acquire_average(self, empty_qasm_program_qrm):
        # arrange
        qasm = empty_qasm_program_qrm
        data = {
            "bin_mode": None,
            "acq_channel": 0,
            "acq_index": 0,
            "duration": 100e-6,
        }
        strategy = acquisitions.TriggerCountAcquisitionStrategy(
            types.OpInfo(name="", data=data, timing=0)
        )
        strategy.generate_data({})

        # act
        strategy._acquire_average(qasm)

        # assert
        assert qasm.instructions == [
            [
                "",
                "acquire_ttl",
                "0,0,1,4",
                "# Enable TTL acquisition of acq_channel:0, bin_mode:average",
            ],
            ["", "wait", "65532", "# auto generated wait (99996 ns)"],
            ["", "wait", "34464", "# auto generated wait (99996 ns)"],
            [
                "",
                "acquire_ttl",
                "0,0,0,4",
                "# Disable TTL acquisition of acq_channel:0, bin_mode:average",
            ],
        ]

    def test_acquire_append(self, empty_qasm_program_qrm):
        # arrange
        qasm = empty_qasm_program_qrm
        data = {
            "bin_mode": None,
            "acq_channel": 0,
            "acq_index": 5,
            "duration": 100e-6,
        }
        strategy = acquisitions.TriggerCountAcquisitionStrategy(
            types.OpInfo(name="", data=data, timing=0)
        )
        strategy.bin_idx_register = qasm.register_manager.allocate_register()
        strategy.generate_data({})

        # act
        strategy._acquire_append(qasm)

        # assert
        assert qasm.instructions == [
            [
                "",
                "acquire_ttl",
                "0,R0,1,4",
                "# Enable TTL acquisition of acq_channel:0, store in bin:R0",
            ],
            ["", "wait", "65532", "# auto generated wait (99996 ns)"],
            ["", "wait", "34464", "# auto generated wait (99996 ns)"],
            [
                "",
                "acquire_ttl",
                "0,R0,0,4",
                "# Disable TTL acquisition of acq_channel:0, store in bin:R0",
            ],
            ["", "add", "R0,1,R0", "# Increment bin_idx for ch0 by 1"],
        ]


@pytest.mark.parametrize(
    "acquisition_strategy",
    [
        acquisitions.SquareAcquisitionStrategy,
        acquisitions.WeightedAcquisitionStrategy,
        acquisitions.TriggerCountAcquisitionStrategy,
    ],
)
def test_acquire_append_invalid_bin_idx(acquisition_strategy, empty_qasm_program_qrm):
    # arrange
    data = {
        "bin_mode": BinMode.APPEND,
        "acq_channel": 0,
        "acq_index": 5,
        "duration": 100e-6,
    }
    strategy = acquisition_strategy(types.OpInfo(name="", data=data, timing=0))

    # act
    with pytest.raises(ValueError) as exc:
        strategy.insert_qasm(empty_qasm_program_qrm)

    assert (
        exc.value.args[0] == "Attempting to add acquisition with append binmode. "
        "bin_idx_register cannot be None."
    )


def test_trace_acquisition_measurement_control(
    mock_setup_basic_transmon_with_standard_params, mocker, make_cluster_component
):
    hardware_cfg = {
        "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
        "hardware_description": {
            "cluster0": {
                "instrument_type": "Cluster",
                "modules": {"4": {"instrument_type": "QRM_RF"}},
                "ref": "internal",
            }
        },
        "hardware_options": {
            "modulation_frequencies": {"q2:res-q2.ro": {"interm_freq": 50000000.0}}
        },
        "connectivity": {"graph": [["cluster0.module4.complex_output_0", "q2:res"]]},
    }

    mock_setup = mock_setup_basic_transmon_with_standard_params
    ic_cluster0 = make_cluster_component("cluster0")
    instr_coordinator = mock_setup["instrument_coordinator"]
    instr_coordinator.add_component(ic_cluster0)

    quantum_device = mock_setup["quantum_device"]
    quantum_device.hardware_config(hardware_cfg)
    quantum_device.cfg_sched_repetitions(1)

    acq_duration = 5e-6  # retrieve 5000 samples
    q2 = mock_setup["q2"]
    q2.measure.acq_delay(600e-9)
    q2.clock_freqs.readout(7404000000.0)
    q2.measure.integration_time(acq_duration)

    sample_param = ManualParameter("sample", label="Sample time", unit="s")
    sample_param.batched = True

    sampling_rate = constants.SAMPLING_RATE
    sample_times = np.arange(start=0, stop=acq_duration, step=1 / sampling_rate)

    sched_gettable = ScheduleGettable(
        quantum_device=quantum_device,
        schedule_function=trace_schedule_circuit_layer,
        schedule_kwargs={"qubit_name": q2.name},
        batched=True,
    )

    # Setup dummy acquisition data
    dummy_scope_acquisition_data = DummyScopeAcquisitionData(
        data=[(0, 1)] * 15000, out_of_range=(False, False), avg_cnt=(0, 0)
    )
    ic_cluster0.instrument.set_dummy_scope_acquisition_data(
        slot_idx=4, sequencer=None, data=dummy_scope_acquisition_data
    )

    meas_ctrl = quantum_device.instr_measurement_control.get_instr()
    meas_ctrl.settables(sample_param)
    meas_ctrl.setpoints(sample_times)
    meas_ctrl.gettables(sched_gettable)

    with mocker.patch.object(
        meas_ctrl,
        "_get_fracdone",
        side_effect=np.linspace(start=0, stop=1.0, num=4).tolist()
        + 3 * [1.0],  # Prevent StopIteration upon more calls than elem in side_effect
    ):
        try:
            dataset = meas_ctrl.run(f"Readout trace schedule of {q2.name}")
        except:
            pprint.pprint(sched_gettable.compiled_schedule.compiled_instructions)
            raise
    assert dataset.sizes == {"dim_0": acq_duration * sampling_rate}

    instr_coordinator.remove_component(ic_cluster0.name)


def test_custom_long_trace_acquisition_measurement_control(
    mock_setup_basic_transmon_with_standard_params, make_cluster_component
):
    hardware_cfg = {
        "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
        "hardware_description": {
            "cluster0": {
                "instrument_type": "Cluster",
                "modules": {"4": {"instrument_type": "QRM"}},
                "ref": "internal",
            }
        },
        "hardware_options": {},
        "connectivity": {"graph": [["cluster0.module4.real_output_0", "q2:res"]]},
    }

    mock_setup = mock_setup_basic_transmon_with_standard_params
    ic_cluster0 = make_cluster_component("cluster0")
    instr_coordinator = mock_setup["instrument_coordinator"]
    instr_coordinator.add_component(ic_cluster0)

    quantum_device = mock_setup["quantum_device"]
    quantum_device.hardware_config(hardware_cfg)
    quantum_device.cfg_sched_repetitions(1)

    acq_duration = 1e-6
    q2 = mock_setup["q2"]
    q2.measure.pulse_amp(0.2)
    q2.measure.acq_delay(600e-9)
    q2.clock_freqs.readout(300e6)
    q2.reset.duration(252e-9)
    q2.measure.integration_time(acq_duration)

    sample_param = ManualParameter("sample", label="Dummy Sample", unit="None")
    sample_param.batched = True
    num_points = 1000
    sample_setpoints = np.arange(start=0, stop=num_points, step=1)

    sched_gettable = ScheduleGettable(
        quantum_device=quantum_device,
        schedule_function=long_time_trace_with_qubit,
        schedule_kwargs={"qubit": q2, "num_points": num_points},
        batched=True,
    )

    meas_ctrl = quantum_device.instr_measurement_control.get_instr()
    meas_ctrl.settables(sample_param)
    meas_ctrl.setpoints(sample_setpoints)
    meas_ctrl.gettables(sched_gettable)
    with pytest.warns(
        FutureWarning,
        match="The format of acquisition data of looped measurements in APPEND mode will change in quantify-scheduler>=0.18.0",
    ):
        dataset = meas_ctrl.run(f"Readout long trace schedule of {q2.name}")

    assert dataset.y0.size == num_points
    assert dataset.y1.size == num_points
    instr_coordinator.remove_component(ic_cluster0.name)


@pytest.mark.parametrize(
    argnames=["qubit_name", "rotation", "threshold"],
    argvalues=[
        [qubit_name, rotation, threshold]
        for qubit_name in ["q0", "q4"]
        for rotation in [10, 340]
        for threshold in [0.5, -0.9]
    ],
)
def test_thresholded_acquisition(
    mock_setup_basic_transmon_with_standard_params,
    hardware_compilation_config_qblox_example,
    qubit_name,
    rotation,
    threshold,
):
    hardware_config = hardware_compilation_config_qblox_example
    mock_setup = mock_setup_basic_transmon_with_standard_params
    quantum_device = mock_setup["quantum_device"]
    quantum_device.hardware_config(hardware_config)
    qubit_to_device_map = {
        "q4": "cluster0_module3",
        "q0": "cluster0_module4",
    }

    qubit = mock_setup[qubit_name]
    qubit.measure.acq_rotation(rotation)
    qubit.measure.acq_threshold(threshold)

    schedule = Schedule("Thresholded acquisition")
    schedule.add(Measure(qubit_name, acq_protocol="ThresholdedAcquisition"))

    compiler = SerialCompiler("compiler", quantum_device=quantum_device)
    compiled_schedule = compiler.compile(schedule)

    compiled_instructions = compiled_schedule.compiled_instructions["cluster0"][
        qubit_to_device_map[qubit_name]
    ]
    sequencer_compiled_instructions = compiled_instructions["sequencers"]["seq0"]
    sequencer_acquisition_metadata = compiled_instructions["acq_metadata"]["seq0"]

    assert (
        sequencer_compiled_instructions["thresholded_acq_threshold"]
        == threshold * qubit.measure.integration_time() * 1e9
    )
    assert sequencer_compiled_instructions["thresholded_acq_rotation"] == rotation
    assert sequencer_acquisition_metadata.acq_protocol == "ThresholdedAcquisition"


def test_thresholded_acquisition_multiplex(
    mock_setup_basic_transmon_with_standard_params,
):
    hardware_config = {
        "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
        "hardware_description": {
            "cluster0": {
                "instrument_type": "Cluster",
                "modules": {"3": {"instrument_type": "QRM"}},
                "ref": "internal",
            },
            "iq_mixer_lo": {"instrument_type": "IQMixer"},
            "lo": {"instrument_type": "LocalOscillator", "power": 1},
        },
        "hardware_options": {
            "modulation_frequencies": {
                "q0:res-q0.ro": {"lo_freq": 7200000000.0},
                "q1:res-q1.ro": {"lo_freq": 7200000000.0},
            }
        },
        "connectivity": {
            "graph": [
                ["cluster0.module3.complex_output_0", "iq_mixer_lo.if"],
                ["lo.output", "iq_mixer_lo.lo"],
                ["iq_mixer_lo.rf", "q0:res"],
                ["iq_mixer_lo.rf", "q1:res"],
            ]
        },
    }

    mock_setup = mock_setup_basic_transmon_with_standard_params
    q0 = mock_setup["q0"]
    q1 = mock_setup["q1"]

    quantum_device = mock_setup["quantum_device"]
    quantum_device.hardware_config(hardware_config)

    rotation_q0, rotation_q1 = 350, 222
    threshold_q0, threshold_q1 = 0.2, -0.5

    q0.measure.acq_rotation(rotation_q0)
    q0.measure.acq_threshold(threshold_q0)
    q1.measure.acq_rotation(rotation_q1)
    q1.measure.acq_threshold(threshold_q1)

    schedule = Schedule("Thresholded acquisition")
    schedule.add(Measure("q0", "q1", acq_protocol="ThresholdedAcquisition"))

    compiler = SerialCompiler("compiler", quantum_device=quantum_device)
    compiled_schedule = compiler.compile(schedule)

    for index, (phase, threshold) in enumerate(
        zip((rotation_q0, rotation_q1), (threshold_q0, threshold_q1))
    ):
        sequencer_compiled_instructions = compiled_schedule.compiled_instructions[
            "cluster0"
        ]["cluster0_module3"]["sequencers"][f"seq{index}"]
        sequencer_acquisition_metadata = compiled_schedule.compiled_instructions[
            "cluster0"
        ]["cluster0_module3"]["acq_metadata"][f"seq{index}"]

        if index == 0:
            integration_length = q0.measure.integration_time() * 1e9
        else:
            integration_length = q1.measure.integration_time() * 1e9

        assert (
            sequencer_compiled_instructions["thresholded_acq_threshold"]
            == threshold * integration_length
        )
        assert sequencer_compiled_instructions["thresholded_acq_rotation"] == phase
        assert sequencer_acquisition_metadata.acq_protocol == "ThresholdedAcquisition"


def test_trigger_count_append(
    mock_setup_basic_nv, make_cluster_component, hardware_cfg_trigger_count_legacy
):
    # Setup objects needed for experiment
    ic_cluster0 = make_cluster_component("cluster0")
    laser_red = MockLocalOscillator("laser_red")
    ic_laser_red = GenericInstrumentCoordinatorComponent(laser_red)
    ic_generic = GenericInstrumentCoordinatorComponent("generic")

    instr_coordinator = mock_setup_basic_nv["instrument_coordinator"]
    instr_coordinator.add_component(ic_cluster0)
    instr_coordinator.add_component(ic_laser_red)
    instr_coordinator.add_component(ic_generic)

    quantum_device = mock_setup_basic_nv["quantum_device"]
    quantum_device.hardware_config(hardware_cfg_trigger_count_legacy)

    # Define experiment schedule
    schedule = Schedule("test multiple measurements")
    schedule.add(
        Measure(
            "qe0", acq_index=0, acq_protocol="TriggerCount", bin_mode=BinMode.APPEND
        )
    )
    schedule.add(
        Measure(
            "qe0", acq_index=1, acq_protocol="TriggerCount", bin_mode=BinMode.APPEND
        )
    )
    schedule.add(
        Measure(
            "qe0", acq_index=2, acq_protocol="TriggerCount", bin_mode=BinMode.APPEND
        )
    )

    # Setup dummy acquisition data
    ic_cluster0.instrument.set_dummy_binned_acquisition_data(
        slot_idx=3,
        sequencer=0,
        acq_index_name="0",
        data=[
            DummyBinnedAcquisitionData(data=(10000, 15000), thres=0, avg_cnt=100),
            DummyBinnedAcquisitionData(data=(20000, 25000), thres=0, avg_cnt=200),
            DummyBinnedAcquisitionData(data=(20000, 25000), thres=0, avg_cnt=300),
        ],
    )

    # Generate compiled schedule
    compiler = SerialCompiler(name="compiler")
    compiled_sched = compiler.compile(
        schedule=schedule, config=quantum_device.generate_compilation_config()
    )

    # Upload schedule and run experiment
    instr_coordinator.prepare(compiled_sched)
    instr_coordinator.start()
    data = instr_coordinator.retrieve_acquisition()
    instr_coordinator.stop()

    # Assert intended behaviour
    assert isinstance(data, Dataset)
    expected_dataarray = DataArray(
        [[100, 200, 300]],
        coords=[[0], [0, 1, 2]],
        dims=["repetition", "acq_index_0"],
        attrs={"acq_protocol": "TriggerCount"},
    )
    expected_dataset = Dataset({0: expected_dataarray})

    xr.testing.assert_identical(data, expected_dataset)

    instr_coordinator.remove_component("ic_cluster0")


def test_trigger_count_append_gettables(
    mock_setup_basic_nv, make_cluster_component, hardware_cfg_trigger_count_legacy
):
    # Setup objects needed for experiment
    ic_cluster0 = make_cluster_component("cluster0")
    laser_red = MockLocalOscillator("laser_red")
    ic_laser_red = GenericInstrumentCoordinatorComponent(laser_red)
    ic_generic = GenericInstrumentCoordinatorComponent("generic")

    instr_coordinator = mock_setup_basic_nv["instrument_coordinator"]
    instr_coordinator.add_component(ic_cluster0)
    instr_coordinator.add_component(ic_laser_red)
    instr_coordinator.add_component(ic_generic)

    quantum_device = mock_setup_basic_nv["quantum_device"]
    quantum_device.hardware_config(hardware_cfg_trigger_count_legacy)

    # Define experiment schedule
    def _schedule_function(repetitions):
        schedule = Schedule("test multiple measurements", repetitions=repetitions)
        schedule.add(
            Measure(
                "qe0", acq_index=0, acq_protocol="TriggerCount", bin_mode=BinMode.APPEND
            )
        )
        schedule.add(
            Measure(
                "qe0", acq_index=1, acq_protocol="TriggerCount", bin_mode=BinMode.APPEND
            )
        )
        schedule.add(
            Measure(
                "qe0", acq_index=2, acq_protocol="TriggerCount", bin_mode=BinMode.APPEND
            )
        )
        return schedule

    # Setup dummy acquisition data
    ic_cluster0.instrument.set_dummy_binned_acquisition_data(
        slot_idx=3,
        sequencer=0,
        acq_index_name="0",
        data=[
            DummyBinnedAcquisitionData(data=(10000, 15000), thres=0, avg_cnt=100),
            DummyBinnedAcquisitionData(data=(20000, 25000), thres=0, avg_cnt=200),
            DummyBinnedAcquisitionData(data=(20000, 25000), thres=0, avg_cnt=300),
        ],
    )

    sched_gettable = ScheduleGettable(
        quantum_device=quantum_device,
        schedule_function=_schedule_function,
        schedule_kwargs={},
        batched=True,
    )
    data = sched_gettable.get()

    # Assert intended behaviour
    np.testing.assert_array_equal(data, [[100, 200, 300]])

    instr_coordinator.remove_component("ic_cluster0")


def test_trigger_count_average(
    mock_setup_basic_nv, make_cluster_component, hardware_cfg_trigger_count_legacy
):
    # Setup objects needed for experiment
    ic_cluster0 = make_cluster_component("cluster0")
    laser_red = MockLocalOscillator("laser_red")
    ic_laser_red = GenericInstrumentCoordinatorComponent(laser_red)
    ic_generic = GenericInstrumentCoordinatorComponent("generic")

    instr_coordinator = mock_setup_basic_nv["instrument_coordinator"]
    instr_coordinator.add_component(ic_cluster0)
    instr_coordinator.add_component(ic_laser_red)
    instr_coordinator.add_component(ic_generic)

    quantum_device = mock_setup_basic_nv["quantum_device"]
    quantum_device.hardware_config(hardware_cfg_trigger_count_legacy)

    # Define experiment schedule
    schedule = Schedule("test multiple measurements")
    meas0 = Measure("qe0", acq_protocol="TriggerCount", bin_mode=BinMode.AVERAGE)
    schedule.add(meas0)

    # Setup dummy acquisition data
    ic_cluster0.instrument.set_dummy_binned_acquisition_data(
        slot_idx=3,
        sequencer=0,
        acq_index_name="0",
        data=[
            DummyBinnedAcquisitionData(data=(10000, 15000), thres=0, avg_cnt=100),
            DummyBinnedAcquisitionData(data=(20000, 25000), thres=0, avg_cnt=100),
            DummyBinnedAcquisitionData(data=(30000, 35000), thres=0, avg_cnt=75),
            DummyBinnedAcquisitionData(data=(40000, 45000), thres=0, avg_cnt=50),
            DummyBinnedAcquisitionData(data=(50000, 55000), thres=0, avg_cnt=25),
            DummyBinnedAcquisitionData(data=(60000, 65000), thres=0, avg_cnt=25),
            DummyBinnedAcquisitionData(data=(70000, 75000), thres=0, avg_cnt=5),
        ],
    )

    # Generate compiled schedule
    compiler = SerialCompiler(name="compiler")
    compiled_sched = compiler.compile(
        schedule=schedule, config=quantum_device.generate_compilation_config()
    )

    # Upload schedule and run experiment
    instr_coordinator.prepare(compiled_sched)
    instr_coordinator.start()
    data = instr_coordinator.retrieve_acquisition()
    instr_coordinator.stop()

    # Assert intended behaviour
    assert isinstance(data, Dataset)
    expected_dataarray = DataArray(
        [[25, 25, 25, 20, 5]],
        coords=[[0], [2, 3, 4, 6, 7]],
        dims=["repetition", "counts"],
        attrs={"acq_protocol": "TriggerCount"},
    )
    expected_dataset = Dataset({0: expected_dataarray})

    xr.testing.assert_identical(data, expected_dataset)

    instr_coordinator.remove_component("ic_cluster0")


def test_trigger_count_average_gettables(
    mock_setup_basic_nv, make_cluster_component, hardware_cfg_trigger_count_legacy
):
    # Setup objects needed for experiment
    ic_cluster0 = make_cluster_component("cluster0")
    laser_red = MockLocalOscillator("laser_red")
    ic_laser_red = GenericInstrumentCoordinatorComponent(laser_red)
    ic_generic = GenericInstrumentCoordinatorComponent("generic")

    instr_coordinator = mock_setup_basic_nv["instrument_coordinator"]
    instr_coordinator.add_component(ic_cluster0)
    instr_coordinator.add_component(ic_laser_red)
    instr_coordinator.add_component(ic_generic)

    quantum_device = mock_setup_basic_nv["quantum_device"]
    quantum_device.hardware_config(hardware_cfg_trigger_count_legacy)

    # Define experiment schedule
    def _schedule_function(repetitions):
        schedule = Schedule("test multiple measurements", repetitions=repetitions)
        meas0 = Measure("qe0", acq_protocol="TriggerCount", bin_mode=BinMode.AVERAGE)
        schedule.add(meas0)
        return schedule

    # Setup dummy acquisition data
    ic_cluster0.instrument.set_dummy_binned_acquisition_data(
        slot_idx=3,
        sequencer=0,
        acq_index_name="0",
        data=[
            DummyBinnedAcquisitionData(data=(10000, 15000), thres=0, avg_cnt=100),
            DummyBinnedAcquisitionData(data=(20000, 25000), thres=0, avg_cnt=100),
            DummyBinnedAcquisitionData(data=(30000, 35000), thres=0, avg_cnt=75),
            DummyBinnedAcquisitionData(data=(40000, 45000), thres=0, avg_cnt=50),
            DummyBinnedAcquisitionData(data=(50000, 55000), thres=0, avg_cnt=25),
            DummyBinnedAcquisitionData(data=(60000, 65000), thres=0, avg_cnt=25),
            DummyBinnedAcquisitionData(data=(70000, 75000), thres=0, avg_cnt=5),
        ],
    )

    # Generate compiled schedule
    sched_gettable = ScheduleGettable(
        quantum_device=quantum_device,
        schedule_function=_schedule_function,
        schedule_kwargs={},
        batched=True,
    )
    data = sched_gettable.get()

    np.testing.assert_array_equal(data, [[25, 25, 25, 20, 5]])

    instr_coordinator.remove_component("ic_cluster0")


def test_mixed_binned_trace_measurements(
    mock_setup_basic_transmon, make_cluster_component
):
    hardware_cfg = {
        "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
        "hardware_description": {
            "cluster0": {
                "instrument_type": "Cluster",
                "modules": {"3": {"instrument_type": "QRM"}},
                "ref": "internal",
            }
        },
        "hardware_options": {
            "modulation_frequencies": {"q0:res-q0.ro": {"interm_freq": 50000000.0}}
        },
        "connectivity": {
            "graph": [
                ["cluster0.module3.complex_output_0", "q0:res"],
                ["cluster0.module3.real_output_0", "q1:res"],
            ]
        },
    }

    # Setup objects needed for experiment
    mock_setup = mock_setup_basic_transmon
    ic_cluster0 = make_cluster_component("cluster0")

    instr_coordinator = mock_setup["instrument_coordinator"]
    instr_coordinator.add_component(ic_cluster0)

    quantum_device = mock_setup["quantum_device"]
    quantum_device.hardware_config(hardware_cfg)

    q0 = mock_setup["q0"]
    q1 = mock_setup["q1"]
    q0.clock_freqs.readout(50e6)
    q1.clock_freqs.readout(50e6)

    # Define experiment schedule
    schedule = Schedule("test multiple measurements")
    meas0 = Measure("q0", acq_protocol="SSBIntegrationComplex")
    meas1 = Measure("q1", acq_protocol="Trace")
    schedule.add(meas0)
    schedule.add(meas1)

    # Change acq delay, duration and channel
    q0.measure.acq_delay(1e-6)
    q1.measure.acq_delay(1e-6)
    q1.measure.acq_channel(1)
    q0.measure.integration_time(5e-6)
    q1.measure.integration_time(3e-6)

    # Setup dummy acquisition data
    dummy_scope_acquisition_data = DummyScopeAcquisitionData(
        data=[(0, 1)] * 15000, out_of_range=(False, False), avg_cnt=(0, 0)
    )
    ic_cluster0.instrument.set_dummy_scope_acquisition_data(
        slot_idx=3, sequencer=None, data=dummy_scope_acquisition_data
    )
    dummy_binned_acquisition_data = [
        DummyBinnedAcquisitionData(data=(100.0, 200.0), thres=0, avg_cnt=0),
    ]
    ic_cluster0.instrument.set_dummy_binned_acquisition_data(
        slot_idx=3, sequencer=0, acq_index_name="0", data=dummy_binned_acquisition_data
    )

    # Generate compiled schedule
    compiler = SerialCompiler(name="compiler")
    compiled_sched = compiler.compile(
        schedule=schedule, config=quantum_device.generate_compilation_config()
    )

    # Upload schedule and run experiment
    instr_coordinator.prepare(compiled_sched)
    instr_coordinator.start()
    data = instr_coordinator.retrieve_acquisition()
    instr_coordinator.stop()

    # Assert intended behaviour
    assert isinstance(data, Dataset)
    expected_dataarray_trace = DataArray(
        [[1j] * 3000],
        coords=[[0], range(3000)],
        dims=["acq_index_1", "trace_index_1"],
        attrs={"acq_protocol": "Trace"},
    )
    expected_dataarray_binned = DataArray(
        [0.02 + 0.04j],
        coords=[[0]],
        dims=["acq_index_0"],
        attrs={"acq_protocol": "SSBIntegrationComplex"},
    )
    expected_dataset = Dataset(
        {0: expected_dataarray_binned, 1: expected_dataarray_trace}
    )

    xr.testing.assert_identical(data, expected_dataset)

    instr_coordinator.remove_component("ic_cluster0")


def test_multiple_trace_raises(
    mock_setup_basic_transmon_with_standard_params, make_cluster_component
):
    hardware_cfg = {
        "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
        "hardware_description": {
            "cluster0": {
                "instrument_type": "Cluster",
                "modules": {"3": {"instrument_type": "QRM_RF"}},
                "ref": "internal",
            }
        },
        "hardware_options": {
            "modulation_frequencies": {"q0:res-q0.ro": {"interm_freq": 50000000.0}}
        },
        "connectivity": {"graph": [["cluster0.module3.complex_output_0", "q0:res"]]},
    }

    # Setup objects needed for experiment
    mock_setup = mock_setup_basic_transmon_with_standard_params
    ic_cluster0 = make_cluster_component("cluster0")
    instr_coordinator = mock_setup["instrument_coordinator"]
    instr_coordinator.add_component(ic_cluster0)

    quantum_device = mock_setup["quantum_device"]
    quantum_device.hardware_config(hardware_cfg)

    q0 = mock_setup["q0"]

    # Define experiment schedule
    schedule = Schedule("test multiple measurements")
    meas0 = Measure("q0", acq_protocol="Trace")
    schedule.add(meas0)

    # Change acq delay, duration and channel
    q0.measure.acq_delay(1e-6)
    q0.measure.integration_time(5e-6)

    # Generate compiled schedule
    compiler = SerialCompiler(name="compiler")
    compiled_sched = compiler.compile(
        schedule=schedule, config=quantum_device.generate_compilation_config()
    )

    # Imitate a compiled schedule which contains multiple trace acquisition for one module.
    acq_metadata = compiled_sched.compiled_instructions["cluster0"]["cluster0_module3"][
        "acq_metadata"
    ]
    acq_metadata_trace = acq_metadata["seq0"]
    acq_metadata = {"seq0": acq_metadata_trace, "seq1": acq_metadata_trace}
    compiled_sched.compiled_instructions["cluster0"]["cluster0_module3"][
        "acq_metadata"
    ] = acq_metadata

    with pytest.raises(ValueError) as exc:
        instr_coordinator.prepare(compiled_sched)

    # assert
    assert exc.value.args[0] == (
        "Both sequencer '1' and '0' "
        "of 'ic_cluster0_module3' attempts to perform scope mode acquisitions. "
        "Only one sequencer per device can "
        "trigger raw trace capture.\n\nPlease ensure that "
        "only one port-clock combination performs "
        "raw trace acquisition per instrument."
    )

    instr_coordinator.remove_component("ic_cluster0")


@pytest.mark.parametrize(
    "qubit_to_overwrite",
    ["q1", "q2"],
)
def test_same_index_in_module_and_cluster_measurement_error(
    mocker,
    mock_setup_basic_transmon_with_standard_params,
    make_cluster_component,
    qubit_to_overwrite,
):
    hardware_cfg = {
        "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
        "hardware_description": {
            "cluster0": {
                "instrument_type": "Cluster",
                "modules": {
                    "3": {"instrument_type": "QRM"},
                    "4": {"instrument_type": "QRM_RF"},
                },
                "ref": "internal",
            }
        },
        "hardware_options": {
            "modulation_frequencies": {
                "q0:res-q0.ro": {"interm_freq": 50000000.0},
                "q1:res-q1.ro": {"interm_freq": 50000000.0},
                "q2:res-q2.ro": {"interm_freq": 50000000.0},
            }
        },
        "connectivity": {
            "graph": [
                ["cluster0.module3.complex_output_0", "q0:res"],
                ["cluster0.module3.complex_output_0", "q1:res"],
                ["cluster0.module4.complex_output_0", "q2:res"],
            ]
        },
    }

    # Setup objects needed for experiment
    mock_setup = mock_setup_basic_transmon_with_standard_params
    ic_cluster0 = make_cluster_component("cluster0")
    instr_coordinator = mock_setup["instrument_coordinator"]
    instr_coordinator.add_component(ic_cluster0)

    for comp in ic_cluster0._cluster_modules.values():
        instrument = comp.instrument
        mock_acquisition_data = {
            "0": {
                "index": 0,
                "acquisition": {"bins": {"integration": {"path0": [0], "path1": [0]}}},
            }
        }
        mocker.patch.object(
            instrument, "get_acquisitions", return_value=mock_acquisition_data
        )

    quantum_device = mock_setup["quantum_device"]
    quantum_device.hardware_config(hardware_cfg)

    # Define experiment schedule
    schedule = Schedule("test multiple measurements")
    schedule.add(Measure("q0", acq_protocol="SSBIntegrationComplex", acq_index=0))
    schedule.add(
        Measure(qubit_to_overwrite, acq_protocol="SSBIntegrationComplex", acq_index=0)
    )
    schedule.add_resource(ClockResource(name="q0.ro", freq=50e6))
    schedule.add_resource(ClockResource(name="q1.ro", freq=50e6))

    # Change acq delay, duration and channel
    q0 = mock_setup["q0"]
    q0.measure.acq_delay(1e-6)
    q0.measure.integration_time(5e-6)
    q1 = mock_setup["q1"]
    q1.measure.acq_delay(1e-6)
    q1.measure.integration_time(5e-6)
    q2 = mock_setup["q2"]
    q2.measure.acq_delay(600e-9)
    q2.clock_freqs.readout(7404000000.0)
    q2.measure.integration_time(5e-6)

    # Generate compiled schedule
    compiler = SerialCompiler(name="compiler")
    compiled_sched = compiler.compile(
        schedule=schedule, config=quantum_device.generate_compilation_config()
    )

    # Upload schedule and run experiment
    instr_coordinator.prepare(compiled_sched)
    instr_coordinator.start()

    with pytest.raises(RuntimeError) as exc:
        instr_coordinator.retrieve_acquisition()

    # assert
    assert (
        exc.value.args[0] == "Attempting to gather acquisitions. "
        "Make sure an acq_channel, acq_index corresponds to not more than one acquisition.\n"
        "The following indices are defined multiple times.\n"
        "acq_channel=0; acq_index_0=0"
    )

    instr_coordinator.stop()

    instr_coordinator.remove_component("ic_cluster0")


def test_complex_input_hardware_cfg(make_cluster_component, mock_setup_basic_transmon):
    # for a transmon measurement now both input and output can be used to run it.
    # if we like to take these apart, dispersive_measurement should be adjusted.
    hardware_cfg = {
        "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
        "hardware_description": {
            "cluster0": {
                "instrument_type": "Cluster",
                "modules": {"3": {"instrument_type": "QRM"}},
                "ref": "internal",
            }
        },
        "hardware_options": {
            "modulation_frequencies": {
                "q0:res-q0.ro": {"interm_freq": 50000000.0},
                "q1:res-q1.ro": {"interm_freq": 50000000.0},
            }
        },
        "connectivity": {
            "graph": [
                ["cluster0.module3.complex_input_0", "q0:res"],
                ["cluster0.module3.complex_output_0", "q1:res"],
            ]
        },
    }

    # Setup objects needed for experiment
    ic_cluster0 = make_cluster_component("cluster0")
    instr_coordinator = mock_setup_basic_transmon["instrument_coordinator"]
    instr_coordinator.add_component(ic_cluster0)

    quantum_device = mock_setup_basic_transmon["quantum_device"]
    quantum_device.hardware_config(hardware_cfg)

    q0 = quantum_device.get_element("q0")
    q1 = quantum_device.get_element("q1")

    # Define experiment schedule
    schedule = Schedule("test complex input")
    schedule.add_resource(ClockResource(name="q1.ro", freq=50e6))
    schedule.add_resource(ClockResource(name="q0.ro", freq=50e6))
    schedule.add(Measure("q0", acq_protocol="SSBIntegrationComplex"))
    schedule.add(Measure("q1", acq_protocol="SSBIntegrationComplex"))

    # Change acq delay
    q0.measure.acq_delay(4e-9)
    q1.measure.acq_delay(4e-9)
    q1.measure.acq_channel(1)

    # Generate compiled schedule
    compiler = SerialCompiler(name="compiler")
    compiled_sched = compiler.compile(
        schedule=schedule, config=quantum_device.generate_compilation_config()
    )

    # Setup dummy acquisition data
    dummy_binned_acquisition_data = [
        DummyBinnedAcquisitionData(data=(100.0, 200.0), thres=0, avg_cnt=0),
    ]
    ic_cluster0.instrument.set_dummy_binned_acquisition_data(
        slot_idx=3, sequencer=0, acq_index_name="0", data=dummy_binned_acquisition_data
    )
    ic_cluster0.instrument.set_dummy_binned_acquisition_data(
        slot_idx=3, sequencer=1, acq_index_name="0", data=dummy_binned_acquisition_data
    )

    # Upload schedule and run experiment
    instr_coordinator.prepare(compiled_sched)
    instr_coordinator.start()
    data = instr_coordinator.retrieve_acquisition()
    instr_coordinator.stop()

    # Assert intended behaviour
    assert isinstance(data, Dataset)
    expected_dataarray_0 = DataArray(
        [0.1 + 0.2j],
        coords=[[0]],
        dims=["acq_index_0"],
        attrs={"acq_protocol": "SSBIntegrationComplex"},
    )
    expected_dataarray_1 = DataArray(
        [0.1 + 0.2j],
        coords=[[0]],
        dims=["acq_index_1"],
        attrs={"acq_protocol": "SSBIntegrationComplex"},
    )
    expected_dataset = Dataset({0: expected_dataarray_0, 1: expected_dataarray_1})
    xr.testing.assert_identical(data, expected_dataset)
    assert compiled_sched.compiled_instructions["cluster0"]["cluster0_module3"][
        "sequencers"
    ]["seq0"]["connected_input_indices"] == [0, 1]

    instr_coordinator.remove_component("ic_cluster0")


def test_multi_real_input_hardware_cfg_trigger_count(
    make_cluster_component, mock_setup_basic_nv
):
    hardware_cfg = {
        "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
        "cluster0": {
            "ref": "internal",
            "instrument_type": "Cluster",
            "cluster0_module3": {
                "instrument_type": "QRM",
                "real_output_0": {
                    "lo_name": "laser_red",
                    "mix_lo": False,
                    "portclock_configs": [
                        {
                            "port": "qe0:optical_control",
                            "clock": "qe0.ge0",
                            "interm_freq": 200e6,
                        },
                    ],
                },
                "real_output_1": {
                    "lo_name": "laser_red",
                    "mix_lo": False,
                    "portclock_configs": [
                        {
                            "port": "qe1:optical_control",
                            "clock": "qe1.ge0",
                            "interm_freq": 200e6,
                        },
                    ],
                },
                "real_input_0": {
                    "portclock_configs": [
                        {
                            "port": "qe0:optical_readout",
                            "clock": "qe0.ge0",
                            "interm_freq": 0,
                            "ttl_acq_threshold": 0.5,
                        },
                    ],
                },
                "real_input_1": {
                    "portclock_configs": [
                        {
                            "port": "qe1:optical_readout",
                            "clock": "qe1.ge0",
                            "interm_freq": 0,
                            "ttl_acq_threshold": 0.5,
                        },
                    ],
                },
            },
        },
        "laser_red": {
            "instrument_type": "LocalOscillator",
            "frequency": None,
            "power": 1,
        },
    }

    # Setup objects needed for experiment
    ic_cluster0 = make_cluster_component("cluster0")
    laser_red = MockLocalOscillator("laser_red")
    ic_laser_red = GenericInstrumentCoordinatorComponent(laser_red)
    ic_generic = GenericInstrumentCoordinatorComponent("generic")

    instr_coordinator = mock_setup_basic_nv["instrument_coordinator"]
    instr_coordinator.add_component(ic_cluster0)
    instr_coordinator.add_component(ic_laser_red)
    instr_coordinator.add_component(ic_generic)

    quantum_device = mock_setup_basic_nv["quantum_device"]
    quantum_device.hardware_config(hardware_cfg)

    # Define experiment schedule
    schedule = Schedule("test NV measurement with real output and input")
    schedule.add(Measure("qe0", acq_protocol="TriggerCount", bin_mode=BinMode.APPEND))
    schedule.add(Measure("qe1", acq_protocol="TriggerCount", bin_mode=BinMode.AVERAGE))

    # Generate compiled schedule
    compiler = SerialCompiler(name="compiler")
    compiled_sched = compiler.compile(
        schedule=schedule, config=quantum_device.generate_compilation_config()
    )

    # Assert intended behaviour
    seq_0 = compiled_sched.compiled_instructions["cluster0"]["cluster0_module3"][
        "sequencers"
    ]["seq0"]
    seq_1 = compiled_sched.compiled_instructions["cluster0"]["cluster0_module3"][
        "sequencers"
    ]["seq1"]
    seq_2 = compiled_sched.compiled_instructions["cluster0"]["cluster0_module3"][
        "sequencers"
    ]["seq2"]
    seq_3 = compiled_sched.compiled_instructions["cluster0"]["cluster0_module3"][
        "sequencers"
    ]["seq3"]

    assert seq_0["connected_input_indices"] == [0]
    assert seq_0["ttl_acq_auto_bin_incr_en"] is False
    assert seq_1["connected_input_indices"] == [1]
    assert seq_1["ttl_acq_auto_bin_incr_en"] is True
    assert seq_2["connected_output_indices"] == [0]
    assert seq_2["nco_en"] is True
    assert seq_3["connected_output_indices"] == [1]
    assert seq_3["nco_en"] is True

    instr_coordinator.remove_component("ic_cluster0")


@pytest.mark.parametrize(
    "module_under_test",
    [ClusterType.CLUSTER_QRM_RF, ClusterType.CLUSTER_QRM],
)
def test_trace_acquisition_instrument_coordinator(
    mocker,
    mock_setup_basic_transmon_with_standard_params,
    make_cluster_component,
    module_under_test,
):
    hardware_cfgs = {}
    hardware_cfgs[ClusterType.CLUSTER_QRM_RF] = {
        "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
        "hardware_description": {
            "cluster0": {
                "instrument_type": "Cluster",
                "modules": {"4": {"instrument_type": "QRM_RF"}},
                "ref": "internal",
            }
        },
        "hardware_options": {
            "modulation_frequencies": {"q2:res-q2.ro": {"interm_freq": 50000000.0}}
        },
        "connectivity": {"graph": [["cluster0.module4.complex_output_0", "q2:res"]]},
    }

    hardware_cfgs[ClusterType.CLUSTER_QRM] = {
        "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
        "hardware_description": {
            "cluster0": {
                "instrument_type": "Cluster",
                "modules": {"3": {"instrument_type": "QRM"}},
                "ref": "internal",
            }
        },
        "hardware_options": {},
        "connectivity": {"graph": [["cluster0.module3.complex_output_0", "q2:res"]]},
    }

    hardware_cfg = hardware_cfgs[module_under_test]

    mock_setup = mock_setup_basic_transmon_with_standard_params
    instr_coordinator = mock_setup["instrument_coordinator"]

    name = "cluster0"

    try:
        ic_component = make_cluster_component(name)
    except KeyError:
        close_instruments([name])

    hardware_cfg_module_names = set(
        f"{name}_module{idx}"
        for idx in hardware_cfg["hardware_description"]["cluster0"]["modules"]
    )
    module_name = hardware_cfg_module_names.intersection(
        ic_component._cluster_modules
    ).pop()

    try:
        instr_coordinator.add_component(ic_component)
    except ValueError:
        ic_component.instrument.reset()

    quantum_device = mock_setup["quantum_device"]
    quantum_device.hardware_config(hardware_cfg)

    q2 = mock_setup["q2"]
    q2.measure.acq_delay(600e-9)
    q2.clock_freqs.readout(
        7.404e9 if module_under_test is ClusterType.CLUSTER_QRM_RF else 3e8
    )

    schedule = trace_schedule_circuit_layer(qubit_name="q2")

    # Generate compiled schedule
    compiler = SerialCompiler(name="compiler")
    compiled_sched = compiler.compile(
        schedule=schedule, config=quantum_device.generate_compilation_config()
    )

    module = (
        ic_component._cluster_modules[module_name]
        if isinstance(module_under_test, ClusterType)
        else ic_component
    )

    # Setup dummy acquisition data
    dummy_scope_acquisition_data = DummyScopeAcquisitionData(
        data=[(0, 1)] * 15000, out_of_range=(False, False), avg_cnt=(0, 0)
    )
    module.instrument.set_dummy_scope_acquisition_data(
        sequencer=None, data=dummy_scope_acquisition_data
    )

    wrapped = _ModuleComponentBase._set_parameter
    called_with = None

    def wrapper(*args, **kwargs):
        nonlocal called_with
        if "scope_acq_sequencer_select" in args + tuple(kwargs.values()):
            called_with = args + tuple(kwargs.values())
        wrapped(module, *args, **kwargs)

    with mocker.patch(
        "quantify_scheduler.instrument_coordinator.components.qblox."
        "_ModuleComponentBase._set_parameter",
        wraps=wrapper,
    ):
        try:
            instr_coordinator.prepare(compiled_sched)
        except:
            pprint.pprint(compiled_sched.compiled_instructions)
            raise

    assert called_with == (module.instrument, "scope_acq_sequencer_select", 0)

    instr_coordinator.start()
    acquired_data = instr_coordinator.retrieve_acquisition()
    instr_coordinator.stop()

    module.instrument.store_scope_acquisition.assert_called_with(0, "0")

    assert isinstance(acquired_data, Dataset)
    expected_dataarray = DataArray(
        [[1j] * 1000],
        coords=[[0], range(1000)],
        dims=["acq_index_0", "trace_index_0"],
        attrs={"acq_protocol": "Trace"},
    )
    expected_dataset = Dataset({0: expected_dataarray})
    xr.testing.assert_identical(acquired_data, expected_dataset)
    instr_coordinator.remove_component(ic_component.name)


def test_mix_lo_flag(
    mock_setup_basic_transmon_with_standard_params, make_cluster_component
):
    hardware_cfg = {
        "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
        "hardware_description": {
            "cluster0": {
                "instrument_type": "Cluster",
                "modules": {
                    "1": {
                        "instrument_type": "QCM",
                        "complex_output_0": {"mix_lo": True},
                    }
                },
                "ref": "internal",
            },
            "iq_mixer_lo0": {"instrument_type": "IQMixer"},
            "lo0": {"instrument_type": "LocalOscillator", "power": 1},
        },
        "hardware_options": {
            "modulation_frequencies": {
                "q0:res-q0.ro": {"lo_freq": None, "interm_freq": 50000000.0}
            }
        },
        "connectivity": {
            "graph": [
                ["cluster0.module1.complex_output_0", "iq_mixer_lo0.if"],
                ["lo0.output", "iq_mixer_lo0.lo"],
                ["iq_mixer_lo0.rf", "q0:res"],
            ]
        },
    }

    # Setup objects needed for experiment
    mock_setup = mock_setup_basic_transmon_with_standard_params
    ic_cluster0 = make_cluster_component("cluster0")
    instr_coordinator = mock_setup["instrument_coordinator"]
    instr_coordinator.add_component(ic_cluster0)
    quantum_device = mock_setup["quantum_device"]
    quantum_device.hardware_config(hardware_cfg)

    # Define experiment schedule
    schedule = Schedule("test mix_lo flag")
    schedule.add(SquarePulse(amp=0.2, duration=1e-6, port="q0:res", clock="q0.ro"))

    # Generate compiled schedule where mix_lo is true
    compiler = SerialCompiler(name="compiler")
    compiled_sched_mix_lo_true = compiler.compile(
        schedule=schedule, config=quantum_device.generate_compilation_config()
    )

    # Change mix_lo to false, set new LO freq and generate new compiled schedule
    hardware_cfg["hardware_description"]["cluster0"]["modules"]["1"][
        "complex_output_0"
    ]["mix_lo"] = False
    compiled_sched_mix_lo_false = compiler.compile(
        schedule=schedule, config=quantum_device.generate_compilation_config()
    )

    # Assert LO freq got set if mix_lo is true.
    assert (
        compiled_sched_mix_lo_true.compiled_instructions["generic"]["lo0.frequency"]
        == 7.95e9
    )
    # Assert LO freq got set if mix_lo is false.
    assert (
        compiled_sched_mix_lo_false.compiled_instructions["generic"]["lo0.frequency"]
        == 8e9
    )
    # Assert NCO freq got set if mix_lo is false.
    assert (
        compiled_sched_mix_lo_false.compiled_instructions["cluster0"][
            "cluster0_module1"
        ]["sequencers"]["seq0"]["modulation_freq"]
        == 50e6
    )
    instr_coordinator.remove_component("ic_cluster0")


def test_marker_debug_mode_enable(
    mock_setup_basic_transmon_with_standard_params, make_cluster_component
):
    hardware_cfg = {
        "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
        "hardware_description": {
            "cluster0": {
                "instrument_type": "Cluster",
                "modules": {
                    "1": {
                        "instrument_type": "QRM",
                        "complex_input_0": {"marker_debug_mode_enable": True},
                    }
                },
                "ref": "internal",
            }
        },
        "hardware_options": {
            "modulation_frequencies": {"q0:res-q0.ro": {"interm_freq": 0}}
        },
        "connectivity": {"graph": [["cluster0.module1.complex_input_0", "q0:res"]]},
    }

    # Setup objects needed for experiment
    mock_setup = mock_setup_basic_transmon_with_standard_params
    ic_cluster0 = make_cluster_component("cluster0")
    instr_coordinator = mock_setup["instrument_coordinator"]
    instr_coordinator.add_component(ic_cluster0)
    quantum_device = mock_setup["quantum_device"]
    quantum_device.hardware_config(hardware_cfg)

    # Define experiment schedule
    schedule = Schedule("test marker_enable")
    schedule.add(Measure("q0", acq_protocol="SSBIntegrationComplex"))
    schedule.add_resource(ClockResource(name="q0.res", freq=50e6))

    # Generate compiled schedule for QRM
    compiler = SerialCompiler(name="compiler")
    compiled_sched_qrm = compiler.compile(
        schedule=schedule, config=quantum_device.generate_compilation_config()
    )

    # Generate compiled schedule for QRM-RF
    hardware_cfg["hardware_description"]["cluster0"]["modules"]["1"][
        "instrument_type"
    ] = "QRM_RF"
    compiled_sched_qrm_rf = compiler.compile(
        schedule=schedule, config=quantum_device.generate_compilation_config()
    )

    # Assert markers were set correctly, and wait time is correct for both modules.
    seq0_qrm = compiled_sched_qrm.compiled_instructions["cluster0"]["cluster0_module1"][
        "sequencers"
    ]["seq0"]["sequence"]["program"].splitlines()
    idx = 0
    for i, string in enumerate(seq0_qrm):
        if re.search(r"^\s*set_mrk\s+3", string):
            idx = i
            break
    # 2 instructions after set mrk are set_awg_gain and play
    assert re.search(r"^\s*set_mrk\s+0", seq0_qrm[idx + 3])
    assert re.search(r"^\s*upd_param\s+4", seq0_qrm[idx + 4])
    assert re.search(r"^\s*wait\s+92", seq0_qrm[idx + 5])

    seq0_qrm_rf = compiled_sched_qrm_rf.compiled_instructions["cluster0"][
        "cluster0_module1"
    ]["sequencers"]["seq0"]["sequence"]["program"].splitlines()
    idx = 0
    for i, string in enumerate(seq0_qrm_rf):
        if re.search(r"^\s*set_mrk\s+7", string):
            idx = i
            break
    # 2 instructions after set mrk are set_awg_gain and play
    assert re.search(r"^\s*set_mrk\s+3", seq0_qrm_rf[idx + 3])
    assert re.search(r"^\s*upd_param\s+4", seq0_qrm_rf[idx + 4])
    assert re.search(r"^\s*wait\s+92", seq0_qrm_rf[idx + 5])

    instr_coordinator.remove_component("ic_cluster0")


def test_multiple_binned_measurements(
    mock_setup_basic_transmon, make_cluster_component
):
    hardware_cfg = {
        "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
        "hardware_description": {
            "cluster0": {
                "instrument_type": "Cluster",
                "modules": {
                    "3": {"instrument_type": "QRM"},
                    "4": {"instrument_type": "QRM_RF"},
                },
                "ref": "internal",
            }
        },
        "hardware_options": {
            "modulation_frequencies": {
                "q0:res-q0.ro": {"interm_freq": 50000000.0},
                "q1:res-q1.ro": {"interm_freq": 50000000.0},
            }
        },
        "connectivity": {
            "graph": [
                ["cluster0.module3.complex_output_0", "q0:res"],
                ["cluster0.module4.complex_output_0", "q1:res"],
            ]
        },
    }

    # Setup objects needed for experiment
    mock_setup = mock_setup_basic_transmon
    ic_cluster0 = make_cluster_component("cluster0")

    instr_coordinator = mock_setup["instrument_coordinator"]
    instr_coordinator.add_component(ic_cluster0)

    quantum_device = mock_setup["quantum_device"]
    quantum_device.hardware_config(hardware_cfg)

    q0 = mock_setup["q0"]
    q1 = mock_setup["q1"]
    q0.clock_freqs.readout(50e6)
    q1.clock_freqs.readout(50e6)

    # Define experiment schedule
    schedule = Schedule("test multiple measurements")
    schedule.add(Measure("q0", acq_index=0, acq_protocol="SSBIntegrationComplex"))
    schedule.add(Measure("q0", acq_index=1, acq_protocol="SSBIntegrationComplex"))
    schedule.add(
        SSBIntegrationComplex(
            port="q0:res", clock="q0.ro", duration=5e-6, acq_channel=0, acq_index=2
        )
    )
    schedule.add(
        SSBIntegrationComplex(
            port="q0:res", clock="q0.ro", duration=5e-6, acq_channel=0, acq_index=3
        )
    )
    schedule.add(
        SSBIntegrationComplex(
            port="q0:res", clock="q0.ro", duration=5e-6, acq_channel=2, acq_index=0
        )
    )
    schedule.add(
        SSBIntegrationComplex(
            port="q0:res", clock="q0.ro", duration=5e-6, acq_channel=2, acq_index=1
        )
    )
    schedule.add(
        Measure(
            "q1", acq_channel="ch_1", acq_index=0, acq_protocol="SSBIntegrationComplex"
        )
    )
    schedule.add(
        Measure(
            "q1", acq_channel="ch_1", acq_index=1, acq_protocol="SSBIntegrationComplex"
        )
    )
    schedule.add(
        SSBIntegrationComplex(
            port="q1:res", clock="q1.ro", duration=5e-6, acq_channel="ch_1", acq_index=2
        )
    )
    schedule.add(
        SSBIntegrationComplex(
            port="q1:res", clock="q1.ro", duration=5e-6, acq_channel="ch_1", acq_index=3
        )
    )
    schedule.add(
        SSBIntegrationComplex(
            port="q1:res", clock="q1.ro", duration=5e-6, acq_channel=3, acq_index=0
        )
    )
    schedule.add(
        SSBIntegrationComplex(
            port="q1:res", clock="q1.ro", duration=5e-6, acq_channel=3, acq_index=1
        )
    )

    # Change acq delay, duration and channel
    q0.measure.acq_delay(1e-6)
    q1.measure.acq_delay(1e-6)
    q0.measure.integration_time(5e-6)
    q1.measure.integration_time(5e-6)
    q0.measure.acq_channel(0)
    q1.measure.acq_channel(1)
    q1.clock_freqs.readout(7404000000.0)

    # Setup dummy acquisition data
    ic_cluster0.instrument.set_dummy_binned_acquisition_data(
        slot_idx=3,
        sequencer=0,
        acq_index_name="0",
        data=[
            DummyBinnedAcquisitionData(data=(10000, 15000), thres=0, avg_cnt=0),
            DummyBinnedAcquisitionData(data=(20000, 25000), thres=0, avg_cnt=0),
            DummyBinnedAcquisitionData(data=(30000, 35000), thres=0, avg_cnt=0),
            DummyBinnedAcquisitionData(data=(40000, 45000), thres=0, avg_cnt=0),
        ],
    )
    ic_cluster0.instrument.set_dummy_binned_acquisition_data(
        slot_idx=3,
        sequencer=0,
        acq_index_name="1",
        data=[
            DummyBinnedAcquisitionData(data=(50000, 55000), thres=0, avg_cnt=0),
            DummyBinnedAcquisitionData(data=(60000, 65000), thres=0, avg_cnt=0),
        ],
    )
    ic_cluster0.instrument.set_dummy_binned_acquisition_data(
        slot_idx=4,
        sequencer=0,
        acq_index_name="0",
        data=[
            DummyBinnedAcquisitionData(data=(100000, 150000), thres=0, avg_cnt=0),
            DummyBinnedAcquisitionData(data=(200000, 250000), thres=0, avg_cnt=0),
            DummyBinnedAcquisitionData(data=(300000, 350000), thres=0, avg_cnt=0),
            DummyBinnedAcquisitionData(data=(400000, 450000), thres=0, avg_cnt=0),
        ],
    )
    ic_cluster0.instrument.set_dummy_binned_acquisition_data(
        slot_idx=4,
        sequencer=0,
        acq_index_name="1",
        data=[
            DummyBinnedAcquisitionData(data=(500000, 550000), thres=0, avg_cnt=0),
            DummyBinnedAcquisitionData(data=(600000, 650000), thres=0, avg_cnt=0),
        ],
    )

    # Generate compiled schedule
    compiler = SerialCompiler(name="compiler")
    compiled_sched = compiler.compile(
        schedule=schedule, config=quantum_device.generate_compilation_config()
    )

    # Upload schedule and run experiment
    instr_coordinator.prepare(compiled_sched)
    instr_coordinator.start()
    data = instr_coordinator.retrieve_acquisition()
    instr_coordinator.stop()

    # Assert intended behaviour
    assert isinstance(data, Dataset)
    expected_dataset = Dataset(
        {
            0: DataArray(
                [2 + 3j, 4 + 5j, 6 + 7j, 8 + 9j],
                coords=[[0, 1, 2, 3]],
                dims=["acq_index_0"],
                attrs={"acq_protocol": "SSBIntegrationComplex"},
            ),
            2: DataArray(
                [10 + 11j, 12 + 13j],
                coords=[[0, 1]],
                dims=["acq_index_2"],
                attrs={"acq_protocol": "SSBIntegrationComplex"},
            ),
            "ch_1": DataArray(
                [20 + 30j, 40 + 50j, 60 + 70j, 80 + 90j],
                coords=[[0, 1, 2, 3]],
                dims=["acq_index_ch_1"],
                attrs={"acq_protocol": "SSBIntegrationComplex"},
            ),
            3: DataArray(
                [100 + 110j, 120 + 130j],
                coords=[[0, 1]],
                dims=["acq_index_3"],
                attrs={"acq_protocol": "SSBIntegrationComplex"},
            ),
        }
    )

    xr.testing.assert_identical(data, expected_dataset)

    instr_coordinator.remove_component("ic_cluster0")


def test_append_measurements(mock_setup_basic_transmon, make_cluster_component):
    hardware_cfg = {
        "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
        "hardware_description": {
            "cluster0": {
                "instrument_type": "Cluster",
                "modules": {"3": {"instrument_type": "QRM"}},
                "ref": "internal",
            }
        },
        "hardware_options": {
            "modulation_frequencies": {"q0:res-q0.ro": {"interm_freq": 50000000.0}}
        },
        "connectivity": {"graph": [["cluster0.module3.complex_output_0", "q0:res"]]},
    }

    # Setup objects needed for experiment
    mock_setup = mock_setup_basic_transmon
    ic_cluster0 = make_cluster_component("cluster0")

    instr_coordinator = mock_setup["instrument_coordinator"]
    instr_coordinator.add_component(ic_cluster0)

    quantum_device = mock_setup["quantum_device"]
    quantum_device.hardware_config(hardware_cfg)

    q0 = mock_setup["q0"]
    q0.clock_freqs.readout(50e6)

    # Define experiment schedule
    schedule = Schedule("test multiple measurements", repetitions=3)
    schedule.add(
        Measure(
            "q0",
            acq_index=0,
            acq_protocol="SSBIntegrationComplex",
            bin_mode=BinMode.APPEND,
        )
    )
    schedule.add(
        Measure(
            "q0",
            acq_index=1,
            acq_protocol="SSBIntegrationComplex",
            bin_mode=BinMode.APPEND,
        )
    )

    # Change acq delay, duration and channel
    q0.measure.acq_delay(1e-6)
    q0.measure.integration_time(5e-6)
    q0.measure.acq_channel(1)

    # Setup dummy acquisition data
    ic_cluster0.instrument.set_dummy_binned_acquisition_data(
        slot_idx=3,
        sequencer=0,
        acq_index_name="0",
        data=[
            DummyBinnedAcquisitionData(data=(10000, 15000), thres=0, avg_cnt=0),
            DummyBinnedAcquisitionData(data=(20000, 25000), thres=0, avg_cnt=0),
            DummyBinnedAcquisitionData(data=(30000, 35000), thres=0, avg_cnt=0),
            DummyBinnedAcquisitionData(data=(40000, 45000), thres=0, avg_cnt=0),
            DummyBinnedAcquisitionData(data=(50000, 55000), thres=0, avg_cnt=0),
            DummyBinnedAcquisitionData(data=(60000, 65000), thres=0, avg_cnt=0),
        ],
    )

    # Generate compiled schedule
    compiler = SerialCompiler(name="compiler")
    compiled_sched = compiler.compile(
        schedule=schedule, config=quantum_device.generate_compilation_config()
    )

    # Upload schedule and run experiment
    instr_coordinator.prepare(compiled_sched)
    instr_coordinator.start()
    data = instr_coordinator.retrieve_acquisition()
    instr_coordinator.stop()

    # Assert intended behaviour
    assert isinstance(data, Dataset)
    expected_dataset = Dataset(
        {
            1: DataArray(
                [[2 + 3j, 4 + 5j], [6 + 7j, 8 + 9j], [10 + 11j, 12 + 13j]],
                coords={"acq_index_1": [0, 1]},
                dims=["repetition", "acq_index_1"],
                attrs={"acq_protocol": "SSBIntegrationComplex"},
            ),
        }
    )

    xr.testing.assert_identical(data, expected_dataset)

    instr_coordinator.remove_component("ic_cluster0")


def test_looped_measurements(mock_setup_basic_transmon, make_cluster_component):
    hardware_cfg = {
        "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
        "hardware_description": {
            "cluster0": {
                "instrument_type": "Cluster",
                "ref": "internal",
                "modules": {
                    "3": {"instrument_type": "QRM"},
                },
            }
        },
        "hardware_options": {
            "modulation_frequencies": {
                "q0:res-q0.ro": {
                    "interm_freq": 50e6,
                }
            }
        },
        "connectivity": {
            "graph": [
                ["cluster0.module3.complex_output_0", "q0:res"],
            ]
        },
    }

    # Setup objects needed for experiment
    mock_setup = mock_setup_basic_transmon
    ic_cluster0 = make_cluster_component("cluster0")

    instr_coordinator = mock_setup["instrument_coordinator"]
    instr_coordinator.add_component(ic_cluster0)

    quantum_device = mock_setup["quantum_device"]
    quantum_device.hardware_config(hardware_cfg)

    q0 = mock_setup["q0"]
    q0.clock_freqs.readout(50e6)

    # Define experiment schedule
    schedule = Schedule("test multiple measurements", repetitions=2)
    schedule.add(
        Measure(
            "q0",
            acq_index=0,
            acq_protocol="SSBIntegrationComplex",
            bin_mode=BinMode.APPEND,
        ),
        control_flow=Loop(3),
    )

    # Change acq delay, duration and channel
    q0.measure.acq_delay(1e-6)
    q0.measure.integration_time(5e-6)
    q0.measure.acq_channel(0)

    # Setup dummy acquisition data
    ic_cluster0.instrument.set_dummy_binned_acquisition_data(
        slot_idx=3,
        sequencer=0,
        acq_index_name="0",
        data=[
            DummyBinnedAcquisitionData(data=(10000, 15000), thres=0, avg_cnt=0),
            DummyBinnedAcquisitionData(data=(20000, 25000), thres=0, avg_cnt=0),
            DummyBinnedAcquisitionData(data=(30000, 35000), thres=0, avg_cnt=0),
            DummyBinnedAcquisitionData(data=(40000, 45000), thres=0, avg_cnt=0),
            DummyBinnedAcquisitionData(data=(50000, 55000), thres=0, avg_cnt=0),
            DummyBinnedAcquisitionData(data=(60000, 65000), thres=0, avg_cnt=0),
        ],
    )

    # Generate compiled schedule
    compiler = SerialCompiler(name="compiler")
    compiled_sched = compiler.compile(
        schedule=schedule, config=quantum_device.generate_compilation_config()
    )

    # Upload schedule and run experiment
    instr_coordinator.prepare(compiled_sched)
    instr_coordinator.start()

    with pytest.warns(
        FutureWarning,
        match="The format of acquisition data of looped measurements in APPEND mode will change in quantify-scheduler>=0.18.0",
    ):
        data = instr_coordinator.retrieve_acquisition()

    instr_coordinator.stop()

    # Assert intended behaviour
    assert isinstance(data, Dataset)
    expected_dataset = Dataset(
        {
            0: DataArray(
                [[2 + 3j, 4 + 5j, 6 + 7j], [8 + 9j, 10 + 11j, 12 + 13j]],
                coords=None,
                dims=["repetition", "loop_repetition"],
                attrs={"acq_protocol": "SSBIntegrationComplex"},
            ),
        }
    )

    xr.testing.assert_identical(data, expected_dataset)

    instr_coordinator.remove_component("ic_cluster0")
