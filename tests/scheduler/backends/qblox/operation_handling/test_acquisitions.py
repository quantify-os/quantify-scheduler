# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=missing-module-docstring
# pylint: disable=no-name-in-module
# pylint: disable=no-self-use
# pylint: disable=redefined-outer-name
# pylint: disable=unused-argument

# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Tests for acquisitions module."""
import pprint
from typing import Dict, Any

import pytest
import numpy as np
from qcodes.instrument.parameter import ManualParameter
from qblox_instruments import ClusterType, PulsarType

from quantify_scheduler import waveforms

from quantify_scheduler.enums import BinMode

from quantify_scheduler.backends.qblox import constants
from quantify_scheduler.backends.qblox.instrument_compilers import QrmModule
from quantify_scheduler.backends.qblox.operation_handling import acquisitions
from quantify_scheduler.backends.qblox.qasm_program import QASMProgram
from quantify_scheduler.backends.qblox.register_manager import RegisterManager
from quantify_scheduler.backends.types import qblox as types
from quantify_scheduler.compilation import qcompile
from quantify_scheduler.gettables import ScheduleGettable
from quantify_scheduler.instrument_coordinator.components.qblox import (
    QbloxInstrumentCoordinatorComponentBase,
)
from quantify_scheduler.schedules.trace_schedules import trace_schedule_circuit_layer

from tests.fixtures.mock_setup import close_instruments
from tests.scheduler.instrument_coordinator.components.test_qblox import (  # pylint: disable=unused-import
    make_cluster_component,
    make_qrm_component,
)


@pytest.fixture(name="empty_qasm_program_qrm")
def fixture_empty_qasm_program():
    yield QASMProgram(QrmModule.static_hw_properties, RegisterManager())


class MockAcquisition(acquisitions.AcquisitionStrategyPartial):
    """Used for TestAcquisitionStrategyPartial."""

    def generate_data(self, wf_dict: Dict[str, Any]):
        pass

    def acquire_append(self, qasm_program: QASMProgram):
        pass

    def acquire_average(self, qasm_program: QASMProgram):
        pass


class TestAcquisitionStrategyPartial:
    """
    There is some logic in the AcquisitionStrategyPartial class that deserves
    testing.
    """

    def test_operation_info_property(self):
        # arrange
        data = {"bin_mode": BinMode.AVERAGE, "acq_channel": 0, "acq_index": 0}
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
        append_mock = mocker.patch.object(strategy, "acquire_append")
        average_mock = mocker.patch.object(strategy, "acquire_average")
        # pylint: disable=attribute-defined-outside-init
        # what pylint claims here is simply not true
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
            "1000 ns between acquisitions.\n\nError caused by acquisition:\n"
            "Acquisition  (t=0 to 1e-06)\ndata={'bin_mode': 'nonsense', "
            "'acq_channel': 0, 'acq_index': 0, 'duration': 1e-06}."
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
        data = {"bin_mode": BinMode.AVERAGE, "acq_channel": 0, "acq_index": 0}
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
            "bin_mode": BinMode.AVERAGE,
            "acq_channel": 0,
            "acq_index": 0,
            "duration": 1e-6,
        }
        strategy = acquisitions.SquareAcquisitionStrategy(
            types.OpInfo(name="", data=data, timing=0)
        )
        strategy.generate_data({})

        # act
        strategy.acquire_average(qasm)

        # assert
        assert qasm.instructions == [["", "acquire", "0,0,4", ""]]

    def test_acquire_append(self, empty_qasm_program_qrm):
        # arrange
        qasm = empty_qasm_program_qrm
        data = {
            "bin_mode": BinMode.APPEND,
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
        strategy.acquire_append(qasm)

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
            "bin_mode": BinMode.AVERAGE,
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
            "bin_mode": BinMode.AVERAGE,
            "acq_channel": 2,
            "acq_index": 12,
            "waveforms": weights,
        }
        strategy = acquisitions.WeightedAcquisitionStrategy(
            types.OpInfo(name="", data=data, timing=0)
        )
        strategy.generate_data({})

        # act
        strategy.acquire_average(qasm)

        # assert
        assert qasm.instructions == [
            [
                "",
                "acquire_weighed",
                "2,12,0,1,4",
                "# Store acq in acq_channel:2, bin_idx:12",
            ]
        ]

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
            "bin_mode": BinMode.AVERAGE,
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
        strategy.acquire_append(qasm)

        # assert
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
            ["", "", "", ""],
        ]


def test_trace_acquisition_measurement_control(
    mock_setup_basic_transmon, mocker, make_cluster_component
):
    hardware_cfg = {
        "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
        "cluster0": {
            "ref": "internal",
            "instrument_type": "Cluster",
            "cluster0_module4": {
                "instrument_type": "QRM_RF",
                "complex_output_0": {
                    "portclock_configs": [
                        {"port": "q2:res", "clock": "q2.ro", "interm_freq": 50e6},
                    ],
                },
            },
        },
    }

    ic_cluster0 = make_cluster_component("cluster0")
    instr_coordinator = mock_setup_basic_transmon["instrument_coordinator"]
    instr_coordinator.add_component(ic_cluster0)

    quantum_device = mock_setup_basic_transmon["quantum_device"]
    quantum_device.hardware_config(hardware_cfg)

    q2 = mock_setup_basic_transmon["q2"]
    q2.measure.acq_delay(600e-9)
    q2.clock_freqs.readout(7404000000.0)

    sample_param = ManualParameter("sample", label="Sample time", unit="s")
    sample_param.batched = True

    sample_size = 16384  # Trace acquisition will always return 16384 samples
    sampling_rate = constants.SAMPLING_RATE
    sample_times = np.arange(
        start=0, stop=sample_size / sampling_rate, step=1 / sampling_rate
    )

    sched_gettable = ScheduleGettable(
        quantum_device=quantum_device,
        schedule_function=trace_schedule_circuit_layer,
        schedule_kwargs=dict(qubit_name=q2.name),
        batched=True,
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

    assert dataset.sizes == {"dim_0": sample_size}

    instr_coordinator.remove_component(ic_cluster0.name)


@pytest.mark.parametrize(
    "module_under_test",
    [ClusterType.CLUSTER_QRM_RF, ClusterType.CLUSTER_QRM, PulsarType.PULSAR_QRM],
)
def test_trace_acquisition_instrument_coordinator(  # pylint: disable=too-many-locals
    mocker,
    mock_setup_basic_transmon,
    make_cluster_component,
    make_qrm_component,
    module_under_test,
):
    hardware_cfgs = {}
    hardware_cfgs[ClusterType.CLUSTER_QRM_RF] = {
        "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
        "cluster0": {
            "ref": "internal",
            "instrument_type": "Cluster",
            "cluster0_module4": {
                "instrument_type": "QRM_RF",
                "complex_output_0": {
                    "portclock_configs": [
                        {"port": "q2:res", "clock": "q2.ro", "interm_freq": 50e6}
                    ],
                },
            },
        },
    }
    hardware_cfgs[ClusterType.CLUSTER_QRM] = {
        "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
        "cluster0": {
            "ref": "internal",
            "instrument_type": "Cluster",
            "cluster0_module3": {
                "instrument_type": "QRM",
                "complex_output_0": {
                    "portclock_configs": [{"port": "q2:res", "clock": "q2.ro"}],
                },
            },
        },
    }
    hardware_cfgs[PulsarType.PULSAR_QRM] = {
        "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
        "qrm0": {
            "instrument_type": "Pulsar_QRM",
            "ref": "internal",
            "complex_output_0": {
                "portclock_configs": [{"port": "q2:res", "clock": "q2.ro"}],
            },
        },
    }
    hardware_cfg = hardware_cfgs[module_under_test]

    instr_coordinator = mock_setup_basic_transmon["instrument_coordinator"]

    if isinstance(module_under_test, ClusterType):
        name = "cluster0"

        try:
            ic_component = make_cluster_component(name)
        except KeyError:
            close_instruments([name])

        module_name = (
            set(hardware_cfg[name].keys())
            .intersection(ic_component._cluster_modules)
            .pop()
        )
    else:
        ic_component = make_qrm_component("qrm0")
        instr_coordinator.add_component(ic_component)

    try:
        instr_coordinator.add_component(ic_component)
    except ValueError:
        ic_component.instrument.reset()

    quantum_device = mock_setup_basic_transmon["quantum_device"]
    quantum_device.hardware_config(hardware_cfg)

    q2 = mock_setup_basic_transmon["q2"]
    q2.measure.acq_delay(600e-9)
    q2.clock_freqs.readout(
        7.404e9 if module_under_test is ClusterType.CLUSTER_QRM_RF else 3e8
    )

    schedule = trace_schedule_circuit_layer(qubit_name="q2")

    compiled_sched = qcompile(
        schedule=schedule,
        device_cfg=quantum_device.generate_device_config(),
        hardware_cfg=hardware_cfg,
    )

    wrappee = (
        ic_component._cluster_modules[module_name]
        if isinstance(module_under_test, ClusterType)
        else ic_component
    )
    wrapped = QbloxInstrumentCoordinatorComponentBase._set_parameter
    called_with = None

    def wrapper(*args, **kwargs):
        nonlocal called_with
        if "scope_acq_sequencer_select" in args + tuple(kwargs.values()):
            called_with = args + tuple(kwargs.values())
        wrapped(wrappee, *args, **kwargs)

    with mocker.patch(
        "quantify_scheduler.instrument_coordinator.components.qblox."
        "QbloxInstrumentCoordinatorComponentBase._set_parameter",
        wraps=wrapper,
    ):
        try:
            instr_coordinator.prepare(compiled_sched)
        except:
            pprint.pprint(compiled_sched.compiled_instructions)
            raise

    assert called_with == (wrappee.instrument, "scope_acq_sequencer_select", 0)

    instr_coordinator.start()
    acquired_data = instr_coordinator.retrieve_acquisition()
    instr_coordinator.stop()

    acquired_data = list(acquired_data.values())[0]
    assert isinstance(acquired_data, tuple)
    assert tuple(map(type, acquired_data)) == (np.ndarray, np.ndarray)

    instr_coordinator.remove_component(ic_component.name)
