# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch

from typing import List

import pytest

from quantify_scheduler.backends import SerialCompiler
from quantify_scheduler.enums import BinMode
from quantify_scheduler.helpers.generate_acq_channels_data import (
    generate_acq_channels_data,
)
from quantify_scheduler.operations.acquisition_library import (
    NumericalSeparatedWeightedIntegration,
    NumericalWeightedIntegration,
    SSBIntegrationComplex,
    ThresholdedAcquisition,
    Trace,
    TriggerCount,
    WeightedIntegratedSeparated,
)
from quantify_scheduler.operations.control_flow_library import LoopOperation
from quantify_scheduler.schedules.schedule import (
    AcquisitionChannelData,
    Schedulable,
    Schedule,
)


@pytest.mark.parametrize(
    "protocol,protocol_str,protocol_opt_args",
    [
        (
            SSBIntegrationComplex,
            "SSBIntegrationComplex",
            {"duration": 1e-6},
        ),
        (
            ThresholdedAcquisition,
            "ThresholdedAcquisition",
            {"duration": 1e-6},
        ),
        (
            WeightedIntegratedSeparated,
            "WeightedIntegratedSeparated",
            {"duration": 1e-6, "waveform_a": [0.5], "waveform_b": [0.5]},
        ),
        (
            NumericalSeparatedWeightedIntegration,
            "NumericalSeparatedWeightedIntegration",
            {"weights_a": [0.5], "weights_b": [0.5]},
        ),
        (
            NumericalWeightedIntegration,
            "NumericalWeightedIntegration",
            {"weights_a": [0.5], "weights_b": [0.5]},
        ),
    ],
)
def test_binned_average(
    mock_setup_basic_transmon_with_standard_params,
    protocol,
    protocol_str,
    protocol_opt_args,
):
    quantum_device = mock_setup_basic_transmon_with_standard_params["quantum_device"]

    schedule = Schedule("Test schedule", repetitions=2)
    schedulables: List[Schedulable] = []

    schedulables.append(
        schedule.add(
            protocol(
                acq_channel=0,
                acq_index=0,
                bin_mode=BinMode.AVERAGE,
                port="q0:res",
                clock="q0.ro",
                **protocol_opt_args,
            )
        )
    )
    schedulables.append(
        schedule.add(
            protocol(
                acq_channel=0,
                acq_index=1,
                bin_mode=BinMode.AVERAGE,
                port="q0:res",
                clock="q0.ro",
                **protocol_opt_args,
            )
        )
    )
    schedulables.append(
        schedule.add(
            protocol(
                acq_channel=1,
                acq_index=0,
                bin_mode=BinMode.AVERAGE,
                port="q0:res",
                clock="q0.ro",
                **protocol_opt_args,
            )
        )
    )
    schedulables.append(
        schedule.add(
            protocol(
                acq_channel=1,
                acq_index=1,
                bin_mode=BinMode.AVERAGE,
                port="q0:res",
                clock="q0.ro",
                **protocol_opt_args,
            )
        )
    )
    schedulables.append(
        schedule.add(
            protocol(
                acq_channel=1,
                acq_index=2,
                bin_mode=BinMode.AVERAGE,
                port="q0:res",
                clock="q0.ro",
                **protocol_opt_args,
            )
        )
    )
    schedulables.append(
        schedule.add(
            protocol(
                acq_channel=1,
                acq_index=3,
                bin_mode=BinMode.AVERAGE,
                port="q0:res",
                clock="q0.ro",
                **protocol_opt_args,
            )
        )
    )
    schedulables.append(
        schedule.add(
            protocol(
                acq_channel=2,
                acq_index=0,
                bin_mode=BinMode.AVERAGE,
                port="q0:res",
                clock="q0.ro",
                **protocol_opt_args,
            )
        )
    )

    compiler = SerialCompiler("test")
    partially_compiled_sched = compiler.compile(
        schedule=schedule, config=quantum_device.generate_compilation_config()
    )

    acq_channels_data, schedulable_label_to_acq_index = generate_acq_channels_data(
        partially_compiled_sched
    )

    expected_acq_channels_data = {
        0: AcquisitionChannelData(
            acq_index_dim_name="acq_index_0",
            protocol=protocol_str,
            bin_mode=BinMode.AVERAGE,
            coords=[{}, {}],
        ),
        1: AcquisitionChannelData(
            acq_index_dim_name="acq_index_1",
            protocol=protocol_str,
            bin_mode=BinMode.AVERAGE,
            coords=[{}, {}, {}, {}],
        ),
        2: AcquisitionChannelData(
            acq_index_dim_name="acq_index_2",
            protocol=protocol_str,
            bin_mode=BinMode.AVERAGE,
            coords=[{}],
        ),
    }

    expected_schedulable_label_to_acq_index = {
        ((schedulables[0]["name"],), 0): 0,
        ((schedulables[1]["name"],), 0): 1,
        ((schedulables[2]["name"],), 0): 0,
        ((schedulables[3]["name"],), 0): 1,
        ((schedulables[4]["name"],), 0): 2,
        ((schedulables[5]["name"],), 0): 3,
        ((schedulables[6]["name"],), 0): 0,
    }

    assert expected_acq_channels_data == acq_channels_data
    assert expected_schedulable_label_to_acq_index == schedulable_label_to_acq_index


@pytest.mark.parametrize(
    "protocol,protocol_str,protocol_opt_args",
    [
        (
            SSBIntegrationComplex,
            "SSBIntegrationComplex",
            {"duration": 1e-6},
        ),
        (
            ThresholdedAcquisition,
            "ThresholdedAcquisition",
            {"duration": 1e-6},
        ),
        (
            WeightedIntegratedSeparated,
            "WeightedIntegratedSeparated",
            {"duration": 1e-6, "waveform_a": [0.5], "waveform_b": [0.5]},
        ),
        (
            NumericalSeparatedWeightedIntegration,
            "NumericalSeparatedWeightedIntegration",
            {"weights_a": [0.5], "weights_b": [0.5]},
        ),
        (
            NumericalWeightedIntegration,
            "NumericalWeightedIntegration",
            {"weights_a": [0.5], "weights_b": [0.5]},
        ),
    ],
)
def test_binned_append(
    mock_setup_basic_transmon_with_standard_params,
    protocol,
    protocol_str,
    protocol_opt_args,
):
    quantum_device = mock_setup_basic_transmon_with_standard_params["quantum_device"]

    schedulables: List[Schedulable] = []

    schedule = Schedule("Test schedule", repetitions=2)

    schedulables.append(
        schedule.add(
            protocol(
                acq_channel=0,
                acq_index=0,
                bin_mode=BinMode.APPEND,
                port="q0:res",
                clock="q0.ro",
                **protocol_opt_args,
            )
        )
    )
    schedulables.append(
        schedule.add(
            protocol(
                acq_channel=0,
                acq_index=1,
                bin_mode=BinMode.APPEND,
                port="q0:res",
                clock="q0.ro",
                **protocol_opt_args,
            )
        )
    )
    schedulables.append(
        schedule.add(
            protocol(
                acq_channel=0,
                acq_index=1,
                bin_mode=BinMode.APPEND,
                port="q0:res",
                clock="q0.ro",
                **protocol_opt_args,
            )
        )
    )

    compiler = SerialCompiler("test")
    partially_compiled_sched = compiler.compile(
        schedule=schedule, config=quantum_device.generate_compilation_config()
    )

    acq_channels_data, schedulable_label_to_acq_index = generate_acq_channels_data(
        partially_compiled_sched
    )

    expected_acq_channels_data = {
        0: AcquisitionChannelData(
            acq_index_dim_name="acq_index_0",
            protocol=protocol_str,
            bin_mode=BinMode.APPEND,
            coords=[
                {},
                {},
                {},
            ],
        ),
    }

    expected_schedulable_label_to_acq_index = {
        ((schedulables[0]["name"],), 0): [0],
        ((schedulables[1]["name"],), 0): [1],
        ((schedulables[2]["name"],), 0): [2],
    }

    assert expected_acq_channels_data == acq_channels_data
    assert expected_schedulable_label_to_acq_index == schedulable_label_to_acq_index


@pytest.mark.parametrize(
    "protocol,protocol_str,protocol_opt_args",
    [
        (
            SSBIntegrationComplex,
            "SSBIntegrationComplex",
            {"duration": 1e-6},
        ),
        (
            ThresholdedAcquisition,
            "ThresholdedAcquisition",
            {"duration": 1e-6},
        ),
        (
            WeightedIntegratedSeparated,
            "WeightedIntegratedSeparated",
            {"duration": 1e-6, "waveform_a": [0.5], "waveform_b": [0.5]},
        ),
        (
            NumericalSeparatedWeightedIntegration,
            "NumericalSeparatedWeightedIntegration",
            {"weights_a": [0.5], "weights_b": [0.5]},
        ),
        (
            NumericalWeightedIntegration,
            "NumericalWeightedIntegration",
            {"weights_a": [0.5], "weights_b": [0.5]},
        ),
    ],
)
def test_binned_append_loop(
    mock_setup_basic_transmon_with_standard_params,
    protocol,
    protocol_str,
    protocol_opt_args,
):
    quantum_device = mock_setup_basic_transmon_with_standard_params["quantum_device"]

    schedule = Schedule("Schedule", repetitions=2)

    inner_sched = Schedule("Inner schedule")

    inner_inner_sched = Schedule("Inner inner schedule")
    inner_inner_sched.add(
        protocol(
            acq_channel=0,
            acq_index=0,
            bin_mode=BinMode.APPEND,
            port="q0:res",
            clock="q0.ro",
            **protocol_opt_args,
        )
    )
    inner_sched.add(LoopOperation(inner_inner_sched, repetitions=4))
    schedule.add(LoopOperation(inner_sched, repetitions=3))

    compiler = SerialCompiler("test")
    partially_compiled_sched = compiler.compile(
        schedule=schedule, config=quantum_device.generate_compilation_config()
    )

    acq_channels_data, schedulable_label_to_acq_index = generate_acq_channels_data(
        partially_compiled_sched
    )

    expected_acq_channels_data = {
        0: AcquisitionChannelData(
            acq_index_dim_name="acq_index_0",
            protocol=protocol_str,
            bin_mode=BinMode.APPEND,
            coords=[{"loop_repetition": lr} for lr in range(4 * 3)],
        ),
    }

    def _first_key(d: dict):
        return list(d.keys())[0]

    def _first_val(d: dict):
        return list(d.values())[0]

    outer_schedulable_name = _first_key(partially_compiled_sched.schedulables)
    inner_schedulable_name = _first_key(
        _first_val(partially_compiled_sched.operations).body.schedulables
    )
    inner_inner_schedulable_name = _first_key(
        _first_val(
            _first_val(partially_compiled_sched.operations).body.operations
        ).body.schedulables
    )
    expected_schedulable_label_to_acq_index = {
        (
            (
                outer_schedulable_name,
                None,
                inner_schedulable_name,
                None,
                inner_inner_schedulable_name,
            ),
            0,
        ): list(range(12)),
    }

    assert expected_acq_channels_data == acq_channels_data
    assert expected_schedulable_label_to_acq_index == schedulable_label_to_acq_index


def test_trace_and_binned(mock_setup_basic_transmon_with_standard_params):
    quantum_device = mock_setup_basic_transmon_with_standard_params["quantum_device"]

    schedulables: List[Schedulable] = []

    schedule = Schedule("Test schedule", repetitions=2)

    schedulables.append(
        schedule.add(
            SSBIntegrationComplex(
                acq_channel=0,
                acq_index=0,
                bin_mode=BinMode.AVERAGE,
                port="q0:res",
                clock="q0.ro",
                duration=1e-6,
            )
        )
    )
    schedulables.append(
        schedule.add(
            Trace(
                acq_channel=1,
                acq_index=0,
                bin_mode=BinMode.AVERAGE,
                port="q0:res",
                clock="q0.ro",
                duration=1e-6,
            )
        )
    )

    compiler = SerialCompiler("test")
    partially_compiled_sched = compiler.compile(
        schedule=schedule, config=quantum_device.generate_compilation_config()
    )

    acq_channels_data, schedulable_label_to_acq_index = generate_acq_channels_data(
        partially_compiled_sched
    )

    expected_acq_channels_data = {
        0: AcquisitionChannelData(
            acq_index_dim_name="acq_index_0",
            protocol="SSBIntegrationComplex",
            bin_mode=BinMode.AVERAGE,
            coords=[{}],
        ),
        1: AcquisitionChannelData(
            acq_index_dim_name="acq_index_1",
            protocol="Trace",
            bin_mode=BinMode.AVERAGE,
            coords={},
        ),
    }

    expected_schedulable_label_to_acq_index = {
        ((schedulables[0]["name"],), 0): 0,
    }

    assert expected_acq_channels_data == acq_channels_data
    assert expected_schedulable_label_to_acq_index == schedulable_label_to_acq_index


@pytest.mark.parametrize(
    "bin_mode",
    [
        BinMode.DISTRIBUTION,
        BinMode.APPEND,
    ],
)
def test_trigger_count(mock_setup_basic_transmon_with_standard_params, bin_mode):
    quantum_device = mock_setup_basic_transmon_with_standard_params["quantum_device"]

    schedulables: List[Schedulable] = []

    schedule = Schedule("Test schedule")

    schedulables.append(
        schedule.add(
            TriggerCount(
                acq_channel=1,
                acq_index=0,
                bin_mode=bin_mode,
                port="q0:res",
                clock="q0.ro",
                duration=1e-6,
            )
        )
    )

    schedulables.append(
        schedule.add(
            TriggerCount(
                acq_channel=1,
                acq_index=0,
                bin_mode=bin_mode,
                port="q0:res",
                clock="q0.ro",
                duration=1e-6,
            )
        )
    )

    compiler = SerialCompiler("test")
    partially_compiled_sched = compiler.compile(
        schedule=schedule, config=quantum_device.generate_compilation_config()
    )

    acq_channels_data, schedulable_label_to_acq_index = generate_acq_channels_data(
        partially_compiled_sched
    )

    expected_acq_channels_data = {
        1: AcquisitionChannelData(
            acq_index_dim_name="acq_index_1",
            protocol="TriggerCount",
            bin_mode=bin_mode,
            coords={},
        ),
    }

    expected_schedulable_label_to_acq_index = {}

    assert expected_acq_channels_data == acq_channels_data
    assert expected_schedulable_label_to_acq_index == schedulable_label_to_acq_index
