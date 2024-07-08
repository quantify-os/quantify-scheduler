# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch


from __future__ import annotations

import numpy as np
from pytest import approx

from quantify_scheduler import Schedule
from quantify_scheduler.backends import SerialCompiler
from quantify_scheduler.enums import BinMode
from quantify_scheduler.helpers.collections import make_hash
from quantify_scheduler.helpers.schedule import (
    _extract_port_clocks_used,
    extract_acquisition_metadata_from_schedule,
    get_acq_info_by_uuid,
    get_acq_uuid,
    get_pulse_uuid,
)
from quantify_scheduler.operations.acquisition_library import SSBIntegrationComplex
from quantify_scheduler.operations.gate_library import X90, Measure, Reset
from quantify_scheduler.schedules import spectroscopy_schedules
from quantify_scheduler.schedules.schedule import ScheduleBase


def test_make_hash() -> None:
    my_test_dict = {"a": 5, "nested_dict": {"a": 2, "c": 4, "B": "str"}, "b": 24}

    same_test_dict_diff_order = {
        "a": 5,
        "b": 24,
        "nested_dict": {"a": 2, "c": 4, "B": "str"},
    }

    diff_test_dict = {"nested_dict": {"a": 2, "c": 4, "B": "str"}, "b": 24}

    test_hash = make_hash(my_test_dict)
    same_test_hash = make_hash(same_test_dict_diff_order)

    assert test_hash == same_test_hash

    diff_hash = make_hash(diff_test_dict)

    assert test_hash != diff_hash

    # modify dict in place, the object id won't change
    my_test_dict["efg"] = 15
    new_hash = make_hash(my_test_dict)
    assert test_hash != new_hash


def test_get_acq_info_by_uuid(
    schedule_with_measurement: Schedule,
    device_compile_config_basic_transmon,
):
    # Arrange
    compiler = SerialCompiler(name="compiler")
    schedule = compiler.compile(
        schedule=schedule_with_measurement, config=device_compile_config_basic_transmon
    )

    measure_operation_id = list(schedule.schedulables.values())[-1]["operation_id"]
    measure_operation = schedule.operations[measure_operation_id]

    acq_operation_id = list(measure_operation.schedulables.values())[-1]["operation_id"]
    acq_operation = measure_operation.operations[acq_operation_id]

    acq_info_0 = acq_operation.data["acquisition_info"][0]
    acq_pulse_infos = acq_info_0["waveforms"]

    acq_id = get_acq_uuid(acq_info_0)
    pulse_id0 = get_pulse_uuid(acq_pulse_infos[0])
    pulse_id1 = get_pulse_uuid(acq_pulse_infos[1])

    # Act
    acqid_acqinfo_dict = get_acq_info_by_uuid(schedule)

    # Assert
    assert acq_id in acqid_acqinfo_dict
    assert pulse_id0 not in acqid_acqinfo_dict
    assert pulse_id1 not in acqid_acqinfo_dict

    assert acqid_acqinfo_dict[acq_id] == acq_info_0


def test_schedule_timing_table(mock_setup_basic_transmon_with_standard_params):
    schedule = Schedule("test_schedule_timing_table")
    schedule.add(Reset("q0"))
    schedule.add(X90("q0"))
    schedule.add(Measure("q0"))

    quantum_device = mock_setup_basic_transmon_with_standard_params["quantum_device"]
    compiler = SerialCompiler(name="compiler")
    schedule = compiler.compile(
        schedule=schedule, config=quantum_device.generate_compilation_config()
    )

    q0 = mock_setup_basic_transmon_with_standard_params["q0"]
    X90_duration = q0.rxy.duration()
    measure_acq_delay = q0.measure.acq_delay()
    reset_duration = q0.reset.duration()

    expected_abs_timing = [
        0.0,
        reset_duration,
        reset_duration + X90_duration,
        reset_duration + X90_duration,
        reset_duration + X90_duration + measure_acq_delay,
    ]
    actual_abs_timing = schedule.timing_table.data.abs_time
    assert all(expected_abs_timing == actual_abs_timing)


def test_extract_acquisition_metadata_from_schedule(compiled_two_qubit_t1_schedule):
    comp_t1_sched = compiled_two_qubit_t1_schedule
    acq_metadata = extract_acquisition_metadata_from_schedule(comp_t1_sched)

    assert acq_metadata.acq_protocol == "SSBIntegrationComplex"
    assert acq_metadata.bin_mode == BinMode.AVERAGE
    assert acq_metadata.acq_return_type is complex

    # keys correspond to acquisition channels
    assert set(acq_metadata.acq_channels_metadata.keys()) == {0, 1}
    assert acq_metadata.acq_channels_metadata[0].acq_channel == 0
    assert acq_metadata.acq_channels_metadata[1].acq_channel == 1
    assert acq_metadata.acq_channels_metadata[0].acq_indices == list(np.arange(20))
    assert acq_metadata.acq_channels_metadata[1].acq_indices == list(np.arange(20))


def test_extract_port_clocks_used(create_schedule_with_pulse_info):
    schedule0 = Schedule("my-schedule")
    schedule0.add(X90("q0"))
    schedule0 = create_schedule_with_pulse_info(schedule0)
    assert _extract_port_clocks_used(schedule0) == {("q0:mw", "q0.01")}

    schedule1 = Schedule("my-schedule")
    schedule1.add(
        SSBIntegrationComplex(
            duration=1e-6,
            port="q0:ro",
            clock="q0.res",
            acq_index=0,
            acq_channel=0,
        )
    )
    schedule1 = create_schedule_with_pulse_info(schedule1)
    assert _extract_port_clocks_used(schedule1) == {("q0:ro", "q0.res")}

    schedule2 = Schedule("my-schedule")
    schedule2.add(schedule0)
    schedule2.add(schedule1)
    schedule2 = create_schedule_with_pulse_info(schedule2)
    assert _extract_port_clocks_used(schedule2) == {
        ("q0:mw", "q0.01"),
        ("q0:ro", "q0.res"),
    }
