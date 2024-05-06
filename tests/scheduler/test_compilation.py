from copy import deepcopy

import numpy as np
import pytest
from quantify_scheduler import Operation, Schedule
from quantify_scheduler.backends import SerialCompiler
from quantify_scheduler.backends.circuit_to_device import ConfigKeyError
from quantify_scheduler.compilation import _determine_absolute_timing, flatten_schedule
from quantify_scheduler.enums import BinMode
from quantify_scheduler.operations.control_flow_library import Loop
from quantify_scheduler.operations.gate_library import (
    CNOT,
    CZ,
    X,
    Measure,
    Reset,
    Rxy,
    H,
)
from quantify_scheduler.operations.composite_factories import hadamard_as_y90z
from quantify_scheduler.operations.pulse_library import SquarePulse, SetClockFrequency
from quantify_scheduler.resources import BasebandClockResource, ClockResource, Resource


def test_determine_absolute_timing_ideal_clock():
    sched = Schedule("Test experiment")

    # define the resources
    # q0, q1 = Qubits(n=2) # assumes all to all connectivity
    q0, q1 = ("q0", "q1")

    ref_label_1 = "my_label"

    sched.add(Reset(q0, q1))
    sched.add(Rxy(90, 0, qubit=q0), label=ref_label_1)
    sched.add(operation=CNOT(qC=q0, qT=q1))
    sched.add(Rxy(theta=90, phi=0, qubit=q0))
    sched.add(Measure(q0, q1), label="M0")

    assert len(sched.data["operation_dict"]) == 4
    assert len(sched.data["schedulables"]) == 5

    for schedulable in sched.data["schedulables"].values():
        assert "abs_time" not in schedulable.keys()
        assert schedulable["timing_constraints"][0]["rel_time"] == 0

    timed_sched = _determine_absolute_timing(sched, time_unit="ideal")

    abs_times = [
        schedulable["abs_time"]
        for schedulable in timed_sched.data["schedulables"].values()
    ]
    assert abs_times == [0, 1, 2, 3, 4]

    # add a pulse and schedule simultaneous with the second pulse
    sched.add(Rxy(90, 0, qubit=q1), ref_pt="start", ref_op=ref_label_1)
    timed_sched = _determine_absolute_timing(sched, time_unit="ideal")

    abs_times = [
        constr["abs_time"] for constr in timed_sched.data["schedulables"].values()
    ]
    assert abs_times == [0, 1, 2, 3, 4, 1]

    sched.add(Rxy(90, 0, qubit=q1), ref_pt="start", ref_op="M0")
    timed_sched = _determine_absolute_timing(sched, time_unit="ideal")

    abs_times = [
        schedulable["abs_time"]
        for schedulable in timed_sched.data["schedulables"].values()
    ]
    assert abs_times == [0, 1, 2, 3, 4, 1, 4]

    sched.add(Rxy(90, 0, qubit=q1), ref_pt="end", ref_op=ref_label_1)
    timed_sched = _determine_absolute_timing(sched, time_unit="ideal")

    abs_times = [
        schedulable["abs_time"]
        for schedulable in timed_sched.data["schedulables"].values()
    ]
    assert abs_times == [0, 1, 2, 3, 4, 1, 4, 2]

    sched.add(Rxy(90, 0, qubit=q1), ref_pt="center", ref_op=ref_label_1)
    timed_sched = _determine_absolute_timing(sched, time_unit="ideal")

    abs_times = [
        schedulable["abs_time"]
        for schedulable in timed_sched.data["schedulables"].values()
    ]
    assert abs_times == [0, 1, 2, 3, 4, 1, 4, 2, 1.5]

    bad_sched = Schedule("no good")
    bad_sched.add(Rxy(180, 0, qubit=q1))
    bad_sched.add(Rxy(90, 0, qubit=q1), ref_pt="bad")
    with pytest.raises(NotImplementedError):
        _ = _determine_absolute_timing(schedule=bad_sched, time_unit="ideal")


def test_determine_absolute_timing_alap_raises(
    mock_setup_basic_transmon_with_standard_params, basic_schedule
):
    quantum_device = mock_setup_basic_transmon_with_standard_params["quantum_device"]
    quantum_device.scheduling_strategy("alap")
    assert quantum_device.scheduling_strategy() == "alap"

    # Assert that an implementation error is raised for alap scheduling_strategy
    with pytest.raises(NotImplementedError):
        _determine_absolute_timing(
            schedule=basic_schedule,
            time_unit="ideal",
            config=quantum_device.generate_compilation_config(),
        )


def test_missing_ref_op():
    sched = Schedule("test")
    q0, q1 = ("q0", "q1")
    ref_label_1 = "test_label"
    with pytest.raises(ValueError):
        sched.add(operation=CNOT(qC=q0, qT=q1), ref_op=ref_label_1)


def test_compile_transmon_program(mock_setup_basic_transmon_with_standard_params):
    sched = Schedule("Test schedule")

    # Define the resources
    q0, q2 = ("q0", "q2")

    sched.add(Reset(q0, q2))
    sched.add(Rxy(90, 0, qubit=q0))
    sched.add(operation=CZ(qC=q0, qT=q2))
    sched.add(Rxy(theta=90, phi=0, qubit=q0))
    sched.add(Measure(q0, q2), label="M0")

    compiler = SerialCompiler(name="compiler")
    compiler.compile(
        sched,
        mock_setup_basic_transmon_with_standard_params[
            "quantum_device"
        ].generate_compilation_config(),
    )


def test_compile_gates_to_subschedule(mock_setup_basic_transmon_with_standard_params):
    compiler = SerialCompiler(name="compiler")

    # Add H composite gate to sched and compile to subschedules
    sched = Schedule("Schedule")
    sched.add(H("q0", "q1"))
    compiled_sched = compiler.compile(
        sched,
        mock_setup_basic_transmon_with_standard_params[
            "quantum_device"
        ].generate_compilation_config(),
    )

    # Add H constituent gates Y90 and Z to sched directly as subschedules
    expected_inner_sched = Schedule("Inner sched H q0 q1")
    ref_h = expected_inner_sched.add(hadamard_as_y90z("q0"))
    expected_inner_sched.add(hadamard_as_y90z("q1"), ref_op=ref_h, ref_pt="start")

    expected_sched = Schedule("Expected sched")
    expected_sched.add(expected_inner_sched)

    expected_compiled_sched = compiler.compile(
        expected_sched,
        mock_setup_basic_transmon_with_standard_params[
            "quantum_device"
        ].generate_compilation_config(),
    )

    assert len(compiled_sched) == len(expected_compiled_sched)

    for schedulable, expected_schedulable in zip(
        compiled_sched.schedulables.values(),
        expected_compiled_sched.schedulables.values(),
    ):
        op = compiled_sched.operations[schedulable["operation_id"]]
        expected_op = expected_compiled_sched.operations[
            expected_schedulable["operation_id"]
        ]
        assert op == expected_op


def test_missing_edge(mock_setup_basic_transmon):
    sched = Schedule("Missing edge")

    quantum_device = mock_setup_basic_transmon["quantum_device"]
    quantum_device.remove_edge("q0_q2")

    q0, q2 = ("q0", "q2")
    sched.add(operation=CZ(qC=q0, qT=q2))
    with pytest.raises(
        ConfigKeyError,
        match='edge "q0_q2" is not present in the configuration file',
    ):
        compiler = SerialCompiler(name="compiler")
        compiler.compile(
            sched,
            quantum_device.generate_compilation_config(),
        )


def test_empty_sched():
    sched = Schedule("empty")
    with pytest.raises(ValueError, match="schedule 'empty' contains no schedulables"):
        _ = _determine_absolute_timing(schedule=sched)


def test_bad_gate(device_compile_config_basic_transmon):
    class NotAGate(Operation):
        def __init__(self, q):
            plot_func = (
                "quantify_scheduler.schedules._visualization.circuit_diagram.cnot"
            )
            data = {
                "gate_info": {
                    "unitary": np.array(
                        [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
                    ),
                    "tex": r"bad",
                    "plot_func": plot_func,
                    "qubits": [q],
                    "operation_type": "bad",
                }
            }
            super().__init__(f"bad ({q})")
            self.data.update(data)

    sched = Schedule("Bell experiment")
    sched.add(Reset("q0"))
    sched.add(NotAGate("q0"))
    with pytest.raises(
        ConfigKeyError,
        match='\'operation "bad" is not present in the configuration file.*',
    ):
        compiler = SerialCompiler(name="compiler")
        compiler.compile(
            sched,
            config=device_compile_config_basic_transmon,
        )


def test_pulse_and_clock(device_compile_config_basic_transmon):
    sched = Schedule("pulse_no_clock")
    mystery_clock = "BigBen"
    schedulable = sched.add(SquarePulse(0.5, 20e-9, "q0:mw_ch", clock=mystery_clock))
    op = sched.operations[schedulable["operation_id"]]

    compiler = SerialCompiler(name="compiler")
    with pytest.raises(ValueError) as execinfo:
        compiler.compile(
            sched,
            config=device_compile_config_basic_transmon,
        )
    assert str(execinfo.value) == (
        f"Operation '{str(op)}' contains an unknown clock '{mystery_clock}'; "
        f"ensure this resource has been added to the schedule "
        f"or to the device config."
    )

    sched.add_resource(ClockResource(mystery_clock, 6e9))
    compiler.compile(
        sched,
        config=device_compile_config_basic_transmon,
    )


def test_resource_resolution(device_compile_config_basic_transmon):
    sched = Schedule("resource_resolution")
    qcm0_s0 = Resource("qcm0.s0")
    qcm0_s0["type"] = "qcm"
    qrm0_s0 = Resource("qrm0.s0")
    qrm0_s0["type"] = "qrm"

    sched.add(Rxy(90, 0, "q0"))
    sched.add(SquarePulse(0.6, 20e-9, "q0:mw_ch", clock=BasebandClockResource.IDENTITY))
    sched.add(SquarePulse(0.4, 20e-9, "q0:ro_ch", clock=BasebandClockResource.IDENTITY))

    sched.add_resources([qcm0_s0, qrm0_s0])
    compiler = SerialCompiler(name="compiler")
    _ = compiler.compile(
        sched,
        config=device_compile_config_basic_transmon,
    )


def test_schedule_modified(device_compile_config_basic_transmon):
    q0, q1 = ("q0", "q1")

    ref_label_1 = "my_label"
    sched = Schedule("Test experiment")
    sched.add(Reset(q0, q1))
    sched.add(Rxy(90, 0, qubit=q0), label=ref_label_1)
    sched.add(Rxy(theta=90, phi=0, qubit=q0))
    sched.add(Measure(q0, q1), label="M0")

    copy_of_sched = deepcopy(sched)
    # to verify equality of schedule object works
    assert copy_of_sched == sched

    config = device_compile_config_basic_transmon
    config.keep_original_schedule = True

    compiler = SerialCompiler(name="compiler")
    _ = compiler.compile(
        sched,
        config=device_compile_config_basic_transmon,
    )
    # Fails if schedule is modified
    assert copy_of_sched == sched


def test_measurement_specification_of_binmode(device_compile_config_basic_transmon):
    qubit = "q0"

    ####################################################################################
    # Append selected
    ####################################################################################

    schedule = Schedule("binmode-test", 1)
    schedule.add(Reset(qubit), label=f"Reset {0}")
    schedule.add(
        Measure(qubit, acq_index=0, bin_mode=BinMode.APPEND), label=f"Measurement {0}"
    )

    compiler = SerialCompiler(name="compiler")
    comp_sched = compiler.compile(
        schedule=schedule,
        config=device_compile_config_basic_transmon,
    )

    for value in comp_sched.data["operation_dict"].values():
        if "Measure" in str(value):
            assert value.data["acquisition_info"][0]["bin_mode"] == BinMode.APPEND

    ####################################################################################
    # AVERAGE selected
    ####################################################################################

    schedule = Schedule("binmode-test", 1)
    schedule.add(Reset(qubit), label=f"Reset {0}")
    schedule.add(
        Measure(qubit, acq_index=0, bin_mode=BinMode.AVERAGE), label=f"Measurement {0}"
    )

    comp_sched = compiler.compile(
        schedule=schedule,
        config=device_compile_config_basic_transmon,
    )

    for value in comp_sched.data["operation_dict"].values():
        if "Measure" in str(value):
            assert value.data["acquisition_info"][0]["bin_mode"] == BinMode.AVERAGE

    ####################################################################################
    # Not specified uses default average mode
    ####################################################################################

    schedule = Schedule("binmode-test", 1)
    schedule.add(Reset(qubit), label=f"Reset {0}")
    schedule.add(Measure(qubit, acq_index=0), label=f"Measurement {0}")

    comp_sched = compiler.compile(
        schedule=schedule,
        config=device_compile_config_basic_transmon,
    )

    for value in comp_sched.data["operation_dict"].values():
        if "Measure" in str(value):
            assert value.data["acquisition_info"][0]["bin_mode"] == BinMode.AVERAGE


def test_compile_trace_acquisition(device_compile_config_basic_transmon):
    sched = Schedule("Test schedule")
    q0 = "q0"
    sched.add(Reset(q0))
    sched.add(Rxy(90, 0, qubit=q0))
    sched.add(Measure(q0, acq_protocol="Trace"), label="M0")

    compiler = SerialCompiler(name="compile")
    sched = compiler.compile(
        schedule=sched, config=device_compile_config_basic_transmon
    )

    measure_repr = list(sched.schedulables.values())[-1]["operation_id"]
    assert sched.operations[measure_repr]["acquisition_info"][0]["protocol"] == "Trace"


def test_compile_weighted_acquisition(
    compile_config_basic_transmon_qblox_hardware_cluster,
):
    sched = Schedule("Test schedule")
    q0 = "q0"
    q1 = "q1"

    sched.add(Reset(q0))
    sched.add(Rxy(90, 0, qubit=q0))
    sched.add(
        Measure(q0, acq_protocol="NumericalSeparatedWeightedIntegration"), label="M0"
    )
    sched.add(Measure(q1, acq_protocol="NumericalWeightedIntegration"), label="M1")

    compiler = SerialCompiler(name="compile")
    sched = compiler.compile(
        schedule=sched,
        config=compile_config_basic_transmon_qblox_hardware_cluster,
    )

    measure_repr = list(sched.schedulables.values())[-2]["operation_id"]
    assert (
        sched.operations[measure_repr]["acquisition_info"][0]["protocol"]
        == "NumericalSeparatedWeightedIntegration"
    )
    measure_repr = list(sched.schedulables.values())[-1]["operation_id"]
    assert (
        sched.operations[measure_repr]["acquisition_info"][0]["protocol"]
        == "NumericalWeightedIntegration"
    )


def test_compile_no_device_cfg_determine_absolute_timing(
    mocker, device_compile_config_basic_transmon
):
    sched = Schedule("One pulse schedule")
    sched.add(SquarePulse(amp=1 / 4, duration=12e-9, port="q0:mw", clock="q0.01"))

    mock = mocker.patch("quantify_scheduler.compilation._determine_absolute_timing")
    compiler = SerialCompiler(name="compile")
    compiler.compile(schedule=sched, config=device_compile_config_basic_transmon)
    assert mock.is_called()


def test_determine_absolute_timing_subschedule():
    sched = Schedule("Outer")
    inner_sched = Schedule("Inner")

    # define the resources
    # q0, q1 = Qubits(n=2) # assumes all to all connectivity
    q0, q1 = ("q0", "q1")

    ref_label_1 = "ref0_2"

    inner_sched.add(Reset(q0, q1), label="ref1_0")
    inner_sched.add(Rxy(90, 0, qubit=q0), label="ref1_1")

    sched.add(operation=CNOT(qC=q0, qT=q1), label="ref0_0")
    sched.add(inner_sched, label="ref0_1")
    sched.add(Measure(q0), label="ref0_2")

    assert len(sched.data["operation_dict"]) == 3
    assert len(sched.data["schedulables"]) == 3

    for schedulable in sched.data["schedulables"].values():
        assert "abs_time" not in schedulable.keys()
        assert schedulable["timing_constraints"][0]["rel_time"] == 0

    timed_sched = _determine_absolute_timing(sched, time_unit="ideal")

    abs_times = [
        schedulable["abs_time"]
        for schedulable in timed_sched.data["schedulables"].values()
    ]
    assert abs_times == [0, 1, 3]
    inner_sched_schedulable = timed_sched.data["schedulables"]["ref0_1"]
    timed_inner = timed_sched.operations[inner_sched_schedulable["operation_id"]]
    abs_times = [
        schedulable["abs_time"]
        for schedulable in timed_inner.data["schedulables"].values()
    ]
    assert abs_times == [0, 1]

    # add a pulse and schedule simultaneous with the second pulse
    sched.add(Rxy(90, 0, qubit=q1), ref_pt="start", ref_op=ref_label_1)
    timed_sched = _determine_absolute_timing(sched, time_unit="ideal")

    abs_times = [
        constr["abs_time"] for constr in timed_sched.data["schedulables"].values()
    ]
    assert abs_times == [0, 1, 3, 3]

    flatten_schedule(timed_sched)
    abs_times = [
        constr["abs_time"] for constr in timed_sched.data["schedulables"].values()
    ]
    assert abs_times == [0, 1, 2, 3, 3]


def test_flatten_schedule():
    inner = Schedule("inner")
    inner.add(SetClockFrequency(clock="q0.01", clock_freq_new=7.501e9))

    inner2 = Schedule("inner2")
    inner2.add(SetClockFrequency(clock="q0.01", clock_freq_new=7.502e9))

    inner.add(inner2)

    outer = Schedule("outer")
    outer.add(SetClockFrequency(clock="q0.01", clock_freq_new=7.5e9))

    outer.add(inner)
    outer.add(inner2)
    timed_sched = _determine_absolute_timing(outer, time_unit="ideal")
    flat = flatten_schedule(timed_sched)
    assert len(flat.data["schedulables"]) == 4


@pytest.mark.parametrize(
    argnames="schedule_kwargs", argvalues=[{}, {"control_flow": Loop(1024)}]
)
def test_flatten_schedule_gets_all_resources(
    compile_config_basic_transmon_qblox_hardware, schedule_kwargs
):
    schedule1 = Schedule("")
    schedule2 = Schedule("")
    schedule2.add(X("q0"), **schedule_kwargs)
    schedule1.add(schedule2)

    compiler = SerialCompiler("")
    compiled_schedule = compiler.compile(
        schedule=schedule1,
        config=compile_config_basic_transmon_qblox_hardware,
    )

    expected_resources = {
        "cl0.baseband": {
            "name": "cl0.baseband",
            "type": "BasebandClockResource",
            "freq": 0,
            "phase": 0,
        },
        "digital": {
            "name": "digital",
            "type": "DigitalClockResource",
            "freq": 0,
            "phase": 0,
        },
        "q0.01": {
            "name": "q0.01",
            "type": "ClockResource",
            "freq": 7300000000.0,
            "phase": 0,
        },
    }
    assert compiled_schedule.resources == expected_resources
