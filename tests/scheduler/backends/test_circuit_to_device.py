from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pytest

from quantify_scheduler.backends import SerialCompiler
from quantify_scheduler.backends.circuit_to_device import (
    ConfigKeyError,
    DeviceCompilationConfig,
    OperationCompilationConfig,
    _clocks_compatible,
    _valid_clock_in_schedule,
    compile_circuit_to_device_with_config_validation,
    set_pulse_and_acquisition_clock,
)
from quantify_scheduler.backends.graph_compilation import SerialCompilationConfig
from quantify_scheduler.backends.qblox.operations.gate_library import ConditionalReset
from quantify_scheduler.device_under_test.composite_square_edge import (
    CompositeSquareEdge,
)
from quantify_scheduler.device_under_test.mock_setup import (
    set_standard_params_transmon,
    set_up_mock_transmon_setup,
)
from quantify_scheduler.device_under_test.quantum_device import QuantumDevice
from quantify_scheduler.device_under_test.spin_edge import SpinEdge
from quantify_scheduler.device_under_test.spin_element import BasicSpinElement
from quantify_scheduler.enums import BinMode
from quantify_scheduler.operations.control_flow_library import LoopOperation
from quantify_scheduler.operations.gate_library import (
    CNOT,
    CZ,
    Y90,
    Z90,
    H,
    Measure,
    Reset,
    Rxy,
    Rz,
    X,
    Y,
    Z,
)
from quantify_scheduler.operations.pulse_compensation_library import (
    PulseCompensation,
)
from quantify_scheduler.operations.pulse_factories import (
    composite_square_pulse,
    rxy_drag_pulse,
)
from quantify_scheduler.operations.pulse_library import (
    IdlePulse,
    RampPulse,
    ReferenceMagnitude,
    SetClockFrequency,
    SquarePulse,
)
from quantify_scheduler.operations.spin_library import SpinInit
from quantify_scheduler.resources import BasebandClockResource, ClockResource
from quantify_scheduler.schedules.schedule import Schedule, ScheduleBase

if TYPE_CHECKING:
    from quantify_scheduler.operations.operation import Operation


def test_compile_all_gates_example_transmon_cfg(device_cfg_transmon_example):
    """
    Test compiling all gates using example_transmon_cfg.
    """

    sched = Schedule("Test schedule")

    # define the resources
    q0, q1 = ("q0", "q1")
    sched.add(ConditionalReset(q0))
    sched.add(Reset(q0, q1))
    sched.add(Rxy(90, 0, qubit=q0))
    sched.add(Rxy(45, 0, qubit=q0))
    sched.add(Rxy(12, 0, qubit=q0))
    sched.add(Rxy(12, 0, qubit=q0))
    sched.add(Rz(90, qubit=q0))
    sched.add(Rz(45, qubit=q0))
    sched.add(Rz(12, qubit=q0))
    sched.add(Rz(12, qubit=q0))
    sched.add(X(qubit=q0))
    sched.add(Y(qubit=q0))
    sched.add(Z(qubit=q0))
    sched.add(Y90(qubit=q0))
    sched.add(Z90(qubit=q0))
    sched.add(operation=CZ(qC=q0, qT=q1))
    sched.add(Rxy(theta=90, phi=0, qubit=q0))
    sched.add(Rz(theta=90, qubit=q0))
    sched.add(H(q0))
    sched.add(Measure(q0, q1), label="M_q0_q1")

    assert len(sched.schedulables) == 20

    # test that all these operations compile correctly.
    _ = compile_circuit_to_device_with_config_validation(
        sched,
        config=SerialCompilationConfig(
            name="test", device_compilation_config=device_cfg_transmon_example
        ),
    )


def test_compile_all_gates_basic_transmon(mock_setup_basic_transmon):
    """
    Test compiling all gates using BasicTransmonElement.
    """

    sched = Schedule("Test schedule")

    # define the resources
    q2, q3 = ("q2", "q3")
    sched.add(ConditionalReset(q2))
    sched.add(Reset(q2, q3))
    sched.add(Rxy(90, 0, qubit=q2))
    sched.add(Rxy(45, 0, qubit=q2))
    sched.add(Rxy(12, 0, qubit=q2))
    sched.add(Rxy(12, 0, qubit=q2))
    sched.add(Rz(90, qubit=q2))
    sched.add(Rz(45, qubit=q2))
    sched.add(Rz(12, qubit=q2))
    sched.add(Rz(12, qubit=q2))
    sched.add(X(qubit=q2))
    sched.add(Y(qubit=q2))
    sched.add(Z(qubit=q2))
    sched.add(Y90(qubit=q2))
    sched.add(Z90(qubit=q2))
    sched.add(operation=CZ(qC=q2, qT=q3))
    sched.add(Rxy(theta=90, phi=0, qubit=q2))
    sched.add(Rz(theta=90, qubit=q2))
    sched.add(H(q2))
    sched.add(Measure(q2, q3), label="M_q2_q3")

    assert len(sched.schedulables) == 20

    # test that all these operations compile correctly.
    quantum_device = mock_setup_basic_transmon["quantum_device"]
    _ = compile_circuit_to_device_with_config_validation(
        sched,
        config=quantum_device.generate_compilation_config(),
    )


def test_compile_asymmetric_gate(mock_setup_basic_transmon):
    """
    Test if compilation fails when performing an asymmetric operation and the
    correct edge defining the parent-child device element connection is missing from
    the device config.
    """
    sched = Schedule("Test schedule")

    # define the resources
    q2, q3 = ("q2", "q3")

    # Deliberately define an asymmetric CZ
    asymmetric_cz = CZ(qC=q3, qT=q2)
    asymmetric_cz.data["gate_info"]["symmetric"] = False

    sched.add(Reset(q2, q3))
    sched.add(operation=asymmetric_cz)
    sched.add(Measure(q2, q3), label="M_q2_q3")

    # test that all these operations compile correctly.
    quantum_device = mock_setup_basic_transmon["quantum_device"]

    with pytest.raises(ConfigKeyError):
        _ = compile_circuit_to_device_with_config_validation(
            sched, config=quantum_device.generate_compilation_config()
        )


def test_compile_symmetric_gate_distinguished_qubits(mock_setup_basic_transmon):
    """
    Test if the compilation finds the exact match of edge
    even if the gate is symmetric by default.
    """

    # define the resources
    q2, q3 = ("q2", "q3")

    edge_q3_q2 = CompositeSquareEdge(parent_element_name=q3, child_element_name=q2)

    # test that all these operations compile correctly.
    quantum_device = mock_setup_basic_transmon["quantum_device"]
    quantum_device.add_edge(edge_q3_q2)

    def compile_schedule(q_c, q_t):
        schedule = Schedule("Test schedule")
        cz = CZ(qC=q_c, qT=q_t)
        cz.data["gate_info"]["symmetric"] = True
        schedule.add(operation=cz)

        return compile_circuit_to_device_with_config_validation(
            schedule, config=quantum_device.generate_compilation_config()
        )

    compiled_schedule_q2_q3 = compile_schedule(q2, q3)
    assert list(compiled_schedule_q2_q3.operations.values())[0]["pulse_info"][0]["port"] == "q2:fl"

    compiled_schedule_q3_q2 = compile_schedule(q3, q2)
    assert list(compiled_schedule_q3_q2.operations.values())[0]["pulse_info"][0]["port"] == "q3:fl"


def test_measurement_compile(device_cfg_transmon_example, get_subschedule_operation):
    sched = Schedule("Test schedule")
    sched.add(Measure("q0", "q1"))  # acq_index should be 0 for both.
    sched.add(Measure("q0", acq_index=1))
    sched.add(Measure("q1", acq_index=2))  # acq_channel should be 1
    sched.add(Measure("q1", acq_channel=2, acq_index=0))
    sched.add(Measure("q0", "q1", acq_index=2))
    new_dev_sched = compile_circuit_to_device_with_config_validation(
        sched,
        config=SerialCompilationConfig(
            name="test", device_compilation_config=device_cfg_transmon_example
        ),
    )

    # Subschedule components for an acquisition are
    # 0th: reset clock phase,
    # 1th: readout pulse,
    # 2nd: acquisition.

    m0_q0_acq = get_subschedule_operation(new_dev_sched, [0, 0, 2])["acquisition_info"]
    assert len(m0_q0_acq) == 1
    assert m0_q0_acq[0]["acq_channel"] == 0
    assert m0_q0_acq[0]["acq_index"] == 0

    m0_q1_acq = get_subschedule_operation(new_dev_sched, [0, 1, 2])["acquisition_info"]
    assert len(m0_q1_acq) == 1
    assert m0_q1_acq[0]["acq_channel"] == 1
    assert m0_q1_acq[0]["acq_index"] == 0

    m1_acq = get_subschedule_operation(new_dev_sched, [1, 2])["acquisition_info"]
    assert len(m1_acq) == 1
    assert m1_acq[0]["acq_channel"] == 0
    assert m1_acq[0]["acq_index"] == 1

    m2_acq = get_subschedule_operation(new_dev_sched, [2, 2])["acquisition_info"]
    assert len(m2_acq) == 1
    assert m2_acq[0]["acq_channel"] == 1
    assert m2_acq[0]["acq_index"] == 2

    m3_acq = get_subschedule_operation(new_dev_sched, [3, 2])["acquisition_info"]
    assert len(m3_acq) == 1
    assert m3_acq[0]["acq_channel"] == 2
    assert m3_acq[0]["acq_index"] == 0

    m4_q0_acq = get_subschedule_operation(new_dev_sched, [4, 0, 2])["acquisition_info"]
    assert len(m4_q0_acq) == 1
    assert m4_q0_acq[0]["acq_channel"] == 0
    assert m4_q0_acq[0]["acq_index"] == 2

    m4_q1_acq = get_subschedule_operation(new_dev_sched, [4, 1, 2])["acquisition_info"]
    assert len(m4_q1_acq) == 1
    assert m4_q1_acq[0]["acq_channel"] == 1
    assert m4_q1_acq[0]["acq_index"] == 2


@pytest.mark.parametrize(
    "operations, subschedule_indices, clocks_used",
    [
        ([], [], ["cl0.baseband", "digital"]),
        ([X(qubit="q0")], [], ["cl0.baseband", "digital", "q0.01"]),
        ([Z(qubit="q0")], [], ["cl0.baseband", "digital", "q0.01"]),
        ([Measure("q0", "q1")], [0, 0], ["cl0.baseband", "digital", "q0.ro"]),
        ([Measure("q0", "q1")], [0, 1], ["cl0.baseband", "digital", "q1.ro"]),
        (
            [X(qubit="q0"), Z(qubit="q1"), Measure("q0", "q1")],
            [2, 0],
            ["cl0.baseband", "digital", "q0.ro"],
        ),
        (
            [X(qubit="q0"), Z(qubit="q1"), Measure("q0", "q1")],
            [2, 1],
            ["cl0.baseband", "digital", "q1.ro"],
        ),
        (
            [X(qubit="q0"), Z(qubit="q1"), Measure("q0", "q1")],
            [],
            ["cl0.baseband", "digital", "q0.01", "q1.01"],
        ),
    ],
)
def test_only_add_clocks_used(
    operations: list[Operation],
    clocks_used: list[str],
    device_cfg_transmon_example,
    subschedule_indices: list[int],
    get_subschedule_operation,
):
    sched = Schedule("Test schedule")
    for operation in operations:
        sched.add(operation)
    dev_sched = compile_circuit_to_device_with_config_validation(
        sched,
        config=SerialCompilationConfig(
            name="test", device_compilation_config=device_cfg_transmon_example
        ),
    )
    checked_dev_sched = set_pulse_and_acquisition_clock(
        dev_sched,
        config=SerialCompilationConfig(
            name="test", device_compilation_config=device_cfg_transmon_example
        ),
    )

    assert set(
        get_subschedule_operation(checked_dev_sched, subschedule_indices).resources.keys()
    ) == set(clocks_used)


def test_set_gate_clock_raises(mock_setup_basic_transmon_with_standard_params):
    sched = Schedule("Test schedule")
    operation = X("q0")
    sched.add(operation)

    compilation_cfg = mock_setup_basic_transmon_with_standard_params[
        "quantum_device"
    ].generate_compilation_config()

    with pytest.raises(RuntimeError) as error:
        _ = set_pulse_and_acquisition_clock(sched, config=compilation_cfg)

    assert (
        error.value.args[0] == f"Operation '{operation}' is a gate-level operation and must be "
        f"compiled from circuit to device; ensure compilation "
        f"is made in the correct order."
    )


def test_multiply_defined_clock_freq_raises(
    mock_setup_basic_transmon_with_standard_params,
):
    clock = "q0.01"
    clock_freq_schedule = 5e9

    compilation_cfg = mock_setup_basic_transmon_with_standard_params[
        "quantum_device"
    ].generate_compilation_config()
    clock_freq_device_cfg = compilation_cfg.device_compilation_config.clocks[clock]

    sched = Schedule("Test schedule")
    sched.add_resource(ClockResource(name="q0.01", freq=clock_freq_schedule))
    operation = X("q0")
    sched.add(operation)
    dev_sched = compile_circuit_to_device_with_config_validation(
        schedule=sched, config=compilation_cfg
    )

    with pytest.warns(
        RuntimeWarning,
        match=(
            f"Clock '{clock}' has conflicting frequency definitions: "
            f"{clock_freq_schedule} Hz in the schedule and "
            f"{clock_freq_device_cfg} Hz in the device config. "
            f"The clock is set to '{clock_freq_schedule}'. "
            f"Ensure the schedule clock resource matches the "
            f"device config clock frequency or set the "
            f"clock frequency in the device config to np.NaN "
            f"to omit this warning."
        ),
    ):
        compiled_sched = set_pulse_and_acquisition_clock(schedule=dev_sched, config=compilation_cfg)
    assert clock_freq_schedule != clock_freq_device_cfg
    assert compiled_sched.resources[clock]["freq"] == clock_freq_schedule


def test_set_device_cfg_clock(
    mock_setup_basic_transmon_with_standard_params,
):
    quantum_device = mock_setup_basic_transmon_with_standard_params["quantum_device"]
    q0 = quantum_device.get_element("q0")
    clock = "q0.01"

    sched = Schedule("Test schedule")
    sched.add(X("q0"))

    compiler = SerialCompiler("test")
    compiled_sched = compiler.compile(
        schedule=sched, config=quantum_device.generate_compilation_config()
    )
    assert compiled_sched.resources[clock]["freq"] == q0.clock_freqs.f01()


def test_clock_not_defined_raises():
    simple_config = DeviceCompilationConfig(
        clocks={
            "q0.01": 6020000000.0,
        },
        elements={
            "q0": {
                "measure": {
                    "factory_func": "quantify_scheduler.operations."
                    + "measurement_factories.dispersive_measurement_transmon",
                    "gate_info_factory_kwargs": [
                        "acq_channel_override",
                        "acq_index",
                        "bin_mode",
                        "acq_protocol",
                    ],
                    "factory_kwargs": {
                        "port": "q0:res",
                        "clock": "q0.ro",
                        "pulse_type": "SquarePulse",
                        "pulse_amp": 0.25,
                        "pulse_duration": 1.6e-07,
                        "acq_delay": 1.2e-07,
                        "acq_duration": 3e-07,
                        "acq_channel": 0,
                    },
                },
            },
            "q1": {},
            "q2": {},
        },
        edges={},
    )

    clock = "q0.ro"
    sched = Schedule("Test schedule")
    operation = Measure("q0", acq_protocol="Trace")
    sched.add(operation)
    dev_sched = compile_circuit_to_device_with_config_validation(
        sched,
        config=SerialCompilationConfig(name="test", device_compilation_config=simple_config),
    )
    with pytest.raises(ValueError) as error:
        _ = set_pulse_and_acquisition_clock(
            dev_sched,
            config=SerialCompilationConfig(name="test", device_compilation_config=simple_config),
        )

    assert (
        error.value.args[0]
        == f"Operation 'ResetClockPhase(clock='q0.ro',t0=0)' contains an unknown clock '{clock}'; "
        f"ensure this resource has been added to the schedule "
        f"or to the device config."
    )


def test_reset_operations_compile(device_cfg_transmon_example):
    sched = Schedule("Test schedule")
    sched.add(Reset("q0"))
    sched.add(Reset("q0", "q1"))
    _ = compile_circuit_to_device_with_config_validation(
        sched,
        config=SerialCompilationConfig(
            name="test", device_compilation_config=device_cfg_transmon_example
        ),
    )


def test_qubit_not_in_config_raises(device_cfg_transmon_example):
    sched = Schedule("Test schedule")
    sched.add(Rxy(90, 0, qubit="q20"))
    with pytest.raises(ConfigKeyError):
        _ = compile_circuit_to_device_with_config_validation(
            sched,
            config=SerialCompilationConfig(
                name="test", device_compilation_config=device_cfg_transmon_example
            ),
        )

    sched = Schedule("Test schedule")
    sched.add(Reset("q2", "q5", "q3"))
    with pytest.raises(ConfigKeyError):
        _ = compile_circuit_to_device_with_config_validation(
            sched,
            config=SerialCompilationConfig(
                name="test", device_compilation_config=device_cfg_transmon_example
            ),
        )

    sched = Schedule("Test schedule")
    sched.add(Reset("q0", "q5"))
    with pytest.raises(ConfigKeyError):
        _ = compile_circuit_to_device_with_config_validation(
            sched,
            config=SerialCompilationConfig(
                name="test", device_compilation_config=device_cfg_transmon_example
            ),
        )


def test_edge_not_in_config_raises(device_cfg_transmon_example):
    sched = Schedule("Test schedule")
    sched.add(CZ("q0", "q3"))
    with pytest.raises(ConfigKeyError):
        _ = compile_circuit_to_device_with_config_validation(
            sched,
            config=SerialCompilationConfig(
                name="test", device_compilation_config=device_cfg_transmon_example
            ),
        )


def test_operation_not_in_config_raises(device_cfg_transmon_example):
    sched = Schedule("Test schedule")
    sched.add(CNOT("q0", "q1"))
    with pytest.raises(ConfigKeyError):
        _ = compile_circuit_to_device_with_config_validation(
            sched,
            config=SerialCompilationConfig(
                name="test", device_compilation_config=device_cfg_transmon_example
            ),
        )


def test_compile_schedule_with_trace_acq_protocol():
    simple_config = DeviceCompilationConfig(
        clocks={
            "q0.01": 6020000000.0,
            "q0.ro": 7040000000.0,
        },
        elements={
            "q0": {
                "measure": {
                    "factory_func": "quantify_scheduler.operations."
                    + "measurement_factories.dispersive_measurement_transmon",
                    "gate_info_factory_kwargs": [
                        "acq_channel_override",
                        "acq_index",
                        "bin_mode",
                        "acq_protocol",
                    ],
                    "factory_kwargs": {
                        "port": "q0:res",
                        "clock": "q0.ro",
                        "pulse_type": "SquarePulse",
                        "pulse_amp": 0.25,
                        "pulse_duration": 1.6e-07,
                        "acq_delay": 1.2e-07,
                        "acq_duration": 3e-07,
                        "acq_channel": 0,
                    },
                },
            },
            "q1": {},
            "q2": {},
        },
        edges={},
    )
    sched = Schedule("Test schedule")
    sched.add(Measure("q0", acq_protocol="Trace"))
    _ = compile_circuit_to_device_with_config_validation(
        sched,
        config=SerialCompilationConfig(name="test", device_compilation_config=simple_config),
    )


def test_compile_schedule_with_invalid_pulse_type_raises():
    simple_config = DeviceCompilationConfig(
        clocks={
            "q0.01": 6020000000.0,
            "q0.ro": 7040000000.0,
        },
        elements={
            "q0": {
                "measure": {
                    "factory_func": "quantify_scheduler.operations."
                    + "measurement_factories.dispersive_measurement_transmon",
                    "gate_info_factory_kwargs": [
                        "acq_channel_override",
                        "acq_index",
                        "bin_mode",
                        "acq_protocol",
                    ],
                    "factory_kwargs": {
                        "port": "q0:res",
                        "clock": "q0.ro",
                        "pulse_type": "SoftSquare",
                        "pulse_amp": 0.25,
                        "pulse_duration": 1.6e-07,
                        "acq_delay": 1.2e-07,
                        "acq_duration": 3e-07,
                        "acq_channel": 0,
                    },
                },
            },
            "q1": {},
            "q2": {},
        },
        edges={},
    )
    sched = Schedule("Test schedule")
    sched.add(Measure("q0", acq_protocol="Trace"))
    with pytest.raises(NotImplementedError):
        _ = compile_circuit_to_device_with_config_validation(
            sched,
            config=SerialCompilationConfig(name="test", device_compilation_config=simple_config),
        )


def test_operation_not_in_config_raises_custom():
    simple_config = DeviceCompilationConfig(
        clocks={
            "q0.01": 6020000000.0,
            "q0.ro": 7040000000.0,
            "q1.01": 5020000000.0,
            "q1.ro": 6900000000.0,
        },
        elements={"q0": {}, "q1": {}, "q2": {}},
        edges={},
    )

    sched = Schedule("Test missing single q op")
    sched.add(Reset("q0"))
    with pytest.raises(ConfigKeyError):
        _ = compile_circuit_to_device_with_config_validation(
            sched,
            config=SerialCompilationConfig(name="test", device_compilation_config=simple_config),
        )

    sched = Schedule("Test schedule mux missing op")
    sched.add(Reset("q0", "q1", "q2"))
    with pytest.raises(ConfigKeyError):
        _ = compile_circuit_to_device_with_config_validation(
            sched,
            config=SerialCompilationConfig(name="test", device_compilation_config=simple_config),
        )

    sched = Schedule("Test missing 2Q op")
    sched.add(CZ("q0", "q1"))
    with pytest.raises(ConfigKeyError):
        _ = compile_circuit_to_device_with_config_validation(
            sched,
            config=SerialCompilationConfig(name="test", device_compilation_config=simple_config),
        )


def test_config_with_callables():
    simple_config = DeviceCompilationConfig(
        clocks={
            "q0.01": 6020000000.0,
            "q0.ro": 7040000000.0,
            "q1.01": 5020000000.0,
            "q1.ro": 6900000000.0,
        },
        elements={
            "q0": {
                "Rxy": OperationCompilationConfig(
                    factory_func=rxy_drag_pulse,
                    gate_info_factory_kwargs=["theta", "phi"],
                    factory_kwargs={
                        "amp180": 0.32,
                        "motzoi": 0.45,
                        "port": "q0:mw",
                        "clock": "q0.01",
                        "duration": 2e-08,
                    },
                ),
                "reset": OperationCompilationConfig(
                    factory_func=IdlePulse,
                    factory_kwargs={"duration": 0.0002},
                ),
            },
            "q1": {
                "reset": OperationCompilationConfig(
                    factory_func=IdlePulse,
                    factory_kwargs={"duration": 0.0002},
                ),
            },
            "q2": {},
        },
        edges={},
    )

    sched = Schedule("Test callable op")
    sched.add(Reset("q0", "q1"))
    _ = compile_circuit_to_device_with_config_validation(
        sched,
        config=SerialCompilationConfig(name="test", device_compilation_config=simple_config),
    )


def test_config_validation():
    # Raises no error
    _ = OperationCompilationConfig(
        factory_func=rxy_drag_pulse,
        gate_info_factory_kwargs=["theta", "phi"],
        factory_kwargs={
            "amp180": 0.32,
            "motzoi": 0.45,
            "port": "q0:mw",
            "clock": "q0.01",
            "duration": 2e-08,
        },
    )


def schedule_for_clock_tests():
    schedule = Schedule("test schedule")
    operation = schedule.add(rxy_drag_pulse(1, 1, 0, 0, "port", 20e-9, "q0.01"))
    setup = set_up_mock_transmon_setup()
    set_standard_params_transmon(setup)
    quantum_device: QuantumDevice = setup["quantum_device"]
    device_cfg = quantum_device.generate_device_config()
    return schedule, device_cfg, operation


@pytest.mark.parametrize(
    "schedule_freq, device_cfg_freq, compatible_exp",
    [
        [1e9, 1e9, True],
        [1e9, np.nan, True],
        [1e9, 2e9, False],
        [np.nan, 2e9, False],
        [np.asarray(1e9), 1e9, True],
        [1e9, np.asarray([1e9]), True],
        [np.asarray(1e9), 2e9, False],
        [2e9, np.asarray(1e9), False],
        [np.asarray(1e9), np.nan, True],
        [np.asarray(1e9), np.asarray([np.nan]), True],
        [np.asarray(1e9), np.asarray(1e9), True],
        [np.asarray([1e9, 2e9]), np.asarray([1e9, 2e9]), True],
        [np.asarray([1e9, 2e9]), np.asarray([np.nan]), True],
        [np.asarray([1e9, 2e9]), np.asarray([1e9, np.nan]), True],
        [np.asarray([1e9, 2e9]), np.asarray([np.nan, np.nan]), True],
        [np.asarray([1e9, 2e9]), np.asarray([1e9, np.nan, np.nan]), False],
        [np.asarray([1e9, 2e9]), np.asarray([np.nan, np.nan, np.nan]), True],
        [np.asarray(1e9), np.asarray(2e9), False],
        [np.asarray([1e9, 2e9]), np.asarray([2e9, 1e9]), False],
        [np.asarray([1e9, np.nan]), np.asarray([2e9, 1e9]), False],
        [np.asarray([np.nan, np.nan]), np.asarray([2e9, 1e9]), False],
    ],
)
def test_clocks_compatible(schedule_freq, device_cfg_freq, compatible_exp: bool):
    # Arrange
    schedule, device_cfg, operation = schedule_for_clock_tests()
    schedule.add_resource(ClockResource("q0.01", schedule_freq))
    schedule_clock_resources = {"q0.01": schedule_freq}
    device_cfg.clocks["q0.01"] = device_cfg_freq

    # Act
    compatible = _clocks_compatible(
        clock="q0.01",
        device_cfg=device_cfg,
        schedule_clock_resources=schedule_clock_resources,
    )
    # Assert
    assert compatible_exp == compatible

    # Act & Assert
    # Even if clocks are not compatible, the schedule can be compiled
    assert _valid_clock_in_schedule(
        clock="q0.01",
        all_clock_freqs=schedule_clock_resources,
        schedule=schedule,
        operation=operation,
    )


def test_valid_clock_in_schedule():
    """Test whether a valid clock is in the schedule if they can be taken from the device config."""
    # Arrange
    schedule, device_cfg, operation = schedule_for_clock_tests()
    all_clock_freqs = {"q0.01": 1e9}

    # Act & Assert
    assert not _valid_clock_in_schedule(
        clock="q0.01",
        all_clock_freqs=all_clock_freqs,
        schedule=schedule,
        operation=operation,
    )

    # Arrange
    all_clock_freqs = {"q0.01": np.asarray(1e9)}
    # Act & Assert
    assert not _valid_clock_in_schedule(
        clock="q0.01",
        all_clock_freqs=all_clock_freqs,
        schedule=schedule,
        operation=operation,
    )

    # Arrange
    all_clock_freqs = {"q0.01": np.nan}
    # Act & Assert
    with pytest.raises(ValueError):
        _valid_clock_in_schedule(
            clock="q0.01",
            all_clock_freqs=all_clock_freqs,
            schedule=schedule,
            operation=operation,
        )

    # Arrange
    all_clock_freqs = {}
    # Act & Assert
    with pytest.raises(ValueError):
        _valid_clock_in_schedule(
            clock="q0.01",
            all_clock_freqs=all_clock_freqs,
            schedule=schedule,
            operation=operation,
        )


def test_set_reference_magnitude(mock_setup_basic_transmon, get_subschedule_operation):
    """
    Test if compilation using the BasicTransmonElement reproduces old behaviour.
    """

    sched = Schedule("Test schedule")

    q2 = mock_setup_basic_transmon["q2"]
    q3 = mock_setup_basic_transmon["q3"]
    q2.rxy.reference_magnitude.V(0.5)
    q2.measure.reference_magnitude.dBm(20)
    q3.rxy.reference_magnitude.A(1e-3)

    # define the resources
    q2, q3 = ("q2", "q3")
    sched.add(Reset(q2, q3))
    sched.add(Rxy(90, 0, qubit=q2))
    sched.add(Rxy(12, 0, qubit=q3))
    sched.add(X(qubit=q2))
    sched.add(Y(qubit=q2))
    sched.add(Y90(qubit=q2))
    sched.add(operation=CZ(qC=q2, qT=q3))
    sched.add(Rxy(theta=90, phi=0, qubit=q2))
    sched.add(Measure(q2, q3), label="M_q2_q3")

    # test that all these operations compile correctly.
    quantum_device = mock_setup_basic_transmon["quantum_device"]
    compiled_schedule = compile_circuit_to_device_with_config_validation(
        sched, config=quantum_device.generate_compilation_config()
    )

    operations_dict_with_repr_keys = {str(op): op for op in compiled_schedule.operations.values()}

    assert operations_dict_with_repr_keys["Rxy(theta=90, phi=0, qubit='q2')"]["pulse_info"][0][
        "reference_magnitude"
    ] == ReferenceMagnitude(0.5, "V")
    assert operations_dict_with_repr_keys["Rxy(theta=12, phi=0, qubit='q3')"]["pulse_info"][0][
        "reference_magnitude"
    ] == ReferenceMagnitude(1e-3, "A")

    measure_q2_operations_dict_with_repr_keys = {
        str(op): op
        for op in get_subschedule_operation(compiled_schedule, [8, 0]).operations.values()
    }
    assert measure_q2_operations_dict_with_repr_keys[
        (
            "SquarePulse(amp=0.25,duration=3e-07,"
            "port='q2:res',"
            "clock='q2.ro',"
            "reference_magnitude=ReferenceMagnitude(value=20, unit='dBm'),t0=0)"
        )
    ]["pulse_info"][0]["reference_magnitude"] == ReferenceMagnitude(20, "dBm")

    measure_q3_operations_dict_with_repr_keys = {
        str(op): op
        for op in get_subschedule_operation(compiled_schedule, [8, 1]).operations.values()
    }
    assert (
        measure_q3_operations_dict_with_repr_keys[
            "SquarePulse(amp=0.25,duration=3e-07,port='q3:res',clock='q3.ro',reference_magnitude=None,t0=0)"
        ]["pulse_info"][0]["reference_magnitude"]
        is None
    )


def test_operation_collision():
    sched = Schedule("test")
    cz1 = composite_square_pulse(
        square_amp=0.1,
        square_duration=40e-9,
        square_port="q0.fl",
        square_clock="q0.01",
        virt_z_parent_qubit_phase=0,
        virt_z_parent_qubit_clock="q0.01",
        virt_z_child_qubit_phase=1,
        virt_z_child_qubit_clock="q1.01",
    )
    cz2 = composite_square_pulse(
        square_amp=0.1,
        square_duration=40e-9,
        square_port="q0.fl",
        square_clock="q0.01",
        virt_z_parent_qubit_phase=2,
        virt_z_parent_qubit_clock="q0.01",
        virt_z_child_qubit_phase=3,  # note the difference in phases
        virt_z_child_qubit_clock="q1.01",
    )
    sched.add(cz1)
    sched.add(cz2)

    assert len(sched.operations) == 2


def test_clock_resources_and_subschedules_compiles():
    schedule = Schedule("test clock resource subschedule")
    schedule.add_resource(ClockResource(name="qubit.ro", freq=50e6))

    simple_config = DeviceCompilationConfig(
        clocks={
            "q0.01": 6020000000.0,
        },
        elements={
            f"q{i}": {
                "measure": {
                    "factory_func": "quantify_scheduler.operations."
                    + "measurement_factories.dispersive_measurement_transmon",
                    "gate_info_factory_kwargs": [
                        "acq_channel_override",
                        "acq_index",
                        "bin_mode",
                        "acq_protocol",
                    ],
                    "factory_kwargs": {
                        "port": f"q{i}:res",
                        "clock": f"q{i}.ro",
                        "pulse_type": "SquarePulse",
                        "pulse_amp": 0.25,
                        "pulse_duration": 1.6e-07,
                        "acq_delay": 1.2e-07,
                        "acq_duration": 3e-07,
                        "acq_channel": 0,
                    },
                },
            }
            for i in range(3)
        },
        edges={},
    )

    sched = Schedule("Test schedule")

    subsched = Schedule("Subschedule")
    subsched.add(Measure("q0"))
    subsched.add(Measure("q1"))
    subsched.add(Measure("q2"))

    subsubsched = Schedule("Subsubschedule")
    subsubsched.add_resource(ClockResource(name="q1.ro", freq=5e9))
    subsched.add(subsubsched)

    siblingloopsched = Schedule("Sibling loop schedule")
    siblingloopsched.add(Measure("q1"))
    siblingloopsched.add_resource(ClockResource(name="q2.ro", freq=5e9))
    sched.add(LoopOperation(body=siblingloopsched, repetitions=3))

    sched.add_resource(ClockResource(name="q0.ro", freq=5e9))
    sched.add(subsched)

    dev_sched = compile_circuit_to_device_with_config_validation(
        sched,
        config=SerialCompilationConfig(name="test", device_compilation_config=simple_config),
    )
    _ = set_pulse_and_acquisition_clock(
        dev_sched,
        config=SerialCompilationConfig(name="test", device_compilation_config=simple_config),
    )


def test_long_time_trace_protocol(
    device_cfg_transmon_example,
    mock_setup_basic_transmon,
    get_subschedule_operation,
):
    schedule = Schedule("LongTimeTrace")
    schedule.add(Measure("q0", acq_protocol="LongTimeTrace", bin_mode=BinMode.APPEND))

    quantum_device = mock_setup_basic_transmon["quantum_device"]
    q0 = mock_setup_basic_transmon["q0"]
    q0.measure.num_points(11)
    compiled_schedule = compile_circuit_to_device_with_config_validation(
        schedule, config=quantum_device.generate_compilation_config()
    )

    assert len(compiled_schedule.schedulables) == 1

    long_time_trace_schedule = get_subschedule_operation(compiled_schedule, [0])
    assert long_time_trace_schedule.name == "dispersive_measurement"
    assert len(long_time_trace_schedule.schedulables) == 4

    voltage_offset_operation = get_subschedule_operation(compiled_schedule, [0, 0])
    assert voltage_offset_operation.name == "VoltageOffset"

    loop_operation = get_subschedule_operation(compiled_schedule, [0, 1])
    assert loop_operation.name == "LoopOperation"
    assert loop_operation["control_flow_info"]["repetitions"] == 11

    loop_schedule = get_subschedule_operation(compiled_schedule, [0, 1]).body
    reset_clock_phase_operation = get_subschedule_operation(loop_schedule, [0])
    assert reset_clock_phase_operation.name == "ResetClockPhase"
    ssb_integration_operation = get_subschedule_operation(loop_schedule, [1])
    assert ssb_integration_operation.name == "SSBIntegrationComplex"

    voltage_offset_0_operation = get_subschedule_operation(compiled_schedule, [0, 2])
    assert voltage_offset_0_operation.name == "VoltageOffset"


def test_long_time_trace_invalid_bin_mode(
    device_cfg_transmon_example,
    mock_setup_basic_transmon,
):
    quantum_device = mock_setup_basic_transmon["quantum_device"]
    q0 = mock_setup_basic_transmon["q0"]
    q0.measure.num_points(11)

    schedule = Schedule("LongTimeTrace")
    schedule.add(Measure("q0", acq_protocol="LongTimeTrace", bin_mode=BinMode.AVERAGE))

    with pytest.raises(
        ValueError,
        match="For measurement protocol 'LongTimeTrace' "
        "bin_mode set to 'average', "
        "but only 'BinMode.APPEND' is supported.",
    ):
        compile_circuit_to_device_with_config_validation(
            schedule, config=quantum_device.generate_compilation_config()
        )


def test_device_overrides_dispersive_measure(
    mock_setup_basic_transmon,
    get_subschedule_operation,
):
    schedule = Schedule("test")
    schedule.add(Measure("q0", acq_protocol="SSBIntegrationComplex"))
    schedule.add(Measure("q0", acq_protocol="SSBIntegrationComplex", acq_duration=4e-6))

    quantum_device = mock_setup_basic_transmon["quantum_device"]
    compiled_schedule = compile_circuit_to_device_with_config_validation(
        schedule, config=quantum_device.generate_compilation_config()
    )

    acq_operation_0 = get_subschedule_operation(compiled_schedule, [0, 2])
    duration_0 = acq_operation_0["acquisition_info"][0]["duration"]
    acq_operation_1 = get_subschedule_operation(compiled_schedule, [1, 2])
    duration_1 = acq_operation_1["acquisition_info"][0]["duration"]

    assert not math.isclose(duration_0, duration_1)
    assert math.isclose(duration_0, 1e-6)
    assert math.isclose(duration_1, 4e-6)


def test_device_overrides_multiple_levels_hamilton(
    mock_setup_basic_transmon,
    get_subschedule_operation,
):
    schedule = Schedule("test")
    schedule.add(H("q0"))
    schedule.add(H("q0", duration=4e-6))

    quantum_device = mock_setup_basic_transmon["quantum_device"]
    compiled_schedule = compile_circuit_to_device_with_config_validation(
        schedule, config=quantum_device.generate_compilation_config()
    )

    operation_0 = get_subschedule_operation(compiled_schedule, [0, 1])
    duration_0 = operation_0["pulse_info"][0]["duration"]
    operation_1 = get_subschedule_operation(compiled_schedule, [1, 1])
    duration_1 = operation_1["pulse_info"][0]["duration"]

    assert not math.isclose(duration_0, duration_1)
    assert math.isclose(duration_0, 2e-8)
    assert math.isclose(duration_1, 4e-6)


def test_measurement_freq_override(device_cfg_transmon_example, get_subschedule_operation):
    sched = Schedule("Test schedule")
    sched.add(Measure("q0"))
    sched.add(Measure("q1", freq=5e9))
    new_dev_sched = compile_circuit_to_device_with_config_validation(
        sched,
        config=SerialCompilationConfig(
            name="test", device_compilation_config=device_cfg_transmon_example
        ),
    )

    m0_without_freq = get_subschedule_operation(new_dev_sched, [0])
    assert all(not isinstance(op, SetClockFrequency) for op in m0_without_freq.operations)

    # Subschedule components for an acquisition are (if there is frequency override):
    # 0th: frequency override,
    # 1st: subschedule for readout pulse, measure operation, etc.,
    # 2th: frequency reset.

    m1_set_freq = get_subschedule_operation(new_dev_sched, [1, 0])
    assert isinstance(m1_set_freq, SetClockFrequency)
    assert m1_set_freq.data["pulse_info"][0]["clock_freq_new"] == 5e9

    m1_reset_freq = get_subschedule_operation(new_dev_sched, [1, 2])
    assert isinstance(m1_reset_freq, SetClockFrequency)
    assert m1_reset_freq.data["pulse_info"][0]["clock_freq_new"] is None


def test_pulse_compensation_error_factory_func(
    mock_setup_basic_transmon_with_standard_params,
):
    body = Schedule("schedule")
    body.add(
        SquarePulse(amp=0.8, duration=1e-8, port="q0:mw", clock=BasebandClockResource.IDENTITY)
    )

    schedule = Schedule("compensated_schedule")
    schedule.add(PulseCompensation(body=body, qubits=["q0"]))

    compiler = SerialCompiler(name="compiler")

    q0 = mock_setup_basic_transmon_with_standard_params["q0"]
    q0.pulse_compensation.max_compensation_amp(0.6)
    q0.pulse_compensation.time_grid(1e-9)
    q0.pulse_compensation.sampling_rate(1e9)

    compilation_config = mock_setup_basic_transmon_with_standard_params[
        "quantum_device"
    ].generate_compilation_config()

    compilation_config.device_compilation_config.elements["q0"][
        "pulse_compensation"
    ].factory_func = lambda x: x

    with pytest.raises(ValueError) as exception:
        compiler.compile(
            schedule,
            config=compilation_config,
        )

    assert exception.value.args[0] == (
        "'factory_func' in the device configuration for pulse compensation "
        "for device element 'q0' is not 'None'. "
        "Only 'None' is allowed for 'factory_func' for pulse compensation."
    )


def test_compile_spin_init():
    """
    Test compilation of spin init.
    """

    q2 = BasicSpinElement("q2")
    q3 = BasicSpinElement("q3")

    edge_q2_q3 = SpinEdge(parent_element_name=q2.name, child_element_name=q3.name)
    edge_q2_q3.spin_init.square_duration(2e-6)
    edge_q2_q3.spin_init.ramp_diff(1e-6)
    edge_q2_q3.spin_init.q2_square_amp(0.5)
    edge_q2_q3.spin_init.q2_ramp_amp(0.25)
    edge_q2_q3.spin_init.q2_ramp_rate(0.25 / 3e-6)
    edge_q2_q3.spin_init.q3_square_amp(0.4)
    edge_q2_q3.spin_init.q3_ramp_amp(0.2)
    edge_q2_q3.spin_init.q3_ramp_rate(0.2 / 4e-6)

    quantum_device = QuantumDevice(name="quantum_device")
    quantum_device.add_element(q2)
    quantum_device.add_element(q3)
    quantum_device.add_edge(edge_q2_q3)

    schedule = Schedule("Test schedule")
    schedule.add(SpinInit(qC=q2.name, qT=q3.name))

    compiled_schedule = compile_circuit_to_device_with_config_validation(
        schedule, config=quantum_device.generate_compilation_config()
    )

    expected_schedule = Schedule("spin_init")
    expected_schedule.add(
        SquarePulse(
            amp=0.5,
            duration=2e-6,
            port="q2:mw",
            clock="q2.f_larmor",
        )
    )
    expected_schedule.add(
        SquarePulse(
            amp=0.4,
            duration=2e-6,
            port="q3:mw",
            clock="q3.f_larmor",
        ),
        ref_pt="start",
    )
    expected_schedule.add(
        RampPulse(
            amp=0.25,
            duration=3e-6,
            port="q2:mw",
            clock="q2.f_larmor",
        ),
        ref_pt="end",
        rel_time=0,
    )
    expected_schedule.add(
        RampPulse(
            amp=0.2,
            duration=4e-6,
            port="q3:mw",
            clock="q3.f_larmor",
        ),
        ref_pt="start",
        rel_time=0,
    )

    assert len(compiled_schedule.schedulables) == 1

    compiled_spin_init = list(compiled_schedule.operations.values())[0]

    for schedulable, expected_schedulable in zip(
        compiled_spin_init.schedulables.values(),
        expected_schedule.schedulables.values(),
    ):
        operation = compiled_spin_init.operations[schedulable["operation_id"]]
        expected_operation = expected_schedule.operations[expected_schedulable["operation_id"]]
        assert operation == expected_operation
