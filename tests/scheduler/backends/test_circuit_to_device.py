from typing import List

import pytest
import numpy as np

from quantify_scheduler import Operation, Schedule
from quantify_scheduler.backends import SerialCompiler
from quantify_scheduler.backends.circuit_to_device import (
    ConfigKeyError,
    DeviceCompilationConfig,
    OperationCompilationConfig,
    compile_circuit_to_device_with_config_validation,
    set_pulse_and_acquisition_clock,
    _clocks_compatible,
    _valid_clock_in_schedule,
)
from quantify_scheduler.backends.qblox.operations.gate_library import ConditionalReset
from quantify_scheduler.backends.graph_compilation import SerialCompilationConfig
from quantify_scheduler.device_under_test.mock_setup import (
    set_up_mock_transmon_setup,
    set_standard_params_transmon,
)
from quantify_scheduler.device_under_test.quantum_device import QuantumDevice
from quantify_scheduler.operations.gate_library import (
    CNOT,
    CZ,
    Measure,
    Reset,
    Rxy,
    Rz,
    X,
    Y,
    Y90,
    Z,
    Z90,
    H,
)
from quantify_scheduler.operations.pulse_factories import (
    rxy_drag_pulse,
    composite_square_pulse,
)
from quantify_scheduler.operations.pulse_library import IdlePulse, ReferenceMagnitude
from quantify_scheduler.resources import ClockResource
from quantify_scheduler.schemas.examples.device_example_cfgs import (
    example_transmon_cfg,
)


def test_compile_all_gates_example_transmon_cfg():
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
            name="test", device_compilation_config=example_transmon_cfg
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


def test_measurement_compile():
    sched = Schedule("Test schedule")
    sched.add(Measure("q0", "q1"))  # acq_index should be 0 for both.
    sched.add(Measure("q0", acq_index=1))
    sched.add(Measure("q1", acq_index=2))  # acq_channel should be 1
    sched.add(Measure("q1", acq_channel=2, acq_index=0))
    sched.add(Measure("q0", "q1", acq_index=2))
    new_dev_sched = compile_circuit_to_device_with_config_validation(
        sched,
        config=SerialCompilationConfig(
            name="test", device_compilation_config=example_transmon_cfg
        ),
    )

    operation_keys_list = list(new_dev_sched.operations.keys())

    m0_acq = new_dev_sched.operations[operation_keys_list[0]]["acquisition_info"]
    assert len(m0_acq) == 2  # both q0 and q1
    assert m0_acq[0]["acq_channel"] == 0
    assert m0_acq[1]["acq_channel"] == 1
    assert m0_acq[0]["acq_index"] == 0
    assert m0_acq[1]["acq_index"] == 0

    m1_acq = new_dev_sched.operations[operation_keys_list[1]]["acquisition_info"]
    assert len(m1_acq) == 1
    assert m1_acq[0]["acq_channel"] == 0
    assert m1_acq[0]["acq_index"] == 1

    m2_acq = new_dev_sched.operations[operation_keys_list[2]]["acquisition_info"]
    assert len(m2_acq) == 1
    assert m2_acq[0]["acq_channel"] == 1
    assert m2_acq[0]["acq_index"] == 2

    m3_acq = new_dev_sched.operations[operation_keys_list[3]]["acquisition_info"]
    assert len(m3_acq) == 1
    assert m3_acq[0]["acq_channel"] == 2
    assert m3_acq[0]["acq_index"] == 0

    m4_acq = new_dev_sched.operations[operation_keys_list[4]]["acquisition_info"]
    assert len(m4_acq) == 2
    assert m4_acq[0]["acq_channel"] == 0
    assert m4_acq[1]["acq_channel"] == 1
    assert m4_acq[0]["acq_index"] == 2
    assert m4_acq[1]["acq_index"] == 2


@pytest.mark.parametrize(
    "operations, clocks_used",
    [
        ([], ["cl0.baseband", "digital"]),
        ([X(qubit="q0")], ["cl0.baseband", "digital", "q0.01"]),
        ([Z(qubit="q0")], ["cl0.baseband", "digital", "q0.01"]),
        ([Measure("q0", "q1")], ["cl0.baseband", "digital", "q0.ro", "q1.ro"]),
        (
            [X(qubit="q0"), Z(qubit="q1"), Measure("q0", "q1")],
            ["cl0.baseband", "digital", "q0.01", "q1.01", "q0.ro", "q1.ro"],
        ),
    ],
)
def test_only_add_clocks_used(operations: List[Operation], clocks_used: List[str]):
    sched = Schedule("Test schedule")
    for operation in operations:
        sched.add(operation)
    dev_sched = compile_circuit_to_device_with_config_validation(
        sched,
        config=SerialCompilationConfig(
            name="test", device_compilation_config=example_transmon_cfg
        ),
    )
    checked_dev_sched = set_pulse_and_acquisition_clock(
        dev_sched,
        config=SerialCompilationConfig(
            name="test", device_compilation_config=example_transmon_cfg
        ),
    )

    assert set(checked_dev_sched.resources.keys()) == set(clocks_used)


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
        error.value.args[0]
        == f"Operation '{operation}' is a gate-level operation and must be "
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
        compiled_sched = set_pulse_and_acquisition_clock(
            schedule=dev_sched, config=compilation_cfg
        )
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
                    + "measurement_factories.dispersive_measurement",
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
        config=SerialCompilationConfig(
            name="test", device_compilation_config=simple_config
        ),
    )
    with pytest.raises(ValueError) as error:
        _ = set_pulse_and_acquisition_clock(
            dev_sched,
            config=SerialCompilationConfig(
                name="test", device_compilation_config=simple_config
            ),
        )

    assert (
        error.value.args[0]
        == f"Operation '{operation}' contains an unknown clock '{clock}'; "
        f"ensure this resource has been added to the schedule "
        f"or to the device config."
    )


def test_reset_operations_compile():
    sched = Schedule("Test schedule")
    sched.add(Reset("q0"))
    sched.add(Reset("q0", "q1"))
    _ = compile_circuit_to_device_with_config_validation(
        sched,
        config=SerialCompilationConfig(
            name="test", device_compilation_config=example_transmon_cfg
        ),
    )


def test_qubit_not_in_config_raises():
    sched = Schedule("Test schedule")
    sched.add(Rxy(90, 0, qubit="q20"))
    with pytest.raises(ConfigKeyError):
        _ = compile_circuit_to_device_with_config_validation(
            sched,
            config=SerialCompilationConfig(
                name="test", device_compilation_config=example_transmon_cfg
            ),
        )

    sched = Schedule("Test schedule")
    sched.add(Reset("q2", "q5", "q3"))
    with pytest.raises(ConfigKeyError):
        _ = compile_circuit_to_device_with_config_validation(
            sched,
            config=SerialCompilationConfig(
                name="test", device_compilation_config=example_transmon_cfg
            ),
        )

    sched = Schedule("Test schedule")
    sched.add(Reset("q0", "q5"))
    with pytest.raises(ConfigKeyError):
        _ = compile_circuit_to_device_with_config_validation(
            sched,
            config=SerialCompilationConfig(
                name="test", device_compilation_config=example_transmon_cfg
            ),
        )


def test_edge_not_in_config_raises():
    sched = Schedule("Test schedule")
    sched.add(CZ("q0", "q3"))
    with pytest.raises(ConfigKeyError):
        _ = compile_circuit_to_device_with_config_validation(
            sched,
            config=SerialCompilationConfig(
                name="test", device_compilation_config=example_transmon_cfg
            ),
        )


def test_operation_not_in_config_raises():
    sched = Schedule("Test schedule")
    sched.add(CNOT("q0", "q1"))
    with pytest.raises(ConfigKeyError):
        _ = compile_circuit_to_device_with_config_validation(
            sched,
            config=SerialCompilationConfig(
                name="test", device_compilation_config=example_transmon_cfg
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
                    + "measurement_factories.dispersive_measurement",
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
        config=SerialCompilationConfig(
            name="test", device_compilation_config=simple_config
        ),
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
                    + "measurement_factories.dispersive_measurement",
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
            config=SerialCompilationConfig(
                name="test", device_compilation_config=simple_config
            ),
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
            config=SerialCompilationConfig(
                name="test", device_compilation_config=simple_config
            ),
        )

    sched = Schedule("Test schedule mux missing op")
    sched.add(Reset("q0", "q1", "q2"))
    with pytest.raises(ConfigKeyError):
        _ = compile_circuit_to_device_with_config_validation(
            sched,
            config=SerialCompilationConfig(
                name="test", device_compilation_config=simple_config
            ),
        )

    sched = Schedule("Test missing 2Q op")
    sched.add(CZ("q0", "q1"))
    with pytest.raises(ConfigKeyError):
        _ = compile_circuit_to_device_with_config_validation(
            sched,
            config=SerialCompilationConfig(
                name="test", device_compilation_config=simple_config
            ),
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
        config=SerialCompilationConfig(
            name="test", device_compilation_config=simple_config
        ),
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
    device_cfg.clocks["q0.01"] = device_cfg_freq

    # Act
    compatible = _clocks_compatible(
        clock="q0.01", device_cfg=device_cfg, schedule=schedule
    )
    # Assert
    assert compatible_exp == compatible

    # Act & Assert
    # Even if clocks are not compatible, the schedule can be compiled
    assert _valid_clock_in_schedule(
        clock="q0.01", device_cfg=device_cfg, schedule=schedule, operation=operation
    )


def test_valid_clock_in_schedule():
    """Test whether a valid clock is in the schedule if they can be taken from the device config."""
    # Arrange
    schedule, device_cfg, operation = schedule_for_clock_tests()
    device_cfg.clocks["q0.01"] = 1e9

    # Act & Assert
    assert not _valid_clock_in_schedule(
        clock="q0.01", device_cfg=device_cfg, schedule=schedule, operation=operation
    )

    # Arrange
    device_cfg.clocks["q0.01"] = np.asarray(1e9)
    # Act & Assert
    assert not _valid_clock_in_schedule(
        clock="q0.01", device_cfg=device_cfg, schedule=schedule, operation=operation
    )

    # Arrange
    device_cfg.clocks["q0.01"] = np.nan
    # Act & Assert
    with pytest.raises(ValueError):
        _valid_clock_in_schedule(
            clock="q0.01", device_cfg=device_cfg, schedule=schedule, operation=operation
        )

    # Arrange
    del device_cfg.clocks["q0.01"]
    # Act & Assert
    with pytest.raises(ValueError):
        _valid_clock_in_schedule(
            clock="q0.01", device_cfg=device_cfg, schedule=schedule, operation=operation
        )


def test_set_reference_magnitude(mock_setup_basic_transmon):
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

    operations_dict_with_repr_keys = {
        str(op): op for op in compiled_schedule.operations.values()
    }

    assert operations_dict_with_repr_keys["Rxy(theta=90, phi=0, qubit='q2')"][
        "pulse_info"
    ][0]["reference_magnitude"] == ReferenceMagnitude(0.5, "V")
    assert operations_dict_with_repr_keys["Rxy(theta=12, phi=0, qubit='q3')"][
        "pulse_info"
    ][0]["reference_magnitude"] == ReferenceMagnitude(1e-3, "A")
    assert operations_dict_with_repr_keys[
        "Measure('q2','q3', acq_channel=None, acq_index=[0, 0], acq_protocol=\"None\", bin_mode=None, feedback_trigger_label=None)"
    ]["pulse_info"][1]["reference_magnitude"] == ReferenceMagnitude(20, "dBm")
    assert (
        operations_dict_with_repr_keys[
            "Measure('q2','q3', acq_channel=None, acq_index=[0, 0], acq_protocol=\"None\", bin_mode=None, feedback_trigger_label=None)"
        ]["pulse_info"][3]["reference_magnitude"]
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
