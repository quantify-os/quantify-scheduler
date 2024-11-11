"""
Testing specifically the device compilation.
This stage should take care of the conversion of gates to pulses and also support hybrid
schedules.
"""

import pytest

from quantify_scheduler.backends.graph_compilation import (
    DeviceCompilationConfig,
    SerialCompilationConfig,
    SerialCompiler,
)

# The module we are interested in testing
from quantify_scheduler.device_under_test.quantum_device import QuantumDevice
from quantify_scheduler.device_under_test.transmon_element import BasicTransmonElement
from quantify_scheduler.operations.operation import Operation
from quantify_scheduler.schedules.schedule import CompiledSchedule, Schedule
from quantify_scheduler.schemas.examples.device_example_cfgs import (
    example_transmon_cfg,
)
from tests.fixtures.mock_setup import close_instruments

from .standard_schedules import (
    hybrid_schedule_rabi,
    parametrized_operation_schedule,
    pulse_only_schedule,
    single_qubit_schedule_circuit_level,
    two_qubit_schedule_with_edge,
    two_qubit_t1_schedule,
)


@pytest.mark.parametrize(
    "schedule",
    [
        single_qubit_schedule_circuit_level(),
        two_qubit_t1_schedule(),
        two_qubit_schedule_with_edge(),
        pulse_only_schedule(),
        parametrized_operation_schedule(),
        hybrid_schedule_rabi(),
    ],
)
def test_compiles_standard_schedules(schedule: Schedule, device_compile_config_basic_transmon):
    """
    Tests if a bunch of standard schedules compile with the SerialCompiler to the
    device layer.
    """

    config = device_compile_config_basic_transmon
    assert config.name == "QuantumDevice-generated SerialCompilationConfig"
    assert config.backend == SerialCompiler

    backend = SerialCompiler(name=config.name)  # assert that no exception is raised.
    comp_sched = backend.compile(schedule=schedule, config=config)

    # Assert that no exception was raised and output is the right type.
    assert isinstance(comp_sched, CompiledSchedule)


def test_compile_in_setting_quantum_device(
    basic_schedule, mock_setup_basic_transmon_with_standard_params
):
    """
    Test that compilation works in setting a default quantum device
    """

    # Test not setting a default quantum device and also not supplying config fails
    backend = SerialCompiler("test_None")
    with pytest.raises(RuntimeError):
        backend.compile(schedule=basic_schedule)

    # Test setting a default quantum device and then not supplying config works
    quantum_device = mock_setup_basic_transmon_with_standard_params["quantum_device"]
    backend = SerialCompiler(
        "test_quantum_device",
        quantum_device=quantum_device,
    )
    compiled_sched = backend.compile(schedule=basic_schedule)
    assert isinstance(compiled_sched, CompiledSchedule)

    # Test setting a default quantum device and then supplying different config works
    close_instruments(mock_setup_basic_transmon_with_standard_params)

    quantum_device = QuantumDevice("different_quantum_device")
    q0 = BasicTransmonElement("q0")
    quantum_device.add_element(q0)
    q0.clock_freqs.f01(6e9)

    compiled_sched = backend.compile(
        schedule=basic_schedule,
        config=quantum_device.generate_compilation_config(),
    )
    assert isinstance(compiled_sched, CompiledSchedule)


def test_device_compile_default_compilation_passes(basic_schedule):
    """
    Test that compiling a compiling a schedule from the circuit-layer to the
    device-layer works when no compilation passes are specified in the device config input.
    The default compilation passes should then be used.
    """
    for op in basic_schedule.operations.values():
        assert not op.valid_pulse

    device_config = DeviceCompilationConfig(
        elements=example_transmon_cfg["elements"],
        clocks=example_transmon_cfg["clocks"],
        edges=example_transmon_cfg["edges"],
    )

    config = SerialCompilationConfig(
        name="test",
        device_compilation_config=device_config,
    )
    compiler = SerialCompiler(name="test")
    compiled_schedule = compiler.compile(schedule=basic_schedule, config=config)

    assert isinstance(compiled_schedule, CompiledSchedule)

    for op in compiled_schedule.operations.values():
        assert isinstance(op, Operation)  # FIXME #461
        assert op.valid_pulse
