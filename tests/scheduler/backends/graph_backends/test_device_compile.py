"""
Testing specifically the device compilation.
This stage should take care of the conversion of gates to pulses and also support hybrid
schedules.
"""
import pytest
from quantify_scheduler import Schedule, CompiledSchedule
from .standard_schedules import (
    single_qubit_schedule_circuit_level,
    two_qubit_t1_schedule,
    two_qubit_schedule_with_edge,
    pulse_only_schedule,
    parametrized_operation_schedule,
    hybrid_schedule_rabi,
)


# The module we are interested in testing
from quantify_scheduler.backends import SerialCompiler


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
def test_compiles_standard_schedules(
    schedule: Schedule, device_compile_config_basic_transmon
):

    config = device_compile_config_basic_transmon
    # Arrange
    assert config.name == "Device compilation"
    assert (
        config.backend == "quantify_scheduler.backends.graph_compilation.SerialCompiler"
    )

    backend = SerialCompiler(
        name="Device compilation"
    )  # assert that no exception is raised.
    # Act
    comp_sched = backend.compile(schedule=schedule, config=config)
    # Assert that no exception was raised and output is the right type.
    assert isinstance(comp_sched, CompiledSchedule)
