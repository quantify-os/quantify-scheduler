"""
Testing focused on the backend for qblox hardware.
This stage should test from the top level down to the hardware instructions.
We need to be careful how we test the output as the internals of the format might
change in the future.
Might be good to mark those tests in detail.
"""

import pytest

from quantify_scheduler import Schedule, CompiledSchedule
from quantify_scheduler.backends import SerialCompiler

from .standard_schedules import (
    single_qubit_schedule_circuit_level,
    two_qubit_t1_schedule,
    two_qubit_schedule_with_edge,
    pulse_only_schedule,
    parametrized_operation_schedule,
    hybrid_schedule_rabi,
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
def test_compiles_standard_schedules(
    schedule: Schedule,
    compile_config_basic_transmon_qblox_hardware,
):
    """
    Tests if a set of standard schedules compile without raising exceptions
    """

    config = compile_config_basic_transmon_qblox_hardware
    assert config.name == "Qblox backend"
    assert (
        config.backend == "quantify_scheduler.backends.graph_compilation.SerialCompiler"
    )

    backend = SerialCompiler("Qblox backend")

    # assert that no exception is raised.
    # Act
    comp_sched = backend.compile(schedule=schedule, config=config)
    # Assert that no exception was raised and output is the right type.
    assert isinstance(comp_sched, CompiledSchedule)
