"""
Testing focused on the backend for ZurichInstruments hardware.
This stage should test from the top level down to the hardware instructions.

We need to be careful how we test the output as the internals of the format might
change in the future.
Might be good to mark those tests in detail.
"""


import pytest
from quantify_scheduler.backends import zhinst_backend


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
from quantify_scheduler.backends.device_compile import DeviceCompile


@pytest.mark.parametrize(
    "schedule",
    [
        single_qubit_schedule_circuit_level(),
        # two_qubit_t1_schedule(),
        # two_qubit_schedule_with_edge(),
        # pulse_only_schedule(),
        parametrized_operation_schedule(),
        hybrid_schedule_rabi(),
    ],
)
def test_compiles_standard_schedules(
    schedule: Schedule,
    compile_config_basic_transmon_zhinst_hardware,
):

    config = compile_config_basic_transmon_zhinst_hardware
    # Arrange
    backend = zhinst_backend.ZhinstBackend()

    # assert that no exception is raised.
    # Act
    comp_sched = backend.compile(schedule=schedule, config=config)
    # Assert that no exception was raised and output is the right type.
    assert isinstance(comp_sched, CompiledSchedule)


# NOTE, the tests below are identical to the one above.
# I have only moved the ones that use features that do not exist yet to a RaiseNotImplementedError

# TODO add issue number
@pytest.mark.xfail(
    reason="Multiplexed readout not fully supported in Zhinst backend #xx"
)
@pytest.mark.parametrize(
    "schedule",
    [
        two_qubit_t1_schedule(),
        # pulse_only_schedule(),
    ],
)
def test_compiles_standard_schedules_mux_ro(
    schedule: Schedule,
    compile_config_basic_transmon_zhinst_hardware,
):

    config = compile_config_basic_transmon_zhinst_hardware
    # Arrange
    backend = zhinst_backend.ZhinstBackend()

    # assert that no exception is raised.
    # Act
    comp_sched = backend.compile(schedule=schedule, config=config)
    # Assert that no exception was raised and output is the right type.
    assert isinstance(comp_sched, CompiledSchedule)


# TODO add issue number
@pytest.mark.xfail(
    reason="Real-valued baseband pulses not fully supported in Zhinst backend #xx"
)
@pytest.mark.parametrize(
    "schedule",
    [
        two_qubit_schedule_with_edge(),
    ],
)
def test_compiles_standard_schedules_edge(
    schedule: Schedule,
    compile_config_basic_transmon_zhinst_hardware,
):

    config = compile_config_basic_transmon_zhinst_hardware
    # Arrange
    backend = zhinst_backend.ZhinstBackend()

    # assert that no exception is raised.
    # Act
    comp_sched = backend.compile(schedule=schedule, config=config)
    # Assert that no exception was raised and output is the right type.
    assert isinstance(comp_sched, CompiledSchedule)


# TODO add issue number
@pytest.mark.xfail(reason="Real-valued baseband pulses only should work")
@pytest.mark.parametrize(
    "schedule",
    [
        pulse_only_schedule(),
    ],
)
def test_compiles_standard_schedules_pulse_only(
    schedule: Schedule,
    compile_config_basic_transmon_zhinst_hardware,
):

    config = compile_config_basic_transmon_zhinst_hardware
    # Arrange
    backend = zhinst_backend.ZhinstBackend()

    # assert that no exception is raised.
    # Act
    comp_sched = backend.compile(schedule=schedule, config=config)
    # Assert that no exception was raised and output is the right type.
    assert isinstance(comp_sched, CompiledSchedule)
