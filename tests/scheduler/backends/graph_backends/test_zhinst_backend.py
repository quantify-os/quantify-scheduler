"""
Testing focused on the backend for ZurichInstruments hardware.
This stage should test from the top level down to the hardware instructions.

We need to be careful how we test the output as the internals of the format might
change in the future.
Might be good to mark those tests in detail.
"""


import pytest

from quantify_scheduler import Schedule, CompiledSchedule
from quantify_scheduler.backends import SerialCompiler
from quantify_scheduler.device_under_test.quantum_device import QuantumDevice
from quantify_scheduler.resources import ClockResource

from .standard_schedules import (
    single_qubit_schedule_circuit_level,
    two_qubit_t1_schedule,
    two_qubit_schedule_with_edge,
    pulse_only_schedule,
    parametrized_operation_schedule,
    hybrid_schedule_rabi,
)
from ....fixtures.mock_setup import ZHINST_HARDWARE_MAPPING


@pytest.mark.parametrize(
    "schedule",
    [
        single_qubit_schedule_circuit_level(),
        # two_qubit_t1_schedule(),
        # two_qubit_schedule_with_edge(),
        pulse_only_schedule(),
        parametrized_operation_schedule(),
        hybrid_schedule_rabi(),
    ],
)
def test_compiles_standard_schedules(
    schedule: Schedule,
    compile_config_basic_transmon_zhinst_hardware,
):
    """
    Test if a set of standard schedules compile correctly on this backend.
    """

    config = compile_config_basic_transmon_zhinst_hardware
    assert config.name == "Zhinst compiler"
    assert config.backend == SerialCompiler

    backend = SerialCompiler(name=config.name)
    comp_sched = backend.compile(schedule=schedule, config=config)

    # Assert that no exception was raised and output is the right type.
    assert isinstance(comp_sched, CompiledSchedule)


# NOTE, the tests below are identical to the one above.
# I have moved those that fail because they are not supported in the backend yet.
@pytest.mark.xfail(reason="Support for multiplexed readout in Zhinst backend #307")
@pytest.mark.parametrize(
    "schedule",
    [
        two_qubit_t1_schedule(),
    ],
)
def test_compiles_standard_schedules_mux_ro(
    schedule: Schedule,
    compile_config_basic_transmon_zhinst_hardware,
):
    """
    Test if a multiplexed readout schedule compiles, to be moved up once it passes.
    """

    config = compile_config_basic_transmon_zhinst_hardware
    assert config.name == "Zhinst compiler"
    assert (
        config.backend == "quantify_scheduler.backends.graph_compilation.SerialCompiler"
    )

    backend = SerialCompiler(name=config.name)

    comp_sched = backend.compile(schedule=schedule, config=config)

    # Assert that no exception was raised and output is the right type.
    assert isinstance(comp_sched, CompiledSchedule)


@pytest.mark.xfail(reason="Support use of real-valued outputs in Zhinst backend #44")
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
    """
    Test if a schedule with baseband flux pulses compiles, to be moved up once it passes
    """

    config = compile_config_basic_transmon_zhinst_hardware
    assert config.name == "Zhinst compiler"
    assert (
        config.backend == "quantify_scheduler.backends.graph_compilation.SerialCompiler"
    )

    backend = SerialCompiler(name=config.name)
    comp_sched = backend.compile(schedule=schedule, config=config)

    # Assert that no exception was raised and output is the right type.
    assert isinstance(comp_sched, CompiledSchedule)


def test_compile_empty_device():
    """
    Test if compilation works for a pulse only schedule on a freshly initialized
    quantum device object to which only a hardware config has been provided.
    """

    sched = pulse_only_schedule()

    # since the device config is empty, the clock needs to be manually added
    sched.add_resource(ClockResource(name="q0.ro", freq=6.4e9))

    quantum_device = QuantumDevice(name="empty_quantum_device")
    quantum_device.hardware_config(ZHINST_HARDWARE_MAPPING)

    compilation_config = quantum_device.generate_compilation_config()

    backend = SerialCompiler(compilation_config.name)
    comp_sched = backend.compile(schedule=sched, config=compilation_config)

    # Assert that no exception was raised and output is the right type.
    assert isinstance(comp_sched, CompiledSchedule)

    # this will fail if no hardware_config was specified
    assert len(comp_sched.compiled_instructions) > 0

    quantum_device.close()  # need to clean up nicely after the test
