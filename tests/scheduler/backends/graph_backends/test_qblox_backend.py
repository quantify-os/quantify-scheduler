"""
Testing focused on the backend for qblox hardware.
This stage should test from the top level down to the hardware instructions.
We need to be careful how we test the output as the internals of the format might
change in the future.
Might be good to mark those tests in detail.
"""

import pytest
from quantify_core.data.handling import set_datadir
from quantify_scheduler import Schedule, CompiledSchedule
from quantify_scheduler.backends import SerialCompiler
from quantify_scheduler.device_under_test.quantum_device import QuantumDevice

from .standard_schedules import (
    single_qubit_schedule_circuit_level,
    two_qubit_t1_schedule,
    two_qubit_schedule_with_edge,
    pulse_only_schedule,
    parametrized_operation_schedule,
    hybrid_schedule_rabi,
)
from ....fixtures.mock_setup import QBLOX_HARDWARE_MAPPING


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
    assert config.name == "Qblox compiler"
    assert (
        config.backend == "quantify_scheduler.backends.graph_compilation.SerialCompiler"
    )

    backend = SerialCompiler(config.name)
    comp_sched = backend.compile(schedule=schedule, config=config)
    # Assert that no exception was raised and output is the right type.
    assert isinstance(comp_sched, CompiledSchedule)


def test_compile_empty_device(tmp_test_data_dir):
    """
    Test if compilation works for a pulse only schedule on a freshly initialized
    quantum device object to which only a hardware config has been provided.
    """
    # ensures the datadir is set up and files can be written during compilation
    set_datadir(tmp_test_data_dir)

    sched = pulse_only_schedule()

    quantum_device = QuantumDevice(name="empty_quantum_device")
    quantum_device.hardware_config(QBLOX_HARDWARE_MAPPING)

    compilation_config = quantum_device.generate_compilation_config()

    backend = SerialCompiler(compilation_config.name)
    comp_sched = backend.compile(schedule=sched, config=compilation_config)

    # Assert that no exception was raised and output is the right type.
    assert isinstance(comp_sched, CompiledSchedule)

    # this will fail if no hardware_config was specified
    assert len(comp_sched.compiled_instructions) > 0

    quantum_device.close()  # need to clean up nicely after the test
