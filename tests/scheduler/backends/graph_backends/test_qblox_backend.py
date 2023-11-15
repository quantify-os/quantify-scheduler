# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=missing-module-docstring

# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""
Testing focused on the backend for qblox hardware.
This stage should test from the top level down to the hardware instructions.
We need to be careful how we test the output as the internals of the format might
change in the future.
Might be good to mark those tests in detail.
"""
import json
from typing import Union

import pytest

from quantify_scheduler import CompiledSchedule, Schedule
from quantify_scheduler.backends import SerialCompiler
from quantify_scheduler.device_under_test.quantum_device import QuantumDevice
from quantify_scheduler.operations.pulse_library import SetClockFrequency
from quantify_scheduler.resources import ClockResource

from .standard_schedules import (
    hybrid_schedule_rabi,
    parametrized_operation_schedule,
    pulse_only_schedule,
    single_qubit_schedule_circuit_level,
    two_qubit_schedule_with_edge,
    two_qubit_t1_schedule,
)


def clock_only_schedule() -> Schedule:
    sched = Schedule("Clock only schedule")
    sched.add(SetClockFrequency(clock="q0.01", clock_freq_new=7.501e9))

    return sched


@pytest.mark.parametrize(
    "schedule",
    [
        single_qubit_schedule_circuit_level(),
        two_qubit_t1_schedule(),
        two_qubit_schedule_with_edge(),
        pulse_only_schedule(),
        parametrized_operation_schedule(),
        hybrid_schedule_rabi(),
        clock_only_schedule(),
    ],
)
def test_compiles_standard_schedules(
    schedule: Schedule,
    compile_config_basic_transmon_qblox_hardware_pulsar,
):
    """
    Tests if a set of standard schedules compile without raising exceptions
    """

    config = compile_config_basic_transmon_qblox_hardware_pulsar
    assert config.name == "QuantumDevice-generated SerialCompilationConfig"
    assert config.backend == SerialCompiler

    backend = SerialCompiler(config.name)
    compiled_sched = backend.compile(schedule=schedule, config=config)

    # Assert that no exception was raised and output is the right type
    assert isinstance(compiled_sched, CompiledSchedule)


def test_compile_empty_device(hardware_cfg_pulsar):
    """
    Test if compilation works for a pulse only schedule on a freshly initialized
    quantum device object to which only a hardware config has been provided.
    """

    quantum_device = QuantumDevice(name="empty_quantum_device")
    quantum_device.hardware_config(hardware_cfg_pulsar)

    config = quantum_device.generate_compilation_config()
    backend = SerialCompiler(config.name)

    sched = pulse_only_schedule()
    sched.add_resource(ClockResource("q0.ro", 6.2e9))
    compiled_sched = backend.compile(schedule=sched, config=config)

    # Assert that no exception was raised and output is the right type
    assert isinstance(compiled_sched, CompiledSchedule)

    # This will fail if no hardware_config was specified
    assert len(compiled_sched.compiled_instructions) > 0

    quantum_device.close()


@pytest.mark.deprecated
@pytest.mark.parametrize(
    "instrument, sequence_to_file",
    [
        (instrument, sequence_to_file)
        for instrument in ["qrm0", ("cluster0", "cluster0_module1")]
        for sequence_to_file in [True, False, None]
    ],
)
def test_compile_sequence_to_file_deprecated_hardware_config(
    instrument: Union[str, tuple], sequence_to_file: bool
):
    # Arrange

    hardware_cfg = {
        "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile"
    }
    if isinstance(instrument, tuple):
        hardware_cfg[instrument[0]] = {
            "instrument_type": "Cluster",
            "ref": "internal",
            "sequence_to_file": sequence_to_file,
            instrument[1]: {
                "instrument_type": "QRM",
                "complex_output_0": {
                    "portclock_configs": [
                        {
                            "port": "q0:res",
                            "clock": "q0.ro",
                        }
                    ],
                },
            },
        }
        if sequence_to_file is None:
            del hardware_cfg[instrument[0]]["sequence_to_file"]
    else:
        hardware_cfg[instrument] = {
            "instrument_type": "Pulsar_QRM",
            "ref": "internal",
            "sequence_to_file": sequence_to_file,
            "complex_output_0": {
                "portclock_configs": [
                    {
                        "port": "q0:res",
                        "clock": "q0.ro",
                    }
                ],
            },
        }
        if sequence_to_file is None:
            del hardware_cfg[instrument]["sequence_to_file"]

    quantum_device = QuantumDevice(name="empty_quantum_device")
    quantum_device.hardware_config(hardware_cfg)

    config = quantum_device.generate_compilation_config()
    backend = SerialCompiler(config.name)

    # Act
    sched = pulse_only_schedule()
    sched.add_resource(ClockResource("q0.ro", 6.2e9))
    compiled_sched = backend.compile(schedule=sched, config=config)

    # Assert
    compiled_data = compiled_sched.compiled_instructions
    if isinstance(instrument, tuple):
        for key in instrument:
            compiled_data = compiled_data.get(key)
    else:
        compiled_data = compiled_data.get(instrument)

    seq0_json = compiled_data["sequencers"]["seq0"]["sequence"]
    seq_fn = compiled_data["sequencers"]["seq0"]["seq_fn"]
    assert len(seq0_json["program"]) > 0

    if sequence_to_file is True:
        with open(seq_fn) as file:
            seq0_json_from_disk = json.load(file)
        assert seq0_json_from_disk == seq0_json
    else:
        assert seq_fn is None

    quantum_device.close()


@pytest.mark.parametrize(
    "instrument, sequence_to_file",
    [
        (instrument, sequence_to_file)
        for instrument in ["qrm0", ("cluster0", "module1")]
        for sequence_to_file in [True, False, None]
    ],
)
def test_compile_sequence_to_file(
    instrument: Union[str, tuple], sequence_to_file: bool
):
    # Arrange

    hardware_comp_cfg = {
        "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
        "hardware_options": {},
    }
    if isinstance(instrument, tuple):
        hardware_comp_cfg["hardware_description"] = {
            instrument[0]: {
                "instrument_type": "Cluster",
                "ref": "internal",
                "sequence_to_file": sequence_to_file,
                "modules": {
                    instrument[1][-1]: {"instrument_type": "QRM"},
                },
            }
        }
        hardware_comp_cfg["connectivity"] = {
            "graph": [(f"{instrument[0]}.{instrument[1]}.complex_output_0", "q0:res")]
        }
        if sequence_to_file is None:
            del hardware_comp_cfg["hardware_description"][instrument[0]][
                "sequence_to_file"
            ]
    else:
        hardware_comp_cfg["hardware_description"] = {
            instrument: {
                "instrument_type": "Pulsar_QRM",
                "ref": "internal",
                "sequence_to_file": sequence_to_file,
            }
        }
        hardware_comp_cfg["connectivity"] = {
            "graph": [(f"{instrument}.complex_output_0", "q0:res")]
        }
        if sequence_to_file is None:
            del hardware_comp_cfg["hardware_description"][instrument][
                "sequence_to_file"
            ]

    quantum_device = QuantumDevice(name="empty_quantum_device")
    quantum_device.hardware_config(hardware_comp_cfg)

    config = quantum_device.generate_compilation_config()
    backend = SerialCompiler(config.name)

    # Act
    sched = pulse_only_schedule()
    sched.add_resource(ClockResource("q0.ro", 6.2e9))
    compiled_sched = backend.compile(schedule=sched, config=config)

    # Assert
    compiled_data = compiled_sched.compiled_instructions
    if isinstance(instrument, tuple):
        compiled_data = compiled_data[instrument[0]][f"{instrument[0]}_{instrument[1]}"]
    else:
        compiled_data = compiled_data[instrument]

    seq0_json = compiled_data["sequencers"]["seq0"]["sequence"]
    seq_fn = compiled_data["sequencers"]["seq0"]["seq_fn"]
    assert len(seq0_json["program"]) > 0

    if sequence_to_file is True:
        with open(seq_fn) as file:
            seq0_json_from_disk = json.load(file)
        assert seq0_json_from_disk == seq0_json
    else:
        assert seq_fn is None

    quantum_device.close()
