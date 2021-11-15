# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
"""Compiler backend for Qblox hardware."""
from __future__ import annotations

from typing import Any, Dict, Tuple

# pylint: disable=no-name-in-module
from quantify_core.utilities.general import make_hash, without

from quantify_scheduler import CompiledSchedule, Schedule
from quantify_scheduler.backends.qblox import compiler_container, helpers
from quantify_scheduler.backends.types.qblox import OpInfo
from quantify_scheduler.operations.pulse_library import WindowOperation


def generate_port_clock_to_device_map(
    mapping: Dict[str, Any]
) -> Dict[Tuple[str, str], str]:
    """
    Generates a mapping which specifies which port-clock combinations belong to which
    device.

    .. note::
        The same device may contain multiple port-clock combinations, but each
        port-clock combination may only occur once.

    Parameters
    ----------
    mapping:
        The hardware mapping config.

    Returns
    -------
    :
        A dictionary with as key a tuple representing a port-clock combination, and
        as value the name of the device. Note that multiple port-clocks may point to
        the same device.
    """

    portclock_map = dict()
    for device_name, device_info in mapping.items():
        if not isinstance(device_info, dict):
            continue

        portclocks = helpers.find_all_port_clock_combinations(device_info)

        for portclock in portclocks:
            portclock_map[portclock] = device_name

    return portclock_map


# pylint: disable=too-many-locals
def _assign_pulse_and_acq_info_to_devices(
    schedule: Schedule,
    device_compilers: Dict[str, Any],
    portclock_mapping: Dict[Tuple[str, str], str],
):
    """
    Traverses the schedule and generates `OpInfo` objects for every pulse and
    acquisition, and assigns it to the correct `InstrumentCompiler`.

    Parameters
    ----------
    schedule
        The schedule to extract the pulse and acquisition info from.
    device_compilers
        Dictionary containing InstrumentCompilers as values and their names as keys.
    portclock_mapping
        A dictionary that maps tuples containing a port and a clock to names of
        instruments. The port and clock combinations are unique, but multiple portclocks
        can point to the same instrument.

    Raises
    ------
    RuntimeError
        This exception is raised then the function encountered an operation that has no
        pulse or acquisition info assigned to it.
    KeyError
        This exception is raised when attempting to assign a pulse with a port-clock
        combination that is not defined in the hardware configuration.
    KeyError
        This exception is raised when attempting to assign an acquisition with a
        port-clock combination that is not defined in the hardware configuration.
    """

    for op_timing_constraint in schedule.timing_constraints:
        op_hash = op_timing_constraint["operation_repr"]
        op_data = schedule.operations[op_hash]

        if isinstance(op_data, WindowOperation):
            continue

        if not op_data.valid_pulse and not op_data.valid_acquisition:
            raise RuntimeError(
                f"Operation {op_hash} is not a valid pulse or acquisition. Please check"
                f" whether the device compilation been performed successfully. "
                f"Operation data: {repr(op_data)}"
            )

        operation_start_time = op_timing_constraint["abs_time"]
        for pulse_data in op_data.data["pulse_info"]:
            if "t0" in pulse_data:
                pulse_start_time = operation_start_time + pulse_data["t0"]
            else:
                pulse_start_time = operation_start_time

            port = pulse_data["port"]
            clock = pulse_data["clock"]
            if port is None:
                continue  # ignore idle pulses

            combined_data = OpInfo(
                name=op_data.data["name"],
                data=pulse_data,
                timing=pulse_start_time,
            )

            if (port, clock) not in portclock_mapping:
                raise KeyError(
                    f"Could not assign pulse data to device. The combination"
                    f" of port {port} and clock {clock} could not be found "
                    f"in hardware configuration.\n\nAre both the port and clock "
                    f"specified in the hardware configuration?\n\nRelevant operation:\n"
                    f"{combined_data}."
                )
            dev = portclock_mapping[(port, clock)]
            device_compilers[dev].add_pulse(port, clock, pulse_info=combined_data)

        for acq_data in op_data.data["acquisition_info"]:
            if "t0" in acq_data:
                acq_start_time = operation_start_time + acq_data["t0"]
            else:
                acq_start_time = operation_start_time
            port = acq_data["port"]
            clock = acq_data["clock"]
            if port is None:
                continue

            hashed_dict = without(acq_data, ["t0", "waveforms"])
            hashed_dict["waveforms"] = list()
            for acq in acq_data["waveforms"]:
                hashed_dict["waveforms"].append(without(acq, ["t0"]))

            combined_data = OpInfo(
                name=op_data.data["name"],
                data=acq_data,
                timing=acq_start_time,
            )
            if (port, clock) not in portclock_mapping:
                raise KeyError(
                    f"Could not assign acquisition data to device. The combination"
                    f" of port {port} and clock {clock} could not be found "
                    f"in hardware configuration.\n\nAre both the port and clock "
                    f"specified in the hardware configuration?\n\nRelevant operation:\n"
                    f"{combined_data}."
                )
            dev = portclock_mapping[(port, clock)]
            device_compilers[dev].add_acquisition(port, clock, acq_info=combined_data)


def hardware_compile(
    schedule: Schedule, hardware_cfg: Dict[str, Any]
) -> CompiledSchedule:
    """
    Main function driving the compilation. The principle behind the overall compilation
    works as follows:

    For every instrument in the hardware configuration, we instantiate a compiler
    object. Then we assign all the pulses/acquisitions that need to be played by that
    instrument to the compiler, which then compiles for each instrument individually.

    This function then returns all the compiled programs bundled together in a
    dictionary with the QCoDeS name of the instrument as key.

    Parameters
    ----------
    schedule
        The schedule to compile. It is assumed the pulse and acquisition info is
        already added to the operation. Otherwise and exception is raised.
    hardware_cfg
        The hardware configuration of the setup.

    Returns
    -------
    :
        The compiled schedule.
    """
    portclock_map = generate_port_clock_to_device_map(hardware_cfg)

    container = compiler_container.CompilerContainer.from_mapping(
        schedule, hardware_cfg
    )
    _assign_pulse_and_acq_info_to_devices(
        schedule=schedule,
        device_compilers=container.instrument_compilers,
        portclock_mapping=portclock_map,
    )

    compiled_instructions = container.compile(repetitions=schedule.repetitions)
    # add the compiled instructions to the schedule data structure
    schedule["compiled_instructions"] = compiled_instructions
    # Mark the schedule as a compiled schedule
    return CompiledSchedule(schedule)
