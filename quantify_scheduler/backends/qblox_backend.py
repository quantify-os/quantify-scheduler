# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Compiler backend for Qblox hardware."""
from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

from quantify_scheduler import CompiledSchedule, Schedule
from quantify_scheduler.backends.distortions import correct_waveform
from quantify_scheduler.backends.qblox import helpers, compiler_container, constants
from quantify_scheduler.backends.types.qblox import OpInfo
from quantify_scheduler.helpers.collections import without
from quantify_scheduler.operations.pulse_library import WindowOperation


logger = logging.getLogger(__name__)


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

    for schedulable in schedule.schedulables.values():
        op_hash = schedulable["operation_repr"]
        op_data = schedule.operations[op_hash]

        if isinstance(op_data, WindowOperation):
            continue

        if not op_data.valid_pulse and not op_data.valid_acquisition:
            raise RuntimeError(
                f"Operation {op_hash} is not a valid pulse or acquisition. Please check"
                f" whether the device compilation been performed successfully. "
                f"Operation data: {repr(op_data)}"
            )

        operation_start_time = schedulable["abs_time"]
        for pulse_data in op_data.data["pulse_info"]:
            if "t0" in pulse_data:
                pulse_start_time = operation_start_time + pulse_data["t0"]
            else:
                pulse_start_time = operation_start_time

            port = pulse_data["port"]
            clock = pulse_data["clock"]

            combined_data = OpInfo(
                name=op_data.data["name"],
                data=pulse_data,
                timing=pulse_start_time,
            )

            if port is None:
                for (map_port, map_clock), dev in portclock_mapping.items():
                    if map_clock == clock:
                        device_compilers[dev].add_pulse(
                            map_port, clock, pulse_info=combined_data
                        )
            else:
                if (port, clock) not in portclock_mapping:
                    raise KeyError(
                        f"Could not assign pulse data to device. The combination"
                        f" of port {port} and clock {clock} could not be found "
                        f"in hardware configuration.\n\nAre both the port and clock "
                        f"specified in the hardware configuration?\n\n"
                        f"Relevant operation:\n{combined_data}."
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


def hardware_compile_distortion_corrections(
    schedule: Schedule, hardware_cfg: Dict[str, Any]
) -> CompiledSchedule:
    """
    TODO: add general description of functionality

    For waveform components in need of correcting (indicated via their port & clock) we
    are *only* replacing the dict in "pulse_info" associated to the waveform component

    This means that we can have a combination of corrected (i.e., pre-sampled) and
    uncorrected waveform components in the same operation

    Also, we are not updating the "operation_repr" key, used to reference the operation
    from the schedulable

    Parameters
    ----------
    schedule
    hardware_cfg

    Returns
    -------

    """

    distortion_corrections_key = "distortion_corrections"
    if distortion_corrections_key not in hardware_cfg:
        logging.info(
            f'Backend "{hardware_compile_distortion_corrections.__name__}"'
            f'invoked but no "distortion_corrections" supplied in hardware config'
        )
        return hardware_compile(schedule, hardware_cfg)

    for operation_repr in schedule.operations.keys():
        substitute_operation = None

        for pulse_info_idx, pulse_data in enumerate(
            schedule.operations[operation_repr].data["pulse_info"]
        ):
            portclock_key = f"{pulse_data['port']}-{pulse_data['clock']}"

            if portclock_key in hardware_cfg[distortion_corrections_key]:
                correction_cfg = hardware_cfg[distortion_corrections_key][portclock_key]

                substitute_pulse = correct_waveform(
                    pulse_data=pulse_data,
                    sampling_rate=constants.SAMPLING_RATE,
                    correction_cfg=correction_cfg,
                )

                schedule.operations[operation_repr].data["pulse_info"][
                    pulse_info_idx
                ] = substitute_pulse.data["pulse_info"][0]

                if pulse_info_idx == 0:
                    substitute_operation = substitute_pulse

        # Convert to operation type of first entry in pulse_info,
        # required as first entry in pulse_info is used to generate signature in __str__
        if substitute_operation is not None:
            substitute_operation.data["pulse_info"] = schedule.operations[
                operation_repr
            ].data["pulse_info"]
            schedule.operations[operation_repr] = substitute_operation

    return hardware_compile(schedule, hardware_cfg)
