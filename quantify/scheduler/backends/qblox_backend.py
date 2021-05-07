# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
"""Compiler backend for Qblox hardware."""
from __future__ import annotations

from typing import Dict, Any, Tuple, Callable

from quantify.scheduler.helpers.schedule import get_total_duration

# pylint: disable=no-name-in-module
from quantify.utilities.general import (
    make_hash,
    without,
)

from quantify.scheduler.backends.qblox import helpers
from quantify.scheduler.backends.qblox import instrument_compilers
from quantify.scheduler.backends.qblox.instrument_compilers import (
    LocalOscillator,
)
from quantify.scheduler.backends.qblox.compiler_abc import InstrumentCompiler
from quantify.scheduler.backends.types.qblox import OpInfo

from quantify.scheduler.types import Schedule


def generate_ext_local_oscillators(
    total_play_time: float, hardware_cfg: Dict[str, Any]
) -> Dict[str, LocalOscillator]:
    """
    Traverses the `hardware_cfg` dict and extracts the used local oscillators.
    `LocalOscillator` objects are instantiated for each LO and the `lo_freq` is
    assigned if specified.

    Parameters
    ----------
    total_play_time:
        Total time the schedule is played for, not counting repetitions.
    hardware_cfg:
        Hardware mapping dictionary

    Returns
    -------
    :
        A dictionary with the names of the devices as keys and compiler
        objects for the local oscillators as values.
    """
    all_lo_objs = dict()
    lo_dicts = helpers.find_inner_dicts_containing_key(hardware_cfg, "lo_name")
    for lo_dict in lo_dicts:
        lo_name = lo_dict["lo_name"]
        if lo_name not in all_lo_objs:
            lo_obj = LocalOscillator(
                lo_name,
                total_play_time,
            )
            all_lo_objs[lo_name] = lo_obj

        if "lo_freq" in lo_dict:
            all_lo_objs[lo_name].assign_frequency(lo_dict["lo_freq"])

    return all_lo_objs


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
def _assign_frequencies(
    device_compilers: Dict[str, InstrumentCompiler],
    lo_compilers: Dict[str, LocalOscillator],
    hw_mapping: Dict[str, Any],
    portclock_mapping: Dict[Tuple[str, str], str],
    schedule_resources: Dict[str, Any],
):
    """
    Determines the IF or LO frequency based on the clock frequency and assigns it to
    the `InstrumentCompiler`. If the IF is specified the LO frequency is calculated
    based on the constraint `clock_freq = interm_freq + lo_freq`, and vice versa.

    Function removes LOs from lo_compilers if not used by any devices.

    Parameters
    ----------
    device_compilers:
        A dictionary containing all the `InstrumentCompiler` objects for which IQ
        modulation is used. The keys correspond to the QCoDeS names of the instruments.
    lo_compilers:
        A dictionary containing all the `LocalOscillator` objects that are used. The
        keys correspond to the QCoDeS names of the instruments.
    hw_mapping:
        The hardware mapping dictionary describing the whole setup.
    portclock_mapping:
        A dictionary that maps tuples containing a port and a clock to names of
        instruments. The port and clock combinations are unique, but multiple portclocks
        can point to the same instrument.
    schedule_resources:
        The schedule resources containing all the clocks.

    Returns
    -------

    """
    lo_info_dicts = helpers.find_inner_dicts_containing_key(hw_mapping, "lo_name")
    los_used = set()
    for lo_info_dict in lo_info_dicts:
        lo_obj = lo_compilers[lo_info_dict["lo_name"]]
        associated_portclock_dicts = helpers.find_inner_dicts_containing_key(
            lo_info_dict, "port"
        )

        lo_freq = None
        if "lo_freq" in lo_info_dict:
            lo_freq = lo_info_dict["lo_freq"]
        if lo_freq is None:
            for portclock_dict in associated_portclock_dicts:
                port, clock = portclock_dict["port"], portclock_dict["clock"]
                interm_freq = portclock_dict["interm_freq"]
                if clock in schedule_resources:
                    cl_freq = schedule_resources[clock]["freq"]

                    dev_name = portclock_mapping[(port, clock)]
                    if (port, clock) in device_compilers[dev_name].portclocks_with_data:
                        los_used.add(lo_obj.name)
                    assign_frequency = getattr(
                        device_compilers[dev_name], "assign_modulation_frequency"
                    )
                    assign_frequency((port, clock), interm_freq)
                    lo_obj.assign_frequency(cl_freq - interm_freq)
        else:  # lo_freq given
            lo_obj.assign_frequency(lo_freq)
            for portclock_dict in associated_portclock_dicts:
                port, clock = portclock_dict["port"], portclock_dict["clock"]
                dev_name = portclock_mapping[(port, clock)]
                if (port, clock) in device_compilers[dev_name].portclocks_with_data:
                    los_used.add(lo_obj.name)
                assign_frequency = getattr(
                    device_compilers[dev_name], "assign_modulation_frequency"
                )
                if clock in schedule_resources:
                    cl_freq = schedule_resources[clock]["freq"]
                    assign_frequency((port, clock), cl_freq - lo_freq)

    unused_los = set(lo_compilers.keys()).difference(los_used)
    for lo_name in unused_los:
        lo_compilers.pop(lo_name)


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
    schedule:
        The schedule to extract the pulse and acquisition info from.
    device_compilers:
        Dictionary containing InstrumentCompilers as values and their names as keys.
    portclock_mapping:
        A dictionary that maps tuples containing a port and a clock to names of
        instruments. The port and clock combinations are unique, but multiple portclocks
        can point to the same instrument.

    Returns
    -------

    Raises
    ------
    RuntimeError
        This exception is raised then the function encountered an operation that has no
        pulse or acquisition info assigned to it.
    """
    # for op_hash, op_data in schedule.operations.items():
    for op_timing_constraint in schedule.timing_constraints:
        op_hash = op_timing_constraint["operation_hash"]
        op_data = schedule.operations[op_hash]
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
                continue  # ignore idle pulse

            uuid = make_hash(without(pulse_data, "t0"))
            combined_data = OpInfo(data=pulse_data, timing=pulse_start_time, uuid=uuid)

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
            uuid = make_hash(hashed_dict)

            combined_data = OpInfo(data=acq_data, timing=acq_start_time, uuid=uuid)
            dev = portclock_mapping[(port, clock)]
            device_compilers[dev].add_acquisition(port, clock, acq_info=combined_data)


def _construct_compiler_objects(
    total_play_time: float,
    mapping: Dict[str, Any],
) -> Dict[str, InstrumentCompiler]:
    """
    Traverses the hardware mapping dictionary and instantiates the appropriate
    instrument compiler objects for all the devices that make up the setup. Local
    oscillators are excluded from this step due to them being defined implicitly in the
    hardware mapping.

    Parameters
    ----------
    total_play_time:
        Total time that it takes to execute a single repetition of the schedule with the
        current hardware setup as defined in the mapping.
    mapping:
        The hardware mapping dictionary.

    Returns
    -------
    :
        A dictionary with an `InstrumentCompiler` as value and the QCoDeS name of the
        instrument the compiler compiles for as key.
    """
    device_compilers = dict()
    for device, dev_cfg in mapping.items():
        if not isinstance(dev_cfg, dict):
            continue
        device_type = dev_cfg["type"]

        device_compiler: Callable = getattr(instrument_compilers, device_type)
        device_compilers[device] = device_compiler(
            device,
            total_play_time,
            mapping[device],
        )
    return device_compilers


def hardware_compile(
    schedule: Schedule, hardware_map: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Main function driving the compilation. The principle behind the overall compilation
    works as follows:

    For every instrument in the hardware mapping, we instantiate a compiler object. Then
    we assign all the pulses/acquisitions that need to be played by that instrument to
    the compiler, which then compiles for each instrument individually.

    This function then returns all the compiled programs bundled together in a
    dictionary with the QCoDeS name of the instrument as key.

    Parameters
    ----------
    schedule:
        The schedule to compile. It is assumed the pulse and acquisition info is
        already added to the operation. Otherwise and exception is raised.
    hardware_map:
        The hardware mapping of the setup.

    Returns
    -------
    :
        The compiled program
    """
    total_play_time = get_total_duration(schedule)

    portclock_map = generate_port_clock_to_device_map(hardware_map)

    device_compilers = _construct_compiler_objects(
        total_play_time=total_play_time,
        mapping=hardware_map,
    )
    _assign_pulse_and_acq_info_to_devices(
        schedule=schedule,
        device_compilers=device_compilers,
        portclock_mapping=portclock_map,
    )

    lo_compilers = generate_ext_local_oscillators(total_play_time, hardware_map)
    _assign_frequencies(
        device_compilers,
        lo_compilers,
        hw_mapping=hardware_map,
        portclock_mapping=portclock_map,
        schedule_resources=schedule.resources,
    )
    device_compilers.update(lo_compilers)

    compiled_schedule = dict()
    for name, compiler in device_compilers.items():
        compiled_dev_program = compiler.compile(repetitions=schedule.repetitions)

        if compiled_dev_program is not None:
            compiled_schedule[name] = compiled_dev_program

    return compiled_schedule
