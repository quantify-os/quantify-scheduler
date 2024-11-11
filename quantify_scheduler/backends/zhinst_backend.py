# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Backend for Zurich Instruments."""

from __future__ import annotations

from quantify_scheduler.compatibility_check import check_zhinst_compatibility

check_zhinst_compatibility()

import logging
import re
import warnings
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    List,
    Literal,
    get_args,
)
from uuid import uuid4

import numpy as np
from pydantic import Field, TypeAdapter, model_validator
from zhinst.toolkit.helpers import Waveform

from quantify_scheduler import enums
from quantify_scheduler.backends.corrections import (
    apply_software_distortion_corrections,
    determine_relative_latency_corrections,
)
from quantify_scheduler.backends.graph_compilation import (
    CompilationConfig,
    SimpleNodeConfig,
)
from quantify_scheduler.backends.types import common, zhinst
from quantify_scheduler.backends.zhinst import helpers as zi_helpers
from quantify_scheduler.backends.zhinst import resolvers, seqc_il_generator
from quantify_scheduler.backends.zhinst import settings as zi_settings
from quantify_scheduler.helpers import schedule as schedule_helpers
from quantify_scheduler.helpers import waveforms as waveform_helpers
from quantify_scheduler.helpers.collections import (
    find_all_port_clock_combinations,
    find_port_clock_path,
)
from quantify_scheduler.helpers.schedule import _extract_port_clocks_used
from quantify_scheduler.instrument_coordinator.components.generic import (
    DEFAULT_NAME as GENERIC_ICC_DEFAULT_NAME,
)
from quantify_scheduler.operations.control_flow_library import ControlFlowOperation
from quantify_scheduler.operations.pulse_library import SetClockFrequency
from quantify_scheduler.schedules.schedule import (
    CompiledSchedule,
    Schedulable,
    Schedule,
    ScheduleBase,
)

if TYPE_CHECKING:
    import pandas

    from quantify_scheduler.operations.operation import Operation
    from quantify_scheduler.resources import Resource

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    # "%(levelname)-8s | %(module)s | %(funcName)s::%(lineno)s. %(message)s"
)
handler.setFormatter(formatter)
logger.addHandler(handler)


# List of supported zhinst devices
SUPPORTED_DEVICE_TYPES: list[str] = ["HDAWG", "UHFQA"]
SUPPORTED_ACQ_PROTOCOLS = Literal["Trace", "SSBIntegrationComplex"]

# https://www.zhinst.com/sites/default/files/documents/2020-09/ziHDAWG_UserManual_20.07.1.pdf
# Section: 3.4. Basic Qubit Characterization, page 83
# All waveform lengths need to be multiples of 16 sample-clock cycles to comply
# with the waveform granularity specification.
WAVEFORM_GRANULARITY: dict[zhinst.DeviceType, int] = {
    zhinst.DeviceType.HDAWG: 16,
    zhinst.DeviceType.UHFQA: 16,
}

# https://www.zhinst.com/sites/default/files/documents/2020-09/ziHDAWG_UserManual_20.07.2.pdf
# page: 262
HDAWG_DEVICE_TYPE_CHANNEL_GROUPS: dict[str, dict[int, int]] = {
    "HDAWG8": {
        # Use the outputs in groups of 2. One sequencer program controls 2 outputs.
        0: 2,
        # Use the outputs in groups of 4. One sequencer program controls 4 outputs.
        1: 4,
        # Use the outputs in groups of 8. One sequencer program controls 8 outputs.
        2: 8,
    },
    "HDAWG4": {
        # Use the outputs in groups of 2. One sequencer program controls 2 outputs.
        0: 2,
        # Use the outputs in groups of 4. One sequencer program controls 4 outputs.
        1: 4,
    },
}

DEVICE_SAMPLING_RATES: dict[zhinst.DeviceType, dict[int, int]] = {
    zhinst.DeviceType.HDAWG: zi_helpers.get_sampling_rates(2.4e9),
    zhinst.DeviceType.UHFQA: zi_helpers.get_sampling_rates(1.8e9),
}

# The sequencer clock rate always is 1/8 of the sampling rate
# (Same for UHFQA and HDAWG as of 2 Nov 2021)
CLOCK_SAMPLE_FACTOR = 8

NUM_UHFQA_READOUT_CHANNELS = 10
MAX_QAS_INTEGRATION_LENGTH = 4096


def ensure_no_operations_overlap(timing_table: pandas.DataFrame) -> None:
    """
    Iterates over all hardware_channels in a schedule to determine if any of the pulses
    have overlap.

    Parameters
    ----------
    timing_table:
        a timing table containing the absolute time and duration as well as the hardware
        channels on which these pulses are to be applied.

    Raises
    ------
    ValueError
        If there is overlap between operations.

    """
    for output_ch in timing_table.hardware_channel.unique():
        if output_ch is None:
            continue

        tt_output_ch = timing_table[timing_table["hardware_channel"] == output_ch]
        tt_output_ch = tt_output_ch[tt_output_ch.is_acquisition != False]  # noqa: E712

        end_time = tt_output_ch["abs_time"] + tt_output_ch["duration"]
        # if any have overlap
        if (end_time.shift(1) > tt_output_ch["abs_time"]).any():
            clashing_ops = tt_output_ch[(end_time.shift(1) > tt_output_ch["abs_time"])]
            clashing_op = clashing_ops.iloc[0]
            preceding_op = tt_output_ch.loc[clashing_op.name - 1]

            raise ValueError(
                f"Operation {clashing_op.operation} at time"
                f" {clashing_op.abs_time*1e9:.1f} ns "
                f"overlaps with {preceding_op.operation} at "
                f"time {preceding_op.abs_time*1e9:.1f} ns "
                f"on output channel {clashing_op.hardware_channel}."
            )


def _extract_port_clock_channelmapping(hardware_cfg: dict[str, Any]) -> dict[str, str]:
    """
    Take the hardware configuration file and return a dictionary that maps port-clock
    pairs to instrument output channels.

    e.g.: {'q0:mw-q0.01': 'ic_hdawg0.channel_0',
           'q0:res-q0.ro': 'ic_uhfqa0.channel_0'}

    """
    port_clock_dict = {}
    for device in hardware_cfg["devices"]:
        instr_name = device["name"]
        for key, value in device.items():
            if "channel_" in key:
                channel_name = key
                channel_idx = channel_name[-1]
                port = value["port"]
                clock = value["clock"]
                # Zurich instruments hardware has "awgs" modules as the channels
                port_clock_dict[f"{port}-{clock}"] = f"{instr_name}.awg{channel_idx}"

    return port_clock_dict


def _determine_clock_sample_start(
    hardware_channel: str,
    abs_time: float,
    operation_name: str = "",
) -> tuple[int, float]:
    """
    depending on the output channel, select the right clock cycle time and sample rate
    from the channel descriptor for ZI channels.
    the sample is returned as a float to preserve information of incorrect rounding to
    full samples if present.
    """
    if "uhfqa" in hardware_channel:
        hw_sample_rate = DEVICE_SAMPLING_RATES[zhinst.DeviceType.UHFQA][
            0
        ]  # 0 -Assumes the default setting for the clock select
        hw_clock_rate = hw_sample_rate / CLOCK_SAMPLE_FACTOR
        # UHFQA has a 4.444 ns clock cycle (8 samples of ~0.55 ns)
        # 9 clock cycles = 40 ns

    elif "hdawg" in hardware_channel:
        hw_sample_rate = DEVICE_SAMPLING_RATES[zhinst.DeviceType.HDAWG][
            0
        ]  # 0 -Assumes the default setting for the clock select
        hw_clock_rate = hw_sample_rate / CLOCK_SAMPLE_FACTOR
        # HDAWG has a 3.333 ns clock cycle (8 samples of ~0.4 ns)
        # 3 clock cycles is 10 ns

    else:
        error_message = (
            f"Instrument type for channel {hardware_channel} not recognized. "
            + "Could not determine clock and sample start."
        )
        raise NotImplementedError(f"{error_message}")

    # next determine what clock cycle and sample things happen on.
    # here we do a combination of floor and round as the samples are added afterwards.
    # The round to 3 decimals serves to catch an edge when rounding to x.999999 clocks.
    clock_cycle = np.floor(np.round(abs_time * hw_clock_rate, decimals=3))

    sample_time = abs_time - clock_cycle / hw_clock_rate
    # first done using floating point to be able to detect incorrect rounding on samples
    sample_float = sample_time * hw_sample_rate

    sample = round(sample_float)
    if not np.all(np.isclose(sample_float, sample, atol=0.0001)):
        # tip, the common clock cycle of UHF and HDAWG is 40 ns, for HDAWG pulses only
        # 10 ns is a nice clock multiple as well.
        raise ValueError(
            f"Rounding to samples not exact for operation ({operation_name}) at time "
            f"({abs_time*1e9:.1f} ns). Attempting to round ({sample_float}) "
            f"to ({sample}) \n TIP: Try to ensure waveforms start a multiple of"
            " the samlping rate e.g., try multiples of 10 ns for the HDAWG or 40 ns for"
            " UFHQA pulses."
        )

    return (clock_cycle, sample_float)


def _determine_clock_start(hardware_channel: str, abs_time: float, operation_name: str) -> float:
    if hardware_channel is None:
        return np.nan
    clock_start, _ = _determine_clock_sample_start(
        hardware_channel=hardware_channel,
        abs_time=abs_time,
        operation_name=operation_name,
    )
    return clock_start


def _determine_sample_start(hardware_channel: str, abs_time: float, operation_name: str) -> float:
    if hardware_channel is None:
        return np.nan
    _, sample_start = _determine_clock_sample_start(
        hardware_channel=hardware_channel,
        abs_time=abs_time,
        operation_name=operation_name,
    )
    return sample_start


def _add_channel_information(
    timing_table: pandas.DataFrame, port_clock_channelmapping: dict
) -> pandas.DataFrame | None:
    """ """

    def map_port_clock_to_channel(port: str, clock: str) -> str:
        if port is None or clock is None:
            return None
        port_clock = f"{port}-{clock}"
        try:
            return port_clock_channelmapping[port_clock]
        except KeyError as e:
            raise KeyError(
                f"Combination of port ({port}) and clock ({clock}) is not mapped to a "
                "channel. Consider double checking the hardware configuration file."
            ) from e

    timing_table["hardware_channel"] = timing_table.apply(
        lambda row: map_port_clock_to_channel(row["port"], row["clock"]), axis=1
    )
    return timing_table


def _apply_latency_corrections(
    timing_table: pandas.DataFrame, latency_dict: dict
) -> pandas.DataFrame:
    """
    Changes the "abs_time" of a timing table depending on the specified latency
    corrections for each port-clock combination as specified in the latency
    dict. The corrections are added to the abs_time elements fulfilling the
    specific port-clock combination.
    """
    for port_clock_combination_key in latency_dict:
        port, clock = port_clock_combination_key.split("-")
        port_mask = timing_table["port"] == port
        clock_mask = timing_table["clock"] == clock
        full_mask_combination = port_mask * clock_mask
        latency_corr = latency_dict[port_clock_combination_key]
        timing_table.loc[full_mask_combination, "abs_time"] += latency_corr

    return timing_table


def _determine_measurement_fixpoint_correction(
    measurement_start_sample: int, common_frequency: float = 600e6
) -> tuple[float, int]:
    """
    Calculates by how much time to shift all operations to ensure a measurement starts
    at sample 0.

    Parameters
    ----------
    measurement_start_sample:
        the sample at which the measurement starts
    common_frequency:
        The shift needs to be such that it occurs at a multiple of the common frequency.
        A larger common frequency results in a smaller time correction.
        This largest common frequency is the common frequency of the HDAWG and UHFQA and
        is 600 MHz.

    Returns
    -------
    tuple[float, int]
        The time correction to be applied in seconds.
        The correction in the number of samples.

    """
    uhf_sampling_rate = 1.8e9
    samples_per_clock_cycle = 8

    uhf_common_sample = uhf_sampling_rate / common_frequency
    if not uhf_common_sample.is_integer():
        raise ValueError(
            f"Invalid common frequency: The UHF sampling rate ({uhf_sampling_rate}) is "
            f"not a multiple of the common frequency {common_frequency}."
        )

    required_sample_correction = (-measurement_start_sample) % samples_per_clock_cycle

    success = False
    for i in range(10):
        sample_correction = int(i * uhf_common_sample)
        if sample_correction % samples_per_clock_cycle == required_sample_correction:
            success = True
            break

    if not success:
        raise ValueError("Could not identify a measurement fixpoint correction")

    time_shift = sample_correction / uhf_sampling_rate

    return time_shift, sample_correction


def _apply_measurement_fixpoint_correction(
    timing_table: pandas.DataFrame, common_frequency: float = 600e6
) -> pandas.DataFrame:
    """
    Updates the abs_time of all operations based on the measurement fixpoint correction.

    The abs_time is applied to all operations between two acquisitions.
    After that the samples and clocks are re-calculated to reflect this change in time.

    Parameters
    ----------
    timing_table:
        A timing table that has the samples already determined.
    common_frequency:
        The shift needs to be such that it occurs at a multiple of the common frequency.
        A larger common frequency results in a smaller time correction.
        This largest common frequency is the common frequency of the HDAWG and UHFQA and
        is 600 MHz.

    """
    acquisitions = timing_table[timing_table.is_acquisition]
    time_corrections = np.zeros(len(timing_table))
    prev_idx = 0
    cumulative_sample_corrections = 0

    # FIXME: there is an edge-case in the slicing of the operations when
    # the pulse of the measurement is applied after triggering the acquisition.
    # this should be included explicitly in the indices to slice (acquisitions.index)
    for idx, sample_start in zip(acquisitions.index, acquisitions.sample_start):
        effective_start_sample = round(sample_start + cumulative_sample_corrections)
        time_corr, sample_corr = _determine_measurement_fixpoint_correction(
            measurement_start_sample=effective_start_sample,
            common_frequency=common_frequency,
        )

        # all subsequent waveforms are shifted in time
        time_corrections[prev_idx:] += time_corr
        cumulative_sample_corrections += sample_corr
        prev_idx = idx + 1

    timing_table["abs_time"] += time_corrections

    # After shifting operations to align the measurement for the fixpoint correction the
    # clock and sample start needs to be updated.
    timing_table = _add_clock_sample_starts(timing_table=timing_table)

    return timing_table


def _add_clock_sample_starts(timing_table: pandas.DataFrame) -> pandas.DataFrame:
    """
    Adds the sequence clock cycle start and sampling start of each operation for each
    channel.
    """
    timing_table["clock_cycle_start"] = timing_table.apply(
        lambda row: _determine_clock_start(
            hardware_channel=row["hardware_channel"],
            abs_time=row["abs_time"],
            operation_name=row["operation"],
        ),
        axis=1,
    )

    timing_table["sample_start"] = timing_table.apply(
        lambda row: _determine_sample_start(
            hardware_channel=row["hardware_channel"],
            abs_time=row["abs_time"],
            operation_name=row["operation"],
        ),
        axis=1,
    )
    return timing_table


def _add_waveform_ids(timing_table: pandas.DataFrame) -> pandas.DataFrame:
    """
    Multiple (numerical) waveforms might be needed to represent a single operation.

    This waveform_id consists of a concatenation of the waveform_op_id with the
    sample_start and modulation phase added to it.

    """

    def _determine_waveform_id(waveform_op_id: str, sample_start: float, phase: float = 0) -> str:
        # acq_index is not part of the waveform this is filtered out from the
        # waveform_id as it doesn't affect the waveform itself.
        waveform_op_id = re.sub(r"acq_index=\(.*\)", "acq_index=(*)", waveform_op_id)
        waveform_op_id = re.sub(r"acq_index=.*,", "acq_index=*,", waveform_op_id)
        waveform_op_id = re.sub(r"acq_index=.*\)", "acq_index=*,", waveform_op_id)

        # samples should always be positive, the abs is here to catch a rare bug
        # where a very small negative number (e.g., -0.00000000000000013) is printed
        # as -0.0 causing conflicting waveform_ids for the same waveform.
        waveform_id = f"{waveform_op_id}_sample:{abs(sample_start):.1f}_phase:{phase:.1f}"
        return waveform_id

    # N.B. phase is relevant if premodulation is used.
    # calculating the phase is currently not implemented.
    timing_table["waveform_id"] = timing_table.apply(
        lambda row: _determine_waveform_id(row["waveform_op_id"], row["sample_start"], phase=0),
        axis=1,
    )
    return timing_table


def _parse_local_oscillators(data: dict[str, Any]) -> dict[str, zhinst.LocalOscillator]:
    """
    Returns the LocalOscillator domain models parsed from the data dictionary.

    Parameters
    ----------
    data :
        The hardware map "local_oscillators" entry.

    Returns
    -------
    :
        A dictionary of unique LocalOscillator instances.

    Raises
    ------
    RuntimeError
        If duplicate LocalOscillators have been found.

    """
    local_oscillators: dict[str, zhinst.LocalOscillator] = dict()
    lo_list_validator = TypeAdapter(List[zhinst.LocalOscillator])
    lo_list = lo_list_validator.validate_python(data)
    for local_oscillator in lo_list:
        if local_oscillator.unique_name in local_oscillators:
            raise RuntimeError(
                f"Duplicate entry LocalOscillators '{local_oscillator.unique_name}' in "
                "hardware configuration!"
            )

        local_oscillators[local_oscillator.unique_name] = local_oscillator

    return local_oscillators


def _parse_devices(data: dict[str, Any]) -> list[zhinst.Device]:
    device_list_validator = TypeAdapter(List[zhinst.Device])
    device_list = device_list_validator.validate_python(data)

    for device in device_list:
        if device.device_type.value not in SUPPORTED_DEVICE_TYPES:
            raise NotImplementedError(
                f"Unable to create zhinst backend for '{device.device_type.value}'!"
            )

        sample_rates = DEVICE_SAMPLING_RATES[device.device_type]
        if device.clock_select not in sample_rates:
            raise ValueError(
                f"Unknown value clock_select='{device.clock_select}' "
                + f"for device type '{device.device_type.value}'"
            )

        device.sample_rate = sample_rates[device.clock_select]

    return device_list


def _validate_schedule(schedule: Schedule) -> None:
    """
    Validates the CompiledSchedule required values for creating the backend.

    Parameters
    ----------
    schedule :

    Raises
    ------
    ValueError
        The validation error.

    """
    if len(schedule.schedulables) == 0:
        raise ValueError(f"Undefined schedulables for schedule '{schedule.name}'!")

    for schedulable in schedule.schedulables.values():
        if "abs_time" not in schedulable:
            raise ValueError(
                "Absolute timing has not been determined " + f"for the schedule '{schedule.name}'!"
            )
    for op in schedule.operations.values():
        for pulse_data in op.data["pulse_info"]:
            if pulse_data.get("reference_magnitude", None) is not None:
                raise NotImplementedError
        if isinstance(op, (ControlFlowOperation, SetClockFrequency)):
            raise NotImplementedError(f"Operation '{op}' is not supported by the zhinst backend.")


def apply_waveform_corrections(
    output: zhinst.Output,
    waveform: np.ndarray,
    start_and_duration_in_seconds: tuple[float, float],
    instrument_info: zhinst.InstrumentInfo,
    is_pulse: bool,
) -> tuple[int, int, np.ndarray]:
    """
    Add waveform corrections such as modulation, changing the
    waveform starting time by shifting it and resizing it
    based on the Instruments granularity.

    Parameters
    ----------
    output :
    waveform :
    start_and_duration_in_seconds :
    instrument_info :
    is_pulse :
        True if it is a pulse to be up converted, False if it is an integration weight.

    Returns
    -------
    :

    """
    (start_in_seconds, duration_in_seconds) = start_and_duration_in_seconds

    if output.modulation.type == zhinst.ModulationModeType.MODULATE:
        raise NotImplementedError("Hardware real-time modulation is not available yet!")

    if is_pulse:
        # Modulate the waveform
        if output.modulation.type == zhinst.ModulationModeType.PREMODULATE:
            t: np.ndarray = np.arange(0, 0 + duration_in_seconds, 1 / instrument_info.sample_rate)
            waveform = waveform_helpers.modulate_waveform(
                t, waveform, output.modulation.interm_freq
            )
        if output.mixer_corrections is not None:
            waveform = waveform_helpers.apply_mixer_skewness_corrections(
                waveform,
                output.mixer_corrections.amp_ratio,
                output.mixer_corrections.phase_error,
            )
    # in the case where the waveform is an integration weight
    elif output.modulation.type == zhinst.ModulationModeType.PREMODULATE:
        # Modulate the waveform
        t: np.ndarray = np.arange(0, 0 + duration_in_seconds, 1 / instrument_info.sample_rate)
        # N.B. the minus sign with respect to the pulse being applied
        waveform = waveform_helpers.modulate_waveform(
            t, waveform, -1 * output.modulation.interm_freq
        )

        # mixer corrections for the integration are not supported yet.
        # they would belong here.

    start_in_clocks, waveform = waveform_helpers.shift_waveform(
        waveform,
        start_in_seconds,
        instrument_info.sample_rate,
        instrument_info.num_samples_per_clock,
    )
    n_samples_shifted = len(waveform)

    waveform = waveform_helpers.resize_waveform(waveform, instrument_info.granularity)

    return start_in_clocks, n_samples_shifted, waveform


def _get_instruction_list(
    output_timing_table: pandas.DataFrame,
) -> list[zhinst.Instruction]:
    """
    Iterates over a timing table for a specific output for which clock_cycle_start and
    waveform_id have been determined to return a list of all instructions to be played
    on a Zurich Instruments device.
    """
    instruction_list: list[zhinst.Instruction] = []
    for _, row in output_timing_table.iterrows():
        if row.is_acquisition:
            instruction_list.append(
                zhinst.Acquisition(
                    waveform_id=row.waveform_id,
                    abs_time=row.abs_time,
                    duration=row.duration,
                    clock_cycle_start=row.clock_cycle_start,
                )
            )
        else:
            instruction_list.append(
                zhinst.Wave(
                    waveform_id=row.waveform_id,
                    abs_time=row.abs_time,
                    duration=row.duration,
                    clock_cycle_start=row.clock_cycle_start,
                )
            )
    return instruction_list


@dataclass(frozen=True)
class ZIAcquisitionConfig:
    """
    Zurich Instruments acquisition configuration.

    Parameters
    ----------
    bin_mode:

    """

    n_acquisitions: int
    """The number of distinct acquisitions in this experiment."""

    resolvers: dict[int, Callable]
    """
    Resolvers used to retrieve the results from the right UHFQA nodes.

    See also :mod:`~quantify_scheduler.backends.zhinst.resolvers`
    """

    bin_mode: enums.BinMode
    """The bin mode of the acquisitions."""

    acq_protocols: dict[int, SUPPORTED_ACQ_PROTOCOLS]
    """Acquisition protocols for each acquisition channel."""


@dataclass(frozen=True)
class ZIDeviceConfig:
    """Zurich Instruments device configuration."""

    name: str
    """The name of the schedule the config is for."""

    settings_builder: zi_settings.ZISettingsBuilder
    """
    The builder to configure the ZI settings.

    The builder typically includes DAQ and AWG settings.
    """

    acq_config: ZIAcquisitionConfig | None
    """
    Acquisition config for the schedule.

    The acquisition config contains the number of acquisitions and a dictionary of
    resolvers used to retrieve the results from the right UHFQA nodes.
    Note that this part of the config is not needed during prepare, but only during
    the retrieve acquisitions step.
    """


def _generate_legacy_hardware_config(  # noqa: PLR0912, PLR0915
    schedule: Schedule,
    compilation_config: CompilationConfig,
) -> dict:
    """
    Extract the old-style Zhinst hardware config from the CompilationConfig.

    Only the port-clok combinations that are used in the schedule are included in the
    old-style hardware config.

    Parameters
    ----------
    schedule: Schedule
        Schedule from which the port-clock combinations are extracted.
    config: CompilationConfig
        CompilationConfig from which hardware config is extracted.

    Returns
    -------
    hardware_config : dict
        Zhinst hardware configuration.

    Raises
    ------
    KeyError
        If the CompilationConfig.connectivity does not contain a hardware config.
    KeyError
        If the 'frequency_param' is not specified for in a LO config.
    ValueError
        If a value is specified in both the hardware options and the hardware config.
    RuntimeError
        If no external local oscillator is found in the generated zhinst
        hardware configuration.

    """

    hardware_description = compilation_config.hardware_compilation_config.hardware_description
    hardware_options = compilation_config.hardware_compilation_config.hardware_options
    connectivity = compilation_config.hardware_compilation_config.connectivity

    if isinstance(connectivity, dict):
        if "graph" in connectivity:
            raise KeyError(
                "Connectivity contains a dictionary including a 'graph' key, most likely"
                " because the networkx Graph object could not be parsed correctly."
            )
        return connectivity

    port_clocks = _extract_port_clocks_used(operation=schedule)

    hardware_config: dict = {"devices": [], "local_oscillators": []}
    hardware_config["backend"] = "quantify_scheduler.backends.zhinst_backend.compile_backend"

    # Add connectivity information to the hardware config:
    connectivity_graph = compilation_config.hardware_compilation_config.connectivity.graph
    for port, clock in port_clocks:
        connected_nodes = {}
        for node in connectivity_graph:
            if port in node:
                connected_nodes = connectivity_graph[node]
                break

        for connected_node in connected_nodes:
            io_path = connected_node.split(sep=".")
            instrument = io_path[0]

            lo_name = None
            if hardware_description[instrument].instrument_type == "IQMixer":
                # Find which lo is used for this IQ mixer
                lo_name = list(connectivity_graph[instrument + ".lo"])[0].split(sep=".")[0]
                # Find which instrument is connected to if port
                io_path = list(connectivity_graph[instrument + ".if"])[0].split(sep=".")

            # Set port-clock combination in io config:
            instr_indices = [
                i for i, v in enumerate(hardware_config["devices"]) if v["name"] == io_path[0]
            ]
            if len(instr_indices) == 0:
                hardware_config["devices"].append({"name": io_path[0]})
                instr_indices = [-1]
            instr_config = hardware_config["devices"][instr_indices[0]]

            instrument_io = io_path[-1]
            if instrument_io not in instr_config:
                instr_config[instrument_io] = {}
            if (old_port := instr_config[instrument_io].get("port")) or (
                old_clock := instr_config[instrument_io].get("clock")
            ):
                raise ValueError(
                    f"Trying to set port-clock combination {port}-{clock} for "
                    f"instrument channel {io_path} in the hardware config, while "
                    f"it has previously been set to {old_port}-{old_clock}."
                )
            instr_config[instrument_io]["port"] = port
            instr_config[instrument_io]["clock"] = clock
            if lo_name is not None:
                instr_config[instrument_io]["local_oscillator"] = lo_name

    # Sort the devices to ensure deterministic behaviour:
    hardware_config["devices"] = sorted(hardware_config["devices"], key=lambda x: x["name"])

    # Add info from hardware description to hardware config
    if hardware_config.get("local_oscillators") is None:
        hardware_config["local_oscillators"] = []
    if hardware_config.get("devices") is None:
        hardware_config["devices"] = []
    for instr_name, instr_description in hardware_description.items():
        if instr_description.instrument_type in ["UHFQA", "HDAWG4", "HDAWG8"]:
            instr_indices = [
                i for i, v in enumerate(hardware_config["devices"]) if v["name"] == instr_name
            ]
            if len(instr_indices) > 1:
                raise ValueError(
                    f"Device with name {instr_name} can only appear once in "
                    f"the hardware_config['devices'] list."
                )
            if len(instr_indices) == 0:
                # Instrument is not in the hardware config, because it is not
                # used in the schedule.
                continue
            instr_config = hardware_config["devices"][instr_indices[0]]
            instr_config["type"] = instr_description.instrument_type
            instr_config["ref"] = instr_description.ref
            if instr_description.instrument_type in ["HDAWG4", "HDAWG8"]:
                instr_config["channelgrouping"] = instr_description.channelgrouping
                instr_config["clock_select"] = instr_description.clock_select
            if instr_description.channel_0 is not None and "channel_0" in instr_config:
                # Only propagate channel description settings if it was already
                # added based on the Connectivity and the Schedule.
                instr_config["channel_0"]["mode"] = instr_description.channel_0.mode
                instr_config["channel_0"]["markers"] = instr_description.channel_0.markers
                instr_config["channel_0"]["trigger"] = instr_description.channel_0.trigger

            if (
                instr_description.instrument_type in ["HDAWG4", "HDAWG8"]
                and instr_description.channel_1 is not None
                and "channel_1" in instr_config
            ):
                # Only propagate channel description settings if it was already
                # added based on the Connectivity and the Schedule.
                instr_config["channel_1"]["mode"] = instr_description.channel_1.mode
                instr_config["channel_1"]["markers"] = instr_description.channel_1.markers
                instr_config["channel_1"]["trigger"] = instr_description.channel_1.trigger
            if (
                instr_description.instrument_type == "HDAWG8"
                and instr_description.channel_2 is not None
                and "channel_2" in instr_config
            ):
                # Only propagate channel description settings if it was already
                # added based on the Connectivity and the Schedule.
                instr_config["channel_2"]["mode"] = instr_description.channel_2.mode
                instr_config["channel_2"]["markers"] = instr_description.channel_2.markers
                instr_config["channel_2"]["trigger"] = instr_description.channel_2.trigger
            if (
                instr_description.instrument_type == "HDAWG8"
                and instr_description.channel_3 is not None
                and "channel_3" in instr_config
            ):
                # Only propagate channel description settings if it was already
                # added based on the Connectivity and the Schedule.
                instr_config["channel_3"]["mode"] = instr_description.channel_3.mode
                instr_config["channel_3"]["markers"] = instr_description.channel_3.markers
                instr_config["channel_3"]["trigger"] = instr_description.channel_3.trigger

        elif instr_description.instrument_type == "LocalOscillator":
            lo_indices = [
                i
                for i, v in enumerate(hardware_config["local_oscillators"])
                if v["unique_name"] == instr_name
            ]
            if len(lo_indices) > 1:
                raise ValueError(
                    f"LocalOscillator with name {instr_name} appears multiple "
                    f"times in the hardware_config['local_oscillators'] list."
                )
            elif len(lo_indices) == 0:
                hardware_config["local_oscillators"].append({})
                lo_indices = [len(hardware_config["local_oscillators"]) - 1]
            lo_config = hardware_config["local_oscillators"][lo_indices[0]]
            lo_config["unique_name"] = instr_name
            lo_config["instrument_name"] = instr_description.instrument_name
            lo_config["frequency_param"] = instr_description.frequency_param
            # Set the LO power in the LO config:
            if "power" not in lo_config:
                lo_config["power"] = {}
            elif isinstance(lo_config["power"], int):
                lo_config["power"] = {"power": lo_config["power"]}
            lo_config["power"][instr_description.power_param] = instr_description.power

    # Set hardware options in the hardware config
    if hardware_options.latency_corrections is not None:
        hardware_config["latency_corrections"] = hardware_options.model_dump()[
            "latency_corrections"
        ]
    if hardware_options.distortion_corrections is not None:
        hardware_config["distortion_corrections"] = hardware_options.model_dump()[
            "distortion_corrections"
        ]

    modulation_frequencies = hardware_options.modulation_frequencies
    if modulation_frequencies is not None:
        for port, clock in port_clocks:
            if (pc_mod_freqs := modulation_frequencies.get(f"{port}-{clock}")) is None:
                # No modulation frequencies to set for this port-clock.
                continue
            # Find path to port-clock combination in the hardware config, e.g.,
            # ["devices", 0, "channel_0"]
            ch_path = find_port_clock_path(hardware_config=hardware_config, port=port, clock=clock)
            # Extract channel config dict:
            ch_config = hardware_config
            for key in ch_path:
                ch_config = ch_config[key]
            if "modulation" not in ch_config:
                # Initialize modulation config:
                ch_config["modulation"] = {"type": "premod"}
            # Set the interm_freq in the modulation config:
            ch_config["modulation"]["interm_freq"] = pc_mod_freqs.interm_freq
            # Find the LO config and add the frequency config:
            lo_name: str = ch_config["local_oscillator"]
            lo_configs: list = hardware_config.get("local_oscillators", [])
            lo_config_found = False
            for lo_config in lo_configs:
                if lo_config["unique_name"] == lo_name:
                    lo_config_found = True
                    if "frequency_param" not in lo_config:
                        raise KeyError(
                            f"Frequency parameter for {lo_name} not found in the"
                            f" hardware config. Please specify it under the "
                            f" 'frequency_param' key in {lo_config=}."
                        )
                    if "frequency" not in lo_config:
                        # Initialize frequency config dict:
                        lo_config["frequency"] = {}
                    lo_freq_key = lo_config.get("frequency_param")
                    # Set the LO freq in the LO config:
                    lo_config["frequency"][lo_freq_key] = pc_mod_freqs.lo_freq
            if not lo_config_found:
                raise RuntimeError(
                    f"External local oscillator '{lo_name}' set to "
                    f"be used for {port=} and {clock=} not found! Make "
                    f"sure it is present in the hardware configuration."
                )

    mixer_corrections = hardware_options.mixer_corrections
    if mixer_corrections is not None:
        for port, clock in port_clocks:
            if (pc_mix_corr := mixer_corrections.get(f"{port}-{clock}")) is None:
                # No mixer corrections to set for this port-clock.
                continue
            # Find path to port-clock combination in the hardware config, e.g.,
            # ["devices", 0, "channel_0"]
            ch_path = find_port_clock_path(hardware_config=hardware_config, port=port, clock=clock)
            # Extract channel config dict:
            ch_config = hardware_config
            for key in ch_path:
                ch_config = ch_config[key]
            # Set mixer corrections from hw options in channel config dict:
            ch_config["mixer_corrections"] = pc_mix_corr.model_dump()

    output_gain = hardware_options.output_gain
    if output_gain is not None:
        for port, clock in find_all_port_clock_combinations(hardware_config):
            if (pc_output_gain := output_gain.get(f"{port}-{clock}")) is None:
                # No modulation frequencies to set for this port-clock.
                continue
            # Find path to port-clock combination in the hardware config, e.g.,
            # ["devices", 0, "channel_0"]
            ch_path = find_port_clock_path(hardware_config=hardware_config, port=port, clock=clock)
            # Extract instrument config and I/O channel config dicts:
            instr_config = hardware_config
            for key in ch_path[:-1]:
                instr_config = instr_config[key]
            ch_name = ch_path[-1]
            ch_config = instr_config[ch_name]

            # Set the output gain in the channel config:
            ch_config["gain1"] = pc_output_gain.gain_I
            ch_config["gain2"] = pc_output_gain.gain_Q

    return hardware_config


def _generate_new_style_hardware_compilation_config(  # noqa: PLR0912, PLR0915
    old_style_config: dict,
) -> dict:
    """
    Generate a new-style ZIHardwareCompilationConfig from an old-style hardware config.

    Parameters
    ----------
    old_style_config
        Old-style hardware config.

    Returns
    -------
    dict
        New-style hardware compilation config dictionary.

    """
    # Avoid modifying the original config
    old_style_config = deepcopy(old_style_config)

    # Initialize input dicts (before validation)
    hardware_description = {}
    hardware_options = defaultdict(lambda: defaultdict(dict))
    connectivity = {"graph": []}

    # Loop over old-style config and populate new-style input dicts
    for hw_cfg_key, hw_cfg_value in old_style_config.items():
        if hw_cfg_key == "backend":
            pass
        elif hw_cfg_key in ["latency_corrections", "distortion_corrections"]:
            hardware_options[hw_cfg_key] = hw_cfg_value
        elif hw_cfg_key == "local_oscillators":  # noqa: SIM114
            for lo_cfg in hw_cfg_value:
                power_cfg = lo_cfg.get("power")
                if power_cfg:
                    power_param = list(power_cfg.keys())[0]
                    power = power_cfg[power_param]
                else:
                    power_param = "power"
                    power = None
                hardware_description[lo_cfg["unique_name"]] = {
                    "instrument_type": "LocalOscillator",
                    "instrument_name": lo_cfg.get("instrument_name"),
                    "generic_icc_name": lo_cfg.get("generic_icc_name"),
                    "frequency_param": lo_cfg.get("frequency_param", "frequency"),
                    "power_param": power_param,
                    "power": power,
                }
        elif hw_cfg_key == "devices":
            for instr_cfg in hw_cfg_value:
                if instr_cfg["type"] in ["HDAWG4", "HDAWG8"]:
                    # Hardware Description
                    hardware_description[instr_cfg["name"]] = {
                        "instrument_type": instr_cfg["type"],
                        "ref": instr_cfg["ref"],
                        "channelgrouping": instr_cfg["channelgrouping"],
                        "clock_select": instr_cfg["clock_select"],
                    }
                    for ch_idx in [0, 1, 2, 3]:
                        if ch_cfg := instr_cfg.get(f"channel_{ch_idx}"):
                            hardware_description[instr_cfg["name"]][f"channel_{ch_idx}"] = {
                                "mode": ch_cfg["mode"],
                                "markers": ch_cfg.get("markers", []),
                                "trigger": ch_cfg.get("trigger"),
                            }
                        else:
                            continue
                        # Connectivity
                        port = ch_cfg.get("port")
                        clock = ch_cfg.get("clock")
                        port_name = f"{instr_cfg['name']}.channel_{ch_idx}"
                        if ch_cfg.get("local_oscillator") is not None:
                            # Add IQ mixer to the hardware_description:
                            iq_mixer_id = f"iqm_{instr_cfg['name']}_ch{ch_idx}"
                            hardware_description[iq_mixer_id] = {"instrument_type": "IQMixer"}
                            # Add LO and IQ mixer to connectivity graph:
                            connectivity["graph"].extend(
                                [
                                    (
                                        port_name,
                                        f"{iq_mixer_id}.if",
                                    ),
                                    (
                                        f"{ch_cfg['local_oscillator']}.output",
                                        f"{iq_mixer_id}.lo",
                                    ),
                                ]
                            )
                            # Overwrite port_name to IQ mixer RF output:
                            port_name = f"{iq_mixer_id}.rf"
                            # Set LO frequency in hardware options:
                            lo_cfg = [
                                lo_cfg
                                for lo_cfg in old_style_config["local_oscillators"]
                                if lo_cfg["unique_name"] == ch_cfg["local_oscillator"]
                            ][0]
                            if "frequency" in lo_cfg:
                                hardware_options["modulation_frequencies"][f"{port}-{clock}"][
                                    "lo_freq"
                                ] = list(lo_cfg["frequency"].values())[0]
                        connectivity["graph"].append((port_name, port))
                        # Hardware Options
                        if ch_cfg.get("mixer_corrections"):
                            # Decapitalize DC offset keys:
                            mix_corr_lower_case = {}
                            for key in ch_cfg["mixer_corrections"]:
                                mix_corr_lower_case[key.lower()] = ch_cfg["mixer_corrections"][key]
                            hardware_options["mixer_corrections"][
                                f"{port}-{clock}"
                            ] = mix_corr_lower_case
                        if ch_cfg.get("gain1"):
                            hardware_options["output_gain"][f"{port}-{clock}"]["gain_I"] = ch_cfg[
                                "gain1"
                            ]
                        if ch_cfg.get("gain2"):
                            hardware_options["output_gain"][f"{port}-{clock}"]["gain_Q"] = ch_cfg[
                                "gain2"
                            ]
                        if ch_cfg.get("modulation") and ch_cfg["modulation"]["type"] == "premod":
                            hardware_options["modulation_frequencies"][f"{port}-{clock}"][
                                "interm_freq"
                            ] = ch_cfg["modulation"]["interm_freq"]
                elif instr_cfg["type"] == "UHFQA":
                    # Hardware Description
                    hardware_description[instr_cfg["name"]] = {
                        "instrument_type": instr_cfg["type"],
                        "ref": instr_cfg["ref"],
                    }
                    if ch_cfg := instr_cfg.get("channel_0"):
                        hardware_description[instr_cfg["name"]]["channel_0"] = {
                            "mode": ch_cfg["mode"],
                            "markers": ch_cfg.get("markers", []),
                            "trigger": ch_cfg.get("trigger"),
                        }
                        # Connectivity
                        port = ch_cfg.get("port")
                        clock = ch_cfg.get("clock")
                        port_name = f"{instr_cfg['name']}.{'channel_0'}"
                        if ch_cfg.get("local_oscillator") is not None:
                            # Add IQ mixer to the hardware_description:
                            iq_mixer_id = f"iqm_{instr_cfg['name']}_ch0"
                            hardware_description[iq_mixer_id] = {"instrument_type": "IQMixer"}
                            # Add LO and IQ mixer to connectivity graph:
                            connectivity["graph"].extend(
                                [
                                    (
                                        port_name,
                                        f"{iq_mixer_id}.if",
                                    ),
                                    (
                                        f"{ch_cfg['local_oscillator']}.output",
                                        f"{iq_mixer_id}.lo",
                                    ),
                                ]
                            )
                            # Overwrite port_name to IQ mixer RF output:
                            port_name = f"{iq_mixer_id}.rf"
                            # Set LO frequency in hardware options:
                            lo_cfg = [
                                lo_cfg
                                for lo_cfg in old_style_config["local_oscillators"]
                                if lo_cfg["unique_name"] == ch_cfg["local_oscillator"]
                            ][0]
                            if "frequency" in lo_cfg:
                                hardware_options["modulation_frequencies"][f"{port}-{clock}"][
                                    "lo_freq"
                                ] = list(lo_cfg["frequency"].values())[0]
                        connectivity["graph"].append((port_name, port))
                        # Hardware Options
                        if ch_cfg.get("mixer_corrections"):
                            hardware_options["mixer_corrections"][f"{port}-{clock}"] = ch_cfg[
                                "mixer_corrections"
                            ]
                        if ch_cfg.get("gain1"):
                            hardware_options["output_gain"][f"{port}-{clock}"]["gain_I"] = ch_cfg[
                                "gain1"
                            ]
                        if ch_cfg.get("gain2"):
                            hardware_options["output_gain"][f"{port}-{clock}"]["gain_Q"] = ch_cfg[
                                "gain2"
                            ]
                        if ch_cfg.get("modulation") and ch_cfg["modulation"]["type"] == "premod":
                            hardware_options["modulation_frequencies"][f"{port}-{clock}"][
                                "interm_freq"
                            ] = ch_cfg["modulation"]["interm_freq"]
                else:
                    raise ValueError(
                        f"Unexpected instrument type {instr_cfg['type']} in old-style "
                        f"hardware config."
                    )
        elif hw_cfg_key == "mode":
            pass
        else:
            warnings.warn(f"Skipping unknown key {hw_cfg_key} in old-style hardware config.")

    return dict(
        config_type="quantify_scheduler.backends.zhinst_backend.ZIHardwareCompilationConfig",
        hardware_description=hardware_description,
        hardware_options=hardware_options,
        connectivity=connectivity,
    )


def flatten_schedule(schedule: Schedule, config: CompilationConfig | None = None) -> Schedule:
    """
    Recursively flatten subschedules based on the absolute timing.

    Parameters
    ----------
    schedule : Schedule
        schedule to be flattened
    config : CompilationConfig | None, optional
        Compilation config for
        :class:`~quantify_scheduler.backends.graph_compilation.QuantifyCompiler`,
        which is currently not only used to detect if the function is called directly.
        by default None

    Returns
    -------
    Schedule
        Equivalent schedule without subschedules

    """

    def _insert_op_at_time(schedule: Schedule, operation: Operation, abs_time: float) -> None:
        new_key = str(uuid4())
        new_schedulable = Schedulable(
            name=new_key,
            operation_id=operation.hash,
        )
        # Timing constraints in the new schedulable are meaningless, so remove the list
        new_schedulable["timing_constraints"] = None
        new_schedulable["abs_time"] = abs_time
        schedule["operation_dict"][operation.hash] = operation
        schedule["schedulables"][new_key] = new_schedulable

    def _move_to_end(ordered_dict: dict, key: Any) -> None:  # noqa: ANN401
        """
        Moves the element with ``key`` to the end of the dict.

        Note: dictionaries from Python 3.7 are ordered.
        """
        value = ordered_dict.pop(key)
        ordered_dict[key] = value

    all_resources = dict(schedule.resources)
    for op in schedule.operations.values():
        if isinstance(op, ScheduleBase):
            flatten_schedule(op, config)
            all_resources.update(op.resources)

    op_keys_to_pop = set()
    schedulable_keys_to_pop = set()
    # we cannot use .items() directly since we modify schedule.schedulables in the loop
    schedulable_iter = tuple(schedule.schedulables.items())
    for schedulable_key, schedulable in schedulable_iter:
        op_key = schedulable["operation_id"]
        op = schedule.operations[op_key]
        if isinstance(op, ScheduleBase):
            offset = schedulable["abs_time"]

            # insert new schedulables shifted by the correct offset
            for inner_schedulable in op.schedulables.values():
                inner_op = op.operations[inner_schedulable["operation_id"]]
                _insert_op_at_time(schedule, inner_op, inner_schedulable["abs_time"] + offset)

            # mark the inner schedule for removal from the parent
            op_keys_to_pop.add(op_key)
            schedulable_keys_to_pop.add(schedulable_key)
        else:
            _move_to_end(schedule.schedulables, schedulable_key)

    for key in op_keys_to_pop:
        schedule["operation_dict"].pop(key)
    for key in schedulable_keys_to_pop:
        schedule["schedulables"].pop(key)

    for resource in all_resources.values():
        if resource.name not in schedule.resources:
            schedule.add_resource(resource)

    return schedule


def _get_operations_by_repr(schedule: Schedule) -> dict[str, Operation]:
    operations_dict_with_repr_keys = {str(op): op for op in schedule.operations.values()}
    if len(schedule.operations) != len(operations_dict_with_repr_keys):
        all_reprs = set()
        colliding_ops = []
        for operation in schedule.operations.values():
            repr_key = str(operation)
            if repr_key in all_reprs:
                colliding_ops.append(operation)
            all_reprs.add(repr_key)
        raise ValueError(
            "Multiple operations with different content have the same repr."
            f"Colliding operations are {colliding_ops}."
        )
    return operations_dict_with_repr_keys


def compile_backend(  # noqa: PLR0912
    schedule: Schedule,
    config: CompilationConfig | dict[str, Any] | None = None,
    # config can be Dict to support (deprecated) calling with hardware config
    # as positional argument.
    *,  # Support for (deprecated) calling with hardware_cfg as keyword argument:
    hardware_cfg: dict[str, Any] | None = None,
) -> CompiledSchedule:
    """
    Compiles zhinst hardware instructions for a schedule.

    This method generates sequencer programs, waveforms and
    configurations required for the instruments defined in
    the hardware configuration.

    Parameters
    ----------
    schedule :
        The schedule to be compiled.
    config
        Compilation config for
        :class:`~quantify_scheduler.backends.graph_compilation.QuantifyCompiler`.
    hardware_cfg :
        (deprecated) The hardware configuration of the setup. Pass a full compilation
        config instead using ``config`` argument.

    Returns
    -------
    :
        A collection containing the compiled backend
        configuration for each device.

    Raises
    ------
    NotImplementedError
        Thrown when using unsupported ZI Instruments.
    ValueError
        When both ``config`` and ``hardware_cfg`` are supplied.

    """
    if not ((config is not None) ^ (hardware_cfg is not None)):
        raise ValueError(
            f"Zhinst `{compile_backend.__name__}` was called with {config=} and "
            f"{hardware_cfg=}. Please make sure this function is called with "
            f"one of the two (CompilationConfig recommended)."
        )
    if not isinstance(config, CompilationConfig):
        warnings.warn(
            f"Zhinst `{compile_backend.__name__}` will require a full "
            f"CompilationConfig as input as of quantify-scheduler >= 0.19.0",
            FutureWarning,
        )
    if isinstance(config, CompilationConfig):
        # Extract the hardware config from the CompilationConfig
        hardware_cfg = _generate_legacy_hardware_config(
            schedule=schedule, compilation_config=config
        )
    elif config is not None:
        # Support for (deprecated) calling with hardware_cfg as positional argument.
        hardware_cfg = config

    _validate_schedule(schedule)

    if "latency_corrections" in hardware_cfg:
        # Important: currently only used to validate the input, should also be
        # used for storing the latency corrections
        # (see also https://gitlab.com/groups/quantify-os/-/epics/1)
        common.HardwareOptions(latency_corrections=hardware_cfg["latency_corrections"])

    if (distortion_corrections := hardware_cfg.get("distortion_corrections")) is not None:
        replacing_schedule = apply_software_distortion_corrections(schedule, distortion_corrections)
        if replacing_schedule is not None:
            schedule = replacing_schedule

    ################################################
    # Timing table manipulation
    ################################################

    # the schedule has a Styled pandas dataframe as the return type.
    # here we want to manipulate the data directly so we extract the raw dataframe.
    timing_table = schedule.timing_table.data

    # information is added on what output channel is used for every pulse and acq.
    port_clock_channelmapping = _extract_port_clock_channelmapping(hardware_cfg)
    timing_table = _add_channel_information(
        timing_table=timing_table, port_clock_channelmapping=port_clock_channelmapping
    )

    # the timing of all pulses and acquisitions is corrected based on the latency corr.
    latency_dict = determine_relative_latency_corrections(hardware_cfg)
    timing_table = _apply_latency_corrections(timing_table=timing_table, latency_dict=latency_dict)

    # ensure that operations are still sorted by time after applying the latency corr.
    timing_table.sort_values("abs_time", inplace=True)

    # add the sequencer clock cycle start and sampling start for the operations.
    timing_table = _add_clock_sample_starts(timing_table=timing_table)

    # After adjusting for the latencies, the fix-point correction can be applied.
    # the fix-point correction has the goal to ensure that all measurement operations
    # will always start at a multiple of *all* relevant clock domains.
    # this is achieved by shifting all instructions between different measurements
    # by the same amount of samples.
    timing_table = _apply_measurement_fixpoint_correction(
        timing_table=timing_table, common_frequency=600e6
    )

    # because of the shifting in time on a sub-clock delay, up to 8 distinct waveforms
    # may be required to realize the identical pulse. Pre-modulation adds another
    # variant depending on the starting phase of the operation.
    timing_table = _add_waveform_ids(timing_table=timing_table)

    ensure_no_operations_overlap(timing_table)

    # Parse the hardware configuration file, zhinst.Device is a dataclass containing
    # device descriptions (name, type, channels etc. )
    devices: list[zhinst.Device] = _parse_devices(hardware_cfg["devices"])

    local_oscillators: dict[str, zhinst.LocalOscillator] = _parse_local_oscillators(
        hardware_cfg["local_oscillators"]
    )

    ################################################
    # Constructing the waveform table
    ################################################

    device_dict = {}
    for dev in devices:
        device_dict[dev.name] = dev

    operations_dict_with_repr_keys = _get_operations_by_repr(schedule)

    numerical_wf_dict = construct_waveform_table(
        timing_table,
        operations_dict=operations_dict_with_repr_keys,
        device_dict=device_dict,
    )

    ################################################
    # Above here is the layer that translates what should happen at the device to what
    # output needs to be generated to realize that.

    # COMPILATION SHOULD BE SPLIT HERE

    # Below here is the layer that translates the timing table to instructions for the
    # hardware.
    ################################################

    ################################################
    # Assemble waveforms and timeline into seqc
    ################################################

    # keys are instrument names, and the ZIDeviceConfig contain the settings incl seqc
    # to configure.
    device_configs: dict[str, ZIDeviceConfig | float] = dict()

    for device in devices:
        if device.device_type == zhinst.DeviceType.HDAWG:
            builder = _compile_for_hdawg(
                device=device,
                timing_table=timing_table,
                numerical_wf_dict=numerical_wf_dict,
                repetitions=schedule.repetitions,
            )
            acq_config: ZIAcquisitionConfig | None = None

        elif device.device_type == zhinst.DeviceType.UHFQA:
            acq_metadata = schedule_helpers.extract_acquisition_metadata_from_schedule(schedule)
            bin_mode = acq_metadata.bin_mode

            builder, acq_config = _compile_for_uhfqa(
                device=device,
                timing_table=timing_table,
                numerical_wf_dict=numerical_wf_dict,
                repetitions=schedule.repetitions,
                operations=operations_dict_with_repr_keys,
                bin_mode=bin_mode,
            )

        else:
            raise NotImplementedError(f"{device.device_type} not supported.")

        device_configs[device.name] = ZIDeviceConfig(device.name, builder, acq_config)

        # add the local oscillator config by iterating over all output channels.
        # note that not all output channels have an LO associated to them.
        for channel in device.channels:
            _add_lo_config(
                channel=channel,
                local_oscillators=local_oscillators,
                device_configs=device_configs,
                resources=schedule.resources,
            )

    # Create compiled instructions key if not already present. This can happen if this
    # compilation function is called directly instead of through a `QuantifyCompiler`.
    if "compiled_instructions" not in schedule:
        schedule["compiled_instructions"] = {}
    schedule["compiled_instructions"].update(device_configs)
    schedule._hardware_timing_table = timing_table
    schedule._hardware_waveform_dict = numerical_wf_dict
    compiled_schedule = CompiledSchedule(schedule)
    return compiled_schedule


class ZIHardwareCompilationConfig(common.HardwareCompilationConfig):
    """
    Datastructure containing the information needed to compile to the Zurich Instruments backend.

    This information is structured in the same way as in the generic
    :class:`~quantify_scheduler.backends.types.common.HardwareCompilationConfig`, but
    contains fields for hardware-specific settings.
    """

    config_type: type[ZIHardwareCompilationConfig] = Field(
        default="quantify_scheduler.backends.zhinst_backend.ZIHardwareCompilationConfig",
        validate_default=True,
    )
    """
    A reference to the ``HardwareCompilationConfig`` DataStructure for the
    Zurich Instruments backend.
    """
    hardware_description: dict[
        str,
        zhinst.ZIHardwareDescription | common.HardwareDescription,
    ]
    """Description of the instruments in the physical setup."""
    hardware_options: zhinst.ZIHardwareOptions
    """
    Options that are used in compiling the instructions for the hardware, such as
    :class:`~quantify_scheduler.backends.types.common.LatencyCorrection` or
    :class:`~quantify_scheduler.backends.types.zhinst.OutputGain`.
    """
    compilation_passes: list[SimpleNodeConfig] = [
        SimpleNodeConfig(name="flatten_schedule", compilation_func=flatten_schedule),
        SimpleNodeConfig(name="zhinst_hardware_compile", compilation_func=compile_backend),
    ]
    """
    The list of compilation nodes that should be called in succession to compile a
    schedule to instructions for the Zurich Instruments hardware.
    """

    @model_validator(mode="before")
    @classmethod
    def from_old_style_hardware_config(
        cls: type[ZIHardwareCompilationConfig], data: Any  # noqa: ANN401
    ) -> Any:  # noqa: ANN401
        """Convert old style hardware config dict to new style before validation."""
        if (
            isinstance(data, dict)
            and data.get("backend") == "quantify_scheduler.backends.zhinst_backend.compile_backend"
        ):
            # Input is an old style ZI hardware config dict
            data = _generate_new_style_hardware_compilation_config(data)

        return data


def _add_lo_config(  # noqa: PLR0912
    channel: zhinst.Output,
    local_oscillators: dict[str, zhinst.LocalOscillator],
    resources: dict[str, Resource],
    device_configs: dict[str, ZIDeviceConfig | float],
) -> None:
    """
    Adds configuration for a local oscillator required for a specific output channel to
    the device configs.
    """
    if channel.local_oscillator is None:
        # no local oscillator is associated to this channel.
        return
    unique_name = channel.local_oscillator

    if unique_name not in local_oscillators:
        raise KeyError(f'Missing configuration for LocalOscillator "{unique_name}"')

    local_oscillator = local_oscillators[unique_name]

    # the frequencies from the config file
    ((lo_freq_key, lo_freq_val),) = local_oscillator.frequency.items()

    # Get the power of the local oscillator
    if local_oscillator.power:
        ((power_key, power_val),) = local_oscillator.power.items()

    # Get the phase of the local oscillator
    if local_oscillator.phase:
        ((phase_key, phase_val),) = local_oscillator.phase.items()

    interm_freq = channel.modulation.interm_freq

    if (lo_freq_val is not None) and (interm_freq is not None):
        rf_freq = lo_freq_val + interm_freq
    else:
        channel_clock_resource = resources.get(channel.clock)
        if channel_clock_resource is not None:
            rf_freq = channel_clock_resource.get("freq")
        else:
            # no clock is specified for this channel.
            # this can happen for e.g., baseband pulses or when the channel is not used
            # in the schedule.
            return

    if lo_freq_val is None and interm_freq is not None:
        lo_freq_val = rf_freq - interm_freq
        local_oscillator.frequency[lo_freq_key] = lo_freq_val

    elif interm_freq is None and lo_freq_val is not None:
        interm_freq = rf_freq - lo_freq_val
        channel.modulation.interm_freq = interm_freq

    elif interm_freq is None and lo_freq_val is None:
        raise ValueError(
            "Either local oscillator frequency or channel intermediate frequency "
            f'must be set for LocalOscillator "{unique_name}"'
        )

    if (
        local_oscillator.unique_name in device_configs
        and device_configs[local_oscillator.unique_name].get("frequency") != lo_freq_val
    ):
        # the device_config currently only contains the frequency
        raise ValueError(f'Multiple frequencies assigned to LocalOscillator "{unique_name}"')

    lo_config = {
        f"{local_oscillator.instrument_name}.{lo_freq_key}": lo_freq_val,
    }

    if local_oscillator.power:
        lo_config[f"{local_oscillator.instrument_name}.{power_key}"] = power_val

    if local_oscillator.phase:
        lo_config[f"{local_oscillator.instrument_name}.{phase_key}"] = phase_val

    # This line detects if the generic_icc_name exists in the local_oscillator entry of
    # the hardware_config. If it exists, then, it takes the entry value, if not, the
    # generic_icc_name takes the default value from the generic icc base module.
    if local_oscillator.generic_icc_name:
        generic_icc_name = local_oscillator.generic_icc_name
    else:
        generic_icc_name = GENERIC_ICC_DEFAULT_NAME

    if generic_icc_name in device_configs:
        device_configs[generic_icc_name].update(lo_config)
    else:
        device_configs[generic_icc_name] = lo_config


def _add_wave_nodes(
    device_type: zhinst.DeviceType,
    awg_index: int,
    wf_id_mapping: dict[str, int],
    numerical_wf_dict: dict[str, np.ndarray],
    settings_builder: zi_settings.ZISettingsBuilder,
) -> zi_settings.ZISettingsBuilder:
    for wf_id, wf_index in wf_id_mapping.items():
        if wf_id not in numerical_wf_dict:
            # this is to catch an edge-case where certain acquisitions do not set
            # integration weights. Ideally, these should be filtered before the wf_id
            # is added to the wf_id_mapping, but it is easier to catch here.
            continue
        numerical_waveform = numerical_wf_dict[wf_id]
        waveform = Waveform(numerical_waveform.real, numerical_waveform.imag)
        if device_type == zhinst.DeviceType.UHFQA:
            settings_builder.with_csv_wave_vector(awg_index, wf_index, waveform.data)
        else:
            settings_builder.with_wave_vector(awg_index, wf_index, waveform.data)
    return settings_builder


def _compile_for_hdawg(
    device: zhinst.Device,
    timing_table: pandas.DataFrame,
    numerical_wf_dict: dict[str, np.ndarray],
    repetitions: int,
) -> zi_settings.ZISettingsBuilder:
    """

    Parameters
    ----------
    device :
    timing_table :
    numerical_wf_dict :
    repetitions :

    Raises
    ------
    ValueError

    """
    # calculating duration over all operations instead of only the last ensures a
    # long operation near the end does not get overlooked.
    schedule_duration = (timing_table.abs_time + timing_table.duration).max()

    ########################################
    # Add standard settings to builder
    ########################################

    settings_builder = zi_settings.ZISettingsBuilder()

    n_awgs: int = int(device.n_channels / 2)
    settings_builder.with_defaults(
        [
            ("sigouts/*/on", 0),
            ("awgs/*/single", 1),
        ]
    ).with_system_channelgrouping(device.channelgrouping)

    # Set the clock-rate of an AWG
    for awg_index in range(n_awgs):
        settings_builder.with_awg_time(awg_index, device.clock_select)  # type: ignore

    # device.type is either HDAWG8 or HDAWG4
    channelgroups_mode = HDAWG_DEVICE_TYPE_CHANNEL_GROUPS[device.type]
    # Defaults to mode =0 -> value = 2 -> sequencers control pairs of channels
    channelgroups_value = channelgroups_mode[device.channelgrouping]
    sequencer_step = int(channelgroups_value / 2)  # nr of awg pairs per sequencer
    # the index of the last sequencer to configure
    # N.B. 8-11-2021 the min(len(device.channels)) might make the wrong choice when
    # using only awgs 2 and 3. To be tested.
    sequencer_stop = min(len(device.channels), int(n_awgs / sequencer_step))

    logger.debug(
        f"HDAWG[{device.name}] devtype={device.device_type} " + f" awg_count={n_awgs} {str(device)}"
    )

    enabled_outputs: dict[int, zhinst.Output] = dict()

    for i, awg_index in enumerate(range(0, sequencer_stop, sequencer_step)):
        # here Output corresponds to an awg unit or a channel pair
        # and is a dataclass containing info on port, clock, gain etc.
        output = device.channels[i]

        if output is None:
            raise ValueError(f"Required output at index '{i}' is undefined!")
        logger.debug(f"[{device.name}-awg{awg_index}] enabling outputs...")
        mixer_corrections = (
            output.mixer_corrections
            if output.mixer_corrections is not None
            else common.MixerCorrections()
        )
        settings_builder.with_sigouts(awg_index, (1, 1)).with_gain(
            awg_index, (output.gain1, output.gain2)
        ).with_sigout_offset(int(awg_index * 2), mixer_corrections.dc_offset_i).with_sigout_offset(
            int(awg_index * 2) + 1, mixer_corrections.dc_offset_q
        )

        enabled_outputs[awg_index] = output

    ############################################
    # Add seqc instructions and waveform table
    ############################################
    for awg_index, output in enabled_outputs.items():
        # select only the instructions relevant for the output channel.
        output_timing_table = timing_table[
            timing_table["hardware_channel"] == f"{device.name}.awg{awg_index}"
        ]

        instructions: list[zhinst.Instruction] = _get_instruction_list(output_timing_table)

        # enumerate the waveform_ids used in this particular output channel
        unique_wf_ids = output_timing_table.drop_duplicates(subset="waveform_id")["waveform_id"]
        # this table maps waveform ids to indices in the seqc command table.
        wf_id_mapping = {}
        for i, wf_id in enumerate(unique_wf_ids):
            wf_id_mapping[wf_id] = i

        # Step 1: Generate and compile sequencer program AND
        # Step 2: Set CommandTable JSON vector

        (seqc, commandtable_json) = _assemble_hdawg_sequence(
            instructions=instructions,
            wf_id_mapping=wf_id_mapping,
            numerical_wf_dict=numerical_wf_dict,
            repetitions=repetitions,
            schedule_duration=schedule_duration,
            markers=device.channels[awg_index].markers,
            trigger=device.channels[awg_index].trigger,
        )

        logger.debug(seqc)
        logger.debug(commandtable_json)

        settings_builder.with_commandtable_data(awg_index, commandtable_json)
        settings_builder.with_compiler_sourcestring(awg_index, seqc)

        #######################################################
        # Set waveforms to wave nodes in the settings builder
        #######################################################

        # Step 3: Upload waveforms to AWG CommandTable
        _add_wave_nodes(
            device_type=zhinst.DeviceType.HDAWG,
            awg_index=awg_index,
            wf_id_mapping=wf_id_mapping,
            numerical_wf_dict=numerical_wf_dict,
            settings_builder=settings_builder,
        )

    return settings_builder


def _assemble_hdawg_sequence(
    instructions: list[zhinst.Instruction],
    wf_id_mapping: dict[str, int],
    numerical_wf_dict: dict[str, np.ndarray],
    repetitions: int,
    schedule_duration: float,
    markers: str | int | None = None,
    trigger: int = None,
) -> tuple[str, str]:
    """ """
    seqc_instructions = ""
    commandtable_json = str({})

    seqc_gen = seqc_il_generator.SeqcILGenerator()

    # Declare sequence variables
    seqc_gen.declare_var("__repetitions__", repetitions)

    ###############################################################
    # Generate the command table and waveforms
    ###############################################################
    command_table_entries: list[zhinst.CommandTableEntry] = list()
    for waveform_id, waveform_index in wf_id_mapping.items():
        name: str = f"w{waveform_index}"
        waveform = numerical_wf_dict[waveform_id]

        # Create and add variables to the Sequence program
        # as well as assign the variables with operations
        seqc_gen.declare_wave(name)
        seqc_gen.assign_placeholder(name, len(waveform))
        seqc_gen.emit_assign_wave_index(name, name, index=waveform_index)

        # Do bookkeeping for the CommandTable
        command_table_entry = zhinst.CommandTableEntry(
            index=len(command_table_entries),
            waveform=zhinst.CommandTableWaveform(index=waveform_index, length=len(waveform)),
        )
        command_table_entries.append(command_table_entry)
    command_table = zhinst.CommandTable(table=command_table_entries)

    ###############################################################
    # Add the loop that executes the program.
    ###############################################################

    # N.B. All HDAWG markers can be used to trigger a UHFQA or other HDAWGs.
    # marker output is set to 0 before the loop is started
    seqc_il_generator.add_set_trigger(seqc_gen, value=0, device_type=zhinst.DeviceType.HDAWG)

    seqc_gen.emit_begin_repeat("__repetitions__")

    current_clock: int = 0

    # set markers to high at the start of the repeition if this is the primary
    # channel or wait for an external trigger if this is a secondary channel
    if markers is not None and len(markers) > 0:
        current_clock += seqc_il_generator.add_set_trigger(
            seqc_gen,
            value=markers,
            device_type=zhinst.DeviceType.HDAWG,
        )
    elif trigger is not None:
        assert trigger in [1, 2]
        seqc_gen.emit_wait_dig_trigger(
            index=trigger,
            comment=f"\t// clock={current_clock}",
            device_type=zhinst.DeviceType.HDAWG,
        )
    else:
        # If the hardware config does not provide any settings assume this is a
        # primary HDAWG channel and send triggers on all channels
        current_clock += seqc_il_generator.add_set_trigger(
            seqc_gen,
            value=["AWG_MARKER1", "AWG_MARKER2"],
            device_type=zhinst.DeviceType.HDAWG,
        )

    # this is where a longer wait statement is added to allow for latency corrections.
    for instruction in instructions:
        assert isinstance(instruction, zhinst.Wave)

        clock_cycles_to_wait = instruction.clock_cycle_start - current_clock
        if clock_cycles_to_wait < 0:
            # a common mistake if there is no overlap if the instruction needs to start
            # to soon after the start of a new cycle.
            raise ValueError("Negative wait time, please ensure operations do not overlap in time.")

        current_clock += seqc_il_generator.add_wait(
            seqc_gen=seqc_gen,
            delay=int(clock_cycles_to_wait),
            device_type=zhinst.DeviceType.HDAWG,
            comment=f"clock={current_clock}",
        )

        current_clock += seqc_il_generator.add_execute_table_entry(
            seqc_gen=seqc_gen,
            index=wf_id_mapping[instruction.waveform_id],
            device_type=zhinst.DeviceType.HDAWG,
            comment=f"clock={current_clock}",
        )

    current_clock += seqc_il_generator.add_set_trigger(
        seqc_gen,
        value=0,
        device_type=zhinst.DeviceType.HDAWG,
        comment=f"clock={current_clock}",
    )

    # clock rate = 2.4e9/8 for HDAWG
    clock_rate = DEVICE_SAMPLING_RATES[zhinst.DeviceType.HDAWG][0] / CLOCK_SAMPLE_FACTOR
    total_duration_in_clocks = int(schedule_duration * clock_rate)
    clock_cycles_to_wait = total_duration_in_clocks - current_clock

    if trigger is None:
        current_clock += seqc_il_generator.add_wait(
            seqc_gen=seqc_gen,
            delay=int(clock_cycles_to_wait),
            device_type=zhinst.DeviceType.HDAWG,
            comment=(f"clock={current_clock}, dead time to ensure total schedule duration"),
        )

    seqc_gen.emit_end_repeat()

    seqc_instructions = seqc_gen.generate()
    commandtable_json = command_table.model_dump_json()
    return seqc_instructions, commandtable_json


def _compile_for_uhfqa(  # noqa: PLR0915
    device: zhinst.Device,
    timing_table: pandas.DataFrame,
    numerical_wf_dict: dict[str, np.ndarray],
    repetitions: int,
    operations: dict[str, Operation],
    bin_mode: enums.BinMode,
) -> tuple[zi_settings.ZISettingsBuilder, ZIAcquisitionConfig]:
    """
    Initialize programming the UHFQA ZI Instrument.

    Creates a sequence program and converts schedule
    pulses to waveforms for the UHFQA.

    Parameters
    ----------
    device :
    timing_table :
    numerical_wf_dict :
    repetitions :
    operations :

    Returns
    -------
    :

    """
    ########################################
    # Add standard settings to builder
    ########################################

    settings_builder = zi_settings.ZISettingsBuilder()

    instrument_info = zhinst.InstrumentInfo(
        sample_rate=device.sample_rate,
        num_samples_per_clock=CLOCK_SAMPLE_FACTOR,
        granularity=WAVEFORM_GRANULARITY[device.device_type],
    )
    channels = device.channels
    channels = list(filter(lambda c: c.mode == zhinst.SignalModeType.REAL, channels))

    awg_index = 0
    channel = channels[awg_index]
    logger.debug(f"[{device.name}-awg{awg_index}] {str(device)}")
    mixer_corrections = (
        channel.mixer_corrections
        if channel.mixer_corrections is not None
        else common.MixerCorrections()
    )

    # Set all integration weigths to default

    settings_builder.with_defaults(
        [
            ("awgs/0/single", 1),
            ("qas/0/rotations/*", (1 + 1j)),
            ("qas/0/integration/sources/*", 0),
        ]
    ).with_sigouts(0, (1, 1)).with_awg_time(
        0, device.clock_select
    ).with_qas_integration_weights_real(
        channels=list(range(NUM_UHFQA_READOUT_CHANNELS)),
        real=np.zeros(MAX_QAS_INTEGRATION_LENGTH),
    ).with_qas_integration_weights_imag(
        channels=list(range(NUM_UHFQA_READOUT_CHANNELS)),
        imag=np.zeros(MAX_QAS_INTEGRATION_LENGTH),
    ).with_sigout_offset(
        0, mixer_corrections.dc_offset_i
    ).with_sigout_offset(
        1, mixer_corrections.dc_offset_q
    )

    logger.debug(f"[{device.name}-awg{awg_index}] channel={str(channel)}")

    ############################################
    # Add seqc instructions and waveform table
    ############################################

    # select only the instructions relevant for the output channel.
    output_timing_table = timing_table[
        timing_table["hardware_channel"] == f"{device.name}.awg{awg_index}"
    ]

    instructions: list[zhinst.Instruction] = _get_instruction_list(output_timing_table)

    # FIXME ensure unique_wf_ids is only for pulses and not integration weights
    # enumerate the waveform_ids used in this particular output channel
    unique_wf_ids = output_timing_table.drop_duplicates(subset="waveform_id")["waveform_id"]
    # this table maps waveform ids to indices in the seqc command table.
    wf_id_mapping = {}
    for i, wf_id in enumerate(unique_wf_ids):
        wf_id_mapping[wf_id] = i

    # # Generate and apply sequencer program
    seqc = _assemble_uhfqa_sequence(
        instructions=instructions,
        wf_id_mapping=wf_id_mapping,
        repetitions=repetitions,
        device_name=device.name,
        trigger=device.channels[awg_index].trigger,
    )

    settings_builder.with_compiler_sourcestring(awg_index, seqc)
    logger.debug(seqc)

    #######################################################
    # Set waveforms to wave nodes in the settings builder
    #######################################################

    # Apply waveforms to AWG
    settings_builder = _add_wave_nodes(
        device_type=zhinst.DeviceType.UHFQA,
        awg_index=0,
        wf_id_mapping=wf_id_mapping,
        numerical_wf_dict=numerical_wf_dict,
        settings_builder=settings_builder,
    )

    #######################################################
    # Set integration weights and configure acquisitions
    #######################################################

    # Get a list of all acquisition protocol channels
    acq_channel_resolvers_map: dict[int, Callable[..., Any]] = dict()

    # select only the acquisition operations relevant for the output channel.
    timing_table_acquisitions = output_timing_table[output_timing_table.is_acquisition]
    timing_table_unique_acquisitions = timing_table_acquisitions.drop_duplicates(
        subset="waveform_id"
    )

    n_unique_acquisitions = len(timing_table_acquisitions)
    if bin_mode == enums.BinMode.AVERAGE:
        n_acquisitions = n_unique_acquisitions
    elif bin_mode == enums.BinMode.APPEND:
        n_acquisitions = n_unique_acquisitions * repetitions
        repetitions = 1
    else:
        raise NotImplementedError(f"BinMode {bin_mode} is not supported.")

    # These variables have to be identical for all acquisitions.
    # initialized to None here and overwritten while iterating over the acquisitions.
    acq_duration: float = np.nan

    # a list of used acquisition channels, this is used to raise an exception
    # when multiple acquisitions assign to the same channel.
    acq_channels_used: list[int] = []

    acq_protocols: dict[int, SUPPORTED_ACQ_PROTOCOLS] = {}
    for _, acq_row in timing_table_unique_acquisitions.iterrows():
        acquisition = operations[acq_row.operation]
        wf_id = acq_row.wf_idx
        acq_info = acquisition.data["acquisition_info"][acq_row.wf_idx]

        # update acq_duration only if it was not set before
        acq_duration = acq_info["duration"] if np.isnan(acq_duration) else acq_duration
        # verify that the both durations are identical, if not raise an exception
        # this exception relates to a limitation of the hardware.
        if acq_duration != acq_info["duration"]:
            raise ValueError(
                f"Different acquisitions have a different duration "
                f"{acq_duration*1e9:.1f}ns and {acq_info['duration']*1e9:.1f}ns. "
                "The integration lenght needs to be identical for all acquisitions."
            )

        if (acq_protocol := acq_info["protocol"]) not in get_args(SUPPORTED_ACQ_PROTOCOLS):
            raise ValueError(
                f"Acquisition protocol {acq_protocol} is not supported by"
                " Zurich Instruments LabOne backend."
            )

        acq_protocol: SUPPORTED_ACQ_PROTOCOLS
        acq_channel: int = acq_info["acq_channel"]
        acq_protocols[acq_channel] = acq_protocol
        if acq_channel not in acq_channels_used:
            acq_channels_used.append(acq_channel)
        else:
            raise ValueError(
                f"Acquisition channel {acq_channel} is already used by another "
                "acquisition. Different acquisitions should use a unique "
                "acquisition channel."
                f"Offending acquisition ({acq_row.waveform_id})"
            )

        integration_length = round(acq_duration * instrument_info.sample_rate)
        logger.debug(
            f"[{device.name}] acq_info={acq_info} "
            + f" acq_duration={acq_duration} integration_length={integration_length}"
        )

        if acq_protocol == "Trace":
            # Disable Weighted integration because we'd like to see
            # the raw signal.
            settings_builder.with_qas_monitor_enable(True).with_qas_monitor_averages(
                repetitions
            ).with_qas_monitor_length(integration_length).with_qas_integration_weights_real(
                list(range(NUM_UHFQA_READOUT_CHANNELS)),
                np.ones(MAX_QAS_INTEGRATION_LENGTH),
            ).with_qas_integration_weights_imag(
                list(range(NUM_UHFQA_READOUT_CHANNELS)),
                np.ones(MAX_QAS_INTEGRATION_LENGTH),
            )

            monitor_nodes = (
                "qas/0/monitor/inputs/0/wave",
                "qas/0/monitor/inputs/1/wave",
            )
            acq_channel_resolvers_map[acq_channel] = partial(
                resolvers.monitor_acquisition_resolver, monitor_nodes=monitor_nodes
            )
        else:
            # The waveform is slightly larger then the integration_length
            # because of the waveform granularity. This is irrelevant
            # due to the waveform being appended with zeros. Therefore
            # avoiding an extra slice of waveform[0:integration_length]

            acquisition_waveform = numerical_wf_dict[acq_row.waveform_id]

            weights_i = np.zeros(MAX_QAS_INTEGRATION_LENGTH)
            weights_q = np.zeros(MAX_QAS_INTEGRATION_LENGTH)

            weights_i[0 : len(acquisition_waveform)] = acquisition_waveform.real
            weights_q[0 : len(acquisition_waveform)] = acquisition_waveform.imag

            # set the integration weights, note that we need to set 4 weights in order
            # to use a complex valued weight function in the right way.
            # Z = (w0*sI + w1*sQ) + 1j ( w1*sI - w0 * sQ)
            settings_builder.with_qas_integration_weights_real(
                2 * acq_channel, list(weights_i)
            ).with_qas_integration_weights_imag(
                2 * acq_channel, list(weights_q)
            ).with_qas_integration_weights_real(
                2 * acq_channel + 1, list(weights_q)
            ).with_qas_integration_weights_imag(
                2 * acq_channel + 1, list(-1 * weights_i)
            )

            # Create partial function for delayed execution
            acq_channel_resolvers_map[acq_channel] = partial(
                resolvers.result_acquisition_resolver,
                result_nodes=[
                    f"qas/0/result/data/{2*acq_channel}/wave",
                    f"qas/0/result/data/{2*acq_channel+1}/wave",
                ],
            )

    # only configure these variables if there are actually acquisitions present in
    # the schedule.
    if len(timing_table_unique_acquisitions) > 0:
        integration_length = round(acq_duration * instrument_info.sample_rate)
        settings_builder.with_qas_integration_mode(
            zhinst.QasIntegrationMode.NORMAL
        ).with_qas_integration_length(integration_length).with_qas_result_enable(
            False
        ).with_qas_monitor_enable(
            False
        ).with_qas_delay(
            0
        )

        settings_builder.with_qas_result_mode(zhinst.QasResultMode.CYCLIC).with_qas_result_source(
            zhinst.QasResultSource.INTEGRATION
        ).with_qas_result_length(n_acquisitions).with_qas_result_enable(
            True
        ).with_qas_result_averages(
            repetitions
        )

    settings_builder.with_qas_result_reset(0).with_qas_result_reset(1)
    settings_builder.with_qas_monitor_reset(0).with_qas_monitor_reset(1)

    return (
        settings_builder,
        ZIAcquisitionConfig(
            n_unique_acquisitions,
            resolvers=acq_channel_resolvers_map,
            bin_mode=bin_mode,
            acq_protocols=acq_protocols,
        ),
    )


def _assemble_uhfqa_sequence(
    instructions: list[zhinst.Instruction],
    wf_id_mapping: dict[str, int],
    repetitions: int,
    device_name: str,
    trigger: int = 2,
) -> str:
    """ """
    seqc_instructions = ""
    seqc_gen = seqc_il_generator.SeqcILGenerator()

    # Declare sequence variables
    seqc_gen.declare_var("__repetitions__", repetitions)

    current_clock: int = 0

    ###############################################################
    # Generate the .csv based waveform table
    ###############################################################

    seqc_il_generator.declare_csv_waveform_variables(
        seqc_gen=seqc_gen,
        device_name=device_name,
        waveform_indices=list(wf_id_mapping.values()),
        awg_index=0,
    )

    ###############################################################
    # Add the loop that executes the program.
    ###############################################################

    seqc_gen.emit_begin_repeat("__repetitions__")

    # N.B.! The UHFQA will always need to be triggered by an external device such as
    # an HDAWG or a trigger box. It will wait for a trigger.
    # Triggers must be a list but we may only wait for one so lets choose the
    # first one in the list, I guess.

    # This does not account for dio ports. Which are not implemented in the current
    # version.
    assert trigger < 5 and trigger > 0
    assert trigger is not None

    seqc_gen.emit_wait_dig_trigger(
        index=trigger,
        comment=f"\t// clock={current_clock}",
        device_type=zhinst.DeviceType.UHFQA,
    )
    # this is where a longer wait statement is added to allow for latency corrections.
    for instruction in instructions:
        clock_cycles_to_wait = instruction.clock_cycle_start - current_clock
        if clock_cycles_to_wait < 0:
            # a common mistake if there is no overlap if the instruction needs to start
            # to soon after the start of a new cycle.
            raise ValueError("Negative wait time, please ensure operations do not overlap in time.")
        current_clock += seqc_il_generator.add_wait(
            seqc_gen=seqc_gen,
            delay=int(clock_cycles_to_wait),
            device_type=zhinst.DeviceType.UHFQA,
            comment=f"clock={current_clock}",
        )

        # Acquisition
        if isinstance(instruction, zhinst.Acquisition):
            current_clock += seqc_il_generator.add_start_qa(
                seqc_gen=seqc_gen,
                device_type=zhinst.DeviceType.UHFQA,
                comment=f"clock={current_clock}",
            )
        # Waveform
        elif isinstance(instruction, zhinst.Wave):
            current_clock += seqc_il_generator.add_play_wave(
                seqc_gen,
                f"w{wf_id_mapping[instruction.waveform_id]}",
                device_type=zhinst.DeviceType.UHFQA,
                comment=f"clock={current_clock}",
            )

    seqc_gen.emit_end_repeat()

    seqc_instructions = seqc_gen.generate()
    return seqc_instructions


def construct_waveform_table(
    timing_table: pandas.DataFrame,
    operations_dict: dict[str, Operation],
    device_dict: dict[str, zhinst.Device],
) -> dict[str, np.ndarray]:
    """
    Iterates over all unique waveforms in a timing_table dataframe to calculate the
    numerical waveforms.

    Parameters
    ----------
    timing_table:
        A timing table for which the waveform_id has been determined
    operations_dict:
        The Operations contained in a Schedule.
    device_dict:
        A dictionary containing the :class:`~.backends.types.zhinst.Device` objects
        describing the devicesin the hardware configuration.

    Returns
    -------
    :
        numerical_waveform dict, a dictionary containing the  complex valued waveforms
        that will be uploaded to the control hardware.

    """
    # remove all entries for which the port is missing such as a Reset operation.
    filtered_df = timing_table.drop_duplicates(subset="waveform_id").dropna(axis=0, subset=["port"])

    instr_info_dict = {}
    for dev_name, device in device_dict.items():
        instrument_info = zhinst.InstrumentInfo(
            sample_rate=device.sample_rate,  # type: ignore
            num_samples_per_clock=CLOCK_SAMPLE_FACTOR,  # one clock cycle is 8 samples
            # every wf needs to be a multiple of 16 samples
            granularity=WAVEFORM_GRANULARITY[device.device_type],
            mode=device.mode,
        )
        instr_info_dict[dev_name] = instrument_info

    numerical_wf_dict = {}
    for _, row in filtered_df.iterrows():
        device_name, awg = row.hardware_channel.split(".")
        ch_idx = int(awg[-1])  # the name is always awg_x where x is an int
        output = device_dict[device_name].channels[ch_idx]
        instrument_info = instr_info_dict[device_name]

        if row.is_acquisition:
            waveform_info = operations_dict[row["operation"]]["acquisition_info"][row["wf_idx"]][
                "waveforms"
            ]

            # There are acquisitions (e.g., Trace) in which no integration weights are
            #  uploaded. in that case there are no (2) waveforms to be uploaded.
            if len(waveform_info) != 2:
                continue

            # Evaluate waveform
            wf_i = waveform_helpers.get_waveform(
                waveform_info[0], sampling_rate=instrument_info.sample_rate
            )
            wf_q = waveform_helpers.get_waveform(
                waveform_info[1], sampling_rate=instrument_info.sample_rate
            )

            # storing it as a complex waveform, N.B. the wf_q is already imaginary
            waveform = np.array(wf_i) + np.array(wf_q)

            # Apply corrections
            _, _, corr_wf = apply_waveform_corrections(
                output=output,
                waveform=waveform,
                start_and_duration_in_seconds=(0, row["duration"]),
                instrument_info=instrument_info,
                is_pulse=not row.is_acquisition,
            )

            if len(corr_wf) > 4096:
                raise ValueError(
                    f"Attempting to set an integration weight of {len(corr_wf)} samples"
                    " (>4096) corresponding to an integration time of "
                    f"{len(corr_wf)/instrument_info.sample_rate*1e6} us. "
                    "Please double check that your schedule does not contain any "
                    "acquisitions with a duration longer than "
                    f"{4096/instrument_info.sample_rate*1e6:.2f} us.\n"
                    f'Offending operation: "{row["operation"]}"'
                )

            numerical_wf_dict[row["waveform_id"]] = corr_wf

        else:
            waveform_info = operations_dict[row["operation"]]["pulse_info"][row["wf_idx"]]
            waveform = waveform_helpers.get_waveform(
                waveform_info, sampling_rate=instrument_info.sample_rate
            )

            # Apply corrections
            # Apply corrections
            _, _, corr_wf = apply_waveform_corrections(
                output=output,
                waveform=waveform,
                start_and_duration_in_seconds=(0, row["duration"]),
                instrument_info=instrument_info,
                is_pulse=not row.is_acquisition,
            )
            numerical_wf_dict[row["waveform_id"]] = corr_wf
    return numerical_wf_dict
