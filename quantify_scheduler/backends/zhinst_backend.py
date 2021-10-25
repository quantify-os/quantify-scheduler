# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
"""Backend for Zurich Instruments."""
# pylint: disable=too-many-lines
from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union, cast

import numpy as np
from zhinst.toolkit.helpers import Waveform

from quantify_core.utilities.general import make_hash

from quantify_scheduler import enums, types
from quantify_scheduler.backends.types import common, zhinst
from quantify_scheduler.backends.zhinst import helpers as zi_helpers
from quantify_scheduler.backends.zhinst import resolvers, seqc_il_generator
from quantify_scheduler.backends.zhinst import settings as zi_settings
from quantify_scheduler.helpers import schedule as schedule_helpers
from quantify_scheduler.helpers import waveforms as waveform_helpers
from quantify_scheduler.resources import Resource

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    # "%(levelname)-8s | %(module)s | %(funcName)s::%(lineno)s. %(message)s"
)
handler.setFormatter(formatter)
logger.addHandler(handler)


# List of supported zhinst devices
SUPPORTED_DEVICE_TYPES: List[str] = ["HDAWG", "UHFQA"]

# https://www.zhinst.com/sites/default/files/documents/2020-09/ziHDAWG_UserManual_20.07.1.pdf
# Section: 3.4. Basic Qubit Characterization, page 83
# All waveform lengths need to be multiples of 16 sample-clock cycles to comply
# with the waveform granularity specification.
WAVEFORM_GRANULARITY: Dict[zhinst.DeviceType, int] = {
    zhinst.DeviceType.HDAWG: 16,
    zhinst.DeviceType.UHFQA: 16,
}

# https://www.zhinst.com/sites/default/files/documents/2020-09/ziHDAWG_UserManual_20.07.2.pdf
# page: 262
HDAWG_DEVICE_TYPE_CHANNEL_GROUPS: Dict[str, Dict[int, int]] = {
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


DEVICE_CLOCK_RATES: Dict[zhinst.DeviceType, Dict[int, int]] = {
    zhinst.DeviceType.HDAWG: zi_helpers.get_clock_rates(2.4e9),
    zhinst.DeviceType.UHFQA: zi_helpers.get_clock_rates(1.8e9),
}

NUM_UHFQA_READOUT_CHANNELS = 10
MAX_QAS_INTEGRATION_LENGTH = 4096


def _parse_local_oscillators(data: Dict[str, Any]) -> Dict[str, common.LocalOscillator]:
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
    local_oscillators: Dict[str, common.LocalOscillator] = dict()
    lo_list: List[common.LocalOscillator] = common.LocalOscillator.schema().load(
        data, many=True
    )
    for local_oscillator in lo_list:
        if local_oscillator.name in local_oscillators:
            raise RuntimeError(
                f"Duplicate entry LocalOscillators '{local_oscillator.name}' in "
                "hardware configuration!"
            )

        local_oscillators[local_oscillator.name] = local_oscillator

    return local_oscillators


def _parse_devices(data: Dict[str, Any]) -> List[zhinst.Device]:
    device_list: List[zhinst.Device] = zhinst.Device.schema().load(data, many=True)

    for device in device_list:
        if device.device_type.value not in SUPPORTED_DEVICE_TYPES:
            raise NotImplementedError(
                f"Unable to create zhinst backend for '{device.device_type.value}'!"
            )

        clock_rates = DEVICE_CLOCK_RATES[device.device_type]
        if not device.clock_select in clock_rates:
            raise ValueError(
                f"Unknown value clock_select='{device.clock_select}' "
                + f"for device type '{device.device_type.value}'"
            )

        device.clock_rate = clock_rates[device.clock_select]

    return device_list


def _validate_schedule(schedule: types.CompiledSchedule) -> None:
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
    if len(schedule.timing_constraints) == 0:
        raise ValueError(
            f"Undefined timing constraints for schedule '{schedule.name}'!"
        )

    for t_constr in schedule.timing_constraints:

        if "abs_time" not in t_constr:
            raise ValueError(
                "Absolute timing has not been determined "
                + f"for the schedule '{schedule.name}'!"
            )


def apply_waveform_corrections(
    output: zhinst.Output,
    waveform: np.ndarray,
    start_and_duration_in_seconds: Tuple[float, float],
    instrument_info: zhinst.InstrumentInfo,
    is_pulse: bool,
) -> Tuple[int, int, np.ndarray]:
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

    if is_pulse:
        # Modulate the waveform
        if output.modulation.type == enums.ModulationModeType.PREMODULATE:
            t: np.ndarray = np.arange(
                0, 0 + duration_in_seconds, 1 / instrument_info.clock_rate
            )
            waveform = waveform_helpers.modulate_waveform(
                t, waveform, output.modulation.interm_freq
            )
        if not output.mixer_corrections is None:
            waveform = waveform_helpers.apply_mixer_skewness_corrections(
                waveform,
                output.mixer_corrections.amp_ratio,
                output.mixer_corrections.phase_error,
            )

    else:  # in the case where the waveform is an integration weight
        # Modulate the waveform
        if output.modulation.type == enums.ModulationModeType.PREMODULATE:
            t: np.ndarray = np.arange(
                0, 0 + duration_in_seconds, 1 / instrument_info.clock_rate
            )
            # N.B. the minus sign with respect to the pulse being applied
            waveform = waveform_helpers.modulate_waveform(
                t, waveform, -1 * output.modulation.interm_freq
            )
        # mixer corrections for the integration are not supported yet.
        # they would belong here.

    start_in_clocks, waveform = waveform_helpers.shift_waveform(
        waveform,
        start_in_seconds,
        instrument_info.clock_rate,
        instrument_info.resolution,
    )
    n_samples_shifted = len(waveform)

    waveform = waveform_helpers.resize_waveform(waveform, instrument_info.granularity)

    return start_in_clocks, n_samples_shifted, waveform


def _flatten_dict(collection: Dict[Any, Any]) -> Iterable[Tuple[Any, Any]]:
    """
    Flattens a collection to an iterable set of tuples.

    Parameters
    ----------
    collection :

    Returns
    -------
    :
    """

    def expand(key, obj):
        if isinstance(obj, dict):
            for i, value in obj.items():
                yield from expand(i, value)
        elif isinstance(obj, list):
            for value in obj:
                yield (key, value)
        else:
            yield (key, obj)

    return expand(None, collection)


def get_wave_instruction(
    uuid: int,
    timeslot_index: int,
    output: zhinst.Output,
    cached_schedule: schedule_helpers.CachedSchedule,
    instrument_info: zhinst.InstrumentInfo,
) -> zhinst.Wave:
    """
    Returns wave sequence instruction.

    This function returns a record type class
    containing the waveform and timing critical
    information.

    Parameters
    ----------
    uuid :
    timeslot_index :
    output :
    cached_schedule :
    instrument_info :

    Returns
    -------
    :
    """
    pulse_info = cached_schedule.pulseid_pulseinfo_dict[uuid]

    t_constr = cached_schedule.schedule.timing_constraints[timeslot_index]
    abs_time = t_constr["abs_time"] - cached_schedule.start_offset_in_seconds
    t0: float = abs_time + pulse_info["t0"]

    duration_in_seconds: float = pulse_info["duration"]
    waveform = waveform_helpers.exec_waveform_partial(
        uuid, cached_schedule.pulseid_waveformfn_dict, instrument_info.clock_rate
    )
    n_samples = len(waveform)

    if instrument_info.mode == enums.InstrumentOperationMode.CALIBRATING:
        # Set all the numeric pulse values to one in calibration mode.
        waveform = np.ones(n_samples)

    (
        corrected_start_in_clocks,
        n_samples_shifted,
        waveform,
    ) = apply_waveform_corrections(
        output=output,
        waveform=waveform,
        start_and_duration_in_seconds=(t0, duration_in_seconds),
        instrument_info=instrument_info,
        is_pulse=True,
    )

    duration_in_clocks: float = duration_in_seconds / instrument_info.low_res_clock

    if n_samples_shifted != n_samples:
        # If the slope start waveform shifts with a number of samples,
        # the base waveform is altered therefore a new uuid is required.
        pulse_info["_n_samples_shifted"] = n_samples_shifted
        uuid = schedule_helpers.get_pulse_uuid(pulse_info, [])
        del pulse_info["_n_samples_shifted"]

    # Overwrite or add newly created uuid's pulse_info
    cached_schedule.pulseid_pulseinfo_dict[uuid] = pulse_info

    return zhinst.Wave(
        uuid,
        abs_time,
        timeslot_index,
        t0,
        corrected_start_in_clocks,
        duration_in_seconds,
        round(duration_in_clocks),
        waveform,
        n_samples,
        n_samples_shifted,
    )


# pylint: disable=too-many-locals
def get_measure_instruction(
    uuid: int,
    timeslot_index: int,
    output: zhinst.Output,
    cached_schedule: schedule_helpers.CachedSchedule,
    instrument_info: zhinst.InstrumentInfo,
) -> zhinst.Measure:
    """
    Returns the measurement sequence instruction.

    Parameters
    ----------
    uuid :
    timeslot_index :
    output :
    cached_schedule :
    instrument_info :

    Returns
    -------
    :
    """
    acq_info = cached_schedule.acqid_acqinfo_dict[uuid]

    t_constr = cached_schedule.schedule.timing_constraints[timeslot_index]
    abs_time = t_constr["abs_time"] - cached_schedule.start_offset_in_seconds

    t0: float = abs_time + acq_info["t0"]
    duration_in_seconds: float = acq_info["duration"]
    duration_in_clocks = round(duration_in_seconds / instrument_info.low_res_clock)

    weights_list: List[np.ndarray] = [np.empty((0,)), np.empty((0,))]
    corrected_start_in_clocks: int = 0

    if len(acq_info["waveforms"]) == 0:
        pass
    elif len(acq_info["waveforms"]) == 2:
        # an acquisition weight has two waveforms, the real and imaginary part
        wf_i = waveform_helpers.exec_waveform_partial(
            schedule_helpers.get_pulse_uuid(acq_info["waveforms"][0]),
            cached_schedule.pulseid_waveformfn_dict,
            instrument_info.clock_rate,
        )
        wf_q = waveform_helpers.exec_waveform_partial(
            schedule_helpers.get_pulse_uuid(acq_info["waveforms"][1]),
            cached_schedule.pulseid_waveformfn_dict,
            instrument_info.clock_rate,
        )

        # the imaginary part is already included in wf_q
        waveform = wf_i + wf_q

        (corrected_start_in_clocks, _, waveform_corr) = apply_waveform_corrections(
            output=output,
            waveform=waveform,
            start_and_duration_in_seconds=(t0, duration_in_seconds),
            instrument_info=instrument_info,
            is_pulse=False,
        )
        weights_list[0] = waveform_corr.real
        weights_list[1] = waveform_corr.imag
    else:
        raise ValueError(
            "A measurement either has no integration weights or 2 "
            "integration weights (real and imaginary)"
        )

    if len(acq_info["waveforms"]) == 0:
        (corrected_start_in_clocks, _) = waveform_helpers.shift_waveform(
            [],
            t0,
            instrument_info.clock_rate,
            instrument_info.resolution,
        )

    return zhinst.Measure(
        uuid=uuid,
        abs_time=abs_time,
        timeslot_index=timeslot_index,
        start_in_seconds=t0,
        start_in_clocks=corrected_start_in_clocks,
        duration_in_seconds=duration_in_seconds,
        duration_in_clocks=duration_in_clocks,
        weights_i=weights_list[0],
        weights_q=weights_list[1],
    )


def get_execution_table(
    cached_schedule: schedule_helpers.CachedSchedule,
    instrument_info: zhinst.InstrumentInfo,
    output: zhinst.Output,
) -> List[zhinst.Instruction]:
    """
    Returns a timing critical execution table of Instructions.

    Parameters
    ----------
    cached_schedule :
    instrument_info :
    output :

    Raises
    ------
    RuntimeError
        Raised if encountered an unknown uuid.

    Returns
    -------
    :
    """

    def filter_uuid(pair: Tuple[int, int]) -> bool:
        (_, uuid) = pair

        if uuid in cached_schedule.pulseid_pulseinfo_dict:
            pulse_info = cached_schedule.pulseid_pulseinfo_dict[uuid]
            if pulse_info["port"] is None:
                # Skip pulses without a port, such as Reset.
                return False

        # Item is added
        return True

    def get_instruction(timeslot_index: int, uuid: int) -> zhinst.Instruction:
        if uuid in cached_schedule.acqid_acqinfo_dict:
            measure_instruction = get_measure_instruction(
                uuid, timeslot_index, output, cached_schedule, instrument_info
            )
            return measure_instruction
        if uuid in cached_schedule.pulseid_pulseinfo_dict:
            wave_instruction = get_wave_instruction(
                uuid, timeslot_index, output, cached_schedule, instrument_info
            )
            return wave_instruction

        raise RuntimeError(
            f"Undefined instruction for uuid={uuid} timeslot={timeslot_index}"
        )

    instr_timeline_list: List[Tuple[int, int]] = list(
        _flatten_dict(cached_schedule.port_timeline_dict[output.port])
    )
    instr_timeline_list = filter(filter_uuid, instr_timeline_list)
    instr_timeline_iter = iter(instr_timeline_list)

    current_instr: zhinst.Instruction = zhinst.Instruction.default()
    previous_instr: zhinst.Instruction = get_instruction(*next(instr_timeline_iter))

    logger.debug(zhinst.Wave.__header__())
    logger.debug(repr(previous_instr))

    instructions: List[zhinst.Instruction] = [previous_instr]
    new_timeslot_uuids: List[int] = [previous_instr.uuid]

    for (timeslot_index, uuid) in instr_timeline_iter:
        current_instr = get_instruction(timeslot_index, uuid)

        logger.debug(repr(current_instr))

        if previous_instr.timeslot_index != current_instr.timeslot_index:
            cached_schedule.port_timeline_dict[output.port][
                previous_instr.timeslot_index
            ] = new_timeslot_uuids
            new_timeslot_uuids = list()

        new_timeslot_uuids.append(current_instr.uuid)
        instructions.append(current_instr)
        previous_instr = current_instr

    # Rectify the last timeslot uuid list
    cached_schedule.port_timeline_dict[output.port][
        previous_instr.timeslot_index
    ] = new_timeslot_uuids

    return instructions


@dataclass(frozen=True)
class ZIAcquisitionConfig:
    """Zurich Instruments acquisition configuration.

    Parameters
    ----------
    n_acquisitions :
        the number of distinct acquisitions in this experiment.
    resolvers:
        resolvers used to retrieve the results from the right UHFQA nodes.
        See also :mod:`~quantify_scheduler.backends.zhinst.resolvers`
    """

    n_acquisitions: int
    resolvers: Dict[int, Callable]


@dataclass(frozen=True)
class ZIDeviceConfig:
    """
    Zurich Instruments device configuration.

    Parameters
    ----------
    name :
        the name of the schedule the config is for.
    schedule:
        the CompiledSchedule from which the config is generated.
    settings_builder:
        the builder to configure the ZI settings. This typically includes AWG and
        AWG settings.
    acq_config:
        the acquisition config contains the number of acquisitions and a dictionary of
        resolvers used to retrieve the results from the right UHFQA nodes.
        Note that this part of the config is not needed during prepare, but only during
        the retrieve acquisitions step.
    """

    name: str
    schedule: types.CompiledSchedule
    settings_builder: zi_settings.ZISettingsBuilder
    acq_config: Optional[ZIAcquisitionConfig]


def compile_backend(
    schedule: types.CompiledSchedule, hardware_map: Dict[str, Any]
) -> Dict[str, Union[ZIDeviceConfig, float]]:

    """
    Compiles backend for Zurich Instruments hardware according
    to the CompiledSchedule and hardware configuration.

    This method generates sequencer programs, waveforms and
    configurations required for the instruments defined in
    the hardware configuration.

    Parameters
    ----------
    schedule :
    hardware_map :

    Returns
    -------
    :
        A collection containing the compiled backend
        configuration for each device.

    Raises
    ------
    NotImplementedError
        Thrown when using unsupported ZI Instruments.
    """
    _validate_schedule(schedule)

    # Parse the hardware configuration file
    devices: List[zhinst.Device] = _parse_devices(hardware_map["devices"])
    local_oscillators: Dict[str, common.LocalOscillator] = _parse_local_oscillators(
        hardware_map["local_oscillators"]
    )

    # Create CachedSchedule to populate schedule lookup dictionaries.
    cached_schedule = schedule_helpers.CachedSchedule(schedule)

    resources = cached_schedule.schedule["resource_dict"]
    device_configs: Dict[str, Union[ZIDeviceConfig, float]] = dict()

    # Program devices
    for device in sorted(
        devices, key=lambda x: x.device_type == zhinst.DeviceType.UHFQA
    ):

        builder = zi_settings.ZISettingsBuilder()
        acq_config: Optional[ZIAcquisitionConfig] = None

        if device.device_type == zhinst.DeviceType.HDAWG:
            builder = _compile_for_hdawg(device, cached_schedule, builder)
        elif device.device_type == zhinst.DeviceType.UHFQA:
            builder, acq_config = _compile_for_uhfqa(device, cached_schedule, builder)

        # add the local oscillator config by iterating over all output channels.
        # note that not all output channels have an LO associated to them.
        for channel in device.channels:
            _add_lo_config(
                channel=channel,
                local_oscillators=local_oscillators,
                device_configs=device_configs,
                resources=resources,
            )

        device_configs[device.name] = ZIDeviceConfig(
            device.name, schedule, builder, acq_config
        )

    return device_configs


def _add_lo_config(
    channel: zhinst.Output,
    local_oscillators: List,
    resources: Dict[str, Resource],
    device_configs: Dict[str, Union[ZIDeviceConfig, float]],
) -> None:
    """
    Adds configuration for a local oscillator required for a specific output channel to
    the device configs.
    """
    # N.B. when using baseband pulses no LO will be associated to the channel.
    # this case is caught in the case where the channel.clock is not specified.
    name = channel.local_oscillator

    if name not in local_oscillators:
        raise KeyError(f'Missing configuration for LocalOscillator "{name}"')

    local_oscillator = local_oscillators[name]

    # the frequencies from the config file
    lo_freq = local_oscillator.frequency
    interm_freq = channel.modulation.interm_freq

    if (lo_freq is not None) and (interm_freq is not None):
        rf_freq = lo_freq + interm_freq
    else:
        channel_clock_resource = resources.get(channel.clock)
        if channel_clock_resource is not None:
            rf_freq = channel_clock_resource.get("freq")
        else:
            # no clock is specified for this channel.
            # this can happen for e.g., baseband pulses or when the channel is not used
            # in the schedule.
            return

    if lo_freq is None and interm_freq is not None:
        lo_freq = rf_freq - interm_freq
        local_oscillator.frequency = lo_freq

    elif interm_freq is None and lo_freq is not None:
        interm_freq = rf_freq - lo_freq
        channel.modulation.interm_freq = interm_freq

    elif interm_freq is None and lo_freq is None:
        raise ValueError(
            "Either local oscillator frequency or channel intermediate frequency "
            f'must be set for LocalOscillator "{name}"'
        )

    if local_oscillator.name in device_configs:
        # the device_config currently only contains the frequency
        if device_configs[local_oscillator.name] != lo_freq:
            raise ValueError(
                f'Multiple frequencies assigned to LocalOscillator "{name}"'
            )
    device_configs[local_oscillator.name] = lo_freq


def _add_wave_nodes(
    device: zhinst.Device,
    awg_index: int,
    waveforms_dict: Dict[int, np.ndarray],
    waveform_table: Dict[int, int],
    settings_builder: zi_settings.ZISettingsBuilder,
) -> None:
    """
    Adds for each waveform a new setting in the ZISettingsBuilder
    which will provide settings to the Instruments hardware nodes.

    Parameters
    ----------
    device :
    awg_index :
    waveforms_dict :
    waveform_table :
    settings_builder :
    """

    for pulse_id, waveform_table_index in waveform_table.items():
        array: np.ndarray = waveforms_dict[pulse_id]
        waveform = Waveform(array.real, array.imag)

        if device.device_type == zhinst.DeviceType.UHFQA:
            settings_builder.with_csv_wave_vector(
                awg_index, waveform_table_index, waveform.data
            )
        else:
            settings_builder.with_wave_vector(
                awg_index, waveform_table_index, waveform.data
            )


# pylint: disable=too-many-locals
def _compile_for_hdawg(
    device: zhinst.Device,
    cached_schedule: schedule_helpers.CachedSchedule,
    settings_builder: zi_settings.ZISettingsBuilder,
) -> zi_settings.ZISettingsBuilder:
    """
    Programs the HDAWG ZI Instrument.

    The Sequencer Program will be generated from the Schedule and
    the waveforms will be played using the CommandTable feature.

    https://www.zhinst.com/sites/default/files/documents/2020-09/ziHDAWG_UserManual_20.07.2.pdf
    section: 3.3.6. Memory-efficient Sequencing with the Command Table, page 74

    .. note::

        The following sequential steps are required
        in order to utilize the commandtable.
        1: Compile seqc program
        2. Set commandtable json vector
        3. Upload waveforms

    Parameters
    ----------
    device :
    cached_schedule :
    settings_builder :

    Raises
    ------
    ValueError
    """
    instrument_info = zhinst.InstrumentInfo(
        device.clock_rate, 8, WAVEFORM_GRANULARITY[device.device_type], device.mode
    )
    n_awgs: int = int(device.n_channels / 2)
    settings_builder.with_defaults(
        [
            ("sigouts/*/on", 0),
            ("awgs/*/single", 1),
        ]
    ).with_system_channelgrouping(device.channelgrouping)

    # Set the clock-rate of an AWG
    for awg_index in range(n_awgs):
        settings_builder.with_awg_time(awg_index, device.clock_select)

    enabled_outputs: Dict[int, zhinst.Output] = dict()

    channelgroups = HDAWG_DEVICE_TYPE_CHANNEL_GROUPS[device.type]
    channelgroups_value = channelgroups[device.channelgrouping]
    sequencer_step = int(channelgroups_value / 2)
    sequencer_stop = min(len(device.channels), int(n_awgs / sequencer_step))

    logger.debug(
        f"HDAWG[{device.name}] devtype={device.device_type} "
        + f" awg_count={n_awgs} {str(device)}"
    )

    i = 0
    for awg_index in range(0, sequencer_stop, sequencer_step):
        output = device.channels[i]
        if output is None:
            raise ValueError(f"Required output at index '{i}' is undefined!")

        logger.debug(f"[{device.name}-awg{awg_index}] enabling outputs...")
        mixer_corrections = (
            output.mixer_corrections
            if not output.mixer_corrections is None
            else common.MixerCorrections()
        )
        settings_builder.with_sigouts(awg_index, (1, 1)).with_gain(
            awg_index, (output.gain1, output.gain2)
        ).with_sigout_offset(
            int(awg_index * 2), mixer_corrections.dc_offset_I
        ).with_sigout_offset(
            int(awg_index * 2) + 1, mixer_corrections.dc_offset_Q
        )

        enabled_outputs[awg_index] = output
        i += 1

    for awg_index, output in enabled_outputs.items():
        if output.port not in cached_schedule.port_timeline_dict:
            # this typically occurs when a channel is not used in a schedule.
            logger.info(
                f"[{device.name}-awg{awg_index}] Skipping! "
                + f"Missing pulses for port={output.port}."
            )
            continue

        # Generate sequence execution table
        instructions: List[zhinst.Wave] = get_execution_table(
            cached_schedule,
            instrument_info,
            output,
        )

        # Get a list of all pulse uuid(s)
        pulse_ids: List[int] = list(map(lambda i: i.uuid, instructions))

        # Generate map containing waveform the location of a pulse_id
        waveform_table: Dict[int, int] = zi_helpers.get_waveform_table(
            pulse_ids, cached_schedule.pulseid_pulseinfo_dict
        )

        # Step 1: Generate and compile sequencer program AND
        # Step 2: Set CommandTable JSON vector
        (seqc, commandtable_json) = _assemble_hdawg_sequence(
            awg_index,
            cached_schedule,
            device,
            instrument_info,
            output,
            waveform_table,
            instructions,
        )
        logger.debug(seqc)
        logger.debug(commandtable_json)

        settings_builder.with_commandtable_data(awg_index, commandtable_json)
        settings_builder.with_compiler_sourcestring(awg_index, seqc)

        # Step 3: Upload waveforms to AWG CommandTable
        waveforms_dict = dict(map(lambda i: (i.uuid, i.waveform), instructions))
        _add_wave_nodes(
            device,
            awg_index,
            waveforms_dict,
            waveform_table,
            settings_builder,
        )

    return settings_builder


# pylint: disable=too-many-arguments
# pylint: disable=too-many-statements
# pylint: disable=too-many-locals
def _assemble_hdawg_sequence(
    awg_index: int,
    cached_schedule: schedule_helpers.CachedSchedule,
    device: zhinst.Device,
    instrument_info: zhinst.InstrumentInfo,
    output: zhinst.Output,
    waveform_table: Dict[int, int],
    instructions: List[zhinst.Instruction],
) -> Tuple[str, str]:
    """
    Assembles a new sequence program for the HDAWG.

    The HDAWG acts as a master device. This means that the
    HDAWG sends a trigger to slave devices which can be used
    to start measuring.

    Parameters
    ----------
    awg_index :
    cached_schedule :
    device :
    instrument_info :
    output :
    waveform_table :
    instructions :

    Returns
    -------
    :
        The sequencer program and CommandTable JSON.
    """

    seqc_gen = seqc_il_generator.SeqcILGenerator()
    seqc_info = seqc_il_generator.SeqcInfo(
        cached_schedule,
        output,
        instrument_info.low_res_clock,
    )

    dead_time_in_clocks = (
        seqc_info.schedule_duration_in_clocks + seqc_info.schedule_offset_in_clocks
    ) - seqc_info.timeline_end_in_clocks

    seqc_il_generator.add_seqc_info(seqc_gen, seqc_info)

    is_master_awg: bool = awg_index == 0
    is_slave_awg: bool = not is_master_awg
    has_markers: bool = len(output.markers) > 0
    has_triggers: bool = len(output.triggers) > 0

    is_marker_source: bool = (
        is_master_awg
        and device.ref == enums.ReferenceSourceType.INTERNAL
        and has_markers
    )
    is_trigger_source = (
        is_slave_awg and device.ref != enums.ReferenceSourceType.NONE and has_triggers
    )
    current_clock: int = 0

    # Declare sequence variables
    seqc_gen.declare_var("__repetitions__", cached_schedule.schedule.repetitions)
    wave_instructions_dict: Dict[int, zhinst.Wave] = dict(
        (i.uuid, i) for i in instructions if isinstance(i, zhinst.Wave)
    )
    command_table_entries: List[zhinst.CommandTableEntry] = list()
    for pulse_id, waveform_index in waveform_table.items():
        instruction = wave_instructions_dict[pulse_id]
        waveform_index = waveform_table[instruction.uuid]
        name: str = f"w{waveform_index}"

        # Create and add variables to the Sequence program
        # aswell as assign the variables with operations
        seqc_gen.declare_wave(name)
        seqc_gen.assign_placeholder(name, len(instruction.waveform))
        seqc_gen.emit_assign_wave_index(name, name, index=waveform_index)

        # Do bookkeeping for the CommandTable
        command_table_entry = zhinst.CommandTableEntry(
            index=len(command_table_entries),
            waveform=zhinst.CommandTableWaveform(
                waveform_index, instruction.n_samples_scaled
            ),
        )
        command_table_entries.append(command_table_entry)

    command_table = zhinst.CommandTable(table=command_table_entries)

    # Reset marker
    if is_marker_source:
        seqc_il_generator.add_set_trigger(seqc_gen, 0, device.device_type)

    seqc_gen.emit_begin_repeat("__repetitions__")

    # here the UFHQA gets triggered. This needs to happen at every measurement.
    if is_marker_source:
        seqc_il_generator.add_set_trigger(seqc_gen, output.markers, device.device_type)

    if is_trigger_source:
        seqc_gen.emit_wait_dig_trigger(
            output.triggers[0], comment=f"\t// clock={current_clock}\n"
        )

    if (
        is_marker_source or is_trigger_source
    ) and seqc_info.line_trigger_delay_in_seconds != -1:
        seqc_il_generator.add_wait(
            seqc_gen,
            seqc_info.line_trigger_delay_in_clocks,
            device.device_type,
            comment=f"clock={current_clock}",
        )

    instructions_iter = iter(instructions)
    current_instr: zhinst.Wave = zhinst.Instruction.default()
    previous_instr: zhinst.Wave = next(instructions_iter)

    for instruction in instructions_iter:
        current_instr = cast(zhinst.Wave, instruction)
        previous_instr_end = (
            previous_instr.start_in_clocks + previous_instr.duration_in_clocks
        )
        current_instr_offset = seqc_il_generator.SEQC_INSTR_CLOCKS[device.device_type][
            seqc_il_generator.SeqcInstructions.EXECUTE_TABLE_ENTRY
        ]

        current_clock += seqc_il_generator.add_execute_table_entry(
            seqc_gen,
            waveform_table[previous_instr.uuid],
            device.device_type,
            f"clock={current_clock}",
        )

        remaining_clocks = max(
            current_instr.start_in_clocks - previous_instr_end,
            current_instr.start_in_clocks - current_clock,
        )

        clock_cycles_to_wait: int = remaining_clocks - current_instr_offset

        current_clock += seqc_il_generator.add_wait(
            seqc_gen,
            clock_cycles_to_wait,
            device.device_type,
            comment=f"\t clock={current_clock}",
        )

        previous_instr = current_instr

    clock_start: int = current_clock

    if previous_instr.uuid != -1:
        previous_instr_end = (
            previous_instr.start_in_clocks + previous_instr.duration_in_clocks
        )
        # Adds the last pulse
        current_clock += seqc_il_generator.add_execute_table_entry(
            seqc_gen,
            waveform_table[previous_instr.uuid],
            device.device_type,
            f"clock={current_clock}",
        )

    # Reset trigger each iteration
    if is_marker_source:
        current_clock += seqc_il_generator.add_set_trigger(
            seqc_gen, 0, device.device_type, comment=f"clock={current_clock}"
        )

    if previous_instr.uuid != -1:
        seqc_gen.emit_comment("Dead time")
        remaining_clocks = max(
            previous_instr.start_in_clocks - previous_instr_end,
            previous_instr.start_in_clocks - current_clock,
        )

        current_clock += seqc_il_generator.add_wait(
            seqc_gen,
            remaining_clocks + (current_clock - clock_start) + dead_time_in_clocks,
            device.device_type,
            comment=f"\t// clock={current_clock}",
        )
    else:
        seqc_gen.emit_comment("Dead time")
        current_clock += seqc_il_generator.add_wait(
            seqc_gen,
            dead_time_in_clocks,
            device.device_type,
            comment=f"\t// clock={current_clock}",
        )

    seqc_gen.emit_end_repeat()

    # Reset trigger
    if is_marker_source:
        seqc_il_generator.add_set_trigger(
            seqc_gen, 0, device.device_type, comment=f"\t// clock={current_clock}"
        )

    return (seqc_gen.generate(), command_table.to_json())


# pylint: disable=too-many-locals
def _compile_for_uhfqa(
    device: zhinst.Device,
    cached_schedule: schedule_helpers.CachedSchedule,
    settings_builder: zi_settings.ZISettingsBuilder,
) -> Tuple[zi_settings.ZISettingsBuilder, ZIAcquisitionConfig]:
    """
    Initialize programming the UHFQA ZI Instrument.

    Creates a sequence program and converts schedule
    pulses to waveforms for the UHFQA.

    Parameters
    ----------
    device :
    cached_schedule :
    settings_builder :

    Returns
    -------
    :
    """

    instrument_info = zhinst.InstrumentInfo(
        clock_rate=device.clock_rate,
        resolution=8,
        granularity=WAVEFORM_GRANULARITY[device.device_type],
    )
    channels = device.channels
    channels = list(filter(lambda c: c.mode == enums.SignalModeType.REAL, channels))

    awg_index = 0
    channel = channels[awg_index]
    logger.debug(f"[{device.name}-awg{awg_index}] {str(device)}")
    mixer_corrections = (
        channel.mixer_corrections
        if not channel.mixer_corrections is None
        else common.MixerCorrections()
    )
    settings_builder.with_defaults(
        [
            ("awgs/0/single", 1),
            ("qas/0/rotations/*", (1 + 1j)),
            ("qas/0/integration/sources/*", 0),
        ]
    ).with_sigouts(0, (1, 1)).with_awg_time(
        0, device.clock_select
    ).with_qas_integration_weights_real(
        range(NUM_UHFQA_READOUT_CHANNELS), np.zeros(MAX_QAS_INTEGRATION_LENGTH)
    ).with_qas_integration_weights_imag(
        range(NUM_UHFQA_READOUT_CHANNELS), np.zeros(MAX_QAS_INTEGRATION_LENGTH)
    ).with_sigout_offset(
        0, mixer_corrections.dc_offset_I
    ).with_sigout_offset(
        1, mixer_corrections.dc_offset_Q
    )

    logger.debug(f"[{device.name}-awg{awg_index}] channel={str(channel)}")

    instructions = get_execution_table(
        cached_schedule,
        instrument_info,
        channel,
    )

    # Generate a dictionary of uuid(s) and zhinst.Wave instructions
    wave_instructions_dict: Dict[int, zhinst.Wave] = dict(
        (i.uuid, i) for i in instructions if isinstance(i, zhinst.Wave)
    )

    # Create a list of all pulse_id(s).
    pulse_ids: List[int] = wave_instructions_dict.keys()

    # Generate map containing waveform the location of a pulse_id.
    waveform_table: Dict[int, int] = zi_helpers.get_waveform_table(
        pulse_ids, cached_schedule.pulseid_pulseinfo_dict
    )

    # Create a dictionary of uuid(s) and numerical waveforms.
    waveforms_dict: Dict[int, np.ndarray] = dict(
        (uuid, wf_instr.waveform) for uuid, wf_instr in wave_instructions_dict.items()
    )

    # Create a dictionary of uuid(s) and zhinst.Measure instructions
    n_acquisitions = sum(isinstance(x, zhinst.Measure) for x in instructions)
    measure_instructions_dict: Dict[int, zhinst.Measure] = dict(
        (i.uuid, i) for i in instructions if isinstance(i, zhinst.Measure)
    )

    # Generate and apply sequencer program
    seqc = _assemble_uhfqa_sequence(
        cached_schedule=cached_schedule,
        device=device,
        instrument_info=instrument_info,
        output=device.channel_0,
        waveform_table=waveform_table,
        instructions=instructions,
    )
    logger.debug(seqc)

    settings_builder.with_compiler_sourcestring(awg_index, seqc)

    # Apply waveforms to AWG
    _add_wave_nodes(device, awg_index, waveforms_dict, waveform_table, settings_builder)

    # Get a list of all acquisition protocol channels
    acq_channel_resolvers_map: Dict[int, Callable[..., Any]] = dict()

    # the unique acquisitions are acquisitions
    unique_acquisition_hashes = []

    for acq_uuid, acq_info in cached_schedule.acqid_acqinfo_dict.items():

        # the acquisition index is not required for configuring the integration weights.
        # we use a hash to identify which acquisitions are identical in this context.
        acq_hash = make_hash(acq_info.copy().pop("acq_index"))
        if acq_hash in unique_acquisition_hashes:
            continue

        unique_acquisition_hashes.append(acq_hash)

        acq_protocol: str = acq_info["protocol"]
        acq_duration: float = acq_info["duration"]
        acq_channel: int = acq_info["acq_channel"]

        integration_length = round(acq_duration * instrument_info.clock_rate)
        logger.debug(
            f"[{device.name}] acq_info={acq_info} "
            + f" acq_duration={acq_duration} integration_length={integration_length}"
        )

        settings_builder.with_qas_integration_mode(
            zhinst.QasIntegrationMode.NORMAL
        ).with_qas_integration_length(integration_length).with_qas_result_enable(
            False
        ).with_qas_monitor_enable(
            False
        ).with_qas_delay(
            0
        )

        if acq_protocol == "trace":
            # Disable Weighted integration because we'd like to see
            # the raw signal.
            settings_builder.with_qas_monitor_enable(True).with_qas_monitor_averages(
                cached_schedule.schedule.repetitions
            ).with_qas_monitor_length(
                integration_length
            ).with_qas_integration_weights_real(
                range(NUM_UHFQA_READOUT_CHANNELS), np.ones(MAX_QAS_INTEGRATION_LENGTH)
            ).with_qas_integration_weights_imag(
                range(NUM_UHFQA_READOUT_CHANNELS), np.ones(MAX_QAS_INTEGRATION_LENGTH)
            )

            monitor_nodes = (
                "qas/0/monitor/inputs/0/wave",
                "qas/0/monitor/inputs/1/wave",
            )
            acq_channel_resolvers_map[acq_channel] = partial(
                resolvers.monitor_acquisition_resolver, monitor_nodes=monitor_nodes
            )
        else:
            measure_instruction: zhinst.Measure = measure_instructions_dict[acq_uuid]
            # Combine a reset and setting acq weights
            # by slicing the length of the waveform I and Q values.
            # This overwrites 0..length with new values.
            # The waveform is slightly larger then the integration_length
            # because of the waveform granularity. This is irrelevant
            # due to the waveform being appended with zeros. Therefore
            # avoiding an extra slice of waveform[0:integration_length]

            weights_i = np.zeros(MAX_QAS_INTEGRATION_LENGTH)
            weights_q = np.zeros(MAX_QAS_INTEGRATION_LENGTH)

            weights_i[
                0 : len(measure_instruction.weights_i)
            ] = measure_instruction.weights_i
            weights_q[
                0 : len(measure_instruction.weights_q)
            ] = measure_instruction.weights_q

            settings_builder.with_qas_result_mode(
                zhinst.QasResultMode.CYCLIC
            ).with_qas_result_source(
                zhinst.QasResultSource.INTEGRATION
            ).with_qas_result_length(
                n_acquisitions
            ).with_qas_result_enable(
                True
            ).with_qas_result_averages(
                cached_schedule.schedule.repetitions
            )

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

    settings_builder.with_qas_result_reset(0).with_qas_result_reset(1)
    settings_builder.with_qas_monitor_reset(0).with_qas_monitor_reset(1)

    return settings_builder, ZIAcquisitionConfig(
        n_acquisitions, acq_channel_resolvers_map
    )


# pylint: disable=too-many-arguments
# pylint: disable=too-many-statements
# pylint: disable=too-many-locals
# pylint: disable=too-many-branches
def _assemble_uhfqa_sequence(
    cached_schedule: schedule_helpers.CachedSchedule,
    device: zhinst.Device,
    instrument_info: zhinst.InstrumentInfo,
    output: zhinst.Output,
    waveform_table: Dict[int, int],
    instructions: List[zhinst.Instruction],
) -> str:
    """
    Assembles a new sequence program for the UHFQA.

    The UHFQA will be treated as a slave device. This means that
    the UHFQA will wait for the HDAWG to send a trigger in order
    to start measuring.

    Parameters
    ----------
    cached_schedule :
    device :
    instrument_info:
    output :
    waveform_table :
    instructions :

    Returns
    -------
    :
    """

    seqc_gen = seqc_il_generator.SeqcILGenerator()
    seqc_info = seqc_il_generator.SeqcInfo(
        cached_schedule,
        output,
        instrument_info.low_res_clock,
    )

    dead_time_in_clocks = (
        seqc_info.schedule_duration_in_clocks + seqc_info.schedule_offset_in_clocks
    ) - seqc_info.timeline_end_in_clocks

    seqc_il_generator.add_seqc_info(seqc_gen, seqc_info)

    has_triggers: bool = len(output.triggers) > 0
    is_awaiting_trigger = (
        device.ref == enums.ReferenceSourceType.EXTERNAL and has_triggers
    )
    is_marker_source = device.ref == enums.ReferenceSourceType.INTERNAL and has_triggers

    current_clock: int = 0

    # Declare sequence variables

    seqc_gen.declare_var("__repetitions__", cached_schedule.schedule.repetitions)

    seqc_il_generator.add_csv_waveform_variables(
        seqc_gen, device.name, 0, waveform_table
    )

    seqc_gen.emit_begin_repeat("__repetitions__")

    if is_awaiting_trigger:
        seqc_gen.emit_wait_dig_trigger(
            output.triggers[0],
            comment=f"\t// clock={current_clock}",
        )

    if is_awaiting_trigger and seqc_info.line_trigger_delay_in_seconds != -1:
        seqc_il_generator.add_wait(
            seqc_gen,
            seqc_info.line_trigger_delay_in_clocks,
            device.device_type,
            comment=f"clock={current_clock}",
        )

    if seqc_info.timeline_start_in_clocks > 0:
        current_clock += seqc_il_generator.add_wait(
            seqc_gen,
            seqc_info.timeline_start_in_clocks,
            device.device_type,
            f"clock={current_clock}",
        )

    instructions_iter = iter(instructions)
    current_instr: zhinst.Instruction = zhinst.Instruction.default()
    previous_instr: zhinst.Instruction = next(instructions_iter)

    for instruction in instructions_iter:
        current_instr = instruction
        previous_instr_end = (
            previous_instr.start_in_clocks + previous_instr.duration_in_clocks
        )
        current_instr_offset = (
            seqc_il_generator.SEQC_INSTR_CLOCKS[device.device_type][
                seqc_il_generator.SeqcInstructions.SET_TRIGGER
            ]
            if isinstance(current_instr, zhinst.Measure)
            else seqc_il_generator.SEQC_INSTR_CLOCKS[device.device_type][
                seqc_il_generator.SeqcInstructions.PLAY_WAVE
            ]
        )

        if isinstance(previous_instr, zhinst.Measure):
            seqc_gen.emit_start_qa_monitor()
            # currently the generic (measure all channels) QAResult is used.
            # by generating a bitmask from the Measure instruction this can be improved.
            seqc_gen.emit_start_qa_result()
            seqc_gen.emit_blankline()  # add in a white line for visual separation
        else:
            current_clock += seqc_il_generator.add_play_wave(
                seqc_gen,
                f"w{waveform_table[previous_instr.uuid]:d}",
                device_type=device.device_type,
                comment=f"clock={current_clock}",
            )

        remaining_clocks = max(
            current_instr.start_in_clocks - previous_instr_end,
            current_instr.start_in_clocks - current_clock,
        )
        clock_cycles_to_wait: int = remaining_clocks - current_instr_offset
        if clock_cycles_to_wait > 0:
            # UHFQA wait instruction uses a 1 based number of clock-cycles
            current_clock += seqc_il_generator.add_wait(
                seqc_gen,
                clock_cycles_to_wait,
                device.device_type,
                comment=f"\t// clock={current_clock}",
            )

        previous_instr = current_instr

    # Add the last operation to the sequencer program.
    clock_start: int = current_clock
    if previous_instr.uuid != -1:
        previous_instr_end = (
            previous_instr.start_in_clocks + previous_instr.duration_in_clocks
        )

        if isinstance(previous_instr, zhinst.Measure):
            # Reset the integration
            seqc_gen.emit_start_qa_monitor()
            # currently the generic (measure all channels) QAResult is used.
            # by generating a bitmask from the Measure instruction this can be improved.
            seqc_gen.emit_start_qa_result()
            seqc_gen.emit_blankline()  # add in a white line for visual separation

            # Adds a waiting time after playing the last sequence to wait
            # for the QAS to process and not time out.

            if device.last_seq_wait_clocks < 2000:
                logger.warning(
                    f"The last_seq_wait_clocks={device.last_seq_wait_clocks}\n"
                    + "is less than 2000!\n"
                    + "Proceed with caution. Terminate and increase the\n"
                    + "number if the QAS has an integration error!"
                )
            current_clock += seqc_il_generator.add_wait(
                seqc_gen,
                device.last_seq_wait_clocks,
                device.device_type,
                comment=f"\t// clock={current_clock}",
            )
        else:
            current_clock += seqc_il_generator.add_play_wave(
                seqc_gen,
                f"w{waveform_table[previous_instr.uuid]:d}",
                device_type=device.device_type,
                comment=f"clock={current_clock}",
            )

        remaining_clocks = max(
            previous_instr.start_in_clocks - previous_instr_end,
            previous_instr.start_in_clocks - current_clock,
        )

        if is_marker_source:
            # Wait with additional dead-time
            seqc_gen.emit_comment("Final wait + dead time")

            current_clock += seqc_il_generator.add_wait(
                seqc_gen,
                remaining_clocks + (current_clock - clock_start) + dead_time_in_clocks,
                device.device_type,
                comment=f"\t// clock={current_clock}",
            )
    elif is_marker_source:
        current_clock += seqc_il_generator.add_wait(
            seqc_gen,
            dead_time_in_clocks,
            device.device_type,
            comment=f"\t// clock={current_clock}",
        )

    seqc_gen.emit_end_repeat()

    seqc_il_generator.add_set_trigger(
        seqc_gen,
        0,
        device.type,
        comment="Reset triggers",
    )

    return seqc_gen.generate()
