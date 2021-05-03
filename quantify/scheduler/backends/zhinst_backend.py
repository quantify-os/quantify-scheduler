# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
"""Backend for Zurich Instruments."""
from __future__ import annotations

import logging
from functools import partial
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Tuple,
    Union,
    cast,
)

import numpy as np
from zhinst.toolkit.helpers import Waveform
from qcodes.instrument.base import Instrument
import quantify.scheduler.waveforms as waveforms
from quantify.scheduler import enums
from quantify.scheduler import types
from quantify.scheduler.backends.types import zhinst
from quantify.scheduler.backends.zhinst import helpers as zi_helpers
from quantify.scheduler.backends.zhinst import seqc_il_generator
from quantify.scheduler.backends.zhinst import resolvers
from quantify.scheduler.helpers import schedule as schedule_helpers
from quantify.scheduler.helpers import waveforms as waveform_helpers

if TYPE_CHECKING:
    from zhinst.qcodes import UHFQA
    from zhinst.qcodes import HDAWG
    from zhinst.qcodes.base import ZIBaseInstrument
    from zhinst.qcodes.hdawg import AWG as HDAWG_CORE
    from zhinst.qcodes.uhfqa import AWG as UHFQA_CORE


logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter(
    # "%(levelname)-8s | %(module)s | %(funcName)s::%(lineno)s. %(message)s"
)
handler.setFormatter(formatter)
logger.addHandler(handler)

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

SEQUENCE_DEAD_TIME = 5e-06  # 5 us


def _validate_schedule(schedule: types.Schedule) -> None:
    """
    Validates the Schedule required values for creating the backend.

    Parameters
    ----------
    schedule :
        The :class:`~quantify.scheduler.types.Schedule`

    Raises
    ------
    ValueError
        The validation error.
    """
    if len(schedule.timing_constraints) == 0:
        raise ValueError(f"Undefined timing contraints for schedule '{schedule.name}'!")

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

    Returns
    -------
    Tuple[int, int, np.ndarray]
    """

    (start_in_seconds, duration_in_seconds) = start_and_duration_in_seconds

    if is_pulse:
        # Modulate the waveform
        if output.modulation == enums.ModulationModeType.PREMODULATE:
            t: np.ndarray = np.arange(
                0, 0 + duration_in_seconds, 1 / instrument_info.clock_rate
            )
            waveform = waveforms.modulate_wave(
                t=t, wave=waveform, freq_mod=output.interm_freq
            )

        # TODO [TReynders] Add mixer corrections here.  pylint: disable=fixme

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
    collection : Dict[Any, Any]

    Returns
    -------
    Iterable[Tuple[Any, Any]]
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
    timeslot_index :t
    output :
    cached_schedule :
    instrument_info :

    Returns
    -------
    zhinst._Wave
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

    (
        corrected_start_in_clocks,
        n_samples_shifted,
        waveform,
    ) = apply_waveform_corrections(
        output,
        waveform,
        (t0, duration_in_seconds),
        instrument_info,
        True,
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


def get_measure_instruction(
    uuid: int,
    timeslot_index: int,
    output: zhinst.Output,
    cached_schedule: schedule_helpers.CachedSchedule,
    instrument_info: zhinst.InstrumentInfo,
) -> zhinst.Measure:
    """
    Returns the measurement sequence instruction.

    This

    Parameters
    ----------
    uuid :
    timeslot_index :
    output :
    cached_schedule :
    instrument_info :

    Returns
    -------
    zhinst._Measure
    """
    acq_info = cached_schedule.acqid_acqinfo_dict[uuid]
    abs_time = (
        cached_schedule.schedule.timing_constraints[timeslot_index]["abs_time"]
        - cached_schedule.start_offset_in_seconds
    )

    t0: float = abs_time + acq_info["t0"]
    duration_in_seconds: float = acq_info["duration"]

    weights_list: List[np.ndarray] = [np.empty((0,)), np.empty((0,))]
    corrected_start_in_clocks: int = 0
    for i, pulse_info in enumerate(acq_info["waveforms"]):
        waveform = waveform_helpers.exec_waveform_partial(
            schedule_helpers.get_pulse_uuid(pulse_info),
            cached_schedule.pulseid_waveformfn_dict,
            instrument_info.clock_rate,
        )
        (corrected_start_in_clocks, _, waveform) = apply_waveform_corrections(
            output,
            waveform,
            (t0, duration_in_seconds),
            instrument_info,
            False,
        )
        weights_list[i] = waveform

    return zhinst.Measure(
        uuid,
        abs_time,
        timeslot_index,
        t0,
        corrected_start_in_clocks,
        duration_in_seconds,
        round(duration_in_seconds / instrument_info.low_res_clock),
        *weights_list,
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

    Returns
    -------
    List[zhinst._Instruction]

    Raises
    ------
    RuntimeError
        Raised if encountered an unknown uuid.
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
            return get_measure_instruction(
                uuid, timeslot_index, output, cached_schedule, instrument_info
            )
        if uuid in cached_schedule.pulseid_pulseinfo_dict:
            return get_wave_instruction(
                uuid, timeslot_index, output, cached_schedule, instrument_info
            )

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
        current_instr.timeslot_index
    ] = new_timeslot_uuids

    return instructions


def setup_zhinst_backend(
    schedule: types.Schedule, hardware_map: Dict[str, Any]
) -> Dict[int, Callable[..., Any]]:
    """
    Initialize setting up Schedule for Zurich Instruments hardware.

    This method generates sequencer programs, waveforms and
    configures instruments defined in the hardware configuration
    dictionary.

    Parameters
    ----------
    schedule :
    hardware_map :

    Returns
    -------
    :
        The acquisition channel resolvers mapping.

    Raises
    ------
    NotImplementedError
        Thrown when using unsupported ZI Instruments.
    """
    _validate_schedule(schedule)

    # List of supported zhinst devices
    supported_devices: List[str] = ["HDAWG", "UHFQA"]

    # Create the context
    devices: Dict[str, Tuple[zhinst.Device, ZIBaseInstrument]] = dict()

    # Parse json hardware config and find qcodes instruments
    for device_dict in hardware_map["devices"]:
        device = zhinst.Device.from_dict(device_dict)

        qinstrument = Instrument.find_instrument(device.name)
        qinstrument_classname = qinstrument.__class__.__name__

        if qinstrument_classname not in supported_devices:
            raise NotImplementedError(
                f"Unable to create zhinst backend for '{qinstrument_classname}'!"
            )

        device.type = zhinst.DeviceType(qinstrument_classname)
        devices[device.name] = (device, qinstrument)

    acq_channel_resolvers_map: Dict[int, Callable[..., Any]] = dict()
    cached_schedule = schedule_helpers.CachedSchedule(schedule)

    # Program devices
    for (device, instrument) in devices.values():
        if device.type == zhinst.DeviceType.HDAWG:
            _program_hdawg(
                instrument,
                device,
                cached_schedule,
            )
        elif device.type == zhinst.DeviceType.UHFQA:
            acq_channel_resolvers_map.update(
                _program_uhfqa(instrument, device, cached_schedule)
            )

    return acq_channel_resolvers_map


def _program_modulation(
    awg: Union[HDAWG_CORE, UHFQA_CORE],
    device: zhinst.Device,
    output: zhinst.Output,
    waveforms_dict: Dict[int, np.ndarray],
    pulseid_pulseinfo_dict: Dict[int, Dict[str, Any]],
) -> None:
    """
    Programs modulation type to AWG core.

    Modulation type can be hardware of software modulation.

    Parameters
    ----------
    awg :
    device :
    output :
    waveforms_dict :
    pulseid_pulseinfo_dict :
    """

    if output.modulation == enums.ModulationModeType.PREMODULATE:
        logger.debug(f"[{awg.name}] pre-modulation enabled!")
        clock_rate: int = zi_helpers.get_clock_rate(device.type)

        # Pre-modulate the waveforms
        for pulse_id, waveform in waveforms_dict.items():
            pulse_info = pulseid_pulseinfo_dict[pulse_id]

            t: np.ndarray = np.arange(0, 0 + pulse_info["duration"], 1 / clock_rate)
            wave = waveforms.modulate_wave(
                t=t, wave=waveform, freq_mod=output.interm_freq
            )
            waveforms_dict[pulse_id] = wave

    elif output.modulation == enums.ModulationModeType.MODULATE:
        if device.type == zhinst.DeviceType.HDAWG:
            # Enabled hardware IQ modulation
            logger.debug(f"[{awg.name}] hardware modulation enabled!")
            awg.enable_iq_modulation()
            awg.modulation_freq(output.lo_freq + output.interm_freq)
            awg.modulation_phase_shift(output.phase_shift)
            awg.gain1(output.gain1)
            awg.gain2(output.gain2)
    else:
        if device.type == zhinst.DeviceType.HDAWG:
            awg.disable_iq_modulation()


def _set_waveforms(
    instrument: ZIBaseInstrument,
    awg: Union[HDAWG_CORE, UHFQA_CORE],
    waveforms_dict: Dict[int, np.ndarray],
    commandtable_map: Dict[int, int],
    destination: zhinst.WaveformDestination,
) -> None:
    """
    Sets the waveforms to WaveformDestination.

    Waveform destination is either CSV file based or via
    setting a wave vector to a AWG node.

    Parameters
    ----------
    instrument :
    awg :
    waveforms_dict :
    commandtable_map :
    pulseid_pulseinfo_dict :
    destination :
    """
    awg_directory = awg._awg._module.get_string("directory")
    csv_data_dir = Path(awg_directory).joinpath("awg", "waves")

    for pulse_id, commandtable_index in commandtable_map.items():
        array: np.ndarray = waveforms_dict[pulse_id]

        # https://www.zhinst.com/sites/default/files/documents/2020-08/LabOneProgrammingManual_20.07.0.pdf
        # page 229
        waveform_data = []
        # if signal_mode_type == enums.SignalModeType.COMPLEX:
        # Add I And Q complex values to the Waveform class
        # which will interleave the values converting them to a
        # native ZI vector.
        waveform = Waveform(array.real, array.imag)
        waveform_data = waveform.data
        # elif signal_mode_type == enums.SignalModeType.REAL:
        # waveform_data = array.real

        # logger.debug(waveform_data)

        if destination == zhinst.WaveformDestination.WAVEFORM_TABLE:
            zi_helpers.set_wave_vector(
                instrument, awg._awg._index, commandtable_index, waveform_data
            )
        elif destination == zhinst.WaveformDestination.CSV:
            csv_file = csv_data_dir.joinpath(
                f"{instrument._serial}_wave{commandtable_index}.csv"
            )
            waveform_data = np.reshape(waveform.data, (len(array), -1))
            np.savetxt(csv_file, waveform_data, delimiter=";")


def _program_hdawg(
    hdawg: HDAWG,
    device: zhinst.Device,
    cached_schedule: schedule_helpers.CachedSchedule,
) -> None:
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
    hdawg :
    device :
    cached_schedule :

    Raises
    ------
    ValueError
    """
    instrument_info = zhinst.InstrumentInfo(
        zi_helpers.get_clock_rate(device.type), 8, WAVEFORM_GRANULARITY[device.type]
    )
    devtype: str = hdawg.features.parameters["devtype"]()
    awg_count: int = 4 if devtype == "HDAWG8" else 2

    enabled_outputs: Dict[int, zhinst.Output] = dict()

    channelgroups = HDAWG_DEVICE_TYPE_CHANNEL_GROUPS[devtype]
    channelgroups_value = channelgroups[device.channelgrouping]
    sequencer_step = int(channelgroups_value / 2)
    sequencer_stop = min(len(device.channels), int(awg_count / sequencer_step))

    logger.debug(
        f"HDAWG[{hdawg.name}] devtype={devtype} awg_count={awg_count} {str(device)}"
    )

    zi_helpers.set_value(hdawg, "system/awg/channelgrouping", device.channelgrouping)

    logger.debug(f"[{hdawg.name}] resetting outputs")
    for i in range(awg_count):
        awg = hdawg.awgs[i]
        awg.output1("off")
        awg.output2("off")

    logger.debug(f"[{hdawg.name}] step={sequencer_step} stop={sequencer_stop}")
    i = 0
    for awg_index in range(0, sequencer_stop, sequencer_step):
        output = device.channels[i]
        if output is None:
            raise ValueError(f"Required output at index '{i}' is undefined!")

        awg = hdawg.awgs[awg_index]
        logger.debug(f"[{awg.name}] enabling outputs...")
        awg.output1("on")
        awg.output2("on")
        enabled_outputs[awg_index] = output
        i += 1

    for i, output in enabled_outputs.items():
        awg = hdawg.awgs[i]

        if output.port not in cached_schedule.port_timeline_dict:
            logging.warning(
                f"[{awg.name}] Skipping! Missing pulses for port={output.port}."
            )
            continue

        instructions = get_execution_table(
            cached_schedule,
            instrument_info,
            output,
        )
        pulse_ids: List[int] = list(map(lambda i: i.uuid, instructions))

        # Gets a dictionary of pulse_id by commandtable_indexes
        commandtable_map: Dict[int, int] = zi_helpers.get_commandtable_map(
            pulse_ids, cached_schedule.pulseid_pulseinfo_dict
        )

        # Step 1: Generate and compile sequencer program AND
        # Step 2: Set CommandTable JSON vector
        _program_sequences_hdawg(
            hdawg,
            awg,
            cached_schedule,
            device,
            instrument_info,
            output,
            commandtable_map,
            instructions,
        )

        # Step 3: Upload waveforms to AWG CommandTable
        waveforms_dict = dict(map(lambda i: (i.uuid, i.waveform), instructions))
        _set_waveforms(
            hdawg,
            awg,
            waveforms_dict,
            commandtable_map,
            zhinst.WaveformDestination.WAVEFORM_TABLE,
        )


def _program_sequences_hdawg(
    hdawg: HDAWG,
    awg: HDAWG_CORE,
    cached_schedule: schedule_helpers.CachedSchedule,
    device: zhinst.Device,
    instrument_info: zhinst.InstrumentInfo,
    output: zhinst.Output,
    commandtable_map: Dict[int, int],
    instructions: List[zhinst.Instruction],
) -> None:
    """
    Assembles a new sequence program for the HDAWG.

    The HDAWG acts as a master device. This means that the
    HDAWG sends a trigger to slave devices which can be used
    to start measuring.

    Parameters
    ----------
    hdawg :
    awg :
    cached_schedule :
    device :
    instrument_info :
    output :
    commandtable_map :
    instructions :
    """

    seqc_gen = seqc_il_generator.SeqcILGenerator()
    seqc_info = seqc_il_generator.SeqcInfo(
        cached_schedule,
        output,
        instrument_info.low_res_clock,
    )

    dead_time_in_clocks = (
        seqc_info.schedule_offset_in_clocks
        if seqc_info.schedule_offset_in_seconds > 0
        else SEQUENCE_DEAD_TIME
    )

    seqc_il_generator.add_seqc_info(seqc_gen, seqc_info)

    is_master_awg: bool = awg.index == 0
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
    for pulse_id, waveform_index in commandtable_map.items():
        instruction = wave_instructions_dict[pulse_id]
        waveform_index = commandtable_map[instruction.uuid]
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

    # Reset marker
    if is_marker_source:
        seqc_il_generator.add_set_trigger(seqc_gen, 0, device.type)

    seqc_gen.emit_begin_repeat("__repetitions__")

    if is_marker_source:
        seqc_il_generator.add_set_trigger(seqc_gen, output.markers, device.type)

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
            device.type,
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
        current_instr_offset = seqc_il_generator.SEQC_INSTR_CLOCKS[device.type][
            seqc_il_generator.SeqcInstructions.EXECUTE_TABLE_ENTRY
        ]

        current_clock += seqc_il_generator.add_execute_table_entry(
            seqc_gen,
            commandtable_map[previous_instr.uuid],
            device.type,
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
            device.type,
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
            commandtable_map[previous_instr.uuid],
            device.type,
            f"clock={current_clock}",
        )

    # Reset trigger each iteration
    if is_marker_source:
        current_clock += seqc_il_generator.add_set_trigger(
            seqc_gen, 0, device.type, comment=f"clock={current_clock}"
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
            device.type,
            comment=f"\t// clock={current_clock}",
        )
    else:
        seqc_gen.emit_comment("Dead time")
        current_clock += seqc_il_generator.add_wait(
            seqc_gen,
            dead_time_in_clocks,
            device.type,
            comment=f"\t// clock={current_clock}",
        )

    seqc_gen.emit_end_repeat()

    # Reset trigger
    if is_marker_source:
        seqc_il_generator.add_set_trigger(
            seqc_gen, 0, device.type, comment=f"\t// clock={current_clock}"
        )

    seqc_program = seqc_gen.generate()
    seqc_path: Path = zi_helpers.write_seqc_file(awg, seqc_program, f"{awg.name}.seqc")
    logger.debug(seqc_program)

    awg.set_sequence_params(
        sequence_type="Custom",
        path=str(seqc_path),
    )
    awg.compile()

    json_str: str = zhinst.CommandTable(table=command_table_entries).to_json()
    zi_helpers.set_commandtable_data(hdawg, awg._awg._index, json_str)
    logger.debug(json_str)


def _program_uhfqa(
    uhfqa: UHFQA,
    device: zhinst.Device,
    cached_schedule: schedule_helpers.CachedSchedule,
) -> Dict[int, Callable[..., Any]]:
    """
    Initialize programming the UHFQA ZI Instrument.

    Creates a sequence program and converts schedule
    pulses to waveforms for the UHFQA.

    Parameters
    ----------
    uhfqa :
    device :
    cached_schedule :
    """
    instrument_info = zhinst.InstrumentInfo(
        zi_helpers.get_clock_rate(device.type), 8, WAVEFORM_GRANULARITY[device.type]
    )
    channels = device.channels
    channels = list(filter(lambda c: c.mode == enums.SignalModeType.REAL, channels))

    logger.debug(f"UHFQA[{uhfqa.name}] {str(device)}")

    awg = uhfqa.awg
    logger.debug(f"[{awg.name}] enabling outputs...")
    awg.output1("on")
    awg.output2("on")

    uhfqa.disable_readout_channels()

    channel = channels[0]
    logger.debug(f"[{awg.name}] channel={str(channel)}")

    instructions = get_execution_table(
        cached_schedule,
        instrument_info,
        channel,
    )

    # Build a list of unique pulse ids for all channels
    wave_instructions_dict: Dict[int, zhinst.Wave] = dict(
        (i.uuid, i) for i in instructions if isinstance(i, zhinst.Wave)
    )
    pulse_ids: List[int] = wave_instructions_dict.keys()

    # Gets a dictionary of pulse_id by commandtable_indexes
    commandtable_map: Dict[int, int] = zi_helpers.get_commandtable_map(
        pulse_ids, cached_schedule.pulseid_pulseinfo_dict
    )
    waveforms_dict: Dict[int, np.ndarray] = dict(
        (k, v.waveform) for k, v in wave_instructions_dict.items()
    )

    measure_instructions_dict: Dict[int, zhinst.Measure] = dict(
        (i.uuid, i) for i in instructions if isinstance(i, zhinst.Measure)
    )

    # Apply waveforms to AWG
    _set_waveforms(
        uhfqa,
        awg,
        waveforms_dict,
        commandtable_map,
        zhinst.WaveformDestination.CSV,
    )

    # Get a list of all acquisition protocol channels
    acq_channel_resolvers_map: Dict[int, Callable[..., Any]] = dict()
    readout_channel_index: int = 0

    for acq_uuid, acq_info in cached_schedule.acqid_acqinfo_dict.items():
        acq_protocol: str = acq_info["protocol"]
        acq_duration: float = acq_info["duration"]
        acq_channel: int = acq_info["acq_channel"]

        integration_length = round(acq_duration * instrument_info.clock_rate)
        logger.debug(
            f"[{uhfqa.name}] acq_info={acq_info} "
            + f" acq_duration={acq_duration} integration_length={integration_length}"
        )

        zi_helpers.set_qas_parameters(
            uhfqa,
            integration_length,
            zhinst.QasIntegrationMode.NORMAL,
        )

        if acq_protocol == "trace":
            weights = np.ones(4096)
            # Set the input monitor length
            zi_helpers.set_value(uhfqa, "qas/0/monitor/length", integration_length)

            for i in range(10):
                # Disables weighted integration for this channel
                zi_helpers.set_integration_weights(uhfqa, i, weights, weights)

            monitor_nodes = (
                "qas/0/monitor/inputs/0/wave",
                "qas/0/monitor/inputs/1/wave",
            )
            acq_channel_resolvers_map[acq_channel] = partial(
                resolvers.monitor_acquisition_resolver, uhfqa, monitor_nodes
            )
        else:
            uhfqa.result_source("Integration")

            assert readout_channel_index < 10
            readout_channel = uhfqa.channels[readout_channel_index]

            # Set readout channel rotation
            readout_channel.rotation(0)

            # Enables weighted integration for this channel
            weights_i = [0] * 4096
            weights_q = [0] * 4096

            measure_instruction: zhinst.Measure = measure_instructions_dict[acq_uuid]

            # Combine a reset and setting acq weights
            # by slicing the length of the waveform I and Q values.
            # This overwrites 0..length with new values.
            # The waveform is slightly larger then the integration_length
            # because of the waveform granularity. This is irrelevant
            # due to the waveform being appended with zeros. Theirfore
            # avoiding an extra slice of waveform[0:integration_length]
            waveform_i = measure_instruction.weights_i
            waveform_q = measure_instruction.weights_q

            weights_i[0 : len(waveform_i)] = np.real(waveform_i)
            weights_q[0 : len(waveform_q)] = np.imag(waveform_q)

            zi_helpers.set_integration_weights(
                uhfqa, readout_channel_index, weights_i, weights_q
            )

            # Create partial function for delayed execution
            acq_channel_resolvers_map[acq_channel] = partial(
                resolvers.result_acquisition_resolver,
                uhfqa,
                f"qas/0/result/data/{readout_channel_index}/wave",
            )

            readout_channel_index += 1

    # Generate and apply sequencer program
    _program_sequences_uhfqa(
        uhfqa,
        awg,
        cached_schedule,
        device,
        instrument_info,
        device.channel_0,
        commandtable_map,
        instructions,
    )

    return acq_channel_resolvers_map


def _program_sequences_uhfqa(
    uhfqa: UHFQA,
    awg: UHFQA_CORE,
    cached_schedule: schedule_helpers.CachedSchedule,
    device: zhinst.Device,
    instrument_info: zhinst.InstrumentInfo,
    output: zhinst.Output,
    commandtable_map: Dict[int, int],
    instructions: List[zhinst.Instruction],
) -> None:
    """
    Assembles a new sequence program for the UHFQA.

    The UHFQA will be treated as a slave device. This means that
    the UHFQA will wait for the HDAWG to send a trigger in order
    to start measuring.

    Parameters
    ----------
    uhfqa :
    awg :
    cached_schedule :
    device :
    instrument_info:
    output :
    commandtable_map :
    instructions :
    """

    seqc_gen = seqc_il_generator.SeqcILGenerator()
    seqc_info = seqc_il_generator.SeqcInfo(
        cached_schedule,
        output,
        instrument_info.low_res_clock,
    )

    dead_time_in_clocks = (
        seqc_info.schedule_offset_in_clocks
        if seqc_info.schedule_offset_in_seconds > 0
        else SEQUENCE_DEAD_TIME
    )
    seqc_il_generator.add_seqc_info(seqc_gen, seqc_info)
    acquisition_triggers = [
        "AWG_INTEGRATION_ARM",
        "AWG_INTEGRATION_TRIGGER",
        "AWG_MONITOR_TRIGGER",
    ]
    has_triggers: bool = len(output.triggers) > 0
    is_trigger_source = (
        device.ref == enums.ReferenceSourceType.EXTERNAL and has_triggers
    )
    is_marker_source = device.ref == enums.ReferenceSourceType.INTERNAL and has_triggers

    current_clock: int = 0

    # Declare sequence variables
    seqc_gen.declare_var("__repetitions__", cached_schedule.schedule.repetitions)
    seqc_gen.declare_var("integration_trigger", acquisition_triggers)

    seqc_il_generator.add_csv_waveform_variables(
        seqc_gen, uhfqa._serial, commandtable_map
    )
    seqc_gen.emit_begin_repeat("__repetitions__")

    if is_trigger_source:
        seqc_gen.emit_wait_dig_trigger(
            output.triggers[0],
            comment=f"\t// clock={current_clock}",
        )

    if is_trigger_source and seqc_info.line_trigger_delay_in_seconds != -1:
        seqc_il_generator.add_wait(
            seqc_gen,
            seqc_info.line_trigger_delay_in_clocks,
            device.type,
            comment=f"clock={current_clock}",
        )

    if seqc_info.timeline_start_in_clocks > 0:
        current_clock += seqc_il_generator.add_wait(
            seqc_gen,
            seqc_info.timeline_start_in_clocks,
            device.type,
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
            seqc_il_generator.SEQC_INSTR_CLOCKS[device.type][
                seqc_il_generator.SeqcInstructions.ARM_INTEGRATION
            ]
            if isinstance(current_instr, zhinst.Measure)
            else seqc_il_generator.SEQC_INSTR_CLOCKS[device.type][
                seqc_il_generator.SeqcInstructions.PLAY_WAVE
            ]
        )

        if isinstance(previous_instr, zhinst.Measure):
            current_clock += seqc_il_generator.add_set_trigger(
                seqc_gen,
                "integration_trigger",
                device.type,
                comment=f"clock={current_clock}",
            )
        else:
            current_clock += seqc_il_generator.add_play_wave(
                seqc_gen,
                f"w{commandtable_map[previous_instr.uuid]:d}",
                device.type,
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
                device.type,
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
            current_clock += seqc_il_generator.add_set_trigger(
                seqc_gen,
                "integration_trigger",
                device.type,
                comment=f"clock={current_clock}",
            )
        else:
            current_clock += seqc_il_generator.add_play_wave(
                seqc_gen,
                f"w{commandtable_map[previous_instr.uuid]:d}",
                device.type,
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
                device.type,
                comment=f"\t// clock={current_clock}",
            )
    elif is_marker_source:
        current_clock += seqc_il_generator.add_wait(
            seqc_gen,
            dead_time_in_clocks,
            device.type,
            comment=f"\t// clock={current_clock}",
        )

    seqc_gen.emit_end_repeat()

    seqc_program = seqc_gen.generate()
    seqc_path: Path = zi_helpers.write_seqc_file(awg, seqc_program, f"{awg.name}.seqc")
    logger.debug(seqc_program)

    awg.set_sequence_params(
        sequence_type="Custom",
        path=str(seqc_path),
    )
    awg.compile()
