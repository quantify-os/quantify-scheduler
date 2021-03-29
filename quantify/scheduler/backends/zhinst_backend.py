# -----------------------------------------------------------------------------
# Description:    Backend for Zurich Instruments.
# Repository:     https://gitlab.com/quantify-os/quantify-scheduler
# Copyright (C) Qblox BV & Orange Quantum Systems Holding BV (2020-2021)
# -----------------------------------------------------------------------------
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Tuple, Union

import numpy as np
from qcodes.instrument.base import Instrument
from zhinst.toolkit.helpers import Waveform

import quantify.scheduler.waveforms as waveforms
from quantify.scheduler import enums, math, types
from quantify.scheduler.backends.types import zhinst
from quantify.scheduler.backends.zhinst import helpers as zi_helpers
from quantify.scheduler.backends.zhinst import resolvers, seqc_il_generator
from quantify.scheduler.helpers import schedule as schedule_helpers
from quantify.scheduler.helpers import waveforms as waveform_helpers

if TYPE_CHECKING:
    from zhinst.qcodes import HDAWG, UHFQA
    from zhinst.qcodes.base import ZIBaseInstrument
    from zhinst.qcodes.hdawg import AWG as HDAWG_CORE
    from zhinst.qcodes.uhfqa import AWG as UHFQA_CORE


logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter(
    # "%(levelname)-8s | %(module)s | %(funcName)s::%(lineno)s. %(message)s"
    # "%(asctime)s | %(levelname)-8s | %(module)s | %(funcName)s::%(lineno)s. %(message)s"
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


class _WaveformDestination(Enum):

    """The waveform destination enum type."""

    CSV = 0
    WAVEFORM_TABLE = 1


@dataclass(frozen=True)
class _Pulse:
    id: int
    abs_time: float
    t0: float
    duration: float
    start_in_clocks: int
    duration_in_clocks: int
    size: int

    def __repr__(self):
        return "%-20i | %-16f | %-16f | %-16f | %-16f | %-14i | %-18i | %-6i" % (
            self.id,
            self.t0,
            self.abs_time * 1e9,
            (self.abs_time + self.t0) * 1e9,
            self.duration * 1e9,
            self.start_in_clocks,
            self.duration_in_clocks,
            self.size,
        )


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


class _Context(object):
    def __init__(self):
        self._devices: Dict[str, Tuple[zhinst.Device, ZIBaseInstrument]] = dict()

    def add_device(self, device: zhinst.Device, instrument: ZIBaseInstrument):
        self._devices[device.name] = (device, instrument)

    @property
    def devices(self) -> Iterable[Tuple[zhinst.Device, ZIBaseInstrument]]:
        return self._devices.values()


def _get_waveform_size(waveform: np.ndarray, granularity: int) -> int:
    size: int = len(waveform)
    if size % granularity != 0:
        size = math.closest_number_ceil(size, granularity)

    return size


def setup_zhinst_backend(
    schedule: types.Schedule, hardware_map: Dict[str, Any]
) -> Dict[int, Callable]:
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
    Dict[int, Callable]
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
    context: _Context = _Context()

    pulseid_waveformfn_dict: Dict[
        int, waveform_helpers.GetWaveformPartial
    ] = waveform_helpers.get_waveform_by_pulseid(schedule)
    pulseid_pulseinfo_dict: Dict[
        int, Dict[str, Any]
    ] = schedule_helpers.get_pulse_info_by_uuid(schedule)
    port_timeline_dict: Dict[
        str, Dict[int, List[int]]
    ] = schedule_helpers.get_port_timeline(schedule)
    acqid_acqinfo_dict = schedule_helpers.get_acq_info_by_uuid(schedule)

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
        context.add_device(device, qinstrument)

    acq_channel_resolvers_map: Dict[int, Callable] = dict()

    # Program devices
    for (device, instrument) in context.devices:
        if device.type == zhinst.DeviceType.HDAWG:
            _program_hdawg(
                instrument,
                device,
                schedule,
                pulseid_pulseinfo_dict,
                pulseid_waveformfn_dict,
                port_timeline_dict,
            )
        elif device.type == zhinst.DeviceType.UHFQA:
            acq_channel_resolvers_map.update(
                _program_uhfqa(
                    instrument,
                    device,
                    schedule,
                    pulseid_pulseinfo_dict,
                    acqid_acqinfo_dict,
                    pulseid_waveformfn_dict,
                    port_timeline_dict,
                )
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
    pulseid_pulseinfo_dict: Dict[int, Dict[str, Any]],
    destination: _WaveformDestination,
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
        pulse_info = pulseid_pulseinfo_dict[pulse_id]
        logger.debug(
            f"[{awg.name}] wave={pulse_info['wf_func']} I.length={len(array.real)} Q.length={len(array.imag)}"
        )

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

        logger.debug(waveform_data)

        if destination == _WaveformDestination.WAVEFORM_TABLE:
            zi_helpers.set_wave_vector(
                instrument, awg._awg._index, commandtable_index, waveform_data
            )
        elif destination == _WaveformDestination.CSV:
            csv_file = csv_data_dir.joinpath(
                f"{instrument._serial}_wave{commandtable_index}.csv"
            )
            waveform_data = np.reshape(waveform.data, (len(array), -1))
            np.savetxt(csv_file, waveform_data, delimiter=";")


def _program_hdawg(
    hdawg: HDAWG,
    device: zhinst.Device,
    schedule: types.Schedule,
    pulseid_pulseinfo_dict: Dict[int, Dict[str, Any]],
    pulseid_waveformfn_dict: Dict[int, waveform_helpers.GetWaveformPartial],
    port_timeline_dict: Dict[str, Dict[int, List[int]]],
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
    schedule :
    pulseid_pulseinfo_dict :
    pulseid_waveformfn_dict :
    port_timeline_dict :

    Raises
    ------
    ValueError
    """

    devtype: str = hdawg.features.parameters["devtype"]()
    awg_count: int = 4 if devtype == "HDAWG8" else 2
    clock_rate = zi_helpers.get_clock_rate(device.type)

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

        if output.port not in port_timeline_dict:
            logging.warning(
                f"[{awg.name}] Skipping! Missing pulses for port={output.port}."
            )
            continue

        # Create a dictionary containing unique waveforms by pulse_id
        waveforms_dict: Dict[int, np.ndarray] = dict()
        pulses_timeline_dict = port_timeline_dict[output.port]

        # Gets a list of unique pulse ids for this output
        pulse_ids: List[int] = sorted(
            {x for v in pulses_timeline_dict.values() for x in v}
        )

        # Gets a dictionary of pulse_id by commandtable_indexes
        commandtable_map: Dict[int, int] = zi_helpers.get_commandtable_map(
            pulse_ids, pulseid_pulseinfo_dict
        )

        # Execute partial functions to get the actual waveform
        for pulse_id in commandtable_map.keys():
            if pulse_id in waveforms_dict:
                # The unique waveform is already added to the dictionary.
                continue

            waveforms_dict[pulse_id] = waveform_helpers.exec_waveform_partial(
                pulse_id, pulseid_waveformfn_dict, clock_rate
            )

        # Step 1: Generate and compile sequencer program AND
        # Step 2: Set CommandTable JSON vector
        _program_sequences_hdawg(
            hdawg,
            awg,
            schedule,
            device,
            output,
            commandtable_map,
            waveforms_dict,
            port_timeline_dict,
            pulses_timeline_dict,
            pulseid_pulseinfo_dict,
        )

        # Apply modulation to waveforms
        _program_modulation(awg, device, output, waveforms_dict, pulseid_pulseinfo_dict)

        # Resize the waveforms
        waveform_helpers.resize_waveforms(
            waveforms_dict, WAVEFORM_GRANULARITY[device.type]
        )

        # Step 3: Upload waveforms to AWG CommandTable
        _set_waveforms(
            hdawg,
            awg,
            waveforms_dict,
            commandtable_map,
            pulseid_pulseinfo_dict,
            _WaveformDestination.WAVEFORM_TABLE,
        )


def _program_sequences_hdawg(
    hdawg: HDAWG,
    awg: HDAWG_CORE,
    schedule: types.Schedule,
    device: zhinst.Device,
    output: zhinst.Output,
    commandtable_map: Dict[int, int],
    waveforms_dict: Dict[int, np.ndarray],
    port_timeline_dict: Dict[str, Dict[int, List[int]]],
    pulses_timeline_dict: Dict[int, List[int]],
    pulseid_pulseinfo_dict: Dict[int, Dict[str, Any]],
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
    schedule :
    device :
    output :
    commandtable_map :
    waveforms_dict :
    port_timeline_dict :
    pulses_timeline_dict :
    pulseid_pulseinfo_dict :
    """
    command_table_entries: List[zhinst.CommandTableEntry] = list()
    command_table_index: int = 0
    granularity = WAVEFORM_GRANULARITY[device.type]

    seqc_gen = seqc_il_generator.SeqcILGenerator()
    clock_rate = zi_helpers.get_clock_rate(device.type)

    seqc_info = seqc_il_generator.SeqcInfo(
        clock_rate,
        SEQUENCE_DEAD_TIME,
        output.line_trigger_delay,
        schedule,
        port_timeline_dict,
        pulses_timeline_dict,
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
    seqc_gen.declare_var("__repetitions__", schedule.repetitions)

    for pulse_id, waveform_index in commandtable_map.items():

        name: str = f"w{waveform_index}"
        wave_size: int = len(waveforms_dict[pulse_id])
        if wave_size % granularity != 0:
            wave_size = math.closest_number_ceil(wave_size, granularity)

        # Create and add variables to the Sequence program
        # aswell as assign the variables with operations
        seqc_gen.declare_wave(name)
        seqc_gen.assign_placeholder(name, wave_size)
        seqc_gen.emit_assign_wave_index(name, name, index=waveform_index)

        # Do bookkeeping for the CommandTable
        command_table_entry = zhinst.CommandTableEntry(
            index=command_table_index,
            waveform=zhinst.CommandTableEntryIndex(waveform_index),
        )
        command_table_entries.append(command_table_entry)
        command_table_index += 1

    # Reset marker
    if is_marker_source:
        seqc_gen.emit_set_trigger(0)

    seqc_gen.emit_begin_repeat("__repetitions__")

    if is_marker_source:
        seqc_gen.emit_set_trigger(" + ".join(output.markers))

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
            comment=f"clock={current_clock}",
        )

    logger.debug(
        "%-20s | %-16s | %-16s | %-16s | %-16s | %-14s | %-18s | %-6s",
        "pulse_id",
        "pulse_t0",
        "abs_time",
        "t0",
        "duration",
        "start (clocks)",
        "duration (clocks)",
        "waveform size",
    )

    current_pulse = _Pulse(0, 0, 0, 0, 0, 0, 0)
    previous_pulse = _Pulse(-1, -1, -1, -1, -1, -1, -1)

    for timeslot_index, pulse_ids in pulses_timeline_dict.items():
        # Create the operations section of the Sequence program.
        t_constr = schedule.timing_constraints[timeslot_index]
        abs_time = t_constr["abs_time"]

        for pulse_id in pulse_ids:
            pulse_info = pulseid_pulseinfo_dict[pulse_id]

            t0 = abs_time + pulse_info["t0"] - seqc_info.schedule_offset_in_seconds
            duration = pulse_info["duration"]
            start_in_clocks = seqc_info.to_clocks(t0)
            n_samples: int = _get_waveform_size(waveforms_dict[pulse_id], granularity)

            duration_in_seconds = (1 / clock_rate) * n_samples
            duration_in_clocks = seqc_info.to_clocks(duration_in_seconds)

            current_pulse = _Pulse(
                pulse_id,
                abs_time - seqc_info.schedule_offset_in_seconds,
                pulse_info["t0"],
                duration,
                start_in_clocks,
                duration_in_clocks,
                n_samples,
            )

            logger.debug(repr(current_pulse))

            if previous_pulse.id == -1:
                previous_pulse = current_pulse
                continue

            elapsed_clocks: int = (
                current_pulse.start_in_clocks - previous_pulse.start_in_clocks
            )

            n_assembly_instructions = seqc_il_generator.SEQC_INSTR_CLOCKS[
                seqc_il_generator.SeqcInstructions.EXECUTE_TABLE_ENTRY
            ]
            waveform_index = commandtable_map[previous_pulse.id]
            seqc_gen.emit_execute_table_entry(
                waveform_index,
                comment=f"\t// clock={current_clock} pulse={waveform_index} n_instr={n_assembly_instructions}",
            )

            if elapsed_clocks == previous_pulse.duration_in_clocks:
                n_assembly_instructions += seqc_il_generator.SEQC_INSTR_CLOCKS[
                    seqc_il_generator.SeqcInstructions.WAIT_WAVE
                ]
                seqc_gen.emit_wait_wave(comment=f"\t// clock={current_clock}")
                current_clock += previous_pulse.duration_in_clocks
            else:
                n_assembly_instructions = seqc_il_generator.SEQC_INSTR_CLOCKS[
                    seqc_il_generator.SeqcInstructions.WAIT
                ]
                seqc_il_generator.add_wait(
                    seqc_gen,
                    elapsed_clocks - n_assembly_instructions,
                    comment=f"\t// clock={current_clock}",
                )
                current_clock += elapsed_clocks

            previous_pulse = current_pulse

    # Adds the last pulse
    waveform_index = commandtable_map[current_pulse.id]
    seqc_gen.emit_execute_table_entry(
        waveform_index,
        comment=f"\t// clock={current_clock} pulse={waveform_index}",
    )
    seqc_gen.emit_wait_wave(comment=f"\t// clock={current_clock}")
    current_clock += current_pulse.duration_in_clocks

    # Reset trigger each iteration
    if is_marker_source:
        seqc_gen.emit_set_trigger(0, comment=f"\t// clock={current_clock}")
        current_clock += seqc_il_generator.SEQC_INSTR_CLOCKS[
            seqc_il_generator.SeqcInstructions.SET_TRIGGER
        ]

    seqc_gen.emit_comment("Dead time")
    seqc_il_generator.add_wait(
        seqc_gen,
        (seqc_info.schedule_duration_in_clocks - current_clock)
        + seqc_info.dead_time_in_clocks,
    )

    seqc_gen.emit_end_repeat()

    # Reset trigger
    if is_marker_source:
        seqc_gen.emit_set_trigger(0)

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
    schedule: types.Schedule,
    pulseid_pulseinfo_dict: Dict[int, Dict[str, Any]],
    acqid_acqinfo_dict: Dict[int, Dict[str, Any]],
    pulseid_waveformfn_dict: Dict[int, waveform_helpers.GetWaveformPartial],
    port_timeline_dict: Dict[str, Dict[int, List[int]]],
) -> Dict[int, Callable]:
    """
    Initialize programming the UHFQA ZI Instrument.

    Creates a sequence program and converts schedule
    pulses to waveforms for the UHFQA.

    Parameters
    ----------
    uhfqa :
    device :
    schedule :
    pulseid_pulseinfo_dict :
    acqid_acqinfo_dict :
    pulseid_waveformfn_dict :
    port_timeline_dict :
    """
    clock_rate = zi_helpers.get_clock_rate(device.type)
    channels = device.channels
    channels = list(filter(lambda c: c.mode == enums.SignalModeType.REAL, channels))

    logger.debug(f"UHFQA[{uhfqa.name}] {str(device)}")

    awg = uhfqa.awg
    logger.debug(f"[{awg.name}] enabling outputs...")
    awg.output1("on")
    awg.output2("on")

    uhfqa.disable_readout_channels()

    # Create a dictionary containing unique waveforms by pulse_id
    waveforms_dict: Dict[int, np.ndarray] = dict()

    for channel in channels:
        logger.debug(f"[{awg.name}] channel={str(channel)}")

        # Build a list of unique pulse ids for all channels
        pulse_ids: List[int] = list()
        channel_timeline_dict = port_timeline_dict[channel.port]
        for timeslot_uuids in channel_timeline_dict.values():
            for uuid in timeslot_uuids:
                if uuid in acqid_acqinfo_dict:
                    # Skip acquisition protocols
                    continue

                pulse_ids.append(uuid)

        # Gets a dictionary of pulse_id by commandtable_indexes
        commandtable_map: Dict[int, int] = zi_helpers.get_commandtable_map(
            pulse_ids, pulseid_pulseinfo_dict
        )

        # Execute partial functions to get the actual waveform
        for pulse_id in commandtable_map:
            if pulse_id in waveforms_dict:
                # The unique waveform is already added to the dictionary.
                continue

            waveforms_dict[pulse_id] = waveform_helpers.exec_waveform_partial(
                pulse_id, pulseid_waveformfn_dict, clock_rate
            )

        # Execute partial functions to get acquisition weights.
        # Note that weights are threated the same as pulses.
        for acq_info in acqid_acqinfo_dict.values():
            acq_pulse_infos = acq_info["waveforms"]
            for pulse_info in acq_pulse_infos:
                pulse_id = schedule_helpers.get_pulse_uuid(pulse_info)
                waveforms_dict[pulse_id] = waveform_helpers.exec_waveform_partial(
                    pulse_id, pulseid_waveformfn_dict, clock_rate
                )

        # Apply modulation to waveforms
        _program_modulation(
            awg, device, channel, waveforms_dict, pulseid_pulseinfo_dict
        )

        # Resize the waveforms
        waveform_helpers.resize_waveforms(
            waveforms_dict, WAVEFORM_GRANULARITY[device.type]
        )

        # Apply waveforms to AWG
        _set_waveforms(
            uhfqa,
            awg,
            waveforms_dict,
            commandtable_map,
            pulseid_pulseinfo_dict,
            _WaveformDestination.CSV,
        )

    # Get a list of all acquisition protocol channels
    acq_channel_resolvers_map: Dict[int, Callable] = dict()
    readout_channel_index: int = 0

    for acq_info in acqid_acqinfo_dict.values():
        acq_pulse_infos = acq_info["waveforms"]
        acq_protocol: str = acq_info["protocol"]
        acq_duration: float = acq_info["duration"]
        acq_channel: int = acq_info["acq_channel"]

        integration_length = round(acq_duration * clock_rate)
        logger.debug(
            f"[{uhfqa.name}] acq_info={acq_info} acq_duration={acq_duration} integration_length={integration_length}"
        )

        zi_helpers.set_qas_parameters(
            uhfqa,
            integration_length,
            zhinst.QAS_IntegrationMode.NORMAL,
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

            for i, pulse_info in enumerate(acq_pulse_infos):
                pulse_id = schedule_helpers.get_pulse_uuid(pulse_info)
                waveform: np.ndarray = waveforms_dict[pulse_id]

                # Combine a reset and setting acq weights
                # by slicing the length of the waveform I and Q values.
                # This overwrites 0..length with new values.
                # The waveform is slightly larger then the integration_length
                # because of the waveform granularity. This is irrelevant
                # due to the waveform being appended with zeros. Theirfore
                # avoiding an extra slice of waveform[0:integration_length]
                if i % 2 == 0:
                    weights_i[0 : len(waveform)] = np.real(waveform)
                else:
                    weights_q[0 : len(waveform)] = np.imag(waveform)

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
        schedule,
        device,
        device.channel_0,
        commandtable_map,
        waveforms_dict,
        port_timeline_dict,
        channel_timeline_dict,
        pulseid_pulseinfo_dict,
        acqid_acqinfo_dict,
    )

    return acq_channel_resolvers_map


def _program_sequences_uhfqa(
    uhfqa: UHFQA,
    awg: UHFQA_CORE,
    schedule: types.Schedule,
    device: zhinst.Device,
    output: zhinst.Output,
    commandtable_map: Dict[int, int],
    waveforms_dict: Dict[int, np.ndarray],
    port_timeline_dict: Dict[str, Dict[int, List[int]]],
    pulses_timeline_dict: Dict[int, List[int]],
    pulseid_pulseinfo_dict: Dict[int, Dict[str, Any]],
    acqid_acqinfo_dict: Dict[int, Dict[str, Any]],
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
    schedule :
    device :
    output :
    commandtable_map :
    waveforms_dict :
    port_timeline_dict :
    pulses_timeline_dict :
    pulseid_pulseinfo_dict :
    acqid_acqinfo_dict :
    """
    granularity = WAVEFORM_GRANULARITY[device.type]

    seqc_gen = seqc_il_generator.SeqcILGenerator()
    clock_rate = zi_helpers.get_clock_rate(device.type)

    seqc_info = seqc_il_generator.SeqcInfo(
        clock_rate,
        SEQUENCE_DEAD_TIME,
        output.line_trigger_delay,
        schedule,
        port_timeline_dict,
        pulses_timeline_dict,
    )
    seqc_il_generator.add_seqc_info(seqc_gen, seqc_info)

    has_triggers: bool = len(output.triggers) > 0
    is_trigger_source = (
        device.ref == enums.ReferenceSourceType.EXTERNAL and has_triggers
    )

    current_clock: int = 0

    # Declare sequence variables
    seqc_gen.declare_var("__repetitions__", schedule.repetitions)
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
            comment=f"clock={current_clock}",
        )

    logger.debug(  # pylint: disable=logging-too-many-args
        "%-20s | %-16s | %-16s | %-16s | %-16s | %-14s | %-18s | %-6s",
        "pulse_id",
        "pulse_t0",
        "abs_time",
        "t0",
        "duration",
        "start (clocks)",
        "duration (clocks)",
        "waveform size",
    )

    if seqc_info.timeline_start_in_clocks > 0:
        current_clock += seqc_il_generator.add_wait(
            seqc_gen, seqc_info.timeline_start_in_clocks, f"clock={current_clock}"
        )

    current_pulse = _Pulse(0, 0, 0, 0, 0, 0, 0)
    previous_pulse = _Pulse(-1, 0, 0, 0, current_clock, 0, 0)

    for timeslot_index, timeslot_uuids in pulses_timeline_dict.items():
        t_constr = schedule.timing_constraints[timeslot_index]
        abs_time = t_constr["abs_time"]

        for uuid in timeslot_uuids:
            is_acq: bool = uuid in acqid_acqinfo_dict
            info_dict = (
                acqid_acqinfo_dict[uuid] if is_acq else pulseid_pulseinfo_dict[uuid]
            )

            t0 = abs_time + info_dict["t0"] - seqc_info.schedule_offset_in_seconds
            duration = info_dict["duration"]

            start_in_clocks = seqc_info.to_clocks(t0)
            duration_in_clocks: int = 0
            n_samples: int = 0
            if is_acq:
                duration_in_clocks = seqc_info.to_clocks(duration)
            else:
                n_samples: int = _get_waveform_size(waveforms_dict[uuid], granularity)
                duration_in_sec = (1 / clock_rate) * n_samples
                duration_in_clocks = seqc_info.to_clocks(duration_in_sec)

            current_pulse = _Pulse(
                uuid,
                abs_time - seqc_info.schedule_offset_in_seconds,
                info_dict["t0"],
                duration,
                start_in_clocks,
                duration_in_clocks,
                n_samples,
            )

            if previous_pulse.id == -1:
                logger.debug(repr(current_pulse))
                previous_pulse = current_pulse
                continue

            was_acq: bool = previous_pulse.id in acqid_acqinfo_dict

            instruction_offset_in_clocks: int = (
                current_pulse.start_in_clocks - previous_pulse.start_in_clocks
            )
            elapsed_clocks = max(
                previous_pulse.start_in_clocks - current_clock,
                instruction_offset_in_clocks,
            )
            logger.debug(repr(current_pulse))

            n_assembly_instructions: int = 0

            if was_acq:
                n_assembly_instructions = seqc_il_generator.SEQC_INSTR_CLOCKS[
                    seqc_il_generator.SeqcInstructions.ARM_INTEGRATION
                ]
                seqc_gen.emit_set_trigger(
                    seqc_il_generator.SeqcInstructions.ARM_INTEGRATION.value,
                    comment=f"\t// clock={current_clock} n_instr={n_assembly_instructions}",
                )
            else:
                n_assembly_instructions = seqc_il_generator.SEQC_INSTR_CLOCKS[
                    seqc_il_generator.SeqcInstructions.PLAY_WAVE
                ]
                seqc_gen.emit_play_wave(
                    f"w{commandtable_map[previous_pulse.id]:d}",
                    comment=f"\t// clock={current_clock} n_instr={n_assembly_instructions}",
                )

            if (
                elapsed_clocks != 0
                and elapsed_clocks
                >= seqc_il_generator.SEQC_INSTR_CLOCKS[
                    seqc_il_generator.SeqcInstructions.WAIT
                ]
            ):
                seqc_il_generator.add_wait(
                    seqc_gen,
                    elapsed_clocks - n_assembly_instructions,
                    comment=f"clock={current_clock}",
                )
                current_clock += elapsed_clocks

            previous_pulse = current_pulse

    # Add the last operation to the sequencer program.
    was_acq: bool = previous_pulse.id in acqid_acqinfo_dict
    n_assembly_instructions: int = 0

    if was_acq:
        n_assembly_instructions = seqc_il_generator.SEQC_INSTR_CLOCKS[
            seqc_il_generator.SeqcInstructions.ARM_INTEGRATION
        ]
        seqc_gen.emit_set_trigger(
            seqc_il_generator.SeqcInstructions.ARM_INTEGRATION.value,
            comment=f"\t// clock={current_clock} n_instr={n_assembly_instructions}",
        )
    else:
        n_assembly_instructions = seqc_il_generator.SEQC_INSTR_CLOCKS[
            seqc_il_generator.SeqcInstructions.PLAY_WAVE
        ]
        seqc_gen.emit_play_wave(
            f"w{commandtable_map[previous_pulse.id]:d}",
            comment=f"\t// clock={current_clock} n_instr={n_assembly_instructions}",
        )

    elapsed_clocks = previous_pulse.duration_in_clocks

    wait_duration: int = (
        elapsed_clocks
        - seqc_il_generator.SEQC_INSTR_CLOCKS[seqc_il_generator.SeqcInstructions.WAIT]
    ) + seqc_info.dead_time_in_clocks
    current_clock += elapsed_clocks

    # Wait with additional dead-time
    seqc_gen.emit_comment("Final wait + dead time")
    seqc_gen.emit_wait(wait_duration, comment=f"\t// clock={current_clock}")

    seqc_gen.emit_end_repeat()

    seqc_program = seqc_gen.generate()
    seqc_path: Path = zi_helpers.write_seqc_file(awg, seqc_program, f"{awg.name}.seqc")
    logger.debug(seqc_program)

    awg.set_sequence_params(
        sequence_type="Custom",
        path=str(seqc_path),
    )
    awg.compile()
