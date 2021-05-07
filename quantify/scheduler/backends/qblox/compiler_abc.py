# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
"""Compiler base and utility classes for Qblox backend."""

from __future__ import annotations

import json
from os import path, makedirs
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from typing import Optional, Dict, Any, Set, Tuple, List

import numpy as np
from pathvalidate import sanitize_filename
from qcodes.utils.helpers import NumpyJSONEncoder

# pylint: disable=no-name-in-module
from quantify.data.handling import (
    get_datadir,
    gen_tuid,
)

from quantify.scheduler.helpers.waveforms import modulate_waveform

from quantify.scheduler.backends.qblox import q1asm_instructions
from quantify.scheduler.backends.qblox.helpers import (
    generate_waveform_data,
    _generate_waveform_dict,
    generate_waveform_names_from_uuid,
    verify_qblox_instruments_version,
    find_all_port_clock_combinations,
    find_inner_dicts_containing_key,
)
from quantify.scheduler.backends.qblox.constants import (
    GRID_TIME,
    SAMPLING_RATE,
)
from quantify.scheduler.backends.qblox.qasm_program import QASMProgram
from quantify.scheduler.backends.types.qblox import (
    OpInfo,
    SequencerSettings,
    QASMRuntimeSettings,
    PulsarSettings,
    MixerCorrections,
)
from quantify.scheduler.helpers.waveforms import normalize_waveform_data


class InstrumentCompiler(ABC):
    """
    Abstract base class that defines a generic instrument compiler. The subclasses that
    inherit from this are meant to implement the compilation steps needed to compile the
    lists of `OpInfo` representing the pulse and acquisition info to device specific
    instructions.

    For each device that needs to be part of the compilation process such a
    `InstrumentCompiler` should be implemented.
    """

    def __init__(
        self,
        name: str,
        total_play_time: float,
        hw_mapping: Optional[Dict[str, Any]] = None,
    ):
        """
        Constructor for an InstrumentCompiler object.

        Parameters
        ----------
        name:
            Name of the `QCoDeS` instrument this compiler object corresponds to.
        total_play_time:
            Total time execution of the schedule should go on for. This parameter is
            used to ensure that the different devices, potentially with different clock
            rates, can work in a synchronized way when performing multiple executions of
            the schedule.
        hw_mapping:
            The hardware configuration dictionary for this specific device. This is one
            of the inner dictionaries of the overall hardware config.
        """
        self.name = name
        self.total_play_time = total_play_time
        self.hw_mapping = hw_mapping
        self._pulses = defaultdict(list)
        self._acquisitions = defaultdict(list)

    def add_pulse(self, port: str, clock: str, pulse_info: OpInfo):
        """
        Assigns a certain pulse to this device.

        Parameters
        ----------
        port:
            The port the pulse needs to be sent to.
        clock:
            The clock for modulation of the pulse. Can be a BasebandClock.
        pulse_info:
            Data structure containing all the information regarding this specific pulse
            operation.
        """
        self._pulses[(port, clock)].append(pulse_info)

    def add_acquisition(self, port: str, clock: str, acq_info: OpInfo):
        """
        Assigns a certain acquisition to this device.

        Parameters
        ----------
        port:
            The port the pulse needs to be sent to.
        clock:
            The clock for modulation of the pulse. Can be a BasebandClock.
        acq_info:
            Data structure containing all the information regarding this specific
            acquisition operation.
        """
        self._acquisitions[(port, clock)].append(acq_info)

    @property
    def portclocks_with_data(self) -> Set[Tuple[str, str]]:
        """
        All the port-clock combinations associated with at least one pulse and/or
        acquisition.

        Returns
        -------
        :
            A set containing all the port-clock combinations that are used by this
            InstrumentCompiler.
        """
        portclocks_used = set()
        portclocks_used.update(self._pulses.keys())
        portclocks_used.update(self._acquisitions.keys())
        return portclocks_used

    @abstractmethod
    def compile(self, repetitions: int = 1) -> Any:
        """
        An abstract method that should be overridden by a subclass to implement the
        actual compilation. Method turns the pulses and acquisitions added to the device
        into device specific instructions.

        Parameters
        ----------
        repetitions:
            Number of times execution the schedule is repeated

        Returns
        -------
        :
            A data structure representing the compiled program. The type is
            dependent on implementation.
        """


# pylint: disable=too-many-instance-attributes
class PulsarSequencerBase(ABC):
    """
    Abstract base class that specify the compilation steps on the sequencer level. The
    distinction between Pulsar QCM and Pulsar QRM is made by the subclasses.
    """

    def __init__(
        self,
        parent: PulsarBase,
        name: str,
        portclock: Tuple[str, str],
        modulation_freq: Optional[float] = None,
    ):
        """
        Constructor for the sequencer compiler.

        Parameters
        ----------
        parent:
            A reference to the parent instrument this sequencer belongs to.
        name:
            Name of the sequencer. This is supposed to match "seq{index}".
        portclock:
            Tuple that specifies the unique port and clock combination for this
            sequencer. The first value is the port, second is the clock.
        modulation_freq:
            The frequency used for modulation. This can either be passed in the
            constructor, or assigned in a later stage using `assign_frequency`.
        """
        self.parent = parent
        self._name = name
        self.port = portclock[0]
        self.clock = portclock[1]
        self.pulses: List[OpInfo] = list()
        self.acquisitions: List[OpInfo] = list()
        self._settings = SequencerSettings(
            nco_en=False, sync_en=True, modulation_freq=modulation_freq
        )
        self.mixer_corrections = None

    @property
    def portclock(self) -> Tuple[str, str]:
        """
        A tuple containing the unique port and clock combination for this sequencer.

        Returns
        -------
        :
            The portclock
        """
        return self.port, self.clock

    @property
    def modulation_freq(self) -> float:
        """
        The frequency used for modulation of the pulses.

        Returns
        -------
        :
            The frequency
        """
        return self._settings.modulation_freq

    @property
    def settings(self) -> SequencerSettings:
        """
        Gives the current settings.

        Returns
        -------
        :
            The settings set to this sequencer.
        """
        return self._settings

    @property
    def name(self) -> str:
        """
        The name assigned to this specific sequencer.

        Returns
        -------
        :
            The name.
        """
        return self._name

    @property
    @abstractmethod
    def awg_output_volt(self) -> float:
        """
        The output range in volts. This is to be overridden by the subclass to account
        for the differences between a QCM and a QRM.

        Returns
        -------
        :
            The output range in volts.
        """

    @property
    def has_data(self) -> bool:
        """
        Whether or not the sequencer has any data (meaning pulses or acquisitions)
        assigned to it or not.

        Returns
        -------
        :
            Has data been assigned to this sequencer?
        """
        return len(self.acquisitions) > 0 or len(self.pulses) > 0

    def assign_frequency(self, freq: float):
        """
        Assigns a modulation frequency to the sequencer.

        Parameters
        ----------
        freq:
            The frequency to be used for modulation.

        Raises
        ------
        ValueError
            Attempting to set the modulation frequency to a new value even though a
            value has been previously assigned.
        """
        if self._settings.modulation_freq != freq:
            if self._settings.modulation_freq is not None:
                raise ValueError(
                    f"Attempting to set the modulation frequency of {self._name} of "
                    f"{self.parent.name} to {freq}, while it has previously been set "
                    f"to {self._settings.modulation_freq}."
                )
        self._settings.modulation_freq = freq

    def _generate_awg_dict(self) -> Dict[str, Any]:
        """
        Generates the dictionary that corresponds that contains the awg waveforms in the
        format accepted by the driver.

        Notes
        -----
        The final dictionary to be included in the json that is uploaded to the pulsar
        is of the form:

        .. code-block::

            program
            awg
                waveform_name
                    data
                    index
            acq
                waveform_name
                    data
                    index

        This function generates the awg dictionary.

        Returns
        -------
        :
            The awg dictionary

        Raises
        ------
        ValueError
            I or Q amplitude is being set outside of maximum range.
        """
        waveforms_complex = dict()
        for pulse in self.pulses:
            # FIXME: Most of this is unnecessary but requires
            #  that we change how we deal with QASMRuntimeSettings
            raw_wf_data = generate_waveform_data(
                pulse.data, sampling_rate=SAMPLING_RATE
            )
            raw_wf_data = self._apply_corrections_to_waveform(
                raw_wf_data, pulse.duration, pulse.timing
            )
            raw_wf_data, amp_i, amp_q = normalize_waveform_data(raw_wf_data)
            if np.abs(amp_i) > self.awg_output_volt:
                raise ValueError(
                    f"Attempting to set amplitude to an invalid value. "
                    f"Maximum voltage range is +-{self.awg_output_volt} V for "
                    f"{self.__class__.__name__}.\n"
                    f"{amp_i} V is set as amplitude for the I channel for "
                    f"{repr(pulse)}"
                )
            if np.abs(amp_q) > self.awg_output_volt:
                raise ValueError(
                    f"Attempting to set amplitude to an invalid value. "
                    f"Maximum voltage range is +-{self.awg_output_volt} V for "
                    f"{self.__class__.__name__}.\n"
                    f"{amp_q} V is set as amplitude for the Q channel for "
                    f"{repr(pulse)}"
                )
            pulse.pulse_settings = QASMRuntimeSettings(
                awg_gain_0=amp_i / self.awg_output_volt,
                awg_gain_1=amp_q / self.awg_output_volt,
            )
            if pulse.uuid not in waveforms_complex:
                waveforms_complex[pulse.uuid] = raw_wf_data
        return _generate_waveform_dict(waveforms_complex)

    def _generate_acq_dict(self) -> Dict[str, Any]:
        """
        Generates the dictionary that corresponds that contains the acq weights
        waveforms in the format accepted by the driver.

        Notes
        -----
        The final dictionary to be included in the json that is uploaded to the pulsar
        is of the form:

        .. code-block::

            program
            awg
                waveform_name
                    data
                    index
            acq
                waveform_name
                    data
                    index

        This function generates the acq dictionary.

        Returns
        -------
        :
            The acq dictionary

        Raises
        ------
        NotImplementedError
            Currently, only two one dimensional waveforms can be used as acquisition
            weights. This exception is raised when either or both waveforms contain
            both a real and imaginary part.
        """
        waveforms_complex = dict()
        for acq in self.acquisitions:
            if acq.uuid not in waveforms_complex:
                raw_wf_data_real = generate_waveform_data(
                    acq.data["waveforms"][0], sampling_rate=SAMPLING_RATE
                )
                raw_wf_data_imag = generate_waveform_data(
                    acq.data["waveforms"][1], sampling_rate=SAMPLING_RATE
                )
                self._settings.duration = len(raw_wf_data_real)
                if not (
                    np.all(np.isreal(raw_wf_data_real))
                    and np.all(np.isreal(1.0j * raw_wf_data_imag))
                ):  # since next step will break if either is complex
                    raise NotImplementedError(
                        f"Complex weights not implemented. Please use two 1d "
                        f"real-valued weights. Exception was triggered because of "
                        f"{repr(acq)}."
                    )
                waveforms_complex[acq.uuid] = raw_wf_data_real + raw_wf_data_imag
        return _generate_waveform_dict(waveforms_complex)

    def _apply_corrections_to_waveform(
        self, waveform_data: np.ndarray, time_duration: float, t0: Optional[float] = 0
    ) -> np.ndarray:
        """
        Applies all the needed pre-processing on the waveform data. This includes mixer
        corrections and modulation.

        Parameters
        ----------
        waveform_data:
            The data to correct.
        time_duration:
            Total time is seconds that the waveform is used.
        t0:
            The start time of the pulse/acquisition. This is used for instance to make
            the make the phase change continuously when the start time is not zero.

        Returns
        -------
        :
            The waveform data after applying all the transformations.
        """
        t = np.linspace(t0, time_duration + t0, int(time_duration * SAMPLING_RATE))
        corrected_wf = modulate_waveform(t, waveform_data, self.modulation_freq)
        if self.mixer_corrections is not None:
            corrected_wf = self.mixer_corrections.correct_skewness(corrected_wf)
        return corrected_wf

    def update_settings(self):
        """
        Updates the sequencer settings to set all parameters that are determined by the
        compiler. Currently, this only changes the offsets based on the mixer
        calibration parameters.
        """
        if self.mixer_corrections is not None:
            self._settings.awg_offset_path_0 = (
                self.mixer_corrections.offset_I / self.awg_output_volt
            )
            self._settings.awg_offset_path_1 = (
                self.mixer_corrections.offset_Q / self.awg_output_volt
            )

    # pylint: disable=too-many-locals
    # pylint: disable=too-many-arguments
    @classmethod
    def generate_qasm_program(
        cls,
        total_sequence_time: float,
        pulses: Optional[List[OpInfo]] = None,
        awg_dict: Optional[Dict[str, Any]] = None,
        acquisitions: Optional[List[OpInfo]] = None,
        acq_dict: Optional[Dict[str, Any]] = None,
        repetitions: Optional[int] = 1,
    ) -> str:
        """
        Generates a QASM program for a sequencer. Requires the awg and acq dicts to
        already have been generated.

        Example of a program generated by this function:

        .. code-block::

                    wait_sync     4
                    set_mrk       1
                    move          10,R0         # iterator for loop with label start
            start:
                    wait          4
                    set_awg_gain  22663,10206  # setting gain for 9056793381316377208
                    play          0,1,4
                    wait          176
                    loop          R0,@start
                    set_mrk       0
                    upd_param     4
                    stop


        Parameters
        ----------
        total_sequence_time:
            Total time the program needs to play for. If the sequencer would be done
            before this time, a wait is added at the end to ensure synchronization.
        pulses:
            A list containing all the pulses that are to be played.
        awg_dict:
            Dictionary containing the pulse waveform data and the index that is assigned
            to the I and Q waveforms, as generated by the `generate_awg_dict` function.
            This is used to extract the relevant indexes when adding a play instruction.
        acquisitions:
            A list containing all the acquisitions that are to be performed.
        acq_dict:
            Dictionary containing the acquisition waveform data and the index that is
            assigned to the I and Q waveforms, as generated by the `generate_acq_dict`
            function. This is used to extract the relevant indexes when adding an
            acquire instruction.
        repetitions:
            Number of times to repeat execution of the schedule.

        Returns
        -------
        :
            The generated QASM program.
        """
        loop_label = "start"
        loop_register = "R0"

        qasm = QASMProgram()
        # program header
        qasm.emit(q1asm_instructions.WAIT_SYNC, GRID_TIME)
        qasm.emit(q1asm_instructions.SET_MARKER, 1)

        # program body
        pulses = list() if pulses is None else pulses
        acquisitions = list() if acquisitions is None else acquisitions
        op_list = pulses + acquisitions
        op_list = sorted(op_list, key=lambda p: (p.timing, p.is_acquisition))

        with qasm.loop(
            label=loop_label, register=loop_register, repetitions=repetitions
        ):
            op_queue = deque(op_list)
            while len(op_queue) > 0:
                operation = op_queue.popleft()
                if operation.is_acquisition:
                    idx0, idx1 = cls.get_indices_from_wf_dict(operation.uuid, acq_dict)
                    qasm.wait_till_start_then_acquire(operation, idx0, idx1)
                else:
                    idx0, idx1 = cls.get_indices_from_wf_dict(operation.uuid, awg_dict)
                    qasm.wait_till_start_then_play(operation, idx0, idx1)

            end_time = qasm.to_pulsar_time(total_sequence_time)
            wait_time = end_time - qasm.elapsed_time
            if wait_time <= 0:
                raise RuntimeError(
                    f"Invalid timing detected, attempting to insert wait "
                    f"of {wait_time} ns. The total duration of the "
                    f"schedule is {end_time} but {qasm.elapsed_time} ns "
                    f"already processed."
                )
            qasm.auto_wait(wait_time)

        # program footer
        qasm.emit(q1asm_instructions.SET_MARKER, 0)
        qasm.emit(q1asm_instructions.UPDATE_PARAMETERS, GRID_TIME)
        qasm.emit(q1asm_instructions.STOP)
        return str(qasm)

    @staticmethod
    def get_indices_from_wf_dict(uuid: int, wf_dict: Dict[str, Any]) -> Tuple[int, int]:
        """
        Takes a awg_dict or acq_dict and extracts the waveform indices based off of the
        uuid of the pulse/acquisition.

        Parameters
        ----------
        uuid:
            The unique identifier of the pulse/acquisition.
        wf_dict:
            The awg or acq dict that holds the waveform data and indices.

        Returns
        -------
        :
            Index of the I waveform.
        :
            Index of the Q waveform.
        """
        name_real, name_imag = generate_waveform_names_from_uuid(uuid)
        return wf_dict[name_real]["index"], wf_dict[name_imag]["index"]

    @staticmethod
    def _generate_waveforms_and_program_dict(
        program: str,
        awg_dict: Dict[str, Any],
        acq_dict: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generates the full waveforms and program dict that is to be uploaded to the
        sequencer from the program string and the awg and acq dicts, by combining them
        and assigning the appropriate keys.

        Parameters
        ----------
        program:
            The compiled QASM program as a string.
        awg_dict:
            The dictionary containing all the awg data and indices. This is expected to
            be of the form generated by the `generate_awg_dict` method.
        acq_dict:
            The dictionary containing all the acq data and indices. This is expected to
            be of the form generated by the `generate_acq_dict` method.

        Returns
        -------
        :
            The combined program.
        """
        compiled_dict = dict()
        compiled_dict["program"] = program
        compiled_dict["waveforms"] = dict()
        compiled_dict["waveforms"]["awg"] = awg_dict
        if acq_dict is not None:
            compiled_dict["waveforms"]["acq"] = acq_dict
        return compiled_dict

    @staticmethod
    def _dump_waveforms_and_program_json(
        wf_and_pr_dict: Dict[str, Any], label: Optional[str] = None
    ) -> str:
        """
        Takes a combined waveforms and program dict and dumps it as a json file.

        Parameters
        ----------
        wf_and_pr_dict:
            The dict to dump as a json file.
        label:
            A label that is appended to the filename.

        Returns
        -------
        :
            The full absolute path where the json file is stored.
        """
        data_dir = get_datadir()
        folder = path.join(data_dir, "schedules")
        makedirs(folder, exist_ok=True)

        filename = (
            f"{gen_tuid()}.json" if label is None else f"{gen_tuid()}_{label}.json"
        )
        filename = sanitize_filename(filename)
        file_path = path.join(folder, filename)

        with open(file_path, "w") as file:
            json.dump(wf_and_pr_dict, file, cls=NumpyJSONEncoder, indent=4)

        return file_path

    def compile(self, repetitions: int = 1) -> Optional[Dict[str, Any]]:
        """
        Performs the full sequencer level compilation based on the assigned data and
        settings. If no data is assigned to this sequencer, the compilation is skipped
        and None is returned instead.

        Parameters
        ----------
        repetitions:
            Number of times execution the schedule is repeated

        Returns
        -------
        :
            The compiled program. If no data is assigned to this sequencer, the
            compilation is skipped and None is returned instead.
        """
        if not self.has_data:
            return None

        awg_dict = self._generate_awg_dict()
        acq_dict = self._generate_acq_dict() if len(self.acquisitions) > 0 else None

        qasm_program = self.generate_qasm_program(
            self.parent.total_play_time,
            self.pulses,
            awg_dict,
            self.acquisitions,
            acq_dict,
            repetitions=repetitions,
        )

        wf_and_pr_dict = self._generate_waveforms_and_program_dict(
            qasm_program, awg_dict, acq_dict
        )

        json_filename = self._dump_waveforms_and_program_json(
            wf_and_pr_dict, f"{self.port}_{self.clock}"
        )
        self.update_settings()
        settings_dict = self.settings.to_dict()
        return {"seq_fn": json_filename, "settings": settings_dict}


class PulsarBase(InstrumentCompiler, ABC):
    """
    Pulsar specific implementation of`InstrumentCompiler`. The class is defined as an
    abstract base class since the distinction between Pulsar QRM and Pulsar QCM specific
    implementations are defined in subclasses. Effectively, this base class contains the
    functionality shared by the Pulsar QRM and Pulsar QCM.

    Attributes
    ----------
    output_to_sequencer_idx:
        Dictionary that maps output names to specific sequencer indices. This
        implementation is temporary and will change when multiplexing is supported by
        the hardware.
    """

    output_to_sequencer_idx = {"complex_output_0": 0, "complex_output_1": 1}

    def __init__(
        self,
        name: str,
        total_play_time: float,
        hw_mapping: Dict[str, Any],
    ):
        """
        Constructor function.

        Parameters
        ----------
        name:
            Name of the `QCoDeS` instrument this compiler object corresponds to.
        total_play_time:
            Total time execution of the schedule should go on for. This parameter is
            used to ensure that the different devices, potentially with different clock
            rates, can work in a synchronized way when performing multiple executions of
            the schedule.
        hw_mapping:
            The hardware configuration dictionary for this specific device. This is one
            of the inner dictionaries of the overall hardware config.
        """
        super().__init__(name, total_play_time, hw_mapping)
        verify_qblox_instruments_version()

        self.portclock_map = self._generate_portclock_to_seq_map()
        self.sequencers = self._construct_sequencers()
        self._settings = PulsarSettings.extract_settings_from_mapping(hw_mapping)

    @property
    @abstractmethod
    def sequencer_type(self) -> type(PulsarSequencerBase):
        """
        Specifies whether the sequencers in this pulsar are QCM_sequencers or
        QRM_sequencers.

        Returns
        -------
        :
            A pulsar sequencer type
        """

    @property
    @abstractmethod
    def max_sequencers(self) -> int:
        """
        Specifies the maximum amount of sequencers available to this instrument.

        Returns
        -------
        :
            The maximum amount of sequencers
        """

    def assign_modulation_frequency(self, portclock: Tuple[str, str], freq: float):
        """
        Sets the modulation frequency for a certain portclock belonging to this
        instrument.

        Parameters
        ----------
        portclock:
            A tuple with the port as first element and clock as second.
        freq:
            The modulation frequency to assign to the portclock.
        """
        seq_name = self.portclock_map[portclock]
        seq = self.sequencers[seq_name]
        seq.assign_frequency(freq)

    def _generate_portclock_to_seq_map(self) -> Dict[Tuple[str, str], str]:
        """
        Generates a mapping from portclock tuples to sequencer names.

        Returns
        -------
        :
            A dictionary with as key a portclock tuple and as value the name of a
            sequencer.

        Raises
        ------
        NotImplementedError
            When the hardware mapping contains a dictionary, which is assumed to
            correspond to an output channel, that does not have a name defined in
            self.OUTPUT_TO_SEQ.keys(). Likely this will occur when attempting to use
            real outputs (instead of complex), or when the hardware mapping is invalid.
        """
        output_to_seq = self.output_to_sequencer_idx

        mapping = dict()
        for io, data in self.hw_mapping.items():
            if not isinstance(data, dict):
                continue

            port_clocks = find_all_port_clock_combinations(data)
            if len(port_clocks) > 1:
                raise NotImplementedError(
                    f"{len(port_clocks)} port and clock "
                    f"combinations specified for output {io} "
                    f"(sequencer {output_to_seq[io]}). Multiple "
                    f"sequencers per output is not yet supported "
                    f"by this backend."
                )

            if len(port_clocks) > 0:
                port_clock = port_clocks[0]
                if io not in output_to_seq:
                    raise NotImplementedError(
                        f"Attempting to use non-supported output {io}. "
                        f"Supported output types: "
                        f"{(str(t) for t in output_to_seq)}"
                    )
                mapping[port_clock] = f"seq{output_to_seq[io]}"
        return mapping

    def _construct_sequencers(self) -> Dict[str, PulsarSequencerBase]:
        """
        Constructs `PulsarSequencerBase` objects for each port and clock combination
        belonging to this device.

        Returns
        -------
        :
            A dictionary containing the sequencer objects, the keys correspond to the
            names of the sequencers.
        """
        sequencers = dict()
        for io, io_cfg in self.hw_mapping.items():
            if not isinstance(io_cfg, dict):
                continue

            portclock_dicts = find_inner_dicts_containing_key(io_cfg, "port")
            if len(portclock_dicts) > 1:
                raise NotImplementedError(
                    f"{len(portclock_dicts)} port and clock "
                    f"combinations specified for output {io}. Multiple "
                    f"sequencers per output is not yet supported "
                    f"by this backend."
                )
            portclock_dict = portclock_dicts[0]
            portclock = portclock_dict["port"], portclock_dict["clock"]
            freq = (
                None
                if "interm_freq" in portclock_dict
                else portclock_dict["interm_freq"]
            )

            seq_name = f"seq{self.output_to_sequencer_idx[io]}"
            sequencers[seq_name] = self.sequencer_type(self, seq_name, portclock, freq)
            if "mixer_corrections" in io_cfg:
                sequencers[seq_name].mixer_corrections = MixerCorrections.from_dict(
                    io_cfg["mixer_corrections"]
                )

        if len(sequencers.keys()) > self.max_sequencers:
            raise ValueError(
                f"Attempting to construct too many sequencer compilers. "
                f"Maximum allowed for {self.__class__} is {self.max_sequencers}!"
            )

        return sequencers

    def _distribute_data(self):
        """
        Distributes the pulses and acquisitions assigned to this pulsar over the
        different sequencers based on their portclocks.
        """
        for portclock, pulse_data_list in self._pulses.items():
            for seq in self.sequencers.values():
                if seq.portclock == portclock:
                    seq.pulses = pulse_data_list

        for portclock, acq_data_list in self._acquisitions.items():
            for seq in self.sequencers.values():
                if seq.portclock == portclock:
                    seq.acquisitions = acq_data_list

    def compile(self, repetitions: int = 1) -> Optional[Dict[str, Any]]:
        """
        Performs the actual compilation steps for this pulsar, by calling the sequencer
        level compilation functions and combining them into a single dictionary. The
        compiled program has a settings key, and keys for every sequencer.

        Parameters
        ----------
        repetitions:
            Number of times execution the schedule is repeated

        Returns
        -------
        :
            The compiled program corresponding to this pulsar. It contains an entry for
            every sequencer and general "settings". If the device is not actually used,
            and an empty program is compiled, None is returned instead.
        """
        self._distribute_data()
        program = dict()
        for seq_name, seq in self.sequencers.items():
            seq_program = seq.compile(repetitions=repetitions)
            if seq_program is not None:
                program[seq_name] = seq_program

        if len(program) == 0:
            return None

        self._settings.hardware_averages = repetitions
        program["settings"] = self._settings.to_dict()
        return program
