# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
"""Compiler base and utility classes for Qblox backend."""
# pylint: disable=too-many-lines
from __future__ import annotations

import json
from os import path, makedirs
from abc import ABC, abstractmethod, ABCMeta
from collections import defaultdict, deque
from typing import Optional, Dict, Any, Set, Tuple, List

import numpy as np
from pathvalidate import sanitize_filename
from qcodes.utils.helpers import NumpyJSONEncoder

# pylint: disable=no-name-in-module
from quantify_core.data.handling import (
    get_datadir,
    gen_tuid,
)

from quantify_scheduler.backends.qblox import non_generic
from quantify_scheduler.backends.qblox import q1asm_instructions
from quantify_scheduler.backends.qblox.helpers import (
    generate_waveform_data,
    _generate_waveform_dict,
    generate_waveform_names_from_uuid,
    verify_qblox_instruments_version,
    to_grid_time,
)
from quantify_scheduler.backends.qblox.constants import (
    GRID_TIME,
    SAMPLING_RATE,
)
from quantify_scheduler.backends.qblox.qasm_program import QASMProgram
from quantify_scheduler.backends.qblox import compiler_container
from quantify_scheduler.backends.types.qblox import (
    OpInfo,
    PulsarSettings,
    PulsarRFSettings,
    SequencerSettings,
    QASMRuntimeSettings,
    MixerCorrections,
)
from quantify_scheduler.helpers.waveforms import normalize_waveform_data


class InstrumentCompiler(ABC):
    """
    Abstract base class that defines a generic instrument compiler. The subclasses that
    inherit from this are meant to implement the compilation steps needed to compile the
    lists of :class:`quantify_scheduler.backends.types.qblox.OpInfo` representing the
    pulse and acquisition information to device-specific instructions.

    Each device that needs to be part of the compilation process requires an associated
    `InstrumentCompiler`.
    """

    def __init__(
        self,
        parent: compiler_container.CompilerContainer,
        name: str,
        total_play_time: float,
        hw_mapping: Optional[Dict[str, Any]] = None,
    ):
        # pylint: disable=line-too-long
        """
        Constructor for an InstrumentCompiler object.

        Parameters
        ----------
        parent
            Reference to the parent
            :class:`quantify_scheduler.backends.qblox.compiler_container.CompilerContainer`
            object.
        name
            Name of the `QCoDeS` instrument this compiler object corresponds to.
        total_play_time
            Total time execution of the schedule should go on for. This parameter is
            used to ensure that the different devices, potentially with different clock
            rates, can work in a synchronized way when performing multiple executions of
            the schedule.
        hw_mapping
            The hardware configuration dictionary for this specific device. This is one
            of the inner dictionaries of the overall hardware config.
        """
        self.parent = parent
        self.name = name
        self.total_play_time = total_play_time
        self.hw_mapping = hw_mapping

    def prepare(self) -> None:
        """
        Method that can be overridden to implement logic before the main compilation
        starts. This step is to extract all settings for the devices that are dependent
        on settings of other devices. This step happens after instantiation of the
        compiler object but before the start of the main compilation.
        """

    @abstractmethod
    def compile(self, repetitions: int) -> Any:
        """
        An abstract method that should be overridden in a subclass to implement the
        actual compilation. It should turn the pulses and acquisitions added to the
        device into device-specific instructions.

        Parameters
        ----------
        repetitions
            Number of times execution of the schedule is repeated.

        Returns
        -------
        :
            A data structure representing the compiled program. The type is
            dependent on implementation.
        """


class ControlDeviceCompiler(InstrumentCompiler, metaclass=ABCMeta):
    """
    Abstract class for any device requiring logic for acquisition and playback of
    pulses.
    """

    def __init__(
        self,
        parent: compiler_container.CompilerContainer,
        name: str,
        total_play_time: float,
        hw_mapping: Dict[str, Any],
    ):
        # pylint: disable=line-too-long
        """
        Constructor for a ControlDeviceCompiler object.

        Parameters
        ----------
        parent
            Reference to the parent
            :class:`quantify_scheduler.backends.qblox.compiler_container.CompilerContainer`
            object.
        name
            Name of the `QCoDeS` instrument this compiler object corresponds to.
        total_play_time
            Total time execution of the schedule should go on for. This parameter is
            used to ensure that the different devices, potentially with different clock
            rates, can work in a synchronized way when performing multiple executions of
            the schedule.
        hw_mapping
            The hardware configuration dictionary for this specific device. This is one
            of the inner dictionaries of the overall hardware config.
        """
        super().__init__(parent, name, total_play_time, hw_mapping)
        self._pulses = defaultdict(list)
        self._acquisitions = defaultdict(list)

    @property
    @abstractmethod
    def supports_acquisition(self) -> bool:
        """
        Specifies whether the device can perform acquisitions.

        Returns
        -------
        :
            The maximum amount of sequencers
        """

    def add_pulse(self, port: str, clock: str, pulse_info: OpInfo):
        """
        Assigns a certain pulse to this device.

        Parameters
        ----------
        port
            The port the pulse needs to be sent to.
        clock
            The clock for modulation of the pulse. Can be a BasebandClock.
        pulse_info
            Data structure containing all the information regarding this specific pulse
            operation.
        """
        self._pulses[(port, clock)].append(pulse_info)

    def add_acquisition(self, port: str, clock: str, acq_info: OpInfo):
        """
        Assigns a certain acquisition to this device.

        Parameters
        ----------
        port
            The port the pulse needs to be sent to.
        clock
            The clock for modulation of the pulse. Can be a BasebandClock.
        acq_info
            Data structure containing all the information regarding this specific
            acquisition operation.
        """

        if not self.supports_acquisition:
            raise RuntimeError(
                f"{self.__class__.__name__} {self.name} does not support acquisitions. "
                f"Attempting to add acquisition {repr(acq_info)} "
                f"on port {port} with clock {clock}."
            )

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
    def compile(self, repetitions: int = 1) -> Dict[str, Any]:
        """
        An abstract method that should be overridden by a subclass to implement the
        actual compilation. Method turns the pulses and acquisitions added to the device
        into device-specific instructions.

        Parameters
        ----------
        repetitions
            Number of times execution the schedule is repeated.

        Returns
        -------
        :
            A data structure representing the compiled program.
        """


# pylint: disable=too-many-instance-attributes
class Sequencer:
    """
    Abstract base class that specify the compilation steps on the sequencer level. The
    distinction between Pulsar QCM and Pulsar QRM is made by the subclasses.
    """

    def __init__(
        self,
        parent: PulsarBase,
        name: str,
        portclock: Tuple[str, str],
        seq_settings: dict,
        lo_name: Optional[str] = None,
    ):
        """
        Constructor for the sequencer compiler.

        Parameters
        ----------
        parent
            A reference to the parent instrument this sequencer belongs to.
        name
            Name of the sequencer. This is supposed to match "seq{index}".
        portclock
            Tuple that specifies the unique port and clock combination for this
            sequencer. The first value is the port, second is the clock.
        seq_settings
            Sequencer settings dictionary.
        """
        self.parent = parent
        self._name = name
        self.port = portclock[0]
        self.clock = portclock[1]
        self.pulses: List[OpInfo] = list()
        self.acquisitions: List[OpInfo] = list()
        self.associated_ext_lo = lo_name

        self.instruction_generated_pulses_enabled = seq_settings.get(
            "instruction_generated_pulses_enabled", False
        )

        modulation_freq = seq_settings.get("interm_freq", None)
        nco_en: bool = not (modulation_freq == 0 or modulation_freq is None)
        self._settings = SequencerSettings(
            nco_en=nco_en,
            sync_en=True,
            modulation_freq=modulation_freq,
        )
        self.mixer_corrections = None

    @property
    def portclock(self) -> Tuple[str, str]:
        """
        A tuple containing the unique port and clock combination for this sequencer.

        Returns
        -------
        :
            The portclock.
        """
        return self.port, self.clock

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

    @property
    def frequency(self) -> float:
        """
        The frequency used for modulation of the pulses.

        Returns
        -------
        :
            The frequency.
        """
        return self._settings.modulation_freq

    @frequency.setter
    def frequency(self, freq: float):
        """
        Assigns a modulation frequency to the sequencer.

        Parameters
        ----------
        freq
            The frequency to be used for modulation.

        Raises
        ------
        ValueError
            Attempting to set the modulation frequency to a new value even though a
            different value has been previously assigned.
        """
        if self._settings.modulation_freq != freq:
            if self._settings.modulation_freq is not None:
                raise ValueError(
                    f"Attempting to set the modulation frequency of {self._name} of "
                    f"{self.parent.name} to {freq}, while it has previously been set "
                    f"to {self._settings.modulation_freq}."
                )
        self._settings.modulation_freq = freq
        if freq != 0:
            self._settings.nco_en = True

    def _generate_awg_dict(self) -> Dict[str, Any]:
        """
        Generates the dictionary that contains the awg waveforms in the
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
            The awg dictionary.

        Raises
        ------
        ValueError
            I or Q amplitude is being set outside of maximum range.
        """
        waveforms_complex = dict()
        for pulse in self.pulses:
            reserved_pulse_id = (
                non_generic.check_reserved_pulse_id(pulse)
                if self.instruction_generated_pulses_enabled
                else None
            )
            if reserved_pulse_id is None:
                raw_wf_data = generate_waveform_data(
                    pulse.data, sampling_rate=SAMPLING_RATE
                )
                raw_wf_data, amp_i, amp_q = normalize_waveform_data(raw_wf_data)
            else:
                pulse.uuid = reserved_pulse_id
                raw_wf_data, amp_i, amp_q = non_generic.generate_reserved_waveform_data(
                    reserved_pulse_id, pulse.data, sampling_rate=SAMPLING_RATE
                )

            if np.abs(amp_i) > self.parent.awg_output_volt:
                raise ValueError(
                    f"Attempting to set amplitude to an invalid value. "
                    f"Maximum voltage range is +-{self.parent.awg_output_volt} V for "
                    f"{self.parent.__class__.__name__}.\n"
                    f"{amp_i} V is set as amplitude for the I channel for "
                    f"{repr(pulse)}"
                )
            if np.abs(amp_q) > self.parent.awg_output_volt:
                raise ValueError(
                    f"Attempting to set amplitude to an invalid value. "
                    f"Maximum voltage range is +-{self.parent.awg_output_volt} V for "
                    f"{self.parent.__class__.__name__}.\n"
                    f"{amp_q} V is set as amplitude for the Q channel for "
                    f"{repr(pulse)}"
                )
            pulse.pulse_settings = QASMRuntimeSettings(
                awg_gain_0=amp_i / self.parent.awg_output_volt,
                awg_gain_1=amp_q / self.parent.awg_output_volt,
            )
            if pulse.uuid not in waveforms_complex and raw_wf_data is not None:
                waveforms_complex[pulse.uuid] = raw_wf_data
        return _generate_waveform_dict(waveforms_complex)

    def _generate_weights_dict(self) -> Dict[str, Any]:
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
            The acq dictionary.

        Raises
        ------
        NotImplementedError
            Currently, only two one dimensional waveforms can be used as acquisition
            weights. This exception is raised when either or both waveforms contain
            both a real and imaginary part.
        """
        waveforms_complex = dict()
        for acq in self.acquisitions:
            waveforms_data = acq.data["waveforms"]
            if len(waveforms_data) == 0:
                continue  # e.g. scope acquisition
            if acq.name == "SSBIntegrationComplex":
                continue
            if len(waveforms_data) != 2:
                raise ValueError(
                    f"Acquisitions need 2 waveforms (one for the I quadrature and one "
                    f"for the Q quadrature).\n\n{acq} has {len(waveforms_data)}"
                    "waveforms."
                )
            if acq.uuid not in waveforms_complex:
                raw_wf_data_real = generate_waveform_data(
                    waveforms_data[0], sampling_rate=SAMPLING_RATE
                )
                raw_wf_data_imag = generate_waveform_data(
                    waveforms_data[1], sampling_rate=SAMPLING_RATE
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

    def _generate_acq_declaration_dict(
        self, acquisitions: List[OpInfo]
    ) -> Dict[str, Any]:
        """
        Generates the "acquisitions" entry of the program json. It contains declaration
        of the acquisitions along with the number of bins and the corresponding index.

        For the name of the acquisition (in the hardware), the acquisition channel
        (cast to str) is used, and is thus identical to the index. Number of bins is
        taken to be the highest acq_index specified for that channel.

        Parameters
        ----------
        acquisitions:
            List of the acquisitions assigned to this sequencer.

        Returns
        -------
        :
            The "acquisitions" entry of the program json as a dict. The keys correspond
            to the names of the acquisitions (i.e. the acq_channel in the scheduler).
        """

        def get_acq_channel(acq: OpInfo) -> int:
            """Helper to extract the acq_channel."""
            return acq.data["acq_channel"]

        unique_channels = set(map(get_acq_channel, acquisitions))

        acq_declaration_dict = dict()
        for channel in unique_channels:
            indices = list()
            for acq in acquisitions:
                if get_acq_channel(acq) == channel:
                    indices.append(acq.data["acq_index"])
            if min(indices) != 0:
                raise ValueError(
                    f"Please make sure the lowest bin index used is 0. "
                    f"Found: {min(indices)} as lowest bin for channel {channel}. "
                    f"Problem occurred for port {self.port} with clock {self.clock},"
                    f"which corresponds to {self.name} of {self.parent.name}."
                )
            acq_declaration_dict[str(channel)] = {
                "num_bins": max(indices) + 1,
                "index": channel,
            }

        return acq_declaration_dict

    def update_settings(self):
        """
        Updates the sequencer settings to set all parameters that are determined by the
        compiler.
        """

    # pylint: disable=too-many-locals
    def generate_qasm_program(
        self,
        total_sequence_time: float,
        awg_dict: Optional[Dict[str, Any]] = None,
        weights_dict: Optional[Dict[str, Any]] = None,
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
        total_sequence_time
            Total time the program needs to play for. If the sequencer would be done
            before this time, a wait is added at the end to ensure synchronization.
        awg_dict
            Dictionary containing the pulse waveform data and the index that is assigned
            to the I and Q waveforms, as generated by the `generate_awg_dict` function.
            This is used to extract the relevant indexes when adding a play instruction.
        weights_dict
            Dictionary containing the acquisition waveform data and the index that is
            assigned to the I and Q waveforms, as generated by the `generate_acq_dict`
            function. This is used to extract the relevant indexes when adding an
            acquire instruction.
        repetitions
            Number of times to repeat execution of the schedule.

        Returns
        -------
        :
            The generated QASM program.
        """
        loop_label = "start"
        loop_register = "R0"

        qasm = QASMProgram(parent=self)
        # program header
        qasm.emit(q1asm_instructions.WAIT_SYNC, GRID_TIME)
        qasm.set_marker(self.parent.marker_configuration["start"])

        # program body
        pulses = list() if self.pulses is None else self.pulses
        acquisitions = list() if self.acquisitions is None else self.acquisitions
        op_list = pulses + acquisitions
        op_list = sorted(op_list, key=lambda p: (p.timing, p.is_acquisition))

        with qasm.loop(
            label=loop_label, register=loop_register, repetitions=repetitions
        ):
            op_queue = deque(op_list)
            while len(op_queue) > 0:
                operation = op_queue.popleft()
                if operation.is_acquisition:
                    idx0, idx1 = self.get_indices_from_wf_dict(
                        operation.uuid, weights_dict
                    )
                    qasm.wait_till_start_then_acquire(operation, idx0, idx1)
                else:
                    idx0, idx1 = self.get_indices_from_wf_dict(operation.uuid, awg_dict)
                    qasm.wait_till_start_then_play(operation, idx0, idx1)

            end_time = to_grid_time(total_sequence_time)
            wait_time = end_time - qasm.elapsed_time
            if wait_time < 0:
                raise RuntimeError(
                    f"Invalid timing detected, attempting to insert wait "
                    f"of {wait_time} ns. The total duration of the "
                    f"schedule is {end_time} but {qasm.elapsed_time} ns "
                    f"already processed."
                )
            qasm.auto_wait(wait_time)

        # program footer
        qasm.set_marker(self.parent.marker_configuration["end"])
        qasm.emit(q1asm_instructions.UPDATE_PARAMETERS, GRID_TIME)
        qasm.emit(q1asm_instructions.STOP)
        return str(qasm)

    @staticmethod
    def get_indices_from_wf_dict(uuid: str, wf_dict: Dict[str, Any]) -> Tuple[int, int]:
        """
        Takes a waveforms_dict or weights_dict and extracts the waveform indices based
        off of the uuid of the pulse/acquisition.

        Parameters
        ----------
        uuid
            The unique identifier of the pulse/acquisition.
        wf_dict
            The awg or acq dict that holds the waveform data and indices.

        Returns
        -------
        :
            Index of the I waveform.
        :
            Index of the Q waveform.
        """
        name_real, name_imag = generate_waveform_names_from_uuid(uuid)
        idx_real = None if name_real not in wf_dict else wf_dict[name_real]["index"]
        idx_imag = None if name_imag not in wf_dict else wf_dict[name_imag]["index"]
        return idx_real, idx_imag

    @staticmethod
    def _generate_waveforms_and_program_dict(
        program: str,
        waveforms_dict: Dict[str, Any],
        weights_dict: Optional[Dict[str, Any]] = None,
        acq_decl_dict: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generates the full waveforms and program dict that is to be uploaded to the
        sequencer from the program string and the awg and acq dicts, by combining them
        and assigning the appropriate keys.

        Parameters
        ----------
        program
            The compiled QASM program as a string.
        waveforms_dict
            The dictionary containing all the awg data and indices. This is expected to
            be of the form generated by the `generate_awg_dict` method.
        weights_dict
            The dictionary containing all the acq data and indices. This is expected to
            be of the form generated by the `generate_acq_dict` method.

        Returns
        -------
        :
            The combined program.
        """
        compiled_dict = dict()
        compiled_dict["program"] = program
        compiled_dict["waveforms"] = waveforms_dict
        if weights_dict is not None:
            compiled_dict["weights"] = weights_dict
        if acq_decl_dict is not None:
            compiled_dict["acquisitions"] = acq_decl_dict
        return compiled_dict

    @staticmethod
    def _dump_waveforms_and_program_json(
        wf_and_pr_dict: Dict[str, Any], label: Optional[str] = None
    ) -> str:
        """
        Takes a combined waveforms and program dict and dumps it as a json file.

        Parameters
        ----------
        wf_and_pr_dict
            The dict to dump as a json file.
        label
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
            Number of times execution the schedule is repeated.

        Returns
        -------
        :
            The compiled program. If no data is assigned to this sequencer, the
            compilation is skipped and None is returned instead.
        """
        if not self.has_data:
            return None

        awg_dict = self._generate_awg_dict()
        weights_dict = None
        if self.parent.supports_acquisition:
            weights_dict = (
                self._generate_weights_dict() if len(self.acquisitions) > 0 else dict()
            )
        acq_declaration_dict = (
            self._generate_acq_declaration_dict(self.acquisitions)
            if len(self.acquisitions) > 0
            else None
        )

        qasm_program = self.generate_qasm_program(
            self.parent.total_play_time,
            awg_dict,
            weights_dict,
            repetitions=repetitions,
        )

        wf_and_pr_dict = self._generate_waveforms_and_program_dict(
            qasm_program, awg_dict, weights_dict, acq_declaration_dict
        )

        json_filename = self._dump_waveforms_and_program_json(
            wf_and_pr_dict, f"{self.port}_{self.clock}"
        )
        self.update_settings()
        settings_dict = self.settings.to_dict()
        return {"seq_fn": json_filename, "settings": settings_dict}


class PulsarBase(ControlDeviceCompiler, ABC):
    """
    Pulsar specific implementation of
    :class:`quantify_scheduler.backends.qblox.compiler_abc.InstrumentCompiler`.

    This class is defined as an abstract base class since the distinctions between the
    different Pulsar devices are defined in subclasses.
    Effectively, this base class contains the functionality shared by all Pulsar
    devices and serves to avoid repeated code between them.
    """

    output_to_sequencer_idx = {"complex_output_0": 0, "complex_output_1": 1}
    """
    Dictionary that maps output names to specific sequencer indices. This
    implementation is temporary and will change when multiplexing is supported by
    the hardware.
    """
    sequencer_to_output_idx = {"seq0": 0, "seq1": 1}
    """
    Dictionary that maps sequencer names to specific (complex) output indices. This
    implementation is temporary and will change when multiplexing is supported by
    the hardware.
    """

    def __init__(
        self,
        parent: compiler_container.CompilerContainer,
        name: str,
        total_play_time: float,
        hw_mapping: Dict[str, Any],
    ):
        # pylint: disable=line-too-long
        """
        Constructor function.

        Parameters
        ----------
        parent
            Reference to the parent
            :class:`quantify_scheduler.backends.qblox.compiler_container.CompilerContainer`
            object.
        name
            Name of the `QCoDeS` instrument this compiler object corresponds to.
        total_play_time
            Total time execution of the schedule should go on for. This parameter is
            used to ensure that the different devices, potentially with different clock
            rates, can work in a synchronized way when performing multiple executions of
            the schedule.
        hw_mapping
            The hardware configuration dictionary for this specific device. This is one
            of the inner dictionaries of the overall hardware config.
        """
        super().__init__(parent, name, total_play_time, hw_mapping)
        verify_qblox_instruments_version()

        self.portclock_map = self._generate_portclock_to_seq_map()
        self.sequencers = self._construct_sequencers()
        self._settings = self.settings_type.extract_settings_from_mapping(hw_mapping)

    @property
    @abstractmethod
    def _max_sequencers(self) -> int:
        """
        Specifies the maximum amount of sequencers available to this instrument.

        Returns
        -------
        :
            The maximum amount of sequencers
        """

    @property
    def portclocks(self) -> List[Tuple[str, str]]:
        """Returns all the port and clocks available to this device."""
        return list(self.portclock_map.keys())

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
    @abstractmethod
    def settings_type(self) -> PulsarSettings:
        """
        Specifies the Pulsar Settings class used by the instrument.

        Returns
        -------
        :
            The maximum amount of sequencers
        """

    @property
    @abstractmethod
    def marker_configuration(self) -> dict[str, int]:
        """
        Specifies the values that the markers need to be set to at the start and end
        of each program.

        Returns
        -------
        :
            The maximum amount of sequencers
        """

    def _generate_portclock_to_seq_map(self) -> Dict[Tuple[str, str], str]:
        """
        Generates a mapping from portclock tuples to sequencer names.

        Returns
        -------
        :
            A dictionary with as key a portclock tuple and as value the name of a
            sequencer.
        """
        valid_io = (f"complex_output_{i}" for i in [0, 1])
        valid_seq_names = (f"seq{i}" for i in range(self._max_sequencers))

        mapping = dict()
        for io in valid_io:
            if io not in self.hw_mapping:
                continue

            io_cfg = self.hw_mapping[io]

            for seq_name in valid_seq_names:
                if seq_name not in io_cfg:
                    continue

                seq_cfg = io_cfg[seq_name]
                portclock = seq_cfg["port"], seq_cfg["clock"]

                mapping[portclock] = seq_name
        return mapping

    def _construct_sequencers(self) -> Dict[str, Sequencer]:
        """
        Constructs `Sequencer` objects for each port and clock combination
        belonging to this device.

        Returns
        -------
        :
            A dictionary containing the sequencer objects, the keys correspond to the
            names of the sequencers.

        Raises
        ------
        ValueError
            Raised when multiple definitions for the same sequencer is found, i.e. we
            are attempting to use the same sequencer multiple times in the compilation.
        ValueError
            Attempting to use more sequencers than available.
        """
        sequencers = dict()
        for io, io_cfg in self.hw_mapping.items():
            if not isinstance(io_cfg, dict):
                continue

            lo_name = io_cfg.get("lo_name", None)

            valid_seq_names = (f"seq{i}" for i in range(self._max_sequencers))
            for seq_name in valid_seq_names:
                if seq_name not in io_cfg:
                    continue

                seq_cfg = io_cfg[seq_name]
                portclock = seq_cfg["port"], seq_cfg["clock"]

                if seq_name in sequencers:
                    raise ValueError(
                        f"Attempting to create multiple instances of "
                        f"{seq_name}. Is it defined multiple times in "
                        f"the hardware configuration?"
                    )

                sequencers[seq_name] = Sequencer(
                    self, seq_name, portclock, seq_cfg, lo_name
                )

                if "mixer_corrections" in io_cfg:
                    sequencers[seq_name].mixer_corrections = MixerCorrections.from_dict(
                        io_cfg["mixer_corrections"]
                    )

        if len(sequencers.keys()) > self._max_sequencers:
            raise ValueError(
                f"Attempting to construct too many sequencer compilers. "
                f"Maximum allowed for {self.__class__} is {self._max_sequencers}!"
            )

        return sequencers

    def distribute_data(self):
        """
        Distributes the pulses and acquisitions assigned to this pulsar over the
        different sequencers based on their portclocks. Raises an exception in case
        the device does not support acquisitions.
        """

        if len(self._acquisitions) > 0 and not self.supports_acquisition:
            raise RuntimeError(
                f"Attempting to add acquisitions to {self.__class__} {self.name}, "
                f"which is not supported by hardware."
            )

        for portclock, pulse_data_list in self._pulses.items():
            for seq in self.sequencers.values():
                if seq.portclock == portclock:
                    seq.pulses = pulse_data_list

        for portclock, acq_data_list in self._acquisitions.items():
            for seq in self.sequencers.values():
                if seq.portclock == portclock:
                    seq.acquisitions = acq_data_list

    @abstractmethod
    def assign_frequencies(self, sequencer: Sequencer):
        r"""
        An abstract method that should be overridden. Meant to assign an IF frequency
        to each sequencer, or an LO frequency to each output (if applicable).
        For each sequencer, the following relation is obeyed:
        :math:`f_{RF} = f_{LO} + f_{IF}`.

        In this step it is thus expected that either the IF and/or the LO frequency has
        been set during instantiation. Otherwise an error is thrown. If the frequency
        is overconstraint (i.e. multiple values are somehow specified) an error is
        thrown during assignment.

        Raises
        ------
        ValueError
            Neither the LO nor the IF frequency has been set and thus contain
            :code:`None` values.
        """

    def prepare(self) -> None:
        """
        Performs the logic needed before being able to start the compilation. In effect,
        this means assigning the pulses and acquisitions to the sequencers and
        calculating the relevant frequencies in case an external local oscillator is
        used.
        """
        self.distribute_data()
        self._determine_scope_mode_acquisition_sequencer()
        for seq in self.sequencers.values():
            self.assign_frequencies(seq)

    @abstractmethod
    def update_settings(self):
        """
        Updates the Pulsar settings to set all parameters that are determined by the
        compiler.
        """

    def _determine_scope_mode_acquisition_sequencer(self) -> None:
        """
        Finds which sequencer has to perform raw trace acquisitions and adds it to the
        `scope_mode_sequencer` of the settings.

        Raises
        ------
        ValueError
            Multiple sequencers have to perform trace acquisition. This is not
            supported by the hardware.
        """

        def is_scope_acquisition(acquisition: OpInfo) -> bool:
            return acquisition.data["protocol"] == "trace"

        scope_acq_seq = None
        for seq in self.sequencers.values():
            has_scope = any(map(is_scope_acquisition, seq.acquisitions))
            if has_scope:
                if scope_acq_seq is not None:
                    raise ValueError(
                        f"Both sequencer {seq.name} and {scope_acq_seq} of "
                        f"{self.name} are required to perform scope mode "
                        "acquisitions. Only one sequencer per device can "
                        "trigger raw trace capture.\n\nPlease ensure that "
                        "only one port and clock combination has to "
                        "perform raw trace acquisition per instrument."
                    )
                scope_acq_seq = seq.name

        self._settings.scope_mode_sequencer = scope_acq_seq

    def compile(self, repetitions: int = 1) -> Optional[Dict[str, Any]]:
        """
        Performs the actual compilation steps for this pulsar, by calling the sequencer
        level compilation functions and combining them into a single dictionary. The
        compiled program has a settings key, and keys for every sequencer.

        Parameters
        ----------
        repetitions
            Number of times execution the schedule is repeated.

        Returns
        -------
        :
            The compiled program corresponding to this pulsar. It contains an entry for
            every sequencer and general "settings". If the device is not actually used,
            and an empty program is compiled, None is returned instead.
        """
        program = dict()
        for seq_name, seq in self.sequencers.items():
            seq_program = seq.compile(repetitions=repetitions)
            if seq_program is not None:
                program[seq_name] = seq_program

        if len(program) == 0:
            return None

        self._settings.hardware_averages = repetitions
        self.update_settings()
        program["settings"] = self._settings.to_dict()
        if self.supports_acquisition:
            acq_mapping = self._get_acquisition_mapping()
            if acq_mapping is not None:
                program["acq_mapping"] = acq_mapping
        return program

    def _get_acquisition_mapping(self) -> Optional[dict]:
        """
        Generates a mapping of acq_channel, acq_index to sequencer name, protocol.

        Returns
        -------
        :
            A dictionary containing tuple(acq_channel, acq_index) as keys and
            tuple(sequencer name, protocol) as value.
        """

        def extract_mapping_item(acquisition: OpInfo) -> Tuple[Tuple[int, int], str]:
            return (
                (
                    acquisition.data["acq_channel"],
                    acquisition.data["acq_index"],
                ),
                acquisition.data["protocol"],
            )

        acq_mapping = dict()
        for sequencer in self.sequencers.values():
            mapping_items = map(extract_mapping_item, sequencer.acquisitions)
            for item in mapping_items:
                acq_mapping[item[0]] = (sequencer.name, item[1])

        return acq_mapping if len(acq_mapping) > 0 else None


class PulsarBaseband(PulsarBase):
    """
    Abstract implementation that the Pulsar QCM and Pulsar QRM baseband modules should
    inherit from.
    """

    settings_type = PulsarSettings
    """The settings type used by Pulsar baseband-type devices"""

    def update_settings(self):
        """
        Updates the Pulsar settings to set all parameters that are determined by the
        compiler. Currently, this only changes the offsets based on the mixer
        calibration parameters.
        """

        # Will be changed when LO leakage correction is decoupled from the sequencer
        for seq in self.sequencers.values():
            if seq.mixer_corrections is not None:
                output_index = self.sequencer_to_output_idx[seq.name]
                if output_index == 0:
                    self._settings.offset_ch0_path0 = (
                        seq.mixer_corrections.offset_I / self.awg_output_volt
                    )
                    self._settings.offset_ch0_path1 = (
                        seq.mixer_corrections.offset_Q / self.awg_output_volt
                    )
                elif output_index == 1:
                    self._settings.offset_ch1_path0 = (
                        seq.mixer_corrections.offset_I / self.awg_output_volt
                    )
                    self._settings.offset_ch1_path1 = (
                        seq.mixer_corrections.offset_Q / self.awg_output_volt
                    )

    def assign_frequencies(self, sequencer: Sequencer):
        r"""
        An abstract method that should be overridden. Meant to assign an IF frequency
        to each sequencer, or an LO frequency to each output (if applicable).
        For each sequencer, the following relation is obeyed:
        :math:`f_{RF} = f_{LO} + f_{IF}`.

        In this step it is thus expected that either the IF and/or the LO frequency has
        been set during instantiation. Otherwise an error is thrown. If the frequency
        is overconstraint (i.e. multiple values are somehow specified) an error is
        thrown during assignment.

        Raises
        ------
        ValueError
            Neither the LO nor the IF frequency has been set and thus contain
            :code:`None` values.
        """

        if sequencer.clock not in self.parent.resources:
            return

        clk_freq = self.parent.resources[sequencer.clock]["freq"]
        lo_compiler = self.parent.instrument_compilers.get(
            sequencer.associated_ext_lo, None
        )
        if lo_compiler is None:
            sequencer.frequency = clk_freq
            return

        if_freq = sequencer.frequency
        lo_freq = lo_compiler.frequency

        if lo_freq is None and if_freq is None:
            raise ValueError(
                f"Frequency settings underconstraint for sequencer {sequencer.name} "
                f"with port {sequencer.port} and clock {sequencer.clock}. When using "
                f"an external local oscillator it is required to either supply an "
                f'"lo_freq" or an "interm_freq". Neither was given.'
            )

        if if_freq is not None:
            lo_compiler.frequency = clk_freq - if_freq

        if lo_freq is not None:
            sequencer.frequency = clk_freq - lo_freq


class PulsarRF(PulsarBase):
    r"""
    Abstract implementation that the Pulsar QCM-RF and Pulsar QRM-RF modules should
    inherit from.
    """

    settings_type = PulsarRFSettings
    """The settings type used by Pulsar RF-type devices"""

    def update_settings(self):
        """
        Updates the Pulsar settings to set all parameters that are determined by the
        compiler. Currently, this only changes the offsets based on the mixer
        calibration parameters.
        """

        # Will be changed when LO leakage correction is decoupled from the sequencer
        for seq in self.sequencers.values():
            if seq.mixer_corrections is not None:
                output_index = self.sequencer_to_output_idx[seq.name]
                if output_index == 0:
                    self._settings.offset_ch0_path0 = seq.mixer_corrections.offset_I
                    self._settings.offset_ch0_path1 = seq.mixer_corrections.offset_Q
                elif output_index == 1:
                    self._settings.offset_ch1_path0 = seq.mixer_corrections.offset_I
                    self._settings.offset_ch1_path1 = seq.mixer_corrections.offset_Q

    def assign_frequencies(self, sequencer: Sequencer):
        r"""
        An abstract method that should be overridden. Meant to assign an IF frequency
        to each sequencer, or an LO frequency to each output (if applicable).
        For each sequencer, the following relation is obeyed:
        :math:`f_{RF} = f_{LO} + f_{IF}`.

        In this step it is thus expected that either the IF and/or the LO frequency has
        been set during instantiation. Otherwise an error is thrown. If the frequency
        is overconstraint (i.e. multiple values are somehow specified) an error is
        thrown during assignment.

        Raises
        ------
        ValueError
            Neither the LO nor the IF frequency has been set and thus contain
            :code:`None` values.
        """

        if sequencer.clock not in self.parent.resources:
            return

        clk_freq = self.parent.resources[sequencer.clock]["freq"]

        # Now we have to identify the LO the sequencer is outputting to
        # We can do this by first checking the Sequencer-Output correspondence
        # And then use the fact that LOX is connected to OutputX

        output_index = self.sequencer_to_output_idx[sequencer.name]

        if_freq = sequencer.frequency
        lo_freq = (
            self._settings.lo0_freq if (output_index == 0) else self._settings.lo1_freq
        )

        if lo_freq is None and if_freq is None:
            raise ValueError(
                f"Frequency settings underconstraint for sequencer {sequencer.name} "
                f"with port {sequencer.port} and clock {sequencer.clock}. It is "
                f'required to either supply an "lo_freq" or an "interm_freq". Neither '
                f"was given."
            )

        if if_freq is not None:
            new_lo_freq = clk_freq - if_freq
            if lo_freq is not None and new_lo_freq != lo_freq:
                raise ValueError(
                    f"Attempting to set 'lo{output_index}_freq' to frequency "
                    f"{new_lo_freq}, while it has previously already been set to "
                    f"{lo_freq}!"
                )
            if output_index == 0:
                self._settings.lo0_freq = new_lo_freq
            elif output_index == 1:
                self._settings.lo1_freq = new_lo_freq

        if lo_freq is not None:
            sequencer.frequency = clk_freq - lo_freq
