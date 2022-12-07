# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Compiler base and utility classes for Qblox backend."""
# pylint: disable=too-many-lines
from __future__ import annotations

import json
import logging
from abc import ABC, ABCMeta, abstractmethod
from collections import defaultdict, deque
from functools import partial
from os import makedirs, path
from typing import Any, Callable, Dict, List, Literal, Optional, Set, Tuple, Union

import dataclasses
from pathvalidate import sanitize_filename
from qcodes.utils.helpers import NumpyJSONEncoder
from quantify_core.data.handling import gen_tuid, get_datadir

from quantify_scheduler.backends.qblox import (
    constants,
    driver_version_check,
    helpers,
    q1asm_instructions,
    register_manager,
)
from quantify_scheduler.backends.qblox.operation_handling.acquisitions import (
    AcquisitionStrategyPartial,
)
from quantify_scheduler.backends.qblox.operation_handling.base import IOperationStrategy
from quantify_scheduler.backends.qblox.operation_handling.factory import (
    get_operation_strategy,
)
from quantify_scheduler.backends.qblox.qasm_program import QASMProgram
from quantify_scheduler.backends.types.qblox import (
    BasebandModuleSettings,
    BaseModuleSettings,
    OpInfo,
    PulsarRFSettings,
    PulsarSettings,
    RFModuleSettings,
    SequencerSettings,
    StaticHardwareProperties,
    MarkerConfiguration,
)
from quantify_scheduler.enums import BinMode
from quantify_scheduler.helpers.schedule import (
    _extract_acquisition_metadata_from_acquisitions,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


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
        parent,  # No type hint due to circular import, added to docstring
        name: str,
        total_play_time: float,
        hw_mapping: Dict[str, Any],
        latency_corrections: Optional[Dict[str, float]] = None,
    ):
        # pylint: disable=line-too-long
        """
        Constructor for an InstrumentCompiler object.

        Parameters
        ----------
        parent: :class:`~quantify_scheduler.backends.qblox.compiler_container.CompilerContainer`
            Reference to the parent object.
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
        latency_corrections
            Dict containing the delays for each port-clock combination. This is specified in
            the top layer of hardware config.

        """
        self.parent = parent
        self.name = name
        self.total_play_time = total_play_time
        self.hw_mapping = hw_mapping
        self.latency_corrections = latency_corrections or {}

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
        parent,  # No type hint due to circular import, added to docstring
        name: str,
        total_play_time: float,
        hw_mapping: Dict[str, Any],
        latency_corrections: Optional[Dict[str, float]] = None,
    ):
        # pylint: disable=line-too-long
        """
        Constructor for a ControlDeviceCompiler object.

        Parameters
        ----------
        parent: :class:`~quantify_scheduler.backends.qblox.compiler_container.CompilerContainer`
            Reference to the parent object.
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
        latency_corrections
            Dict containing the delays for each port-clock combination. This is specified in
            the top layer of hardware config.
        """
        super().__init__(parent, name, total_play_time, hw_mapping, latency_corrections)
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
    def _portclocks_with_data(self) -> Set[Tuple[str, str]]:
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

    @property
    def _portclocks_with_pulses(self) -> Set[Tuple[str, str]]:
        """
        All the port-clock combinations associated with at least one pulse.
        Returns
        -------
        :
            A set containing all the port-clock combinations that are used by this
            InstrumentCompiler.
        """
        portclocks_used = set()
        portclocks_used.update(self._pulses.keys())
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
    Class that performs the compilation steps on the sequencer level.
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        parent: QbloxBaseModule,
        index: int,
        portclock: Tuple[str, str],
        static_hw_properties: StaticHardwareProperties,
        connected_outputs: Optional[Union[Tuple[int], Tuple[int, int]]],
        connected_inputs: Optional[Union[Tuple[int], Tuple[int, int]]],
        seq_settings: Dict[str, Any],
        latency_corrections: Dict[str, float],
        lo_name: Optional[str] = None,
        downconverter_freq: float = 0,
        mix_lo: bool = True,
    ):
        """
        Constructor for the sequencer compiler.

        Parameters
        ----------
        parent
            A reference to the parent instrument this sequencer belongs to.
        index
            Index of the sequencer.
        portclock
            Tuple that specifies the unique port and clock combination for this
            sequencer. The first value is the port, second is the clock.
        seq_settings
            Sequencer settings dictionary.
        latency_corrections
            Dict containing the delays for each port-clock combination.
        lo_name
            The name of the local oscillator instrument connected to the same output via
            an IQ mixer. This is used for frequency calculations.
        downconverter_freq
            Frequency of the external downconverter if one is being used.
            Defaults to 0, in which case no downconverter is being used.
        mix_lo
            Boolean flag for IQ mixing with LO.
            Defaults to True meaning IQ mixing is applied.
        """
        self.parent = parent
        self.index = index
        self.port = portclock[0]
        self.clock = portclock[1]
        self.pulses: List[IOperationStrategy] = []
        self.acquisitions: List[IOperationStrategy] = []
        self.associated_ext_lo: str = lo_name
        self.downconverter_freq: float = downconverter_freq
        self.mix_lo: bool = mix_lo

        self.static_hw_properties: StaticHardwareProperties = static_hw_properties

        self.register_manager = register_manager.RegisterManager()

        self.instruction_generated_pulses_enabled = seq_settings.get(
            "instruction_generated_pulses_enabled", False
        )

        self._settings = SequencerSettings.initialize_from_config_dict(
            seq_settings=seq_settings,
            connected_outputs=connected_outputs,
            connected_inputs=connected_inputs,
        )

        self.qasm_hook_func: Optional[Callable] = seq_settings.get(
            "qasm_hook_func", None
        )
        """Allows the user to inject custom Q1ASM code into the compilation, just prior
         to returning the final string."""

        portclock_key = f"{seq_settings['port']}-{seq_settings['clock']}"
        self.latency_correction: float = latency_corrections.get(portclock_key, 0)
        """Latency correction accounted for by delaying the start of the program."""

    @property
    def connected_outputs(self) -> Optional[Union[Tuple[int], Tuple[int, int]]]:
        """
        The indices of the output paths that this sequencer is producing awg
        data for.

        For the baseband modules, these indices correspond directly to a physical output
        (e.g. index 0 corresponds to output 1 etc.).

        For the RF modules, indexes 0 and 1 correspond to path0 and path1 of output 1,
        respectively, while indexes 2 and 3 correspond to path0 and path1 of output 2.
        """
        return self._settings.connected_outputs

    @property
    def connected_inputs(self) -> Optional[Union[Tuple[int], Tuple[int, int]]]:
        """
        The indices of the input paths that this sequencer is collecting
        data for.

        For the baseband modules, these indices correspond directly to a physical input
        (e.g. index 0 corresponds to output 1 etc.).

        For the RF modules, indexes 0 and 1 correspond to path0 and path1 of input 1.
        """
        return self._settings.connected_inputs

    @property
    def io_mode(self) -> Literal["complex", "real", "imag"]:
        """
        Specifies whether the sequencer is using only path0 (real), path1 (imag) or
        both (complex).

        If real or imag, the sequencer is restricted to only using real valued data.
        """
        if self._settings.connected_outputs is not None:
            return helpers.io_mode_from_ios(self._settings.connected_outputs)
        else:
            return helpers.io_mode_from_ios(self._settings.connected_inputs)

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
        return f"seq{self.index}"

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
                    f"Attempting to set the modulation frequency of {self.name} of "
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
        The final dictionary to be included in the json that is uploaded to the module
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
        wf_dict: Dict[str, Any] = {}
        for pulse in self.pulses:
            pulse.generate_data(wf_dict=wf_dict)
        return wf_dict

    def _generate_weights_dict(self) -> Dict[str, Any]:
        """
        Generates the dictionary that corresponds that contains the acq weights
        waveforms in the format accepted by the driver.

        Notes
        -----
        The final dictionary to be included in the json that is uploaded to the module
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
        wf_dict: Dict[str, Any] = {}
        for acq in self.acquisitions:
            acq.generate_data(wf_dict)
        return wf_dict

    def _generate_acq_declaration_dict(
        self,
        acquisitions: List[IOperationStrategy],
        repetitions: int,
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
        repetitions:
            The number of times to repeat execution of the schedule.

        Returns
        -------
        :
            The "acquisitions" entry of the program json as a dict. The keys correspond
            to the names of the acquisitions (i.e. the acq_channel in the scheduler).
        """
        acquisition_infos: List[OpInfo] = list(
            map(lambda acq: acq.operation_info, acquisitions)
        )

        # acquisition metadata for acquisitions relevant to this sequencer only
        acq_metadata = _extract_acquisition_metadata_from_acquisitions(
            acquisition_infos
        )

        # initialize an empty dictionary for the format required by module
        acq_declaration_dict = {}
        for acq_channel, acq_indices in acq_metadata.acq_indices.items():
            # Some sanity checks on the input for easier debugging.
            if min(acq_indices) != 0:
                raise ValueError(
                    f"Please make sure the lowest bin index used is 0. "
                    f"Found: {min(acq_indices)} as lowest bin for channel "
                    f"{acq_channel}. Problem occurred for port {self.port} with"
                    f" clock {self.clock}, which corresponds to {self.name} of "
                    f"{self.parent.name}."
                )
            if len(acq_indices) != max(acq_indices) + 1:
                raise ValueError(
                    f"Please make sure the used bins increment by 1 starting from "
                    f"0. Found: {max(acq_indices)} as the highest bin out of "
                    f"{len(acq_indices)} for channel {acq_channel}, indicating "
                    f"an acquisition_index was skipped. "
                    f"Problem occurred for port {self.port} with clock {self.clock},"
                    f"which corresponds to {self.name} of {self.parent.name}."
                )

            # Add the acquisition metadata to the acquisition declaration dict
            if acq_metadata.bin_mode == BinMode.APPEND:
                num_bins = repetitions * (max(acq_indices) + 1)
            elif acq_metadata.bin_mode == BinMode.AVERAGE:
                num_bins = max(acq_indices) + 1
            else:
                # currently the BinMode enum only has average and append.
                # this check exists to catch unexpected errors if we add more
                # BinModes in the future.
                raise NotImplementedError(f"Unknown bin mode {acq_metadata.bin_mode}.")
            if acq_metadata.acq_protocol == "looped_periodic_acquisition":
                if len(acquisition_infos) > 1:
                    raise ValueError(
                        "only one acquisition allowed if "
                        "looped_periodic_acquisition is used"
                    )
                num_bins = acquisition_infos[0].data["num_times"]
            acq_declaration_dict[str(acq_channel)] = {
                "num_bins": num_bins,
                "index": acq_channel,
            }

        return acq_declaration_dict

    # pylint: disable=too-many-locals
    def generate_qasm_program(
        self,
        total_sequence_time: float,
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
        repetitions
            Number of times to repeat execution of the schedule.

        Returns
        -------
        :
            The generated QASM program.
        """
        loop_label = "start"
        marker_config = self.static_hw_properties.marker_configuration

        qasm = QASMProgram(self.static_hw_properties, self.register_manager)
        if marker_config.init is not None:
            qasm.set_marker(marker_config.init)
            qasm.emit(q1asm_instructions.UPDATE_PARAMETERS, constants.GRID_TIME)
        # program header
        qasm.emit(q1asm_instructions.WAIT_SYNC, constants.GRID_TIME)
        qasm.emit(q1asm_instructions.UPDATE_PARAMETERS, constants.GRID_TIME)
        if marker_config.start is not None:
            qasm.set_marker(marker_config.start)

        pulses = [] if self.pulses is None else self.pulses
        acquisitions = [] if self.acquisitions is None else self.acquisitions

        self._initialize_append_mode_registers(qasm, acquisitions)

        # program body
        op_list = pulses + acquisitions
        op_list = sorted(
            op_list,
            key=lambda p: (p.operation_info.timing, p.operation_info.is_acquisition),
        )

        # Adds the latency correction, this needs to be a minimum of 4 ns,
        # so all sequencers get delayed by at least that.
        latency_correction_ns: int = self._get_latency_correction_ns(
            self.latency_correction
        )
        qasm.auto_wait(
            wait_time=constants.GRID_TIME + latency_correction_ns,
            count_as_elapsed_time=False,
            comment=f"latency correction of {constants.GRID_TIME} + "
            f"{latency_correction_ns} ns",
        )

        with qasm.loop(label=loop_label, repetitions=repetitions):
            qasm.emit(q1asm_instructions.RESET_PHASE)
            qasm.emit(q1asm_instructions.UPDATE_PARAMETERS, constants.GRID_TIME)
            op_queue = deque(op_list)
            while len(op_queue) > 0:
                operation = op_queue.popleft()
                qasm.wait_till_start_operation(operation.operation_info)
                operation.insert_qasm(qasm)

            end_time = helpers.to_grid_time(total_sequence_time)
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
        if marker_config.end is not None:
            qasm.set_marker(marker_config.end)
        qasm.emit(q1asm_instructions.UPDATE_PARAMETERS, constants.GRID_TIME)

        qasm.emit(q1asm_instructions.STOP)

        if self.qasm_hook_func:
            self.qasm_hook_func(qasm)

        self._settings.integration_length_acq = qasm.integration_length_acq
        return str(qasm)

    def _initialize_append_mode_registers(
        self, qasm: QASMProgram, acquisitions: List[AcquisitionStrategyPartial]
    ):
        """
        Adds the instructions to initialize the registers needed to use the append
        bin mode to the program. This should be added in the header.

        Parameters
        ----------
        qasm:
            The program to add the instructions to.
        acquisitions:
            A list with all the acquisitions to consider.
        """
        channel_to_reg = {}
        for acq in acquisitions:
            if acq.operation_info.data["bin_mode"] != BinMode.APPEND:
                continue

            channel = acq.operation_info.data["acq_channel"]
            if channel in channel_to_reg:
                acq_bin_idx_reg = channel_to_reg[channel]
            else:
                acq_bin_idx_reg = self.register_manager.allocate_register()
                channel_to_reg[channel] = acq_bin_idx_reg

                qasm.emit(
                    q1asm_instructions.MOVE,
                    0,
                    acq_bin_idx_reg,
                    comment=f"Initialize acquisition bin_idx for "
                    f"ch{acq.operation_info.data['acq_channel']}",
                )
            acq.bin_idx_register = acq_bin_idx_reg

    def _get_latency_correction_ns(self, latency_correction: float) -> int:
        if latency_correction == 0:
            return 0

        latency_correction_ns = int(round(latency_correction * 1e9))
        if latency_correction_ns % 4 != 0:
            logger.warning(
                f"Latency correction of {latency_correction_ns} ns specified"
                f" for {self.name} of {self.parent.name}, which is not a"
                f" multiple of {constants.GRID_TIME} ns. This feature should"
                f" be considered experimental and stable results are not guaranteed at "
                f"this stage."
            )

        return latency_correction_ns

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
        compiled_dict = {}
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

    def compile(
        self,
        repetitions: int = 1,
        sequence_to_file: Optional[bool] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Performs the full sequencer level compilation based on the assigned data and
        settings. If no data is assigned to this sequencer, the compilation is skipped
        and None is returned instead.

        Parameters
        ----------
        repetitions
            Number of times execution the schedule is repeated.
        sequence_to_file
            Dump waveforms and program dict to JSON file, filename stored in
            `Sequencer.settings.seq_fn`.

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
        acq_declaration_dict = None
        if self.parent.supports_acquisition:
            weights_dict = (
                self._generate_weights_dict() if len(self.acquisitions) > 0 else {}
            )
            acq_declaration_dict = (
                self._generate_acq_declaration_dict(
                    acquisitions=self.acquisitions, repetitions=repetitions
                )
                if len(self.acquisitions) > 0
                else {}
            )

        qasm_program = self.generate_qasm_program(
            self.parent.total_play_time,
            repetitions=repetitions,
        )

        wf_and_prog = self._generate_waveforms_and_program_dict(
            qasm_program, awg_dict, weights_dict, acq_declaration_dict
        )

        self._settings.sequence = wf_and_prog
        self._settings.seq_fn = None
        if sequence_to_file:
            self._settings.seq_fn = self._dump_waveforms_and_program_json(
                wf_and_pr_dict=wf_and_prog, label=f"{self.port}_{self.clock}"
            )

        seq_settings = self._settings.to_dict()
        return seq_settings


class QbloxBaseModule(ControlDeviceCompiler, ABC):
    """
    Qblox specific implementation of
    :class:`quantify_scheduler.backends.qblox.compiler_abc.InstrumentCompiler`.

    This class is defined as an abstract base class since the distinctions between the
    different devices are defined in subclasses.
    Effectively, this base class contains the functionality shared by all Qblox
    devices and serves to avoid repeated code between them.
    """

    def __init__(
        self,
        parent,  # No type hint due to circular import, added to docstring
        name: str,
        total_play_time: float,
        hw_mapping: Dict[str, Any],
        latency_corrections: Optional[Dict[str, float]] = None,
    ):
        # pylint: disable=line-too-long
        """
        Constructor function.

        Parameters
        ----------
        parent: :class:`quantify_scheduler.backends.qblox.compiler_container.CompilerContainer`
            Reference to the parent object.
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
        latency_corrections
            Dict containing the delays for each port-clock combination. This is specified in
            the top layer of hardware config.
        """
        super().__init__(parent, name, total_play_time, hw_mapping, latency_corrections)
        driver_version_check.verify_qblox_instruments_version()

        self.is_pulsar: bool = True
        """Specifies if it is a standalone Pulsar or a cluster module. To be overridden
        by the cluster compiler if needed."""
        self._settings: Union[
            BaseModuleSettings, None
        ] = None  # set in the prepare method.
        self.sequencers: Dict[str, Sequencer] = {}

    @property
    def portclocks(self) -> List[Tuple[str, str]]:
        """Returns all the port-clock combinations that this device can target."""

        portclocks = []

        for io in self.static_hw_properties.valid_ios:
            if io not in self.hw_mapping:
                continue

            portclock_configs = self.hw_mapping[io].get("portclock_configs", [])
            if not portclock_configs:
                raise KeyError(
                    f"No 'portclock_configs' entry found in '{io}' of {self.name}."
                )

            portclocks += [
                (target["port"], target["clock"]) for target in portclock_configs
            ]

        return portclocks

    @property
    @abstractmethod
    def settings_type(self) -> PulsarSettings:
        """
        Specifies the PulsarSettings class used by the instrument.
        """

    @property
    @abstractmethod
    def static_hw_properties(self) -> StaticHardwareProperties:
        """
        The static properties of the hardware. This effectively gathers all the
        differences between the different modules.
        """

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
            When the output names do not conform to the
            `complex_output_X`/`real_output_X` norm,
            where X is the index of the output.
        KeyError
            Raised if no 'portclock_configs' entry is found in the specific outputs of
            the hardware config.
        ValueError
            Raised when the same port-clock is multiply assigned in the hardware config.
        ValueError
            Attempting to use more sequencers than available.

        """
        # Figure out which outputs need to be turned on.
        marker_start_config = self.static_hw_properties.marker_configuration.start
        for io, io_cfg in self.hw_mapping.items():
            if (
                not isinstance(io_cfg, dict)
                or io not in self.static_hw_properties.valid_ios
            ):
                continue

            portclock_configs: List[Dict[str, Any]] = io_cfg.get(
                "portclock_configs", []
            )
            if not portclock_configs:
                continue

            for target in portclock_configs:
                portclock = (target["port"], target["clock"])
                if portclock in self._portclocks_with_pulses:
                    output_map = (
                        self.static_hw_properties.marker_configuration.output_map
                    )
                    if io in output_map:
                        marker_start_config |= output_map[io]

        updated_static_hw_properties = dataclasses.replace(
            self.static_hw_properties,
            marker_configuration=MarkerConfiguration(
                init=self.static_hw_properties.marker_configuration.init,
                start=marker_start_config,
                end=self.static_hw_properties.marker_configuration.end,
            ),
        )

        # Setup each sequencer.
        sequencers: Dict[str, Sequencer] = {}
        portclock_io_map: Dict[Tuple, str] = {}

        for io, io_cfg in self.hw_mapping.items():
            if not isinstance(io_cfg, dict):
                continue
            if io not in self.static_hw_properties.valid_ios:
                raise ValueError(
                    f"Invalid hardware config. '{io}' of {self.name} is not a "
                    f"valid name of an input/output.\n\nSupported names:\n"
                    f"{self.static_hw_properties.valid_ios}"
                )

            lo_name = io_cfg.get("lo_name", None)
            downconverter_freq = io_cfg.get("downconverter_freq", 0)
            mix_lo = io_cfg.get("mix_lo", True)

            portclock_configs: List[Dict[str, Any]] = io_cfg.get(
                "portclock_configs", []
            )

            if not portclock_configs:
                raise KeyError(
                    f"No 'portclock_configs' entry found in '{io}' of {self.name}."
                )

            for target in portclock_configs:
                portclock = (target["port"], target["clock"])

                if portclock in self._portclocks_with_data:
                    connected_outputs = helpers.output_name_to_outputs(io)
                    connected_inputs = helpers.input_name_to_inputs(io)
                    seq_idx = len(sequencers)
                    new_seq = Sequencer(
                        parent=self,
                        index=seq_idx,
                        portclock=portclock,
                        static_hw_properties=updated_static_hw_properties,
                        connected_outputs=connected_outputs,
                        connected_inputs=connected_inputs,
                        seq_settings=target,
                        latency_corrections=self.latency_corrections,
                        lo_name=lo_name,
                        mix_lo=mix_lo,
                        downconverter_freq=downconverter_freq,
                    )
                    sequencers[new_seq.name] = new_seq

                    # Check if the portclock was not multiply specified, which is not allowed
                    if portclock in portclock_io_map:
                        raise ValueError(
                            f"Portclock {portclock} was assigned to multiple "
                            f"portclock_configs of {self.name}. This portclock was "
                            f"used in io '{io}' despite being already previously "
                            f"used in io '{portclock_io_map[portclock]}'. When using "
                            f"the same portclock for output and input, assigning only "
                            f"the output suffices."
                        )

                    portclock_io_map[portclock] = io

        # Check if more portclock_configs than sequencers are active
        if len(sequencers) > self.static_hw_properties.max_sequencers:
            raise ValueError(
                "Number of simultaneously active port-clock combinations exceeds "
                "number of sequencers."
                f"Maximum allowed for {self.name} ({self.__class__.__name__}) is "
                f"{self.static_hw_properties.max_sequencers}!"
            )

        return sequencers

    def distribute_data(self):
        """
        Distributes the pulses and acquisitions assigned to this module over the
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
                instr_gen_pulses = seq.instruction_generated_pulses_enabled
                if seq.portclock == portclock or (
                    portclock[0] is None and portclock[1] == seq.clock
                ):
                    op_info_to_op_strategy_func = partial(
                        get_operation_strategy,
                        instruction_generated_pulses_enabled=instr_gen_pulses,
                        io_mode=seq.io_mode,
                    )
                    strategies_for_pulses = map(
                        op_info_to_op_strategy_func,
                        pulse_data_list,
                    )
                    if seq.pulses is None:
                        seq.pulses = []

                    for pulse_strategy in strategies_for_pulses:
                        seq.pulses.append(pulse_strategy)

        for portclock, acq_data_list in self._acquisitions.items():
            for seq in self.sequencers.values():
                instr_gen_pulses = seq.instruction_generated_pulses_enabled
                if seq.portclock == portclock:
                    partial_func = partial(
                        get_operation_strategy,
                        instruction_generated_pulses_enabled=instr_gen_pulses,
                        io_mode=seq.io_mode,
                    )
                    func_map = map(
                        partial_func,
                        acq_data_list,
                    )
                    seq.acquisitions = list(func_map)

    @abstractmethod
    def assign_frequencies(self, sequencer: Sequencer):
        r"""
        An abstract method that should be overridden. Meant to assign an IF frequency
        to each sequencer, and an LO frequency to each output (if applicable).

        What is executed depends on the mix_lo boolean.

        Raises
        ------
        ValueError
            Neither the LO nor the IF frequency has been set and thus contain
            :code:`None` values.
        """

    @abstractmethod
    def assign_attenuation(self):
        """
        An abstract method that should be overridden. Meant to assign
        attenuation settings from the hardware configuration if there is any.
        """

    @staticmethod
    def downconvert_clock(downconverter_freq: float, clock_freq: float):
        """
        Downconverts clock frequency.

        Parameters
        ----------
        downconverter_freq
            Frequency of the downconverter.
        clock_freq
            clock frequency that is being downconverted.

        Raises
        ------
        ValueError
            When downconverter frequency is negative.
        ValueError
            When downconverter frequency is less than the clock frequency.
        """

        if downconverter_freq == 0:
            return clock_freq

        if downconverter_freq < 0:
            raise ValueError("Downconverter frequency must be positive.")

        if downconverter_freq < clock_freq:
            raise ValueError(
                "Downconverter frequency specified for this port and clock combination"
                "must be greater than its clock frequency."
            )

        return downconverter_freq - clock_freq

    def prepare(self) -> None:
        """
        Performs the logic needed before being able to start the compilation. In effect,
        this means assigning the pulses and acquisitions to the sequencers and
        calculating the relevant frequencies in case an external local oscillator is
        used.
        """
        self._settings = self.settings_type.extract_settings_from_mapping(
            self.hw_mapping
        )
        self._settings = self._configure_mixer_offsets_and_gains(
            self._settings, self.hw_mapping
        )
        self.sequencers = self._construct_sequencers()
        self.distribute_data()
        self._determine_scope_mode_acquisition_sequencer()
        for seq in self.sequencers.values():
            self.assign_frequencies(seq)
        self.assign_attenuation()

    def _configure_mixer_offsets_and_gains(
        self, settings: BaseModuleSettings, hw_mapping: Dict[str, Any]
    ) -> BaseModuleSettings:
        """
        We configure the mixer offsets, gains, attenuations
        after initializing the settings such we can account for
        the differences in the hardware. e.g. the V vs mV encountered here.

        Parameters
        ----------
        settings
            The settings dataclass to which to add the dc offsets and gains.
        hw_mapping
            The hardware configuration.

        Returns
        -------
        :
            The settings dataclass after adding the normalized offsets and gains.

        Raises
        ------
        ValueError
            An offset was used outside of the allowed range.
        """

        def calc_from_units_volt(
            param_name: str, cfg: Dict[str, Any]
        ) -> Optional[float]:

            offset_in_config = cfg.get(param_name, None)  # Always in volts
            if offset_in_config is None:
                return None

            conversion_factor = 1
            voltage_range = self.static_hw_properties.mixer_dc_offset_range
            if voltage_range.units == "mV":
                conversion_factor = 1e3
            elif voltage_range.units != "V":
                raise RuntimeError(
                    f"Parameter {param_name} of {self.name} specifies "
                    f"the units {voltage_range.units}, but this is not "
                    f"supported by the Qblox backend."
                )

            calculated_offset = offset_in_config * conversion_factor
            if (
                calculated_offset < voltage_range.min_val
                or calculated_offset > voltage_range.max_val
            ):
                raise ValueError(
                    f"Attempting to set {param_name} of {self.name} to "
                    f"{offset_in_config} V. {param_name} has to be between "
                    f"{voltage_range.min_val/ conversion_factor} and "
                    f"{voltage_range.max_val/ conversion_factor} V!"
                )

            return calculated_offset

        supported_outputs = ("complex_output_0", "complex_output_1")
        for output_idx, output_label in enumerate(supported_outputs):
            if output_label not in hw_mapping:
                continue

            output_cfg = hw_mapping[output_label]
            if output_idx == 0:
                settings.offset_ch0_path0 = calc_from_units_volt(
                    "dc_mixer_offset_I", output_cfg
                )
                settings.offset_ch0_path1 = calc_from_units_volt(
                    "dc_mixer_offset_Q", output_cfg
                )
            else:
                settings.offset_ch1_path0 = calc_from_units_volt(
                    "dc_mixer_offset_I", output_cfg
                )
                settings.offset_ch1_path1 = calc_from_units_volt(
                    "dc_mixer_offset_Q", output_cfg
                )

        complex_output_0 = hw_mapping.get("complex_output_0", None)
        if complex_output_0 is not None:
            settings.in0_gain = complex_output_0.get("input_gain_I", None)
            settings.in1_gain = complex_output_0.get("input_gain_Q", None)
        else:
            settings.in0_gain = hw_mapping.get("real_output_0", {}).get(
                "input_gain", None
            )
            settings.in1_gain = hw_mapping.get("real_output_1", {}).get(
                "input_gain", None
            )

        return settings

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
            op_infos = [acq.operation_info for acq in seq.acquisitions]

            has_scope = any(map(is_scope_acquisition, op_infos))
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
                scope_acq_seq = seq.index

        self._settings.scope_mode_sequencer = scope_acq_seq

    def compile(
        self,
        repetitions: int = 1,
        sequence_to_file: Optional[bool] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Performs the actual compilation steps for this module, by calling the sequencer
        level compilation functions and combining them into a single dictionary. The
        compiled program has a settings key, and keys for every sequencer.

        Parameters
        ----------
        repetitions
            Number of times execution the schedule is repeated.
        sequence_to_file
            Dump waveforms and program dict to JSON file, filename stored in
            `Sequencer.settings.seq_fn`.

        Returns
        -------
        :
            The compiled program corresponding to this module. It contains an entry for
            every sequencer and general "settings". If the device is not actually used,
            and an empty program is compiled, None is returned instead.
        """
        program = {}

        if sequence_to_file is None:
            sequence_to_file = self.hw_mapping.get("sequence_to_file", True)

        for seq_name, seq in self.sequencers.items():
            seq_program = seq.compile(
                repetitions=repetitions, sequence_to_file=sequence_to_file
            )
            if seq_program is not None:
                program[seq_name] = seq_program

        if len(program) == 0:
            return None

        self._settings.hardware_averages = repetitions
        program["settings"] = self._settings.to_dict()
        if self.supports_acquisition:
            # Add both acquisition metadata (a summary) and acq_mapping
            program["acq_metadata"] = {}

            for sequencer in self.sequencers.values():
                if not sequencer.acquisitions:
                    continue
                acq_metadata = _extract_acquisition_metadata_from_acquisitions(
                    [acq.operation_info for acq in sequencer.acquisitions]
                )
                program["acq_metadata"][sequencer.name] = acq_metadata

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

        acq_mapping = {}
        for sequencer in self.sequencers.values():
            mapping_items = map(
                extract_mapping_item,
                [acq.operation_info for acq in sequencer.acquisitions],
            )
            for item in mapping_items:
                acq_mapping[item[0]] = (sequencer.name, item[1])

        return acq_mapping if len(acq_mapping) > 0 else None


class QbloxBasebandModule(QbloxBaseModule):
    """
    Abstract class with all the shared functionality between the QRM and QCM baseband
    modules.
    """

    @property
    def settings_type(self) -> type:
        """The settings type used by baseband-type devices."""
        return PulsarSettings if self.is_pulsar else BasebandModuleSettings

    def assign_frequencies(self, sequencer: Sequencer):
        """
        Assigns frequencies for baseband modules.
        """
        # ensure to select the top level parent object of the sequencer (pulsar or cluster)
        if self.is_pulsar:
            parent = self.parent
        else:
            parent = self.parent.parent

        if sequencer.associated_ext_lo is None:
            if sequencer.frequency == 0:
                # disable the numerically controlled oscillator (nco)
                sequencer.settings.nco_en = False
            else:
                # set the nco frequency to the clock frequency, and enable the nco.
                clock_freq = parent.resources[sequencer.clock]["freq"]
                sequencer.frequency = clock_freq
                sequencer.settings.nco_en = True
        else:
            self.assign_frequency_with_ext_lo(sequencer, parent)

    @staticmethod
    def assign_frequency_with_ext_lo(sequencer: Sequencer, container):
        r"""
        Meant to assign an IF frequency
        to each sequencer, or an LO frequency to each output (if applicable).
        For each sequencer, the following relation is obeyed
        (if mix_lo is set to True) :math:`f_{RF} = f_{LO} + f_{IF}`.

        If mix_lo is True it is thus expected that either the IF and/or the
        LO frequency has been set during instantiation.
        Otherwise, an error is thrown. If the frequency is overconstraint
        (i.e. multiple values are somehow specified) an error is thrown during
        assignment.

        If mix_lo is False (For example when the LO is used to drive an external
        device) the relation :math:`f_{RF} = f_{LO}` is upheld.
        In this case this function only serves to turn on the NCO, and the LO
        is given the same frequency as the clock.

        Raises
        ------
        ValueError
            Neither the LO nor the IF frequency has been set and thus contain
            :code:`None` values.
        """

        if sequencer.clock not in container.resources:
            return

        clock_freq = container.resources[sequencer.clock]["freq"]
        lo_compiler = container.instrument_compilers.get(
            sequencer.associated_ext_lo, None
        )

        if not sequencer.mix_lo:
            lo_compiler.frequency = clock_freq
            sequencer.settings.nco_en = sequencer.frequency != 0
            return

        if_freq = sequencer.frequency
        lo_freq = lo_compiler.frequency

        clock_freq = QbloxBaseModule.downconvert_clock(
            sequencer.downconverter_freq, clock_freq
        )

        if lo_freq is None and if_freq is None:
            raise ValueError(
                f"Frequency settings underconstraint for sequencer {sequencer.name} "
                f"with port {sequencer.port} and clock {sequencer.clock}. When using "
                f"an external local oscillator it is required to either supply an "
                f'"lo_freq" or an "interm_freq". Neither was given.'
            )

        if if_freq is not None:
            lo_compiler.frequency = clock_freq - if_freq

        if lo_freq is not None:
            if_freq = clock_freq - lo_freq
            sequencer.frequency = if_freq

        if if_freq != 0 and if_freq is not None:
            sequencer.settings.nco_en = True

    def assign_attenuation(self):
        """
        Meant to assign attenuation settings from the hardware configuration, if there
        is any. For baseband modules there is no attenuation parameters currently.
        """


class QbloxRFModule(QbloxBaseModule):
    """
    Abstract class with all the shared functionality between the QRM-RF and QCM-RF
    modules.
    """

    @property
    def settings_type(self) -> type:
        """The settings type used by RF-type devices"""
        return PulsarRFSettings if self.is_pulsar else RFModuleSettings

    def assign_frequencies(self, sequencer: Sequencer):
        r"""
        Meant to assign an IF frequency
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
        resources = (
            self.parent.resources if self.is_pulsar else self.parent.parent.resources
        )

        if sequencer.clock not in resources:
            return

        clock_freq = resources[sequencer.clock]["freq"]

        # Now we have to identify the LO the sequencer is outputting to
        # We can do this by first checking the Sequencer-Output correspondence
        # And then use the fact that LOX is connected to OutputX

        self._validate_io_mode(sequencer)
        for real_output in sequencer.connected_outputs:
            if real_output % 2 != 0:
                # We will only use real output 0 and 2,
                # since 1 and 3 are part of the same
                # complex outputs.
                continue
            complex_output = 0 if real_output == 0 else 1
            if_freq = sequencer.frequency
            lo_freq = (
                self._settings.lo0_freq
                if (complex_output == 0)
                else self._settings.lo1_freq
            )

            if lo_freq is None and if_freq is None:
                raise ValueError(
                    f"Frequency settings underconstraint for sequencer {sequencer.name}"
                    f" with port {sequencer.port} and clock {sequencer.clock}. It is "
                    f'required to either supply an "lo_freq" or an "interm_freq". '
                    f"Neither was given."
                )

            clock_freq = QbloxBaseModule.downconvert_clock(
                sequencer.downconverter_freq, clock_freq
            )

            if if_freq is not None:
                new_lo_freq = clock_freq - if_freq
                if lo_freq is not None and new_lo_freq != lo_freq:
                    raise ValueError(
                        f"Attempting to set 'lo{complex_output}_freq' to frequency "
                        f"{new_lo_freq}, while it has previously already been set to "
                        f"{lo_freq}!"
                    )
                if complex_output == 0:
                    self._settings.lo0_freq = new_lo_freq
                elif complex_output == 1:
                    self._settings.lo1_freq = new_lo_freq

            if lo_freq is not None:
                sequencer.frequency = clock_freq - lo_freq

    def assign_attenuation(self):
        """
        Assigns attenuation settings from the hardware configuration.
        """
        self._settings.in0_att = self.hw_mapping.get("complex_output_0", {}).get(
            "input_att", None
        )
        self._settings.out0_att = self.hw_mapping.get("complex_output_0", {}).get(
            "output_att", None
        )
        self._settings.out1_att = self.hw_mapping.get("complex_output_1", {}).get(
            "output_att", None
        )

    @classmethod
    def _validate_io_mode(cls, sequencer: Sequencer):
        if sequencer.io_mode != "complex":
            raise ValueError(
                f"Attempting to use {cls.__name__} in real "
                f"mode, but this is not supported for Qblox RF modules."
            )
