# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Compiler base and utility classes for Qblox backend."""
# pylint: disable=too-many-lines
from __future__ import annotations

import dataclasses
import json
import logging
import warnings
from abc import ABC, ABCMeta, abstractmethod
from collections import defaultdict, deque
from functools import partial
from os import makedirs, path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
    Union,
)

import numpy as np
from pathvalidate import sanitize_filename
from qcodes.utils.helpers import NumpyJSONEncoder
from quantify_core.data.handling import gen_tuid, get_datadir

from quantify_scheduler.backends.qblox import (
    constants,
    driver_version_check,
    helpers,
    instrument_compilers,
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
from quantify_scheduler.operations.pulse_library import SetClockFrequency

if TYPE_CHECKING:
    from quantify_scheduler.backends.qblox.instrument_compilers import LocalOscillator

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
        self._pulses: Dict[Tuple[str, str], List[OpInfo]] = defaultdict(list)
        self._acquisitions: Dict[Tuple[str, str], List[OpInfo]] = defaultdict(list)

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
        downconverter_freq: Optional[float] = None,
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
            .. warning::
                Using `downconverter_freq` requires custom Qblox hardware, do not use otherwise.
            Frequency of the external downconverter if one is being used.
            Defaults to ``None``, in which case the downconverter is inactive.
        mix_lo
            Boolean flag for IQ mixing with LO.
            Defaults to ``True`` meaning IQ mixing is applied.
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
        if self._settings.modulation_freq is not None and not np.isclose(
            self._settings.modulation_freq, freq
        ):
            raise ValueError(
                f"Attempting to set the modulation frequency of '{self.name}' of "
                f"'{self.parent.name}' to {freq:e}, while it has previously been set "
                f"to {self._settings.modulation_freq:e}."
            )

        self._settings.modulation_freq = freq
        self._settings.nco_en = freq is not None

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

    def _prepare_acq_settings(
        self, acquisitions: List[IOperationStrategy], repetitions: int
    ):
        """
        Sets sequencer settings that are specific to certain acquisitions.
        For example for a TTL acquisition strategy.

        Parameters
        ----------
        acquisitions
            List of the acquisitions assigned to this sequencer.
        repetitions
            The number of times to repeat execution of the schedule.
        """
        acquisition_infos: List[OpInfo] = list(
            map(lambda acq: acq.operation_info, acquisitions)
        )
        acq_metadata = helpers.extract_acquisition_metadata_from_acquisitions(
            acquisitions=acquisition_infos, repetitions=repetitions
        )
        if acq_metadata.acq_protocol == "TriggerCount":
            self._settings.ttl_acq_auto_bin_incr_en = (
                acq_metadata.bin_mode == BinMode.AVERAGE
            )
            if self.connected_inputs is not None:
                if len(self.connected_inputs) == 1:
                    self._settings.ttl_acq_input_select = self.connected_inputs[0]
                else:
                    raise ValueError(
                        f"Please make sure you use a single real input for this "
                        f"portclock combination. "
                        f"Found: {len(self.connected_inputs)} connected. "
                        f"TTL acquisition does not support multiple inputs."
                        f"Problem occurred for port {self.port} with"
                        f"clock {self.clock}, which corresponds to {self.name} of "
                        f"{self.parent.name}."
                    )

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
        acq_metadata = helpers.extract_acquisition_metadata_from_acquisitions(
            acquisitions=acquisition_infos, repetitions=repetitions
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
            if len(acq_indices) < max(acq_indices) + 1:
                raise ValueError(
                    f"Please make sure the used bins increment by 1 starting from "
                    f"0. Found: {max(acq_indices)} as the highest bin out of "
                    f"{len(acq_indices)} for channel {acq_channel}, indicating "
                    f"an acquisition index was skipped. "
                    f"Problem occurred for port {self.port} with clock {self.clock}, "
                    f"which corresponds to {self.name} of {self.parent.name}."
                )

            # Add the acquisition metadata to the acquisition declaration dict
            if acq_metadata.bin_mode == BinMode.APPEND:
                num_bins = repetitions * (max(acq_indices) + 1)
            elif acq_metadata.bin_mode == BinMode.AVERAGE:
                if acq_metadata.acq_protocol == "TriggerCount":
                    num_bins = constants.MAX_NUMBER_OF_BINS
                else:
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
        repetitions: int = 1,
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

        Warns
        -----
        RuntimeWarning
            When number of instructions in the generated QASM program exceeds the
            maximum supported number of instructions for sequencers in the type of
            module.

        Raises
        ------
        RuntimeError
            Upon `total_sequence_time` exceeding :attr:`.QASMProgram.elapsed_time`.
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

        max_instructions = (
            constants.MAX_NUMBER_OF_INSTRUCTIONS_QCM
            if self.parent.__class__
            in [instrument_compilers.QcmModule, instrument_compilers.QcmRfModule]
            else constants.MAX_NUMBER_OF_INSTRUCTIONS_QRM
        )
        if (num_instructions := len(qasm.instructions)) > max_instructions:
            warnings.warn(
                f"Number of instructions ({num_instructions}) compiled for "
                f"'{self.name}' of {self.parent.__class__.__name__} "
                f"'{self.parent.name}' exceeds the maximum supported number of "
                f"instructions in Q1ASM programs for {self.parent.__class__.__name__} "
                f"({max_instructions}).",
                RuntimeWarning,
            )

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
            weights_dict = {}
            acq_declaration_dict = {}
            if len(self.acquisitions) > 0:
                self._prepare_acq_settings(
                    acquisitions=self.acquisitions, repetitions=repetitions
                )
                weights_dict = self._generate_weights_dict()
                acq_declaration_dict = self._generate_acq_declaration_dict(
                    acquisitions=self.acquisitions, repetitions=repetitions
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

    def _construct_sequencers(self):
        """
        Constructs `Sequencer` objects for each port and clock combination
        belonging to this device.

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
                    f"Invalid hardware config: '{io}' of "
                    f"{self.name} ({self.__class__.__name__}) "
                    f"is not a valid name of an input/output."
                    f"\n\nSupported names for {self.__class__.__name__}:\n"
                    f"{self.static_hw_properties.valid_ios}"
                )

            lo_name = io_cfg.get("lo_name", None)
            downconverter_freq = io_cfg.get("downconverter_freq", None)
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
                "number of sequencers. "
                f"Maximum allowed for {self.name} ({self.__class__.__name__}) is "
                f"{self.static_hw_properties.max_sequencers}!"
            )

        self.sequencers = sequencers

    def distribute_data(self):
        """
        Distributes the pulses and acquisitions assigned to this module over the
        different sequencers based on their portclocks. Raises an exception in case
        the device does not support acquisitions.
        """

        if len(self._acquisitions) > 0 and not self.supports_acquisition:
            raise RuntimeError(
                f"Attempting to add acquisitions to  {self.__class__} {self.name}, "
                f"which is not supported by hardware."
            )

        compiler_container = self.parent if self.is_pulsar else self.parent.parent

        portclock: Tuple[str, str]
        pulse_data_list: List[OpInfo]
        for portclock, pulse_data_list in self._pulses.items():
            for seq in self.sequencers.values():
                if seq.portclock == portclock or (
                    portclock[0] is None and portclock[1] == seq.clock
                ):
                    clock_freq = compiler_container.resources.get(seq.clock, {}).get(
                        "freq", None
                    )

                    pulse_data: OpInfo
                    for pulse_data in pulse_data_list:
                        if pulse_data.name == SetClockFrequency.__name__:
                            pulse_data.data.update(
                                {
                                    "clock_freq_old": clock_freq,
                                    "interm_freq_old": seq.frequency,
                                }
                            )

                    op_info_to_op_strategy_func = partial(
                        get_operation_strategy,
                        instruction_generated_pulses_enabled=seq.instruction_generated_pulses_enabled,
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

        acq_data_list: List[OpInfo]
        for portclock, acq_data_list in self._acquisitions.items():
            for seq in self.sequencers.values():
                if seq.portclock == portclock:
                    op_info_to_op_strategy_func = partial(
                        get_operation_strategy,
                        instruction_generated_pulses_enabled=seq.instruction_generated_pulses_enabled,
                        io_mode=seq.io_mode,
                    )
                    strategies_for_acquisitions = map(
                        op_info_to_op_strategy_func,
                        acq_data_list,
                    )
                    seq.acquisitions = list(strategies_for_acquisitions)

    @abstractmethod
    def assign_frequencies(self, sequencer: Sequencer):
        r"""
        An abstract method that should be overridden. Meant to assign an IF frequency
        to each sequencer, and an LO frequency to each output (if applicable).
        """

    def _set_lo_interm_freqs(
        self,
        freqs: helpers.Frequencies,
        sequencer: Sequencer,
        compiler_lo_baseband: Optional[LocalOscillator] = None,
        lo_freq_setting_rf: Optional[str] = None,
    ):
        """
        Sets the LO/IF frequencies, for baseband and RF modules.

        Parameters
        ----------
        freqs
            LO, IF, and clock frequencies, supplied via an :class:`.helpers.Frequencies`
            object.
        sequencer
            The sequencer for which frequences are to be set.
        compiler_lo_baseband
            For baseband modules, supply the :class:`.LocalOscillator` instrument
            compiler of which the frequency is to be set.
        lo_freq_setting_rf
            For RF modules, supply the name of the LO frequency param from the
            :class:`.RFModuleSettings` that is to be set.

        Raises
        ------
        ValueError
            In case neither LO frequency nor IF has been supplied.
        ValueError
            In case both LO frequency and IF have been supplied and do not adhere to
            :math:`f_{RF} = f_{LO} + f_{IF}`.
        ValueError
            In case of RF, when the LO frequency was already set to a different value.
        """
        underconstr = freqs.LO is None and freqs.IF is None
        overconstr = (
            freqs.LO is not None
            and freqs.IF is not None
            and not np.isclose(freqs.LO + freqs.IF, freqs.clock)
        )

        if underconstr or overconstr:
            raise ValueError(
                f"Frequency settings {'under' if underconstr else 'over'}constrained for "
                f"sequencer '{sequencer.name}' of '{self.name}' "
                f"with port '{sequencer.port}' and clock '{sequencer.clock}'. "
                f"It is required to either supply an "
                f"'lo_freq' or an 'interm_freq' "
                f"({'neither' if underconstr else 'both'} supplied)"
                + "{}.".format(
                    ""
                    if sequencer.associated_ext_lo is None
                    else f" in using an external local oscillator "
                    f"({sequencer.associated_ext_lo})"
                )
            )

        if freqs.LO is not None:
            if compiler_lo_baseband is not None:
                compiler_lo_baseband.frequency = freqs.LO

            elif lo_freq_setting_rf is not None:
                previous_lo_freq = getattr(self._settings, lo_freq_setting_rf)

                if previous_lo_freq is not None and not np.isclose(
                    freqs.LO, previous_lo_freq
                ):
                    raise ValueError(
                        f"Attempting to set '{lo_freq_setting_rf}' to frequency "
                        f"'{freqs.LO:e}', while it has previously already been set to "
                        f"'{previous_lo_freq:e}'!"
                    )

                setattr(self._settings, lo_freq_setting_rf, freqs.LO)

        if freqs.IF is not None:
            sequencer.frequency = freqs.IF

    @abstractmethod
    def assign_attenuation(self):
        """
        An abstract method that should be overridden. Meant to assign
        attenuation settings from the hardware configuration if there is any.
        """

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
        self._configure_input_gains()
        self._configure_mixer_offsets()
        self._construct_sequencers()
        for seq in self.sequencers.values():
            self.assign_frequencies(seq)
        self.distribute_data()
        self._ensure_single_scope_mode_acquisition_sequencer()
        self.assign_attenuation()

    def _configure_input_gains(self):
        """
        Configures input gain of module settings.
        Loops through all valid ios and checks for gain values in hw config.
        Throws a ValueError if a gain value gets modified.
        """
        in0_gain, in1_gain = None, None
        for io_name in self.static_hw_properties.valid_ios:
            io_mapping = self.hw_mapping.get(io_name, None)
            if io_mapping is None:
                continue

            if io_name.startswith("complex"):
                in0_gain = io_mapping.get("input_gain_I", None)
                in1_gain = io_mapping.get("input_gain_Q", None)

            elif io_name.startswith("real"):
                # The next code block is for backwards compatibility.
                in_gain = io_mapping.get("input_gain", None)
                if in_gain is None:
                    in0_gain = io_mapping.get("input_gain_0", None)
                    in1_gain = io_mapping.get("input_gain_1", None)
                else:
                    in0_gain = in_gain
                    in1_gain = in_gain

            if in0_gain is not None:
                if (
                    self._settings.in0_gain is None
                    or in0_gain == self._settings.in0_gain
                ):
                    self._settings.in0_gain = in0_gain
                else:
                    raise ValueError(
                        f"Overwriting gain of {io_name} of module {self.name} "
                        f"to in0_gain: {in0_gain}.\nIt was previously set to "
                        f"in0_gain: {self._settings.in0_gain}."
                    )

            if in1_gain is not None:
                if (
                    self._settings.in1_gain is None
                    or in1_gain == self._settings.in1_gain
                ):
                    self._settings.in1_gain = in1_gain
                else:
                    raise ValueError(
                        f"Overwriting gain of {io_name} of module {self.name}"
                        f"to in1_gain: {in1_gain}.\nIt was previously set to "
                        f"in1_gain: {self._settings.in1_gain}."
                    )

    def _configure_mixer_offsets(self):
        """
        Configures offset of input, uses calc_from_units_volt found in helper file.
        Raises an exception if a value outside the accepted voltage range is given.
        """
        supported_outputs = ("complex_output_0", "complex_output_1")
        for output_idx, output_label in enumerate(supported_outputs):
            if output_label not in self.hw_mapping:
                continue

            output_cfg = self.hw_mapping[output_label]
            voltage_range = self.static_hw_properties.mixer_dc_offset_range
            if output_idx == 0:
                self._settings.offset_ch0_path0 = helpers.calc_from_units_volt(
                    voltage_range, self.name, "dc_mixer_offset_I", output_cfg
                )
                self._settings.offset_ch0_path1 = helpers.calc_from_units_volt(
                    voltage_range, self.name, "dc_mixer_offset_Q", output_cfg
                )
            else:
                self._settings.offset_ch1_path0 = helpers.calc_from_units_volt(
                    voltage_range, self.name, "dc_mixer_offset_I", output_cfg
                )
                self._settings.offset_ch1_path1 = helpers.calc_from_units_volt(
                    voltage_range, self.name, "dc_mixer_offset_Q", output_cfg
                )

    def _ensure_single_scope_mode_acquisition_sequencer(self) -> None:
        """
        Raises an error if multiple sequencers use scope mode acquisition,
        because that's not supported by the hardware.
        Also, see
        :func:`~quantify_scheduler.instrument_coordinator.components.qblox.QRMComponent._determine_scope_mode_acquisition_sequencer_and_channel`
        which also ensures the program that gets uploaded to the hardware satisfies this requirement.

        Raises
        ------
        ValueError
            Multiple sequencers have to perform trace acquisition. This is not
            supported by the hardware.
        """

        def is_scope_acquisition(acquisition: OpInfo) -> bool:
            return acquisition.data["protocol"] == "Trace"

        scope_acq_seq = None
        for seq in self.sequencers.values():
            op_infos = [acq.operation_info for acq in seq.acquisitions]

            has_scope = any(map(is_scope_acquisition, op_infos))
            if has_scope:
                if scope_acq_seq is not None:
                    helpers.single_scope_mode_acquisition_raise(
                        sequencer_0=scope_acq_seq,
                        sequencer_1=seq.index,
                        module_name=self.name,
                    )
                scope_acq_seq = seq.index

    def compile(
        self,
        repetitions: int = 1,
        sequence_to_file: Optional[bool] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Performs the actual compilation steps for this module, by calling the sequencer
        level compilation functions and combining them into a single dictionary.

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
            The compiled program corresponding to this module.
            It contains an entry for every sequencer under the key `"sequencers"`,
            and acquisition metadata under the key `"acq_metadata"`,
            and the `"repetitions"` is an integer with
            the number of times the defined schedule is repeated.
            All the other generic settings are under the key `"settings"`.
            If the device is not actually used,
            and an empty program is compiled, None is returned instead.
        """
        program = {}

        if sequence_to_file is None:
            sequence_to_file = self.hw_mapping.get("sequence_to_file", True)

        program["sequencers"] = {}
        for seq_name, seq in self.sequencers.items():
            seq_program = seq.compile(
                repetitions=repetitions, sequence_to_file=sequence_to_file
            )
            if seq_program is not None:
                program["sequencers"][seq_name] = seq_program

        if len(program) == 0:
            return None

        self._settings.hardware_averages = repetitions
        program["settings"] = self._settings.to_dict()
        if self.supports_acquisition:
            program["acq_metadata"] = {}

            for sequencer in self.sequencers.values():
                if not sequencer.acquisitions:
                    continue
                acq_metadata = helpers.extract_acquisition_metadata_from_acquisitions(
                    acquisitions=[acq.operation_info for acq in sequencer.acquisitions],
                    repetitions=repetitions,
                )
                program["acq_metadata"][sequencer.name] = acq_metadata
        program["repetitions"] = repetitions

        return program


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
        Determines LO/IF frequencies and assigns them, for baseband modules.

        In case of **no** external local oscillator, the NCO is given the same
        frequency as the clock -- unless NCO was permanently disabled via
        `"interm_freq": 0` in the hardware config.

        In case of **an** external local oscillator and `sequencer.mix_lo` is
        ``False``, the LO is given the same frequency as the clock
        (via :func:`.helpers.determine_clock_lo_interm_freqs`).
        """
        compiler_container = self.parent if self.is_pulsar else self.parent.parent
        if sequencer.clock not in compiler_container.resources:
            return

        clock_freq = compiler_container.resources[sequencer.clock]["freq"]
        if sequencer.associated_ext_lo is None:
            # Set NCO frequency to the clock frequency, unless NCO was permanently
            # disabled via `"interm_freq": 0` in the hardware config
            if sequencer.frequency != 0:
                sequencer.frequency = clock_freq
        else:
            # In using external local oscillator, determine clock and LO/IF freqs,
            # and then set LO/IF freqs, and enable NCO (via setter)
            if (
                compiler_lo := compiler_container.instrument_compilers.get(
                    sequencer.associated_ext_lo
                )
            ) is None:
                raise RuntimeError(
                    f"External local oscillator '{sequencer.associated_ext_lo}' set to "
                    f"be used by '{sequencer.name}' of '{self.name}' not found! Make "
                    f"sure it is present in the hardware configuration."
                )
            try:
                freqs = helpers.determine_clock_lo_interm_freqs(
                    clock_freq=clock_freq,
                    lo_freq=compiler_lo.frequency,
                    interm_freq=sequencer.frequency,
                    downconverter_freq=sequencer.downconverter_freq,
                    mix_lo=sequencer.mix_lo,
                )
            except Exception as error:  # Adding sequencer info to exception message
                raise error.__class__(
                    f"{error} (for '{sequencer.name}' of '{self.name}' "
                    f"with port '{sequencer.port}' and clock '{sequencer.clock}')."
                )
            self._set_lo_interm_freqs(
                freqs=freqs, sequencer=sequencer, compiler_lo_baseband=compiler_lo
            )

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
        """
        Determines LO/IF frequencies and assigns them for RF modules.
        """
        compiler_container = self.parent if self.is_pulsar else self.parent.parent
        if (
            sequencer.connected_outputs is None
            or sequencer.clock not in compiler_container.resources
        ):
            return

        for lo_idx in QbloxRFModule._get_connected_lo_indices(sequencer):
            lo_freq_setting_name = f"lo{lo_idx}_freq"
            try:
                freqs = helpers.determine_clock_lo_interm_freqs(
                    clock_freq=compiler_container.resources[sequencer.clock]["freq"],
                    lo_freq=getattr(self._settings, lo_freq_setting_name),
                    interm_freq=sequencer.frequency,
                    downconverter_freq=sequencer.downconverter_freq,
                    mix_lo=True,
                )
            except Exception as error:  # Adding sequencer info to exception message
                raise error.__class__(
                    f"{error} (for '{sequencer.name}' of '{self.name}' "
                    f"with port '{sequencer.port}' and clock '{sequencer.clock}')."
                )
            self._set_lo_interm_freqs(
                freqs=freqs,
                sequencer=sequencer,
                lo_freq_setting_rf=lo_freq_setting_name,
            )

    @staticmethod
    def _get_connected_lo_indices(sequencer: Sequencer) -> Generator[int]:
        """
        Identify the LO the sequencer is outputting.
        Use the sequencer output to module output correspondence, and then
        use the fact that LOX is connected to module output X.
        """
        for sequencer_output_index in sequencer.connected_outputs:
            if sequencer_output_index % 2 != 0:
                # We will only use real output 0 and 2, as they are part of the same
                # complex outputs as real output 1 and 3
                continue

            module_output_index = 0 if sequencer_output_index == 0 else 1
            yield module_output_index

    def assign_attenuation(self):
        """
        Assigns attenuation settings from the hardware configuration.

        Floats that are a multiple of 1 are converted to ints.
        This is needed because the :func:`quantify_core.measurement.control.grid_setpoints`
        converts setpoints to floats when using an attenuation as settable.
        """

        def _convert_to_int(value, label: str) -> Optional[int]:
            if value is not None:
                if not np.isclose(value % 1, 0):
                    raise ValueError(
                        f'Trying to set "{label}" to non-integer value {value}'
                    )
                return int(value)
            return None

        complex_input_0 = self.hw_mapping.get("complex_input_0", {})
        complex_output_0 = self.hw_mapping.get("complex_output_0", {})

        input_att = complex_input_0.get("input_att", None)
        if (input_att_output := complex_output_0.get("input_att", None)) is not None:
            if input_att is not None:
                raise ValueError(
                    f"'input_att' is defined for both 'complex_input_0' and "
                    f"'complex_output_0' on module '{self.name}', which is prohibited. "
                    f"Make sure you define it at a single place."
                )
            input_att = input_att_output
        self._settings.in0_att = _convert_to_int(input_att, label="in0_att")

        self._settings.out0_att = _convert_to_int(
            complex_output_0.get("output_att", None),
            label="out0_att",
        )
        complex_output_1 = self.hw_mapping.get("complex_output_1", {})
        self._settings.out1_att = _convert_to_int(
            complex_output_1.get("output_att", None),
            label="out1_att",
        )
