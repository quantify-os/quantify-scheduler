# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Compiler base and utility classes for Qblox backend."""

from __future__ import annotations
from enum import Enum, auto

import json
import logging
import math
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from os import makedirs, path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Hashable,
    Iterator,
)

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
from quantify_scheduler.backends.qblox.enums import ChannelMode
from quantify_scheduler.backends.qblox.operation_handling.acquisitions import (
    AcquisitionStrategyPartial,
    SquareAcquisitionStrategy,
)
from quantify_scheduler.backends.qblox.operation_handling.base import IOperationStrategy
from quantify_scheduler.backends.qblox.operation_handling.factory import (
    get_operation_strategy,
)
from quantify_scheduler.backends.qblox.operation_handling.pulses import (
    MarkerPulseStrategy,
)
from quantify_scheduler.backends.qblox.operation_handling.virtual import (
    ConditionalStrategy,
    ControlFlowReturnStrategy,
    LoopStrategy,
    UpdateParameterStrategy,
    NcoPhaseShiftStrategy,
    NcoResetClockPhaseStrategy,
    NcoSetClockFrequencyStrategy,
)
from quantify_scheduler.backends.qblox.qasm_program import QASMProgram
from quantify_scheduler.backends.types.qblox import (
    BasebandModuleSettings,
    BaseModuleSettings,
    OpInfo,
    RFModuleSettings,
    SequencerSettings,
    StaticHardwareProperties,
)
from quantify_scheduler.enums import BinMode
from quantify_scheduler.helpers.schedule import (
    extract_acquisition_metadata_from_acquisition_protocols,
)
from quantify_scheduler.operations.pulse_library import SetClockFrequency

if TYPE_CHECKING:
    from quantify_scheduler.backends.qblox.instrument_compilers import (
        LocalOscillatorCompiler,
    )
    from quantify_scheduler.schedules.schedule import AcquisitionMetadata

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class InstrumentCompiler(ABC):
    """
    Abstract base class that defines a generic instrument compiler.

    The subclasses that inherit from this are meant to implement the compilation
    steps needed to compile the lists of
    :class:`~quantify_scheduler.backends.types.qblox.OpInfo` representing the
    pulse and acquisition information to device-specific instructions.

    Each device that needs to be part of the compilation process requires an
    associated ``InstrumentCompiler``.

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
    instrument_cfg
        The part of the hardware configuration dictionary referring to this device. This is one
        of the inner dictionaries of the overall hardware config.
    latency_corrections
        Dict containing the delays for each port-clock combination. This is specified in
        the top layer of hardware config.

    """

    def __init__(
        self,
        parent,  # No type hint due to circular import, added to docstring
        name: str,
        total_play_time: float,
        instrument_cfg: dict[str, Any],
        latency_corrections: dict[str, float] | None = None,
    ) -> None:
        self.parent = parent
        self.name = name
        self.total_play_time = total_play_time
        self.instrument_cfg = instrument_cfg
        self.latency_corrections = latency_corrections or {}

    def prepare(self) -> None:
        """
        Method that can be overridden to implement logic before the main compilation
        starts. This step is to extract all settings for the devices that are dependent
        on settings of other devices. This step happens after instantiation of the
        compiler object but before the start of the main compilation.
        """

    @abstractmethod
    def compile(self, debug_mode: bool, repetitions: int) -> Any:
        """
        An abstract method that should be overridden in a subclass to implement the
        actual compilation. It should turn the pulses and acquisitions added to the
        device into device-specific instructions.

        Parameters
        ----------
        debug_mode
            Debug mode can modify the compilation process,
            so that debugging of the compilation process is easier.
        repetitions
            Number of times execution of the schedule is repeated.

        Returns
        -------
        :
            A data structure representing the compiled program. The type is
            dependent on implementation.
        """


class NcoOperationTimingError(ValueError):
    """Exception thrown if there are timing errors for NCO operations."""


class Sequencer:
    """
    Class that performs the compilation steps on the sequencer level.

    Parameters
    ----------
    parent
        A reference to the module compiler this sequencer belongs to.
    index
        Index of the sequencer.
    portclock
        Tuple that specifies the unique port and clock combination for this
        sequencer. The first value is the port, second is the clock.
    static_hw_properties
        The static properties of the hardware. This effectively gathers all the
        differences between the different modules.
    settings
        The settings set to this sequencer.
    latency_corrections
        Dict containing the delays for each port-clock combination.
    qasm_hook_func
        Allows the user to inject custom Q1ASM code into the compilation, just prior to
        returning the final string.
    lo_name
        The name of the local oscillator instrument connected to the same output via
        an IQ mixer. This is used for frequency calculations.
    downconverter_freq
        .. warning::
            Using ``downconverter_freq`` requires custom Qblox hardware, do not use otherwise.

        Frequency of the external downconverter if one is being used.
        Defaults to ``None``, in which case the downconverter is inactive.
    mix_lo
        Boolean flag for IQ mixing with LO.
        Defaults to ``True`` meaning IQ mixing is applied.
    marker_debug_mode_enable
        Boolean flag to indicate if markers should be pulled high at the start of operations.
        Defaults to False, which means the markers will not be used during the sequence.
    """

    def __init__(
        self,
        parent: ClusterModuleCompiler,
        index: int,
        portclock: tuple[str, str],
        static_hw_properties: StaticHardwareProperties,
        settings: SequencerSettings,
        latency_corrections: dict[str, float],
        qasm_hook_func: Callable | None = None,
        lo_name: str | None = None,
        downconverter_freq: float | None = None,
        mix_lo: bool = True,
        marker_debug_mode_enable: bool = False,
    ) -> None:
        self.parent = parent
        self.index = index
        self.port = portclock[0]
        self.clock = portclock[1]
        self.op_strategies: list[IOperationStrategy] = []
        self.associated_ext_lo = lo_name
        self.downconverter_freq = downconverter_freq
        self.mix_lo = mix_lo
        self._marker_debug_mode_enable = marker_debug_mode_enable
        self._num_acquisitions = 0

        self.static_hw_properties = static_hw_properties

        self.register_manager = register_manager.RegisterManager()

        self._settings = settings

        self._default_marker = (
            self.static_hw_properties.channel_name_to_digital_marker.get(
                self._settings.channel_name, self.static_hw_properties.default_marker
            )
        )

        self.qasm_hook_func = qasm_hook_func
        """Allows the user to inject custom Q1ASM code into the compilation, just prior
         to returning the final string."""

        portclock_key = f"{self.port}-{self.clock}"
        self.latency_correction: float = latency_corrections.get(portclock_key, 0)
        """Latency correction accounted for by delaying the start of the program."""

    @property
    def connected_output_indices(self) -> tuple[int, ...]:
        """
        Return the connected output indices associated with the output name
        specified in the hardware config.

        For the baseband modules, output index 'n' corresponds to physical module
        output 'n+1'.

        For RF modules, output indices '0' and '1' (or: '2' and '3') correspond to
        'path_I' and 'path_Q' of some sequencer, and both these paths are routed to the
        **same** physical module output '1' (or: '2').
        """
        return self._settings.connected_output_indices

    @property
    def connected_input_indices(self) -> tuple[int, ...]:
        """
        Return the connected input indices associated with the input name specified
        in the hardware config.

        For the baseband modules, input index 'n' corresponds to physical module input
        'n+1'.

        For RF modules, input indices '0' and '1' correspond to 'path_I' and 'path_Q' of
        some sequencer, and both paths are connected to physical module input '1'.
        """
        return self._settings.connected_input_indices

    @property
    def portclock(self) -> tuple[str, str]:
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
        return len(self.op_strategies) > 0

    @property
    def frequency(self) -> float | None:
        """
        The frequency used for modulation of the pulses.

        Returns
        -------
        :
            The frequency.
        """
        return self._settings.modulation_freq

    @frequency.setter
    def frequency(self, freq: float) -> None:
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
        if (
            self._settings.modulation_freq is not None
            and not math.isnan(self._settings.modulation_freq)
            and not math.isclose(self._settings.modulation_freq, freq)
        ):
            raise ValueError(
                f"Attempting to set the modulation frequency of '{self.name}' of "
                f"'{self.parent.name}' to {freq:e}, while it has previously been set "
                f"to {self._settings.modulation_freq:e}."
            )

        self._settings.modulation_freq = freq
        self._settings.nco_en = freq is not None

    def _generate_awg_dict(self) -> dict[str, Any]:
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

        RuntimeError
            When the total waveform size specified for a port-clock combination exceeds
            the waveform sample limit of the hardware.
        """
        wf_dict: dict[str, Any] = {}
        for op_strategy in self.op_strategies:
            if not op_strategy.operation_info.is_acquisition:
                op_strategy.generate_data(wf_dict=wf_dict)
        self._validate_awg_dict(wf_dict=wf_dict)
        return wf_dict

    def _generate_weights_dict(self) -> dict[str, Any]:
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
        wf_dict: dict[str, Any] = {}
        for op_strategy in self.op_strategies:
            if op_strategy.operation_info.is_acquisition:
                op_strategy.generate_data(wf_dict)
        return wf_dict

    def _validate_awg_dict(self, wf_dict: dict[str, Any]) -> None:
        total_size = 0
        for waveform in wf_dict.values():
            total_size += len(waveform["data"])
        if total_size > constants.MAX_SAMPLE_SIZE_WAVEFORMS:
            raise RuntimeError(
                f"Total waveform size specified for port-clock {self.port}-"
                f"{self.clock} is {total_size} samples, which exceeds the sample "
                f"limit of {constants.MAX_SAMPLE_SIZE_WAVEFORMS}. The compiled "
                f"schedule cannot be uploaded to the sequencer.",
            )

    def _prepare_acq_settings(
        self,
        acquisitions: list[IOperationStrategy],
        acq_metadata: AcquisitionMetadata,
    ) -> None:
        """
        Sets sequencer settings that are specific to certain acquisitions.
        For example for a TTL acquisition strategy.

        Parameters
        ----------
        acquisitions
            List of the acquisitions assigned to this sequencer.
        acq_metadata
            Acquisition metadata.
        """
        acquisition_infos: list[OpInfo] = list(
            map(lambda acq: acq.operation_info, acquisitions)
        )
        if acq_metadata.acq_protocol == "TriggerCount":
            self._settings.ttl_acq_auto_bin_incr_en = (
                acq_metadata.bin_mode == BinMode.AVERAGE
            )
            if len(self.connected_input_indices) == 1:
                self._settings.ttl_acq_input_select = self.connected_input_indices[0]
            elif len(self.connected_input_indices) > 1:
                raise ValueError(
                    f"Please make sure you use a single real input for this "
                    f"portclock combination. "
                    f"Found: {len(self.connected_input_indices)} connected. "
                    f"TTL acquisition does not support multiple inputs."
                    f"Problem occurred for port {self.port} with"
                    f"clock {self.clock}, which corresponds to {self.name} of "
                    f"{self.parent.name}."
                )

        elif acq_metadata.acq_protocol == "ThresholdedAcquisition":
            self._settings.thresholded_acq_rotation = acquisition_infos[0].data.get(
                "acq_rotation"
            )

            integration_length = acquisition_infos[0].data.get("duration") * 1e9
            self._settings.thresholded_acq_threshold = (
                acquisition_infos[0].data.get("acq_threshold") * integration_length
            )
            for info in acquisition_infos:
                if (address := info.data.get("feedback_trigger_address")) is not None:
                    self._settings.thresholded_acq_trigger_en = True
                    self._settings.thresholded_acq_trigger_address = address

        self._settings.integration_length_acq = (
            self._get_integration_length_from_acquisitions()
        )

    def _get_integration_length_from_acquisitions(self) -> int | None:
        """
        Get the (validated) integration_length sequencer setting.

        Get the duration of all SSB integration acquisitions assigned to this sequencer
        and validate that they are all the same.
        """
        integration_length = None
        for op_strat in self.op_strategies:
            if not isinstance(op_strat, SquareAcquisitionStrategy):
                continue

            acq_duration_ns = round(op_strat.operation_info.duration * 1e9)
            if acq_duration_ns % constants.GRID_TIME != 0:
                raise ValueError(
                    "Attempting to perform square acquisition with a duration of "
                    f"{acq_duration_ns} ns. Please ensure the duration is a multiple "
                    f"of {constants.GRID_TIME} ns.\n\nException caused by "
                    f"{repr(op_strat)}."
                )
            if integration_length is None:
                integration_length = acq_duration_ns
            elif integration_length != acq_duration_ns:
                raise ValueError(
                    f"Attempting to set an integration_length of {acq_duration_ns} "
                    f"ns, while this was previously determined to be "
                    f"{integration_length}. Please check whether all square "
                    "acquisitions in the schedule have the same duration."
                )

        return integration_length

    def _generate_acq_declaration_dict(
        self,
        repetitions: int,
        acq_metadata: AcquisitionMetadata,
    ) -> dict[str, Any]:
        """
        Generates the "acquisitions" entry of the program json. It contains declaration
        of the acquisitions along with the number of bins and the corresponding index.

        For the name of the acquisition (in the hardware), the acquisition channel
        (cast to str) is used, and is thus identical to the index. Number of bins is
        taken to be the highest acq_index specified for that channel.

        Parameters
        ----------
        repetitions
            The number of times to repeat execution of the schedule.
        acq_metadata
            Acquisition metadata.

        Returns
        -------
        :
            The "acquisitions" entry of the program json as a dict. The keys correspond
            to the names of the acquisitions (i.e. the acq_channel in the scheduler).
        """
        # initialize an empty dictionary for the format required by module
        acq_declaration_dict = {}
        for (
            qblox_acq_index,
            acq_channel_metadata,
        ) in acq_metadata.acq_channels_metadata.items():
            acq_indices: list[int] = acq_channel_metadata.acq_indices
            acq_channel: Hashable = acq_channel_metadata.acq_channel
            # Some sanity checks on the input for easier debugging.
            if min(acq_indices) != 0:
                raise ValueError(
                    f"Please make sure the lowest acquisition index used is 0. "
                    f"Found: {min(acq_indices)} as lowest index for channel "
                    f"{acq_channel}. Problem occurred for port {self.port} with"
                    f" clock {self.clock}, which corresponds to {self.name} of "
                    f"{self.parent.name}."
                )
            if len(acq_indices) != max(acq_indices) + 1:
                raise ValueError(
                    f"Found {max(acq_indices)} as the highest index out of "
                    f"{len(acq_indices)} for channel {acq_channel}, indicating "
                    f"an acquisition index was skipped or an acquisition index was repeated. "
                    f"Please make sure the used indices increment by 1 starting from 0. "
                    f"Problem occurred for port {self.port} with clock {self.clock}, "
                    f"which corresponds to {self.name} of {self.parent.name}."
                )
            unique_acq_indices = len(set(acq_indices))
            if len(acq_indices) != unique_acq_indices:
                raise ValueError(
                    f"Found {unique_acq_indices} unique indices out of "
                    f"{len(acq_indices)} for channel {acq_channel}, indicating "
                    f"an acquisition index was skipped or an acquisition index was repeated. "
                    f"Please make sure the used indices increment by 1 starting from 0. "
                    f"Problem occurred for port {self.port} with clock {self.clock}, "
                    f"which corresponds to {self.name} of {self.parent.name}."
                )

            # Add the acquisition metadata to the acquisition declaration dict
            if acq_metadata.bin_mode == BinMode.APPEND:
                num_bins = repetitions * self._num_acquisitions
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
            acq_declaration_dict[str(qblox_acq_index)] = {
                "num_bins": num_bins,
                "index": qblox_acq_index,
            }

        return acq_declaration_dict

    def generate_qasm_program(
        self,
        total_sequence_time: float,
        align_qasm_fields: bool,
        acq_metadata: AcquisitionMetadata | None,
        repetitions: int,
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
        align_qasm_fields
            If True, make QASM program more human-readable by aligning its fields.
        acq_metadata
            Acquisition metadata.
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
            Upon ``total_sequence_time`` exceeding :attr:`.QASMProgram.elapsed_time`.
        """
        loop_label = "start"

        qasm = QASMProgram(
            static_hw_properties=self.static_hw_properties,
            register_manager=self.register_manager,
            align_fields=align_qasm_fields,
            acq_metadata=acq_metadata,
        )
        qasm.set_marker(self._default_marker)

        # program header
        qasm.set_latch(self.op_strategies)
        qasm.emit(q1asm_instructions.WAIT_SYNC, constants.MIN_TIME_BETWEEN_OPERATIONS)
        qasm.emit(
            q1asm_instructions.UPDATE_PARAMETERS, constants.MIN_TIME_BETWEEN_OPERATIONS
        )

        self._initialize_append_mode_registers(qasm, self.op_strategies)

        # Program body. The operations are sorted such that real-time IO operations
        # always come after any other operations. E.g., an offset instruction should
        # always come before the parameter update, play, or acquisition instruction.
        op_list = sorted(
            self.op_strategies,
            # Note: round to 12 digits below such that
            # the control flow begin operation is always
            # going to go first, but floating point precision
            # issues do not cause any problems.
            # See compilation.resolve_control_flow,
            # which adds a small negative relative
            # timing for control flow start.
            key=lambda op: (
                round(op.operation_info.timing, ndigits=12),
                op.operation_info.is_real_time_io_operation,
            ),
        )

        self._check_nco_operation_timing(op_list)

        # Adds the latency correction, this needs to be a minimum of 4 ns,
        # so all sequencers get delayed by at least that.
        latency_correction_ns: int = self._get_latency_correction_ns(
            self.latency_correction
        )
        qasm.auto_wait(
            wait_time=constants.MIN_TIME_BETWEEN_OPERATIONS + latency_correction_ns,
            count_as_elapsed_time=False,
            comment=f"latency correction of {constants.MIN_TIME_BETWEEN_OPERATIONS} + "
            f"{latency_correction_ns} ns",
        )

        with qasm.loop(label=loop_label, repetitions=repetitions):
            qasm.emit(q1asm_instructions.RESET_PHASE)
            qasm.emit(
                q1asm_instructions.UPDATE_PARAMETERS,
                constants.MIN_TIME_BETWEEN_OPERATIONS,
            )

            last_operation_end = {True: 0, False: 0}
            for operation in op_list:
                # Check if there is an overlapping pulse or overlapping acquisition
                if operation.operation_info.is_real_time_io_operation:
                    start_time = operation.operation_info.timing
                    is_acquisition = operation.operation_info.is_acquisition
                    if start_time < last_operation_end[is_acquisition]:
                        warnings.warn(
                            f"Operation is interrupting previous"
                            f" {'Acquisition' if is_acquisition else 'Pulse'}"
                            f" because it starts before the previous ends,"
                            f" offending operation:"
                            f" {str(operation.operation_info)}",
                            RuntimeWarning,
                        )
                    last_operation_end[is_acquisition] = (
                        start_time + operation.operation_info.duration
                    )

            self._parse_operations(iter(op_list), qasm, 1)

            end_time = helpers.to_grid_time(total_sequence_time)
            wait_time = end_time - qasm.elapsed_time
            if wait_time < 0:
                raise RuntimeError(
                    f"Invalid timing detected, attempting to insert wait "
                    f"of {wait_time} ns. The total duration of the "
                    f"schedule is {end_time} but {qasm.elapsed_time} ns "
                    f"already processed."
                )
            qasm.auto_wait(wait_time=wait_time)

        # program footer
        qasm.emit(q1asm_instructions.STOP)

        if self.qasm_hook_func:
            self.qasm_hook_func(qasm)

        if (
            num_instructions := len(qasm.instructions)
        ) > self.parent.max_number_of_instructions:
            warnings.warn(
                f"Number of instructions ({num_instructions}) compiled for "
                f"'{self.name}' of {self.parent.__class__.__name__} "
                f"'{self.parent.name}' exceeds the maximum supported number of "
                f"instructions in Q1ASM programs for {self.parent.__class__.__name__} "
                f"({self.parent.max_number_of_instructions}).",
                RuntimeWarning,
            )

        return str(qasm)

    class ParseOperationStatus(Enum):
        """Return status of the stack."""

        COMPLETED_ITERATION = auto()
        """The iterator containing operations is exhausted."""
        EXITED_CONTROL_FLOW = auto()
        """The end of a control flow scope is reached."""

    def _parse_operations(
        self,
        operations_iter: Iterator[IOperationStrategy],
        qasm: QASMProgram,
        acquisition_multiplier: int,
    ) -> ParseOperationStatus:
        """Handle control flow and insert Q1ASM."""
        while (operation := next(operations_iter, None)) is not None:
            qasm.wait_till_start_operation(operation.operation_info)
            if isinstance(operation, LoopStrategy):
                loop_label = f"loop{len(qasm.instructions)}"
                repetitions = operation.operation_info.data["repetitions"]
                with qasm.loop(label=loop_label, repetitions=repetitions):
                    returned_from_return_stack = self._parse_operations(
                        operations_iter=operations_iter,
                        qasm=qasm,
                        acquisition_multiplier=acquisition_multiplier * repetitions,
                    )
                    assert returned_from_return_stack in self.ParseOperationStatus

            elif isinstance(operation, ConditionalStrategy):
                with qasm.conditional(operation):
                    returned_from_return_stack = self._parse_operations(
                        operations_iter=operations_iter,
                        qasm=qasm,
                        acquisition_multiplier=acquisition_multiplier,
                    )
                    assert returned_from_return_stack in self.ParseOperationStatus

            elif isinstance(operation, ControlFlowReturnStrategy):
                return self.ParseOperationStatus.EXITED_CONTROL_FLOW
            else:
                if operation.operation_info.is_acquisition:
                    self._num_acquisitions += acquisition_multiplier
                qasm.conditional_manager.update(operation)
                self._insert_qasm_marker_debug_wrapped(operation, qasm)

        return self.ParseOperationStatus.EXITED_CONTROL_FLOW

    def _insert_qasm_marker_debug_wrapped(
        self, operation: IOperationStrategy, qasm: QASMProgram
    ) -> None:
        if self._marker_debug_mode_enable:
            valid_operation = (
                operation.operation_info.is_acquisition
                or operation.operation_info.data.get("wf_func") is not None
            )
            if valid_operation:
                qasm.set_marker(self._decide_markers(operation))
                operation.insert_qasm(qasm)
                qasm.set_marker(self._default_marker)
                qasm.emit(
                    q1asm_instructions.UPDATE_PARAMETERS,
                    constants.MIN_TIME_BETWEEN_OPERATIONS,
                )
                qasm.elapsed_time += constants.MIN_TIME_BETWEEN_OPERATIONS
        else:
            operation.insert_qasm(qasm)

    def _initialize_append_mode_registers(
        self, qasm: QASMProgram, op_strategies: list[AcquisitionStrategyPartial]
    ) -> None:
        """
        Adds the instructions to initialize the registers needed to use the append
        bin mode to the program. This should be added in the header.

        Parameters
        ----------
        qasm:
            The program to add the instructions to.
        op_strategies:
            An operations list including all the acquisitions to consider.
        """
        channel_to_reg: dict[str, str] = {}
        for op_strategy in op_strategies:
            if not op_strategy.operation_info.is_acquisition:
                continue

            if op_strategy.operation_info.data["bin_mode"] != BinMode.APPEND:
                continue

            channel = op_strategy.operation_info.data["acq_channel"]
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
                    f"ch{op_strategy.operation_info.data['acq_channel']}",
                )
            op_strategy.bin_idx_register = acq_bin_idx_reg

    def _get_latency_correction_ns(self, latency_correction: float) -> int:
        if latency_correction == 0:
            return 0

        latency_correction_ns = int(round(latency_correction * 1e9))
        if latency_correction_ns % 4 != 0:
            logger.warning(
                f"Latency correction of {latency_correction_ns} ns specified"
                f" for {self.name} of {self.parent.name}, which is not a"
                f" multiple of {constants.MIN_TIME_BETWEEN_OPERATIONS} ns. This feature should"
                f" be considered experimental and stable results are not guaranteed at "
                f"this stage."
            )

        return latency_correction_ns

    def _insert_update_parameters(self) -> None:
        """
        Insert update parameter instructions to activate offsets, if they are not
        already activated by a play, acquire or acquire_weighed instruction (see also
        `the Q1ASM reference
        <https://qblox-qblox-instruments.readthedocs-hosted.com/en/main/cluster/q1_sequence_processor.html#instructions>`_).
        """
        upd_params = self._get_new_update_parameters(self.op_strategies)
        # This can be unsorted. The sorting step is in Sequencer.generate_qasm_program()
        self.op_strategies += upd_params

    @staticmethod
    def _any_other_updating_instruction_at_timing_for_parameter_instruction(
        op_index: int, sorted_pulses_and_acqs: list[IOperationStrategy]
    ) -> bool:
        op = sorted_pulses_and_acqs[op_index]
        if not op.operation_info.is_parameter_instruction:
            return False

        def iterate_other_ops(iterate_range, allow_return_stack: bool) -> bool:
            for other_op_index in iterate_range:
                other_op = sorted_pulses_and_acqs[other_op_index]
                if helpers.to_grid_time(
                    other_op.operation_info.timing
                ) != helpers.to_grid_time(op.operation_info.timing):
                    break
                if other_op.operation_info.is_real_time_io_operation:
                    return True
                if not allow_return_stack and other_op.operation_info.is_return_stack:
                    raise RuntimeError(
                        f"Parameter operation {op.operation_info} with start time "
                        f"{op.operation_info.timing} cannot be scheduled at the same "
                        "time as the end of a control-flow block "
                        f"{other_op.operation_info}, which ends at "
                        f"{other_op.operation_info.timing}. The control-flow block can "
                        "be extended by adding an IdlePulse operation with a duration "
                        f"of at least {constants.MIN_TIME_BETWEEN_OPERATIONS} ns, or the Parameter "
                        "operation can be replaced by another operation."
                    )
            return False

        # Check all other operations behind the operation with op_index
        # whether they're within half grid time
        #
        # We specifically allow an offset instruction to be after a return stack:
        # the caller is free to start an offset instruction and a play signal after
        # we return from a loop
        if iterate_other_ops(
            iterate_range=range(op_index - 1, -1, -1), allow_return_stack=True
        ):
            return True
        # Check all other operations in front of the operation with op_index
        # whether they're within half grid time
        #
        # We specifically disallow an offset instruction to be after a return stack:
        # when the caller sets an offset, it might have an unknown effect depending
        # on whether we actually exist the loop, or go the next cycle in the loop
        if iterate_other_ops(
            iterate_range=range(op_index + 1, len(sorted_pulses_and_acqs)),
            allow_return_stack=False,
        ):
            return True

        return False

    def _get_new_update_parameters(
        self,
        pulses_and_acqs: list[IOperationStrategy],
    ) -> list[IOperationStrategy]:
        pulses_and_acqs.sort(key=lambda op: op.operation_info.timing)

        # Collect all times (in ns, so that it's an integer) where an upd_param needs to
        # be inserted.
        upd_param_times_ns: set[int] = set()
        for op_index, op in enumerate(pulses_and_acqs):
            if not op.operation_info.is_parameter_instruction:
                continue
            if helpers.to_grid_time(
                self.parent.total_play_time
            ) == helpers.to_grid_time(op.operation_info.timing):
                raise RuntimeError(
                    f"Parameter operation {op.operation_info} with start time "
                    f"{op.operation_info.timing} cannot be scheduled at the very end "
                    "of a Schedule. The Schedule can be extended by adding an "
                    "IdlePulse operation with a duration of at least "
                    f"{constants.MIN_TIME_BETWEEN_OPERATIONS} ns, or the Parameter operation can be "
                    "replaced by another operation."
                )
            if not self._any_other_updating_instruction_at_timing_for_parameter_instruction(
                op_index=op_index, sorted_pulses_and_acqs=pulses_and_acqs
            ):
                upd_param_times_ns.add(round(op.operation_info.timing * 1e9))

        return [
            UpdateParameterStrategy(
                OpInfo(
                    name="UpdateParameters",
                    data={
                        "t0": 0,
                        "port": self.port,
                        "clock": self.clock,
                        "duration": 0,
                        "instruction": q1asm_instructions.UPDATE_PARAMETERS,
                    },
                    timing=time_ns * 1e-9,
                )
            )
            for time_ns in upd_param_times_ns
        ]

    @staticmethod
    def _generate_waveforms_and_program_dict(
        program: str,
        waveforms_dict: dict[str, Any],
        weights_dict: dict[str, Any] | None = None,
        acq_decl_dict: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
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
            be of the form generated by the ``generate_awg_dict`` method.
        weights_dict
            The dictionary containing all the acq data and indices. This is expected to
            be of the form generated by the ``generate_acq_dict`` method.
        acq_decl_dict
            The dictionary containing all the acq declarations. This is expected to be
            of the form generated by the ``generate_acq_decl_dict`` method.

        Returns
        -------
        :
            The combined program.
        """
        compiled_dict: dict[str, Any] = {}
        compiled_dict["program"] = program
        compiled_dict["waveforms"] = waveforms_dict
        if weights_dict is not None:
            compiled_dict["weights"] = weights_dict
        if acq_decl_dict is not None:
            compiled_dict["acquisitions"] = acq_decl_dict
        return compiled_dict

    @staticmethod
    def _dump_waveforms_and_program_json(
        wf_and_pr_dict: dict[str, Any], label: str | None = None
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

    def prepare(self) -> None:
        """
        Perform necessary operations on this sequencer's data before
        :meth:`~Sequencer.compile` is called.
        """
        self._update_set_clock_frequency_operations()
        self.op_strategies = self._replace_marker_pulses(self.op_strategies)
        self._insert_update_parameters()

    def _update_set_clock_frequency_operations(self) -> None:
        for op_strat in self.op_strategies:
            if op_strat.operation_info.name == SetClockFrequency.__name__:
                op_strat.operation_info.data.update(
                    {
                        "interm_freq_old": self.frequency,
                    }
                )

    def compile(
        self,
        sequence_to_file: bool,
        align_qasm_fields: bool,
        repetitions: int = 1,
    ) -> tuple[dict[str, Any] | None, AcquisitionMetadata | None]:
        """
        Performs the full sequencer level compilation based on the assigned data and
        settings. If no data is assigned to this sequencer, the compilation is skipped
        and None is returned instead.

        Parameters
        ----------
        sequence_to_file
            Dump waveforms and program dict to JSON file, filename stored in
            `Sequencer.settings.seq_fn`.
        align_qasm_fields
            If True, make QASM program more human-readable by aligning its fields.
        repetitions
            Number of times execution the schedule is repeated.

        Returns
        -------
        :
            The compiled program and the acquisition metadata.
            If no data is assigned to this sequencer, the
            compilation is skipped and None is returned instead.
        """
        if not self.has_data:
            return None, None

        awg_dict = self._generate_awg_dict()
        weights_dict = None
        acq_declaration_dict = None
        acq_metadata: AcquisitionMetadata | None = None

        # the program needs _generate_weights_dict for the waveform indices
        if self.parent.supports_acquisition:
            weights_dict = {}
            acquisitions = [
                op_strategy
                for op_strategy in self.op_strategies
                if op_strategy.operation_info.is_acquisition
            ]
            if len(acquisitions) > 0:
                acq_metadata = extract_acquisition_metadata_from_acquisition_protocols(
                    acquisition_protocols=[
                        acq.operation_info.data for acq in acquisitions
                    ],
                    repetitions=repetitions,
                )
                self._prepare_acq_settings(
                    acquisitions=acquisitions,
                    acq_metadata=acq_metadata,
                )
                weights_dict = self._generate_weights_dict()

        # acq_declaration_dict needs to count number of acquires in the program
        qasm_program = self.generate_qasm_program(
            total_sequence_time=self.parent.total_play_time,
            align_qasm_fields=align_qasm_fields,
            acq_metadata=acq_metadata,
            repetitions=repetitions,
        )

        if self.parent.supports_acquisition:
            acq_declaration_dict = {}
            if acq_metadata is not None:
                acq_declaration_dict = self._generate_acq_declaration_dict(
                    repetitions=repetitions,
                    acq_metadata=acq_metadata,
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

        sequencer_cfg = self._settings.to_dict()
        return sequencer_cfg, acq_metadata

    @staticmethod
    def _replace_marker_pulses(
        op_strategies: list[IOperationStrategy],
    ) -> list[IOperationStrategy]:
        """Replaces MarkerPulse operations by explicit high and low operations."""
        new_op_strategies: list[IOperationStrategy] = []
        for op_strategy in op_strategies:
            if isinstance(op_strategy, MarkerPulseStrategy):
                high_op_info = OpInfo(
                    name=op_strategy.operation_info.name,
                    data=op_strategy.operation_info.data.copy(),
                    timing=op_strategy.operation_info.timing,
                )
                duration = op_strategy.operation_info.data["duration"]
                high_op_info.data["enable"] = True
                high_op_info.data["duration"] = 0
                new_op_strategies.append(
                    MarkerPulseStrategy(
                        operation_info=high_op_info,
                        channel_name=op_strategy.channel_name,
                    )
                )

                low_op_info = OpInfo(
                    name=op_strategy.operation_info.name,
                    data=op_strategy.operation_info.data.copy(),
                    timing=op_strategy.operation_info.timing + duration,
                )
                low_op_info.data["enable"] = False
                low_op_info.data["duration"] = 0
                new_op_strategies.append(
                    MarkerPulseStrategy(
                        operation_info=low_op_info,
                        channel_name=op_strategy.channel_name,
                    )
                )
            else:
                new_op_strategies.append(op_strategy)

        return new_op_strategies

    def _decide_markers(self, operation) -> int:
        """
        Helper method to decide what markers should be pulled high when enable_marker is set to True.
        Checks what module and operation are being processed, then builds a bit string accordingly.

        Note that with the current quantify structure a sequencer cannot have connected inputs and outputs simultaneously.
        Therefore, the QRM baseband module pulls both input or output markers high when doing an operation,
        as it is impossible during compile time to find out what physical port is being used.

        Parameters
        ----------
        sequencer
            The sequencer currently in the process of constructing a Q1ASM program.
        operation
            The operation currently being processed by the sequence.

        Returns
        -------
            A bit string passed on to the set_mrk function of the Q1ASM object.
        """
        marker_bit_string = 0
        instrument_type = self.static_hw_properties.instrument_type
        if instrument_type == "QCM":
            for output in self.connected_output_indices:
                marker_bit_string |= 1 << output
        elif instrument_type == "QRM":
            if operation.operation_info.is_acquisition:
                marker_bit_string = 0b1100
            else:
                marker_bit_string = 0b0011

        # For RF modules, the first two indices correspond to path enable/disable.
        # Therefore, the index of the output is shifted by 2.
        elif instrument_type == "QCM_RF":
            for output in self.connected_output_indices:
                marker_bit_string |= 1 << (output + 2)
                marker_bit_string |= self._default_marker
        elif instrument_type == "QRM_RF":
            if operation.operation_info.is_acquisition:
                marker_bit_string = 0b1011
            else:
                marker_bit_string = 0b0111
        return marker_bit_string

    @staticmethod
    def _check_nco_operation_timing(
        sorted_pulses_and_acqs: list[IOperationStrategy],
    ) -> None:
        """Check whether this sequencer's operation adhere to NCO timing restrictions."""
        last_freq_upd_time = -constants.NCO_SET_FREQ_WAIT
        last_phase_upd_time = -constants.NCO_SET_PH_DELTA_WAIT
        for op in sorted_pulses_and_acqs:
            timing = round(op.operation_info.timing * 1e9)
            if isinstance(op, NcoSetClockFrequencyStrategy):
                if (diff := timing - last_freq_upd_time) < constants.NCO_SET_FREQ_WAIT:
                    raise NcoOperationTimingError(
                        f"Operation {op.operation_info} occurred {diff} ns after the "
                        "previous frequency update. The minimum time between frequency "
                        f"updates must be {constants.NCO_SET_FREQ_WAIT} ns."
                    )
                else:
                    last_freq_upd_time = timing

            if isinstance(op, (NcoPhaseShiftStrategy, NcoResetClockPhaseStrategy)):
                timing = round(op.operation_info.timing * 1e9)
                if (
                    diff := timing - last_phase_upd_time
                ) < constants.NCO_SET_PH_DELTA_WAIT:
                    raise NcoOperationTimingError(
                        f"Operation {op.operation_info} occurred {diff} ns after the "
                        "previous phase update. The minimum time between phase "
                        f"updates must be {constants.NCO_SET_PH_DELTA_WAIT} ns."
                    )
                else:
                    last_phase_upd_time = timing

            if isinstance(
                op,
                (
                    NcoSetClockFrequencyStrategy,
                    NcoPhaseShiftStrategy,
                    NcoResetClockPhaseStrategy,
                ),
            ):
                try:
                    helpers.to_grid_time(timing * 1e-9, constants.NCO_TIME_GRID)
                except ValueError as e:
                    raise NcoOperationTimingError(
                        f"NCO related operation {op.operation_info} must be on "
                        f"{constants.NCO_TIME_GRID} ns time grid"
                    ) from e


class ClusterModuleCompiler(InstrumentCompiler, ABC):
    """
    Base class for all cluster modules, and an interface for those modules to the
    :class:`~quantify_scheduler.backends.qblox.instrument_compilers.ClusterCompiler`.

    This class is defined as an abstract base class since the distinctions between the
    different devices are defined in subclasses.
    Effectively, this base class contains the functionality shared by all Qblox
    devices and serves to avoid repeated code between them.

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
    instrument_cfg
        The part of the hardware configuration dictionary referring to this device. This is one
        of the inner dictionaries of the overall hardware config.
    latency_corrections
        Dict containing the delays for each port-clock combination. This is specified in
        the top layer of hardware config.
    """

    def __init__(
        self,
        parent,  # No type hint due to circular import, added to docstring
        name: str,
        total_play_time: float,
        instrument_cfg: dict[str, Any],
        latency_corrections: dict[str, float] | None = None,
    ) -> None:
        driver_version_check.verify_qblox_instruments_version()
        super().__init__(
            parent=parent,
            name=name,
            total_play_time=total_play_time,
            instrument_cfg=instrument_cfg,
            latency_corrections=latency_corrections,
        )
        self._op_infos: dict[tuple[str, str], list[OpInfo]] = defaultdict(list)

        self._settings: BaseModuleSettings | None = None  # set in the prepare method.
        self.sequencers: dict[str, Sequencer] = {}

    @property
    def portclocks(self) -> list[tuple[str, str]]:
        """Returns all the port-clock combinations that this device can target."""
        portclocks = []

        for channel_name in helpers.find_channel_names(self.instrument_cfg):
            portclock_configs = self.instrument_cfg[channel_name].get(
                "portclock_configs", []
            )
            if not portclock_configs:
                raise KeyError(
                    f"No 'portclock_configs' entry found in '{channel_name}' of {self.name}."
                )

            portclocks += [
                (target["port"], target["clock"]) for target in portclock_configs
            ]

        return portclocks

    @property
    @abstractmethod
    def supports_acquisition(self) -> bool:
        """Specifies whether the device can perform acquisitions."""

    @property
    @abstractmethod
    def max_number_of_instructions(self) -> int:
        """The maximum number of Q1ASM instructions supported by this module type."""

    def add_op_info(self, port: str, clock: str, op_info: OpInfo) -> None:
        """
        Assigns a certain pulse or acquisition to this device.

        Parameters
        ----------
        port
            The port this waveform is sent to (or acquired from).
        clock
            The clock for modulation of the pulse or acquisition. Can be a BasebandClock.
        op_info
            Data structure containing all the information regarding this specific
            pulse or acquisition operation.
        """
        if op_info.is_acquisition and not self.supports_acquisition:
            raise RuntimeError(
                f"{self.__class__.__name__} {self.name} does not support acquisitions. "
                f"Attempting to add acquisition {repr(op_info)} "
                f"on port {port} with clock {clock}."
            )

        self._op_infos[(port, clock)].append(op_info)

    @property
    def _portclocks_with_data(self) -> set[tuple[str, str]]:
        """
        All the port-clock combinations associated with at least one pulse and/or
        acquisition.

        Returns
        -------
        :
            A set containing all the port-clock combinations that are used by this
            InstrumentCompiler.
        """
        portclocks_used: set[tuple[str, str]] = {
            portclock
            for portclock, op_infos in self._op_infos.items()
            if not all(op_info.data.get("name") == "LatchReset" for op_info in op_infos)
        }
        return portclocks_used

    @property
    @abstractmethod
    def settings_type(self) -> type[BaseModuleSettings]:
        """Specifies the module settings class used by the instrument."""

    @property
    @abstractmethod
    def static_hw_properties(self) -> StaticHardwareProperties:
        """
        The static properties of the hardware. This effectively gathers all the
        differences between the different modules.
        """

    def _construct_sequencers(self) -> None:
        """
        Constructs :class:`~Sequencer` objects for each port and clock combination
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
        # Setup each sequencer.
        sequencers: dict[str, Sequencer] = {}
        portclock_to_channel: dict[tuple, str] = {}

        for channel_name, channel_cfg in sorted(
            self.instrument_cfg.items()
        ):  # Sort to ensure deterministic sequencer order
            if not isinstance(channel_cfg, dict):
                continue

            portclock_configs: list[dict[str, Any]] = channel_cfg.get(
                "portclock_configs", []
            )

            if not portclock_configs:
                raise KeyError(
                    f"No 'portclock_configs' entry found in '{channel_name}' of {self.name}."
                )

            for sequencer_cfg in portclock_configs:
                portclock = (sequencer_cfg["port"], sequencer_cfg["clock"])

                if portclock in self._portclocks_with_data:
                    settings = SequencerSettings.initialize_from_config_dict(
                        sequencer_cfg=sequencer_cfg,
                        channel_name=channel_name,
                        connected_output_indices=self.static_hw_properties._get_connected_output_indices(
                            channel_name
                        ),
                        connected_input_indices=self.static_hw_properties._get_connected_input_indices(
                            channel_name
                        ),
                    )
                    new_seq = Sequencer(
                        parent=self,
                        index=len(sequencers),
                        portclock=portclock,
                        static_hw_properties=self.static_hw_properties,
                        settings=settings,
                        latency_corrections=self.latency_corrections,
                        qasm_hook_func=sequencer_cfg.get("qasm_hook_func"),
                        lo_name=channel_cfg.get("lo_name"),
                        mix_lo=channel_cfg.get("mix_lo", True),
                        marker_debug_mode_enable=channel_cfg.get(
                            "marker_debug_mode_enable", False
                        ),
                        downconverter_freq=channel_cfg.get("downconverter_freq"),
                    )
                    sequencers[new_seq.name] = new_seq

                    # Check if the portclock was not multiply specified, which is not allowed
                    if portclock in portclock_to_channel:
                        raise ValueError(
                            f"Portclock {portclock} was assigned to multiple "
                            f"portclock_configs of {self.name}. This portclock was "
                            f"used in channel '{channel_name}' despite being already previously "
                            f"used in channel '{portclock_to_channel[portclock]}'. When using "
                            f"the same portclock for output and input, assigning only "
                            f"the output suffices."
                        )

                    portclock_to_channel[portclock] = channel_name

        # Check if more portclock_configs than sequencers are active
        if len(sequencers) > self.static_hw_properties.max_sequencers:
            raise ValueError(
                "Number of simultaneously active port-clock combinations exceeds "
                "number of sequencers. "
                f"Maximum allowed for {self.name} ({self.__class__.__name__}) is "
                f"{self.static_hw_properties.max_sequencers}!"
            )

        self.sequencers = sequencers

    def distribute_data(self) -> None:
        """
        Distributes the pulses and acquisitions assigned to this module over the
        different sequencers based on their portclocks. Raises an exception in case
        the device does not support acquisitions.
        """
        for seq in self.sequencers.values():
            if seq.op_strategies is None:
                seq.op_strategies = []

            for portclock, op_info_list in self._op_infos.items():
                if seq.portclock == portclock or (
                    portclock[0] is None and portclock[1] == seq.clock
                ):
                    for op_info in op_info_list:
                        if not op_info.is_acquisition or not (
                            portclock[0] is None and portclock[1] == seq.clock
                        ):
                            op_strategy = get_operation_strategy(
                                operation_info=op_info,
                                channel_name=seq._settings.channel_name,
                            )

                            if ChannelMode.DIGITAL in seq._settings.channel_name:
                                # A digital pulse always uses one output.
                                op_strategy.operation_info.data["output"] = (
                                    seq.connected_output_indices[0]
                                )
                            seq.op_strategies.append(op_strategy)

    @abstractmethod
    def assign_frequencies(self, sequencer: Sequencer) -> None:
        r"""
        An abstract method that should be overridden. Meant to assign an IF frequency
        to each sequencer, and an LO frequency to each output (if applicable).
        """

    def _set_lo_interm_freqs(
        self,
        freqs: helpers.Frequencies,
        sequencer: Sequencer,
        compiler_lo_baseband: LocalOscillatorCompiler | None = None,
        lo_freq_setting_rf: str | None = None,
    ) -> None:
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
            For baseband modules, supply the
            :class:`.instrument_compilers.LocalOscillatorCompiler` instrument compiler
            of which the frequency is to be set.
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
        if freqs.LO is not None:
            if compiler_lo_baseband is not None:
                compiler_lo_baseband.frequency = freqs.LO

            elif lo_freq_setting_rf is not None:
                previous_lo_freq = getattr(self._settings, lo_freq_setting_rf)

                if (
                    previous_lo_freq is not None
                    and not math.isnan(previous_lo_freq)
                    and not math.isclose(freqs.LO, previous_lo_freq)
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
    def assign_attenuation(self) -> None:
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
            self.instrument_cfg
        )
        self._configure_input_gains()
        self._configure_mixer_offsets()
        self._construct_sequencers()
        for seq in self.sequencers.values():
            self.assign_frequencies(seq)
        self.distribute_data()
        for seq in self.sequencers.values():
            seq.prepare()
        self._ensure_single_scope_mode_acquisition_sequencer()
        self.assign_attenuation()

    def _configure_input_gains(self) -> None:
        """
        Configures input gain of module settings.
        Loops through all valid channel names and checks for gain values in hw config.
        Throws a ValueError if a gain value gets modified.
        """
        in0_gain, in1_gain = None, None

        for channel_name in helpers.find_channel_names(self.instrument_cfg):
            channel_mapping = self.instrument_cfg.get(channel_name, None)

            if channel_name.startswith(ChannelMode.COMPLEX):
                in0_gain = channel_mapping.get("input_gain_I", None)
                in1_gain = channel_mapping.get("input_gain_Q", None)

            elif channel_name.startswith(ChannelMode.REAL):
                # The next code block is for backwards compatibility.
                in_gain = channel_mapping.get("input_gain", None)
                if in_gain is None:
                    in0_gain = channel_mapping.get("input_gain_0", None)
                    in1_gain = channel_mapping.get("input_gain_1", None)
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
                        f"Overwriting gain of {channel_name} of module {self.name} "
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
                        f"Overwriting gain of {channel_name} of module {self.name} "
                        f"to in1_gain: {in1_gain}.\nIt was previously set to "
                        f"in1_gain: {self._settings.in1_gain}."
                    )

    def _configure_mixer_offsets(self) -> None:
        """
        Configures offset of input, uses calc_from_units_volt found in helper file.
        Raises an exception if a value outside the accepted voltage range is given.
        """
        supported_channels = ("complex_output_0", "complex_output_1")
        for output_idx, channel_name in enumerate(supported_channels):
            if channel_name not in self.instrument_cfg:
                continue

            output_cfg = self.instrument_cfg[channel_name]
            voltage_range = self.static_hw_properties.mixer_dc_offset_range
            if output_idx == 0:
                self._settings.offset_ch0_path_I = helpers.calc_from_units_volt(
                    voltage_range, self.name, "dc_mixer_offset_I", output_cfg
                )
                self._settings.offset_ch0_path_Q = helpers.calc_from_units_volt(
                    voltage_range, self.name, "dc_mixer_offset_Q", output_cfg
                )
            else:
                self._settings.offset_ch1_path_I = helpers.calc_from_units_volt(
                    voltage_range, self.name, "dc_mixer_offset_I", output_cfg
                )
                self._settings.offset_ch1_path_Q = helpers.calc_from_units_volt(
                    voltage_range, self.name, "dc_mixer_offset_Q", output_cfg
                )

    def _ensure_single_scope_mode_acquisition_sequencer(self) -> None:
        """
        Raises an error if multiple sequencers use scope mode acquisition,
        because that's not supported by the hardware.
        Also, see
        :func:`~quantify_scheduler.instrument_coordinator.components.qblox._QRMComponent._determine_scope_mode_acquisition_sequencer_and_qblox_acq_index`
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
            op_infos = [
                op.operation_info
                for op in seq.op_strategies
                if op.operation_info.is_acquisition
            ]

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
        debug_mode: bool,
        repetitions: int = 1,
        sequence_to_file: bool | None = None,
    ) -> dict[str, Any]:
        """
        Performs the actual compilation steps for this module, by calling the sequencer
        level compilation functions and combining them into a single dictionary.

        Parameters
        ----------
        debug_mode
            Debug mode can modify the compilation process,
            so that debugging of the compilation process is easier.
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

        # `sequence_to_file` of a module can be `True` even if its `False` for a cluster
        if sequence_to_file is None or sequence_to_file is False:
            sequence_to_file = self.instrument_cfg.get("sequence_to_file", False)

        align_qasm_fields = debug_mode

        if self.supports_acquisition:
            program["acq_metadata"] = {}

        program["sequencers"] = {}
        for seq_name, seq in self.sequencers.items():
            seq_program, acq_metadata = seq.compile(
                repetitions=repetitions,
                sequence_to_file=sequence_to_file,
                align_qasm_fields=align_qasm_fields,
            )
            if seq_program is not None:
                program["sequencers"][seq_name] = seq_program
            if acq_metadata is not None:
                program["acq_metadata"][seq_name] = acq_metadata

        if len(program) == 0:
            return {}

        program["settings"] = self._settings.to_dict()
        program["repetitions"] = repetitions

        return program


class BasebandModuleCompiler(ClusterModuleCompiler):
    """
    Abstract class with all the shared functionality between the QRM and QCM baseband
    modules.
    """

    @property
    def settings_type(self) -> type[BasebandModuleSettings]:
        """The settings type used by baseband-type devices."""
        return BasebandModuleSettings

    def assign_frequencies(self, sequencer: Sequencer) -> None:
        """
        Determines LO/IF frequencies and assigns them, for baseband modules.

        In case of **no** external local oscillator, the NCO is given the same
        frequency as the clock -- unless NCO was permanently disabled via
        `"interm_freq": 0` in the hardware config.

        In case of **an** external local oscillator and `sequencer.mix_lo` is
        ``False``, the LO is given the same frequency as the clock
        (via :func:`.helpers.determine_clock_lo_interm_freqs`).
        """
        compiler_container = self.parent.parent
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
                    freqs=helpers.Frequencies(
                        clock=clock_freq,
                        LO=compiler_lo.frequency,
                        IF=sequencer.frequency,
                    ),
                    downconverter_freq=sequencer.downconverter_freq,
                    mix_lo=sequencer.mix_lo,
                )
            except Exception as error:  # Adding sequencer info to exception message
                raise error.__class__(
                    f"{error} (for '{sequencer.name}' of '{self.name}' "
                    f"with port '{sequencer.port}' and clock '{sequencer.clock}')"
                )
            self._set_lo_interm_freqs(
                freqs=freqs, sequencer=sequencer, compiler_lo_baseband=compiler_lo
            )

    def assign_attenuation(self) -> None:
        """
        Meant to assign attenuation settings from the hardware configuration, if there
        is any. For baseband modules there is no attenuation parameters currently.
        """


class RFModuleCompiler(ClusterModuleCompiler):
    """
    Abstract class with all the shared functionality between the QRM-RF and QCM-RF
    modules.
    """

    @property
    def settings_type(self) -> type[RFModuleSettings]:
        """The settings type used by RF modules."""
        return RFModuleSettings

    def assign_frequencies(self, sequencer: Sequencer) -> None:
        """Determines LO/IF frequencies and assigns them for RF modules."""
        compiler_container = self.parent.parent
        if (
            len(sequencer.connected_output_indices) == 0
            or sequencer.clock not in compiler_container.resources
        ):
            return

        for lo_idx in RFModuleCompiler._get_connected_lo_indices(sequencer):
            lo_freq_setting_name = f"lo{lo_idx}_freq"
            try:
                freqs = helpers.determine_clock_lo_interm_freqs(
                    freqs=helpers.Frequencies(
                        clock=compiler_container.resources[sequencer.clock]["freq"],
                        LO=getattr(self._settings, lo_freq_setting_name),
                        IF=sequencer.frequency,
                    ),
                    downconverter_freq=sequencer.downconverter_freq,
                    mix_lo=True,
                )
            except Exception as error:  # Adding sequencer info to exception message
                raise error.__class__(
                    f"{error} (for '{sequencer.name}' of '{self.name}' "
                    f"with port '{sequencer.port}' and clock '{sequencer.clock}')"
                )
            self._set_lo_interm_freqs(
                freqs=freqs,
                sequencer=sequencer,
                lo_freq_setting_rf=lo_freq_setting_name,
            )

    @staticmethod
    def _get_connected_lo_indices(sequencer: Sequencer) -> Iterator[int]:
        """
        Identify the LO the sequencer is outputting.
        Use the sequencer output to module output correspondence, and then
        use the fact that LOX is connected to module output X.
        """
        for sequencer_output_index in sequencer.connected_output_indices:
            if sequencer_output_index % 2 != 0:
                # We will only use real output 0 and 2, as they are part of the same
                # complex outputs as real output 1 and 3
                continue

            module_output_index = 0 if sequencer_output_index == 0 else 1
            yield module_output_index

    def assign_attenuation(self) -> None:
        """
        Assigns attenuation settings from the hardware configuration.

        Floats that are a multiple of 1 are converted to ints.
        This is needed because the :func:`quantify_core.measurement.control.grid_setpoints`
        converts setpoints to floats when using an attenuation as settable.
        """

        def _convert_to_int(value, label: str) -> int | None:
            if value is not None:
                if not math.isclose(value % 1, 0):
                    raise ValueError(
                        f'Trying to set "{label}" to non-integer value {value}'
                    )
                return int(value)
            return None

        complex_input_0 = self.instrument_cfg.get("complex_input_0", {})
        complex_output_0 = self.instrument_cfg.get("complex_output_0", {})

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
        complex_output_1 = self.instrument_cfg.get("complex_output_1", {})
        self._settings.out1_att = _convert_to_int(
            complex_output_1.get("output_att", None),
            label="out1_att",
        )
