# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Compiler base and utility classes for Qblox backend."""

from __future__ import annotations

import json
import logging
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from enum import Enum, auto
from os import makedirs, path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Hashable,
    Iterator,
    Protocol,
    TypeVar,
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
)
from quantify_scheduler.backends.qblox.operation_handling.base import IOperationStrategy
from quantify_scheduler.backends.qblox.operation_handling.factory import (
    get_operation_strategy,
)
from quantify_scheduler.backends.qblox.operation_handling.virtual import (
    ConditionalStrategy,
    ControlFlowReturnStrategy,
    LoopStrategy,
    UpdateParameterStrategy,
)
from quantify_scheduler.backends.qblox.qasm_program import QASMProgram
from quantify_scheduler.backends.types.qblox import (
    OpInfo,
    SequencerSettings,
    StaticHardwareProperties,
)
from quantify_scheduler.enums import BinMode
from quantify_scheduler.helpers.schedule import (
    extract_acquisition_metadata_from_acquisition_protocols,
)

if TYPE_CHECKING:
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
        distortion_corrections: dict[int, Any] | None = None,
    ) -> None:
        self.parent = parent
        self.name = name
        self.total_play_time = total_play_time
        self.instrument_cfg = instrument_cfg
        self.latency_corrections = latency_corrections or {}
        self.distortion_corrections = distortion_corrections or {}

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


class SequencerCompiler(ABC):
    """
    Class that performs the compilation steps on the sequencer level.

    Abstract base class for different sequencer types.

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
    """

    def __init__(
        self,
        parent: ClusterModuleCompiler,
        index: int,
        portclock: tuple[str, str],
        static_hw_properties: StaticHardwareProperties,
        settings: SequencerSettings,
        latency_corrections: dict[str, float],
        distortion_corrections: dict[int, Any] | None = None,
        qasm_hook_func: Callable | None = None,
    ) -> None:
        self.parent = parent
        self.index = index
        self.port = portclock[0]
        self.clock = portclock[1]
        self.op_strategies: list[IOperationStrategy] = []
        self._num_acquisitions = 0

        self.static_hw_properties = static_hw_properties

        self.register_manager = register_manager.RegisterManager()

        self._settings = settings

        self.qasm_hook_func = qasm_hook_func
        """Allows the user to inject custom Q1ASM code into the compilation, just prior
         to returning the final string."""

        portclock_key = f"{self.port}-{self.clock}"
        self.latency_correction: float = latency_corrections.get(portclock_key, 0)
        """Latency correction accounted for by delaying the start of the program."""

        self.distortion_correction: float = distortion_corrections

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

    @abstractmethod
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
        ordered_op_strategies: list[IOperationStrategy],
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
        ordered_op_strategies
            A sorted list of operations, in order of execution.
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
        self._write_pre_wait_sync_instructions(qasm)

        # program header
        qasm.set_latch(self.op_strategies)
        qasm.emit(q1asm_instructions.WAIT_SYNC, constants.MIN_TIME_BETWEEN_OPERATIONS)
        qasm.emit(
            q1asm_instructions.UPDATE_PARAMETERS, constants.MIN_TIME_BETWEEN_OPERATIONS
        )

        self._initialize_append_mode_registers(qasm, ordered_op_strategies)

        # Program body. The operations must be ordered such that real-time IO operations
        # always come after any other operations. E.g., an offset instruction should
        # always come before the parameter update, play, or acquisition instruction.

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
            self._write_repetition_loop_header(qasm)

            last_operation_end = {True: 0.0, False: 0.0}
            for operation in ordered_op_strategies:
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

            self._parse_operations(iter(ordered_op_strategies), qasm, 1)

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
                self._insert_qasm(operation, qasm)

        return self.ParseOperationStatus.EXITED_CONTROL_FLOW

    @abstractmethod
    def _insert_qasm(
        self, op_strategy: IOperationStrategy, qasm_program: QASMProgram
    ) -> None:
        """Get Q1ASM instruction(s) from ``op_strategy`` and insert them into ``qasm_program``."""

    @abstractmethod
    def _write_pre_wait_sync_instructions(self, qasm: QASMProgram) -> None:
        """
        Write instructions to the QASM program that must come before the first wait_sync.

        The duration must be equal for all module types.
        """

    @abstractmethod
    def _write_repetition_loop_header(self, qasm: QASMProgram) -> None:
        """
        Write the Q1ASM that should appear at the start of the repetition loop.

        The duration must be equal for all module types.
        """

    def _get_ordered_operations(self) -> list[IOperationStrategy]:
        """Get the class' operation strategies in order of scheduled execution."""
        return sorted(
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
        # This can be unsorted. The sorting step is in SequencerCompiler.generate_qasm_program()
        self.op_strategies += upd_params

    @staticmethod
    def _any_other_updating_instruction_at_timing_for_parameter_instruction(
        op_index: int, ordered_op_strategies: list[IOperationStrategy]
    ) -> bool:
        op = ordered_op_strategies[op_index]
        if not op.operation_info.is_parameter_instruction:
            return False

        def iterate_other_ops(iterate_range, allow_return_stack: bool) -> bool:
            for other_op_index in iterate_range:
                other_op = ordered_op_strategies[other_op_index]
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
            iterate_range=range(op_index + 1, len(ordered_op_strategies)),
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
                op_index=op_index, ordered_op_strategies=pulses_and_acqs
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
        :meth:`~SequencerCompiler.compile` is called.
        """
        self._insert_update_parameters()

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
            `SequencerCompiler.settings.seq_fn`.
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
        operation_list = self._get_ordered_operations()
        qasm_program = self.generate_qasm_program(
            ordered_op_strategies=operation_list,
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


_SequencerT_co = TypeVar("_SequencerT_co", bound=SequencerCompiler, covariant=True)
"""
A generic SequencerCompiler type for typehints in :class:`ClusterModuleCompiler`.

Covariant so that subclasses of ClusterModuleCompiler can use subclassses of
:class:`SequencerCompiler` in their typehints.
"""


class _ModuleSettingsType(Protocol):
    """
    A typehint for the various module settings (e.g.
    :class:`~quantify_scheduler.backends.types.qblox.BasebandModuleSettings`) classes.
    """

    def to_dict(self) -> dict[str, Any]:
        """Convert the settings to a dictionary."""


class ClusterModuleCompiler(InstrumentCompiler, Generic[_SequencerT_co], ABC):
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

    _settings: _ModuleSettingsType

    def __init__(
        self,
        parent,  # No type hint due to circular import, added to docstring
        name: str,
        total_play_time: float,
        instrument_cfg: dict[str, Any],
        latency_corrections: dict[str, float] | None = None,
        distortion_corrections: dict[int, Any] | None = None,
    ) -> None:
        driver_version_check.verify_qblox_instruments_version()
        super().__init__(
            parent=parent,
            name=name,
            total_play_time=total_play_time,
            instrument_cfg=instrument_cfg,
            latency_corrections=latency_corrections,
            distortion_corrections=distortion_corrections,
        )
        self._op_infos: dict[tuple[str, str], list[OpInfo]] = defaultdict(list)

        self.sequencers: dict[str, _SequencerT_co] = {}

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
    def static_hw_properties(self) -> StaticHardwareProperties:
        """
        The static properties of the hardware. This effectively gathers all the
        differences between the different modules.
        """

    def _construct_all_sequencer_compilers(self) -> None:
        """
        Constructs :class:`~SequencerCompiler` objects for each port and clock combination
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
        sequencers: dict[str, _SequencerT_co] = {}
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
                    new_seq = self._construct_sequencer_compiler(
                        index=len(sequencers),
                        portclock=portclock,
                        channel_name=channel_name,
                        sequencer_cfg=sequencer_cfg,
                        channel_cfg=channel_cfg,
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

    @abstractmethod
    def _construct_sequencer_compiler(
        self,
        index: int,
        portclock: tuple[str, str],
        channel_name: str,
        sequencer_cfg: dict[str, Any],
        channel_cfg: dict[str, Any],
    ) -> _SequencerT_co:
        """Create the sequencer object of the correct sequencer type belonging to the module."""

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
            `SequencerCompiler.settings.seq_fn`.

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
        program: dict[str, Any] = {}

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
