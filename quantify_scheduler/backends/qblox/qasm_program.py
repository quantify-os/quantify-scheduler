# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch

"""QASM program class for Qblox backend."""
from __future__ import annotations

from contextlib import contextmanager
from typing import (
    TYPE_CHECKING,
    Generator,
    Hashable,
    Iterator,
    Sequence,
)

import numpy as np
from columnar import columnar
from columnar.exceptions import TableOverflowError

from quantify_scheduler.backends.qblox import constants, helpers, q1asm_instructions
from quantify_scheduler.backends.qblox.conditional import (
    ConditionalManager,
)

if TYPE_CHECKING:
    from quantify_scheduler.backends.qblox.operation_handling.base import (
        IOperationStrategy,
    )
    from quantify_scheduler.backends.qblox.operation_handling.virtual import (
        ConditionalStrategy,
    )
    from quantify_scheduler.backends.qblox.register_manager import RegisterManager
    from quantify_scheduler.backends.types.qblox import (
        OpInfo,
        StaticHardwareProperties,
    )
    from quantify_scheduler.schedules.schedule import AcquisitionMetadata


def get_marker_binary(marker_setting: str | int) -> int:
    """
    Sets the marker from a string representing a binary number. Each digit
    corresponds to a marker e.g. '0010' sets the second marker to True.

    If the marker setting is already an integer, the function checks whether it is a
    4-bit integer.

    Parameters
    ----------
    marker_setting
        The string representing a binary number.

    """
    if isinstance(marker_setting, str):
        if len(marker_setting) != 4:
            raise ValueError("4 marker values are expected.")
        return int(marker_setting, 2)
    else:
        if marker_setting > 0b1111:
            raise ValueError(f"Invalid marker setting: {marker_setting=}.")
        return marker_setting


class QASMProgram:
    """
    Class that holds the compiled Q1ASM program that is to be executed by the sequencer.

    Apart from this the class holds some convenience functions that auto generate
    certain instructions with parameters, as well as update the elapsed time.

    Parameters
    ----------
    static_hw_properties
        Dataclass holding the properties of the hardware that this program is to be
        played on.
    register_manager
        The register manager that keeps track of the occupied/available registers.
    align_fields
        If True, make QASM program more human-readable by aligning its fields.
    acq_metadata
        Provides a summary of the used acquisition protocol, bin mode, acquisition
        channels, acquisition indices per channel, and repetitions.

    """

    def __init__(
        self,
        static_hw_properties: StaticHardwareProperties,
        register_manager: RegisterManager,
        align_fields: bool,
        acq_metadata: AcquisitionMetadata | None,
    ) -> None:
        self.static_hw_properties = static_hw_properties
        """Dataclass holding the properties of the hardware that this program is to be
        played on."""
        self.register_manager = register_manager
        """The register manager that keeps track of the occupied/available registers."""
        self.align_fields = align_fields
        """If true, all labels, instructions, arguments and comments
        in the string representation of the program are printed on the same indention level.
        This worsens performance."""
        self.acq_metadata = acq_metadata
        """Provides a summary of the used acquisition protocol, bin mode, acquisition
        channels, acquisition indices per channel, and repetitions."""

        self.time_last_acquisition_triggered: int | None = None
        """Time on which the last acquisition was triggered. Is ``None`` if no previous
        acquisition was triggered."""
        self.time_last_pulse_triggered: int | None = None
        """Time on which the last operation was triggered. Is ``None`` if no previous
        operation was triggered."""
        self.instructions: list[list] = list()
        """A list containing the instructions added to the program. The instructions
        added are in turn a list of the instruction string with arguments."""
        self.conditional_manager = ConditionalManager()
        """The conditional manager that keeps track of the conditionals."""
        self._lock_conditional: bool = False
        """A lock to prevent nested conditionals."""
        self._elapsed_times_in_loops: list[int] = [0]
        """The time elapsed in its current form.
        This is used  to keep track of the overall and nested loop timing and necessary waits."""

    @property
    def elapsed_time(self) -> int:
        """
        Current elapsed time of all the instructions in ns.
        It needs to be manually adjusted after each modifications of the QASM program.
        If the QASM program is in a loop,
        only one repetition's worth of elapsed time should be registered.
        After a loop is ended, ``QASMProgram`` will automatically adjust the correct
        elapsed time with all repetitions.
        """
        return sum(self._elapsed_times_in_loops)

    @elapsed_time.setter
    def elapsed_time(self, value: int) -> None:
        difference: int = value - self.elapsed_time
        self._elapsed_times_in_loops[-1] += difference

    def _find_qblox_acq_index(self, acq_channel: Hashable) -> int:
        """
        Finds the Qblox acq_index corresponding to acq_channel
        in the acq_metadata.
        """
        # This function is a temporary solution.
        # Proper solution: SE-298.
        assert self.acq_metadata is not None
        for (
            qblox_acq_index,
            acq_channel_metadata,
        ) in self.acq_metadata.acq_channels_metadata.items():
            if acq_channel_metadata.acq_channel == acq_channel:
                return qblox_acq_index
        raise ValueError(f"Qblox acquisition index not found for {acq_channel=}.")

    @staticmethod
    def get_instruction_as_list(
        instruction: str,
        *args: int | str,
        label: str | None = None,
        comment: str | None = None,
    ) -> list[str | int]:
        """
        Takes an instruction with arguments, label and comment and turns it into the
        list required by the class.

        Parameters
        ----------
        instruction
            The instruction to use. This should be one specified in
            :mod:`~quantify_scheduler.backends.qblox.q1asm_instructions`
            or the assembler will raise an exception.
        args
            Arguments to be passed.
        label
            Adds a label to the line. Used for jumps and loops.
        comment
            Optionally add a comment to the instruction.

        Returns
        -------
        :
            List that contains all the passed information in the valid format for the
            program.

        Raises
        ------
        SyntaxError
            More arguments passed than the sequencer allows.

        """
        instr_args = ",".join(str(arg) for arg in args)

        label_str = f"{label}:" if label is not None else ""
        comment_str = f"# {comment}" if comment is not None else ""
        return [label_str, instruction, instr_args, comment_str]

    def emit(self, *args, **kwargs) -> list[str | int]:
        """
        Wrapper around the ``get_instruction_as_list`` which adds it to this program.

        Parameters
        ----------
        args
            All arguments to pass to `get_instruction_as_list`.
        **kwargs
            All keyword arguments to pass to `get_instruction_as_list`.

        Returns
        -------
        :
            A list containing instructions.

        """
        # Translating the acquisition channel to qblox acquisition index
        # is intended as a temporary solution.
        # TODO: Proper solution: SE-298.
        instruction = args[0]
        if self.acq_metadata and (
            instruction
            in (
                q1asm_instructions.ACQUIRE,
                q1asm_instructions.ACQUIRE_TTL,
                q1asm_instructions.ACQUIRE_WEIGHED,
            )
        ):
            args = list(args)
            args[1] = self._find_qblox_acq_index(acq_channel=args[1])

        self.instructions.append(self.get_instruction_as_list(*args, **kwargs))
        return self.instructions[-1]

    # --- QOL functions -----

    def set_latch(self, op_strategies: Sequence[IOperationStrategy]) -> None:
        """
        Set the latch that is needed for conditional playback.

        This assumes that the latch address is present inside the pulses'
        `operation_info`. If no latch address is found, nothing is emitted.

        Parameters
        ----------
        op_strategies
            The op_strategies containing the pulses to search the latch address in.

        """
        for op_strategy in op_strategies:
            op_info = op_strategy.operation_info
            if not op_info.is_acquisition and (
                op_info.data.get("feedback_trigger_address") is not None
            ):
                self.emit(q1asm_instructions.FEEDBACK_TRIGGER_EN, 1, 4)
                return

    def auto_wait(
        self,
        wait_time: int,
        count_as_elapsed_time: bool = True,
        comment: str | None = None,
    ) -> None:
        """
        Automatically emits a correct wait command. If the wait time is longer than
        allowed by the sequencer it correctly breaks it up into multiple wait
        instructions. If the number of wait instructions is too high (>4), a loop will
        be used.

        Parameters
        ----------
        wait_time
            Time to wait in ns.
        count_as_elapsed_time
            If true, this wait time is taken into account when keeping track of timing.
            Otherwise, the wait instructions are added but this wait time is ignored in
            the timing calculations in the rest of the program.
        comment
            Allows to override the default comment.

        Raises
        ------
        ValueError
            If ``wait_time <= 0``.

        """
        if wait_time == 0:
            return
        if wait_time < 0:
            raise ValueError(
                f"Invalid wait time. Attempting to wait "
                f"for {wait_time} ns at t={self.elapsed_time}"
                f" ns."
            )

        comment = comment if comment else f"auto generated wait ({wait_time} ns)"
        if wait_time > constants.IMMEDIATE_MAX_WAIT_TIME:
            repetitions = wait_time // constants.IMMEDIATE_MAX_WAIT_TIME

            # number of instructions where it becomes worthwhile to use a loop.
            instr_number_using_loop = 4
            if repetitions > instr_number_using_loop:
                loop_label = f"wait{len(self.instructions)}"
                with self.loop(loop_label, repetitions):
                    self.emit(
                        q1asm_instructions.WAIT,
                        constants.IMMEDIATE_MAX_WAIT_TIME,
                        comment=comment,
                    )
                    if count_as_elapsed_time:
                        self.elapsed_time += constants.IMMEDIATE_MAX_WAIT_TIME
                    self.conditional_manager.num_real_time_instructions += 1
            else:
                for _ in range(repetitions):
                    self.emit(
                        q1asm_instructions.WAIT,
                        constants.IMMEDIATE_MAX_WAIT_TIME,
                        comment=comment,
                    )
                    if count_as_elapsed_time:
                        self.elapsed_time += constants.IMMEDIATE_MAX_WAIT_TIME
                    self.conditional_manager.num_real_time_instructions += 1
            time_left = wait_time % constants.IMMEDIATE_MAX_WAIT_TIME
        else:
            time_left = int(wait_time)

        if time_left > 0:
            self.emit(
                q1asm_instructions.WAIT,
                time_left,
                comment=comment,
            )
            if count_as_elapsed_time:
                self.elapsed_time += time_left
            self.conditional_manager.num_real_time_instructions += 1

    def wait_till_start_operation(self, operation: OpInfo) -> None:
        """
        Waits until the start of a pulse or acquisition.

        Parameters
        ----------
        operation
            The pulse or acquisition that we want to wait for.

        Raises
        ------
        ValueError
            If wait time < 0.

        """
        start_time = helpers.to_grid_time(operation.timing)
        wait_time = start_time - self.elapsed_time
        if wait_time > 0:
            self.auto_wait(wait_time)
        elif wait_time < 0 and operation.is_parameter_instruction:
            raise ValueError(
                f"Invalid timing. {repr(operation)} cannot be started at this order or time. "
                f"Please try to reorder your operations by adding this operation "
                "before any other operation (possibly at the same time) that happens at that time."
            )
        elif wait_time < 0 and operation.name != "IdlePulse":
            # The idle pulse is a no operation, if any other operation
            # is simultaneously running, it is allowed.
            raise ValueError(
                f"Invalid timing. Attempting to wait for {wait_time} "
                f"ns before {repr(operation)}. Please note that a wait time of at least"
                f" {constants.MIN_TIME_BETWEEN_OPERATIONS} ns is required between "
                f"operations.\nAre multiple operations being started at the same time?"
            )

    def set_gain_from_amplitude(
        self,
        amplitude_path_I: float,  # noqa N803 - uppercase in name
        amplitude_path_Q: float,  # noqa N803 - uppercase in name
        operation: OpInfo | None,
    ) -> None:
        """
        Sets the gain such that a 1.0 in waveform memory corresponds to the full awg gain.

        Parameters
        ----------
        amplitude_path_I
            Voltage to set on path_I.
        amplitude_path_Q
            Voltage to set on path_Q.
        operation
            The operation for which this is done. Used for the exception messages.

        """
        awg_gain_path_I_immediate = self.expand_awg_from_normalised_range(
            amplitude_path_I,
            constants.IMMEDIATE_SZ_GAIN,
            "awg_gain_0",
            operation,
        )
        awg_gain_path_Q_immediate = self.expand_awg_from_normalised_range(
            amplitude_path_Q,
            constants.IMMEDIATE_SZ_GAIN,
            "awg_gain_1",
            operation,
        )
        comment = f"setting gain for {operation.name}" if operation else ""
        self.emit(
            q1asm_instructions.SET_AWG_GAIN,
            awg_gain_path_I_immediate,
            awg_gain_path_Q_immediate,
            comment=comment,
        )

    @staticmethod
    def expand_awg_from_normalised_range(
        val: float,
        immediate_size: int,
        param: str | None = None,
        operation: OpInfo | None = None,
    ) -> float:
        """
        Takes the value of an awg gain or offset parameter
        in normalized form (abs(param) <= 1.0),
        and expands it to an integer
        in the appropriate range required by the sequencer.

        Parameters
        ----------
        val
            The value of the parameter to expand.
        immediate_size
            The size of the immediate. Used to find the max int value.
        param
            The name of the parameter, to make a possible exception message more
            descriptive.
        operation
            The operation this value is expanded for, to make a possible exception
            message more descriptive.

        Returns
        -------
        :
            The expanded value of the parameter.

        Raises
        ------
        ValueError
            Parameter is not in the normalized range.

        """
        if np.abs(val) > 1.0:
            raise ValueError(
                f"{param} is set to {val}. Parameter must be in the range "
                f"-1.0 <= {param} <= 1.0 for {repr(operation)}."
            )
        max_gain = immediate_size // 2
        return max(-max_gain, min(round(val * max_gain), max_gain - 1))

    def __str__(self) -> str:
        """
        Returns a string representation of the program. The sequencer expects the program
        to be such a string.

        The conversion to str is done using `columnar`, which expects a list of lists,
        and turns it into a string with rows and columns corresponding to those lists.

        Returns
        -------
        :
            The string representation of the program.

        """
        if self.align_fields:
            try:
                instructions_str = columnar(
                    self.instructions, headers=None, no_borders=True, wrap_max=0
                )
            # running in a sphinx environment can trigger a TableOverFlowError
            except TableOverflowError:
                instructions_str = columnar(
                    self.instructions, headers=None, no_borders=True, terminal_width=120
                )
            # columnar inserts a newline before all the the instruction rows
            return instructions_str.split("\n", 1)[1]
        else:
            return "\n".join(" ".join(instruction) for instruction in self.instructions) + "\n"

    @contextmanager
    def conditional(self, operation: ConditionalStrategy) -> Generator[None, None, None]:
        """
        Defines a conditional block in the QASM program.


        When this context manager is entered/exited it will insert additional
        ``set_cond`` QASM instructions in the program that specify the
        conditionality of a set of instructions.

        The following example should make it clear what is happening.

        .. code-block:: none

            set_cond set_enable=1, mask=0, operator=OR, else_duration=4
            <50 ns duration of instructions that contains 3 real time instructions>

            set_cond set_enable=1, mask=0, operator=NOR, else_duration=4
            wait 50-3*4+4 = 42 ns # adding an additional 4 ns to make math work out

            set_cond set_enable=0, mask=0, operator=OR, else_duration=4

        The `else_duration` is the wait time per real time instruction in the
        conditional block. If a trigger happened, the first block runs normally for
        50 ns, the second block runs for 4 ns. If there is no trigger, the first
        block runs for 3*4 = 12 ns, second block for 42 ns. So the duration in
        both cases is 42 ns. Note that `set_cond` itself has zero duration.

        The exact values that need to be passed to the ``set_cond``
        instructions are determined while the qasm program is generated with the
        help of
        :class:`~quantify_scheduler.backends.qblox.conditional.FeedbackTriggerCondition`
        and
        :class:`~quantify_scheduler.backends.qblox.conditional.ConditionalManager`.

        Parameters
        ----------
        operation: ConditionalStrategy
            The conditional strategy that defines the start of a conditional block.

        """
        trigger_condition = operation.trigger_condition
        if self._lock_conditional:
            raise RuntimeError(
                "Nested conditional playback inside schedules is not supported by "
                f"the Qblox backend. "
                f"This error is caused by the following operation strategy:\n{operation}."
            )
        self._lock_conditional = True

        # This instruction will be replaced when the context manager exits the
        # conditional block.
        self.emit(
            q1asm_instructions.FEEDBACK_SET_COND,
            int(trigger_condition.enable),
            trigger_condition.mask,
            trigger_condition.operator.value,
            constants.MIN_TIME_BETWEEN_OPERATIONS,
            comment="start conditional playback",
        )
        self.conditional_manager.reset()
        self.conditional_manager.start_time = self.elapsed_time

        yield
        # When the context manager exits, add an else branch to fill the correct wait time
        # and add a stop conditional playback and
        # replace the initial FEEDBACK_SET_COND instruction.
        self.conditional_manager.end_time = self.elapsed_time
        self.emit(
            q1asm_instructions.FEEDBACK_SET_COND,
            int(trigger_condition.enable),
            trigger_condition.mask,
            (~trigger_condition.operator).value,
            constants.MIN_TIME_BETWEEN_OPERATIONS,
            comment="else wait",
        )
        # autowait now adds an additional duration to elapsed time that we need to compensate.
        duration = (
            self.conditional_manager.duration
            - constants.MIN_TIME_BETWEEN_OPERATIONS
            * self.conditional_manager.num_real_time_instructions
            + constants.MIN_TIME_BETWEEN_OPERATIONS
        )
        self.auto_wait(duration, count_as_elapsed_time=False)
        self.emit(
            q1asm_instructions.FEEDBACK_SET_COND,
            0,
            0,
            0,
            0,
            comment="stop conditional playback",
        )
        self.elapsed_time += constants.MIN_TIME_BETWEEN_OPERATIONS

        self.conditional_manager.reset()
        self._lock_conditional = False

    @contextmanager
    def loop(self, label: str, repetitions: int = 1) -> Generator[str, None, None]:
        """
        Defines a context manager that can be used to generate a loop in the QASM
        program.

        Parameters
        ----------
        label
            The label to use for the jump.
        repetitions
            The amount of iterations to perform.

        Yields
        ------
        :
            The register used as loop counter.

        Examples
        --------
        This adds a loop to the program that loops 10 times over a wait of 100 ns.

        .. jupyter-execute::

            from quantify_scheduler.backends.qblox.qasm_program import QASMProgram
            from quantify_scheduler.backends.qblox.instrument_compilers import QCMCompiler
            from quantify_scheduler.backends.qblox import register_manager, constants
            from quantify_scheduler.backends.types.qblox import (
                StaticAnalogModuleProperties,
                BoundedParameter
            )

            qasm = QASMProgram(
                static_hw_properties=QCMCompiler.static_hw_properties,
                register_manager=register_manager.RegisterManager(),
                align_fields=True,
                acq_metadata=None,
            )

            with qasm.loop(label="repeat", repetitions=10):
                qasm.auto_wait(100)

            qasm.instructions

        """
        register = self.register_manager.allocate_register()
        comment = f"iterator for loop with label {label}"

        self._elapsed_times_in_loops.append(0)

        self.emit(q1asm_instructions.MOVE, repetitions, register, comment=comment)
        self.emit(q1asm_instructions.NEW_LINE, label=label)

        yield register

        self.emit(q1asm_instructions.LOOP, register, f"@{label}")
        self.register_manager.free_register(register)

        last_elapsed_time = self._elapsed_times_in_loops.pop()

        self._elapsed_times_in_loops[-1] += last_elapsed_time * repetitions

    @contextmanager
    def temp_registers(self, amount: int = 1) -> Iterator[list[str]]:
        """
        Context manager for using a register temporarily. Frees up the register
        afterwards.

        Parameters
        ----------
        amount
            The amount of registers to temporarily use.

        Yields
        ------
        :
            Either a single register or a list of registers.

        """
        registers: list[str] = list()
        for _ in range(amount):
            registers.append(self.register_manager.allocate_register())
        yield registers

        for reg in registers:
            self.register_manager.free_register(reg)
