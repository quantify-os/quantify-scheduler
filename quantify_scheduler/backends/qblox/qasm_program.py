# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
# pylint: disable=comparison-with-callable
"""QASM program class for Qblox backend."""
from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np
from columnar import columnar
from columnar.exceptions import TableOverflowError

from quantify_scheduler.backends.qblox import (
    constants,
    helpers,
    q1asm_instructions,
    register_manager,
)
from quantify_scheduler.backends.types.qblox import OpInfo
from quantify_scheduler.enums import BinMode

if TYPE_CHECKING:
    from quantify_scheduler.backends.qblox import compiler_abc


class QASMProgram:
    """
    Class that holds the compiled Q1ASM program that is to be executed by the sequencer.

    Apart from this the class holds some convenience functions that auto generate
    certain instructions with parameters, as well as update the elapsed time.
    """

    def __init__(self, parent: compiler_abc.Sequencer):
        self.parent = parent
        """A reference to the sequencer for which we are compiling this program."""
        self._register_manager: register_manager.RegisterManager = (
            parent.register_manager
        )
        self.elapsed_time: int = 0
        """The time elapsed after finishing the program in its current form. This is
        used  to keep track of the overall timing and necessary waits."""
        self.time_last_acquisition_triggered: Optional[int] = None
        """Time on which the last acquisition was triggered. Is `None` if no previous
        acquisition was triggered."""
        self.instructions: List[list] = list()
        """A list containing the instructions added to the program. The instructions
        added are in turn a list of the instruction string with arguments."""

    @staticmethod
    def get_instruction_as_list(
        instruction: str,
        *args: Union[int, str],
        label: Optional[str] = None,
        comment: Optional[str] = None,
    ) -> List[Union[str, int]]:
        """
        Takes an instruction with arguments, label and comment and turns it into the
        list required by the class.

        Parameters
        ----------
        instruction
            The instruction to use. This should be one specified in `PulsarInstructions`
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

    def emit(self, *args, **kwargs) -> None:
        """
        Wrapper around the `get_instruction_as_list` which adds it to this program.

        Parameters
        ----------
        args
            All arguments to pass to `get_instruction_as_list`.
        **kwargs
            All keyword arguments to pass to `get_instruction_as_list`.
        """
        self.instructions.append(self.get_instruction_as_list(*args, **kwargs))

    # --- QOL functions -----

    def set_marker(self, marker_setting: Union[str, int] = "0000") -> None:
        """
        Sets the marker from a string representing a binary number. Each digit
        corresponds to a marker e.g. '0010' sets the second marker to True.

        Parameters
        ----------
        marker_setting
            The string representing a binary number.
        """
        if isinstance(marker_setting, str):
            assert len(marker_setting) == 4, "Maximum of 4 markers expected."
            marker_binary = int(marker_setting, 2)
        else:
            assert marker_setting <= 0b1111
            marker_binary = marker_setting
        self.emit(
            q1asm_instructions.SET_MARKER,
            marker_binary,
            comment=f"set markers to {marker_setting}",
        )

    def auto_wait(self, wait_time: int) -> None:
        """
        Automatically emits a correct wait command. If the wait time is longer than
        allowed by the sequencer it correctly breaks it up into multiple wait
        instructions. If the number of wait instructions is too high (>4), a loop will
        be used.

        Parameters
        ----------
        wait_time
            Time to wait in ns.

        Raises
        ------
        ValueError
            If `wait_time` <= 0.
        """
        if wait_time == 0:
            return
        if wait_time < 0:
            raise ValueError(
                f"Invalid wait time. Attempting to wait "
                f"for {wait_time} ns at t={self.elapsed_time}"
                f" ns."
            )

        if wait_time > constants.IMMEDIATE_SZ_WAIT:
            repetitions = wait_time // constants.IMMEDIATE_SZ_WAIT
            instr_number_using_loop = 4
            if repetitions > instr_number_using_loop:
                loop_label = f"wait{len(self.instructions)}"
                with self.loop(loop_label, repetitions):
                    self.emit(
                        q1asm_instructions.WAIT,
                        constants.IMMEDIATE_SZ_WAIT,
                        comment="auto generated wait",
                    )
            else:
                for _ in range(repetitions):
                    self.emit(
                        q1asm_instructions.WAIT,
                        constants.IMMEDIATE_SZ_WAIT,
                        comment="auto generated wait",
                    )
            time_left = wait_time % constants.IMMEDIATE_SZ_WAIT
        else:
            time_left = int(wait_time)

        if time_left > 0:
            self.emit(q1asm_instructions.WAIT, time_left)

        self.elapsed_time += wait_time

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
        if not helpers.is_multiple_of_grid_time(
            operation.timing, grid_time_ns=constants.GRID_TIME
        ):
            raise ValueError(
                f"Start time of operation is invalid. Qblox QcmModule and QRM "
                f"enforce a grid time of {constants.GRID_TIME} ns. Please "
                f"make sure all operations start at an interval of "
                f"{constants.GRID_TIME} ns.\n\nOffending operation:\n"
                f"{repr(operation)}."
            )
        start_time = helpers.to_grid_time(operation.timing)
        wait_time = start_time - self.elapsed_time
        if wait_time > 0:
            self.auto_wait(wait_time)
        elif wait_time < 0:
            raise ValueError(
                f"Invalid timing. Attempting to wait for {wait_time} "
                f"ns before {repr(operation)}. Please note that a wait time of at least"
                f" {constants.GRID_TIME} ns is required between "
                f"operations.\nAre multiple operations being started at the same time?"
            )

    def verify_square_acquisition_duration(self, acquisition: OpInfo, duration: float):
        duration_ns = int(np.round(duration * 1e9))
        if self.parent.settings.integration_length_acq is None:
            if duration_ns % constants.GRID_TIME != 0:
                raise ValueError(
                    f"Attempting to perform square acquisition with a "
                    f"duration of {duration_ns} ns. Please ensure the "
                    f"duration is a multiple of {constants.GRID_TIME} "
                    f"ns.\n\nException caused by {repr(acquisition)}."
                )
            self.parent.settings.integration_length_acq = duration_ns
        elif self.parent.settings.integration_length_acq != duration_ns:
            raise ValueError(
                f"Attempting to set an integration_length of {duration_ns} "
                f"ns, while this was previously determined to be "
                f"{self.parent.settings.integration_length_acq}. Please "
                f"check whether all square acquisitions in the schedule "
                f"have the same duration."
            )

    def _acquire_looped(self, acquisition: OpInfo, bin_idx: Union[int, str]) -> None:
        if bin_idx != 0:
            raise ValueError(
                "looped acquisition currently only works for acquisition "
                "index 0 with binmode average."
            )

        measurement_idx = acquisition.data["acq_channel"]

        duration = acquisition.data["integration_time"]
        self.verify_square_acquisition_duration(acquisition, duration)

        duration_ns = helpers.to_grid_time(duration)

        number_of_times = acquisition.data["num_times"]
        buffer_time = acquisition.data["buffer_time"]
        with self.loop(
            label=f"looped_acq{len(self.instructions)}", repetitions=number_of_times
        ) as loop_register:
            self.emit(
                q1asm_instructions.ACQUIRE,
                measurement_idx,
                loop_register,
                duration_ns,
            )
            buffer_time_ns = helpers.to_grid_time(buffer_time)
            if buffer_time > 0:
                self.emit(q1asm_instructions.WAIT, buffer_time_ns)
            if buffer_time < 0:
                raise ValueError(
                    f"Buffer time cannot be smaller than 0.\n\nException "
                    f"occurred because of {repr(acquisition)}."
                )
        self.elapsed_time += number_of_times * (duration_ns + buffer_time_ns)

    def update_runtime_settings(self, operation: OpInfo):
        """
        Adds the commands needed to correctly set the QASMRuntimeSettings.

        Parameters
        ----------
        operation
            The pulse to prepare the settings for.

        Notes
        -----
            Currently only the AWG gain is set correctly, as that is the only one
            actually used currently by the backend. Will be expanded in the future.
        """
        if operation.pulse_settings is None:
            raise RuntimeError(f"No real-time settings found for {repr(operation)}.")

        awg_gain_path0 = self._expand_from_normalised_range(
            operation.pulse_settings.awg_gain_0,
            constants.IMMEDIATE_SZ_GAIN,
            "awg_gain_0",
            operation,
        )
        awg_gain_path1 = self._expand_from_normalised_range(
            operation.pulse_settings.awg_gain_1,
            constants.IMMEDIATE_SZ_GAIN,
            "awg_gain_1",
            operation,
        )
        self.emit(
            q1asm_instructions.SET_AWG_GAIN,
            awg_gain_path0,
            awg_gain_path1,
            comment=f"setting gain for {operation.uuid}",
        )

    @staticmethod
    def _expand_from_normalised_range(
        val: float,
        immediate_size: int,
        param: Optional[str] = None,
        operation: Optional[OpInfo] = None,
    ):
        """
        Takes a the value of a parameter in normalized form (abs(param) <= 1.0), and
        expands it to an integer in the appropriate range required by the sequencer.

        Parameters
        ----------
        val
            The value of the parameter to expand.
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
                f"-1.0 <= param <= 1.0 for {repr(operation)}."
            )
        return int(val * immediate_size // 2)

    def __str__(self) -> str:
        """
        Returns a string representation of the program. The pulsar expects the program
        to be such a string.

        The conversion to str is done using `columnar`, which expects a list of lists,
        and turns it into a string with rows and columns corresponding to those lists.

        Returns
        -------
        :
            The string representation of the program.
        """
        try:
            return columnar(
                self.instructions, headers=None, no_borders=True, wrap_max=0
            )
        # running in a sphinx environment can trigger a TableOverFlowError
        except TableOverflowError:
            return columnar(
                self.instructions, headers=None, no_borders=True, terminal_width=120
            )

    @contextmanager
    def loop(self, label: str, repetitions: int = 1):
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

            import inspect, json
            from quantify_scheduler import Schedule
            from quantify_scheduler.backends.qblox.qasm_program import QASMProgram
            from quantify_scheduler.schemas.examples import utils
            from quantify_scheduler.backends.qblox import (
                instrument_compilers, compiler_container
            )

            HARDWARE_MAPPING = utils.load_json_example_scheme(
                "qblox_test_mapping.json"
            )

            sched = Schedule("example")
            container = compiler_container.CompilerContainer(sched)
            qcm = instrument_compilers.QcmModule(
                container,
                "qcm0",
                total_play_time=10,
                hw_mapping=HARDWARE_MAPPING["qcm0"]
            )
            qasm = QASMProgram(qcm.sequencers["seq0"])

            with qasm.loop(label='repeat', repetitions=10):
                qasm.auto_wait(100)

            qasm.instructions
        """
        register = self._register_manager.allocate_register()
        comment = f"iterator for loop with label {label}"

        self.emit(q1asm_instructions.MOVE, repetitions, register, comment=comment)
        self.emit(q1asm_instructions.NEW_LINE, label=label)

        yield register

        self.emit(q1asm_instructions.LOOP, register, f"@{label}")
        self._register_manager.free_register(register)

    @contextmanager
    def temp_register(self, amount: int = 1) -> Union[List[str], str]:
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
        registers: List[str] = list()
        for _ in range(amount):
            registers.append(self._register_manager.allocate_register())
        yield registers if len(registers) > 1 else registers[0]

        for reg in registers:
            self._register_manager.free_register(reg)
