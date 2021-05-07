# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
"""QASM program class for Qblox backend."""
from contextlib import contextmanager
from typing import List, Union, Optional

import numpy as np
from columnar import columnar
from columnar.exceptions import TableOverflowError
from quantify.scheduler.backends.qblox import q1asm_instructions
from quantify.scheduler.backends.qblox import constants
from quantify.scheduler.backends.types.qblox import OpInfo


class QASMProgram:
    """
    Class that holds the compiled Q1ASM program that is to be executed by the sequencer.

    Apart from this the class holds some convenience functions that auto generate
    certain instructions with parameters, as well as update the elapsed time.

    Attributes
    ----------
    elapsed_time:
        The time elapsed after finishing the program in its current form. This is used
        to keep track of the overall timing and necessary waits.
    instructions:
        A list containing the instructions added to the program
    """

    def __init__(self):
        self.elapsed_time: int = 0
        self.instructions: List[list] = list()

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
        instruction:
            The instruction to use. This should be one specified in `PulsarInstructions`
            or the assembler will raise an exception.
        args:
            Arguments to be passed.
        label:
            Adds a label to the line. Used for jumps and loops.
        comment:
            Optionally add a comment to the instruction.

        Returns
        -------
        :
            List that contains all the passed information in the valid format for the
            program.

        Raises
        -------
        SyntaxError
            More arguments passed than the sequencer allows.
        """
        max_args_amount = 3
        if len(args) > max_args_amount:
            raise SyntaxError(
                f"Too many arguments supplied to `get_instruction_as_list` for "
                f"instruction {instruction}."
            )
        instr_args = ",".join(str(arg) for arg in args)

        label_str = f"{label}:" if label is not None else ""
        comment_str = f"# {comment}" if comment is not None else ""
        return [label_str, instruction, instr_args, comment_str]

    def emit(self, *args, **kwargs) -> None:
        """
        Wrapper around the `get_instruction_as_list` which adds it to this program.

        Parameters
        ----------
        args:
            All arguments to pass to `get_instruction_as_list`.
        kwargs:
            All keyword arguments to pass to `get_instruction_as_list`.
        """
        self.instructions.append(self.get_instruction_as_list(*args, **kwargs))

    # --- QOL functions -----

    def auto_wait(self, wait_time: int):
        """
        Automatically emits a correct wait command. If the wait time is longer than
        allowed by the sequencer it correctly breaks it up into multiple wait
        instructions.

        Parameters
        ----------
        wait_time:
            Time to wait in ns.

        Returns
        -------

        Raises
        ------
        ValueError
            If `wait_time` <= 0
        """
        if wait_time <= 0:
            raise ValueError(
                f"Invalid wait time. Attempting to wait "
                f"for {wait_time} ns at t={self.elapsed_time}"
                f" ns."
            )

        if wait_time > constants.IMMEDIATE_SZ_WAIT:
            for _ in range(wait_time // constants.IMMEDIATE_SZ_WAIT):
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

    def wait_till_start_operation(self, operation: OpInfo):
        """
        Waits until the start of a pulse or acquisition.

        Parameters
        ----------
        operation:
            The pulse or acquisition that we want to wait for.

        Raises
        ------
        ValueError
            If wait time < 0
        """
        start_time = self.to_pulsar_time(operation.timing)
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

    def wait_till_start_then_play(self, pulse: OpInfo, idx0: int, idx1: int):
        """
        Waits until the start of the pulse, sets the QASMRuntimeSettings and plays the
        pulse.

        Parameters
        ----------
        pulse:
            The pulse to play.
        idx0:
            Index corresponding to the I channel of the pulse in the awg dict.
        idx1:
            Index corresponding to the Q channel of the pulse in the awg dict.

        Returns
        -------

        """
        self.wait_till_start_operation(pulse)
        self.update_runtime_settings(pulse)
        self.emit(q1asm_instructions.PLAY, idx0, idx1, constants.GRID_TIME)
        self.elapsed_time += constants.GRID_TIME

    def wait_till_start_then_acquire(self, acquisition: OpInfo, idx0: int, idx1: int):
        """
        Waits until the start of the acquisition, then starts the acquisition.

        Parameters
        ----------
        acquisition:
            The pulse to perform.
        idx0:
            Index corresponding to the I channel of the acquisition weights in the acq
            dict.
        idx1:
            Index corresponding to the Q channel of the acquisition weights in the acq
            dict.

        Returns
        -------

        """
        self.wait_till_start_operation(acquisition)
        self.emit(q1asm_instructions.ACQUIRE, idx0, idx1, constants.GRID_TIME)
        self.elapsed_time += constants.GRID_TIME

    def update_runtime_settings(self, operation: OpInfo):
        """
        Adds the commands needed to correctly set the QASMRuntimeSettings.

        Parameters
        ----------
        operation:
            The pulse to prepare the settings for.

        Returns
        -------

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
        val:
            The value of the parameter to expand.
        param:
            The name of the parameter, to make a possible exception message more
            descriptive.
        operation:
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

    @staticmethod
    def to_pulsar_time(time: float) -> int:
        """
        Takes a float value representing a time in seconds as used by the schedule, and
        returns the integer valued time in nanoseconds that the sequencer uses.

        Parameters
        ----------
        time:
            The time to convert

        Returns
        -------
        :
            The integer valued nanosecond time
        """
        time_ns = int(round(time * 1e9))
        if time_ns % constants.GRID_TIME != 0:
            raise ValueError(
                f"Attempting to use a time interval of {time_ns} ns. "
                f"Please ensure that the durations of and wait times between "
                f"operations are multiples of {constants.GRID_TIME} ns."
            )
        return time_ns

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
            return columnar(self.instructions, headers=None, no_borders=True)
        # running in a sphinx environment can trigger a TableOverF
        # lowError
        except TableOverflowError:
            return columnar(
                self.instructions, headers=None, no_borders=True, terminal_width=120
            )

    @contextmanager
    def loop(self, register: str, label: str, repetitions: int = 1):
        """
        Defines a context manager that can be used to generate a loop in the QASM
        program.

        Parameters
        ----------
        register:
            The register to use for the loop iterator.
        label:
            The label to use for the jump.
        repetitions:
            The amount of iterations to perform.

        Returns
        -------

        Examples
        --------
        .. jupyter-execute::

            from quantify.scheduler.backends.qblox.instrument_compilers import QASMProgram #pylint: disable=line-too-long

            qasm = QASMProgram()
            with qasm.loop(register='R0', label='repeat', repetitions=10):
                qasm.auto_wait(100)

        This adds a loop to the program that loops 10 times over a wait of 100 ns.
        """
        comment = f"iterator for loop with label {label}"

        def gen_start():
            self.emit(q1asm_instructions.MOVE, repetitions, register, comment=comment)
            self.emit(q1asm_instructions.NEW_LINE, label=label)

        try:
            yield gen_start()
        finally:
            self.emit(q1asm_instructions.LOOP, register, f"@{label}")
