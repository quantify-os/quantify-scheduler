# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
"""QASM program class for Qblox backend."""
from __future__ import annotations
from contextlib import contextmanager
from typing import TYPE_CHECKING
from typing import List, Union, Optional

import numpy as np
from columnar import columnar
from columnar.exceptions import TableOverflowError
from quantify_scheduler.backends.qblox import q1asm_instructions
from quantify_scheduler.backends.qblox import constants
from quantify_scheduler.backends.types.qblox import OpInfo
from quantify_scheduler.backends.qblox import helpers

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
        self.elapsed_time: int = 0
        """The time elapsed after finishing the program in its current form. This is
        used  to keep track of the overall timing and necessary waits."""
        self.instructions: List[list] = list()
        """A list containing the instructions added to the program"""

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
        args
            All arguments to pass to `get_instruction_as_list`.
        **kwargs
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
                loop_reg = "R2"
                loop_label = f"wait{len(self.instructions)}"
                with self.loop(loop_reg, loop_label, repetitions):
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

    def wait_till_start_operation(self, operation: OpInfo):
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
                f"Start time of operation is invalid. Qblox QCM and QRM "
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

    def wait_till_start_then_play(self, pulse: OpInfo, idx0: int, idx1: int):
        """
        Waits until the start of the pulse, sets the QASMRuntimeSettings and plays the
        pulse.

        Parameters
        ----------
        pulse
            The pulse to play.
        idx0
            Index corresponding to the I channel of the pulse in the awg dict.
        idx1
            Index corresponding to the Q channel of the pulse in the awg dict.
        """
        self.wait_till_start_operation(pulse)
        self.auto_play_pulse(pulse, idx0, idx1)

    def _stitched_pulse(self, duration: float, loop_reg: str, idx0: int, idx1: int):
        repetitions = int(duration // constants.PULSE_STITCHING_DURATION)

        if repetitions > 0:
            with self.loop(
                register=loop_reg,
                label=f"stitch{len(self.instructions)}",
                repetitions=repetitions,
            ):
                self.emit(
                    q1asm_instructions.PLAY,
                    idx0,
                    idx1,
                    helpers.to_grid_time(constants.PULSE_STITCHING_DURATION),
                )
                self.elapsed_time += repetitions * helpers.to_grid_time(
                    constants.PULSE_STITCHING_DURATION
                )

        pulse_time_remaining = helpers.to_grid_time(
            duration % constants.PULSE_STITCHING_DURATION
        )
        if pulse_time_remaining > 0:
            self.emit(q1asm_instructions.PLAY, idx0, idx1, pulse_time_remaining)
            self.emit(
                q1asm_instructions.SET_AWG_GAIN,
                0,
                0,
                comment="set to 0 at end of pulse",
            )
        self.elapsed_time += pulse_time_remaining

    def play_stitched_pulse(self, pulse: OpInfo, idx0: int, idx1: int):
        """
        Stitches multiple square pulses together to form one long square pulse.

        Parameters
        ----------
        pulse
            The pulse to play.
        idx0
            Index in the awg_dict corresponding to the waveform for the I channel.
        idx1
            Index in the awg_dict corresponding to the waveform for the Q channel.
        """
        self.update_runtime_settings(pulse)
        self._stitched_pulse(pulse.duration, "R2", idx0, idx1)

    def play_staircase(self, pulse: OpInfo, idx0: int, idx1: int):
        """
        Generates a staircase through offset instructions.

        Parameters
        ----------
        pulse
            The pulse to play.
        idx0
            Not used.
        idx1
            Not used.
        """
        del idx0, idx1  # not used

        loop_reg = "R2"
        offs_reg = "R3"
        offs_reg_zero = "R4"

        num_steps = pulse.data["num_steps"]
        start_amp = pulse.data["start_amp"]
        final_amp = pulse.data["final_amp"]
        step_duration = helpers.to_grid_time(pulse.duration / num_steps)

        amp_step = (final_amp - start_amp) / (num_steps - 1)
        amp_step_immediate = self._expand_from_normalised_range(
            amp_step / self.parent.parent.awg_output_volt,
            constants.IMMEDIATE_SZ_OFFSET,
            "offset_awg_path0",
            pulse,
        )
        start_amp_immediate = self._expand_from_normalised_range(
            start_amp / self.parent.parent.awg_output_volt,
            constants.IMMEDIATE_SZ_OFFSET,
            "offset_awg_path0",
            pulse,
        )
        if start_amp_immediate < 0:
            start_amp_immediate += constants.REGISTER_SIZE  # registers are unsigned

        self.emit(
            q1asm_instructions.SET_AWG_GAIN,
            constants.IMMEDIATE_SZ_GAIN // 2,
            constants.IMMEDIATE_SZ_GAIN // 2,
            comment="set gain to known value",
        )
        self.emit(
            q1asm_instructions.MOVE,
            start_amp_immediate,
            offs_reg,
            comment="keeps track of the offsets",
        )
        self.emit(
            q1asm_instructions.MOVE, 0, offs_reg_zero, comment="zero for Q channel"
        )
        self.emit(q1asm_instructions.NEW_LINE)
        with self.loop(
            loop_reg, f"ramp{len(self.instructions)}", repetitions=num_steps
        ):
            self.emit(q1asm_instructions.SET_AWG_OFFSET, offs_reg, offs_reg_zero)
            self.emit(
                q1asm_instructions.UPDATE_PARAMETERS,
                constants.GRID_TIME,
            )
            self.elapsed_time += constants.GRID_TIME
            if amp_step_immediate >= 0:
                self.emit(
                    q1asm_instructions.ADD,
                    offs_reg,
                    amp_step_immediate,
                    offs_reg,
                    comment=f"next incr offs by {amp_step_immediate}",
                )
            else:
                self.emit(
                    q1asm_instructions.SUB,
                    offs_reg,
                    -amp_step_immediate,
                    offs_reg,
                    comment=f"next incr offs by {amp_step_immediate}",
                )
            self.auto_wait(step_duration - constants.GRID_TIME)
        self.elapsed_time += step_duration * (num_steps - 1) if num_steps > 1 else 0

        self.emit(q1asm_instructions.SET_AWG_OFFSET, 0, 0)
        self.emit(q1asm_instructions.NEW_LINE)

    def auto_play_pulse(self, pulse: OpInfo, idx0: int, idx1: int):
        """
        Generates the instructions to play a pulse and updates the timing. Automatically
        takes care of custom pulse behavior.

        Parameters
        ----------
        pulse
            The pulse to play.
        idx0
            Index in the awg_dict corresponding to the waveform for the I channel.
        idx1
            Index in the awg_dict corresponding to the waveform for the Q channel.
        """
        reserved_pulse_mapping = {
            "stitched_square_pulse": self.play_stitched_pulse,
            "staircase": self.play_staircase,
        }
        if pulse.uuid in reserved_pulse_mapping:
            func = reserved_pulse_mapping[pulse.uuid]
            func(pulse, idx0, idx1)
        else:
            self.update_runtime_settings(pulse)
            self.emit(q1asm_instructions.PLAY, idx0, idx1, constants.GRID_TIME)
            self.elapsed_time += constants.GRID_TIME

    def wait_till_start_then_acquire(self, acquisition: OpInfo, idx0: int, idx1: int):
        """
        Waits until the start of the acquisition, then starts the acquisition.

        Parameters
        ----------
        acquisition
            The pulse to perform.
        idx0
            Index corresponding to the I channel of the acquisition weights in the acq
            dict.
        idx1
            Index corresponding to the Q channel of the acquisition weights in the acq
            dict.
        """
        self.wait_till_start_operation(acquisition)
        self.emit(q1asm_instructions.ACQUIRE, idx0, idx1, constants.GRID_TIME)
        self.elapsed_time += constants.GRID_TIME

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
    def loop(self, register: str, label: str, repetitions: int = 1):
        """
        Defines a context manager that can be used to generate a loop in the QASM
        program.

        Parameters
        ----------
        register
            The register to use for the loop iterator.
        label
            The label to use for the jump.
        repetitions
            The amount of iterations to perform.

        Returns
        -------

        Examples
        --------

        This adds a loop to the program that loops 10 times over a wait of 100 ns.

        .. jupyter-execute::

            import inspect, os, json
            from quantify_scheduler.types import Schedule
            from quantify_scheduler.backends.qblox.qasm_program import QASMProgram
            import quantify_scheduler.schemas.examples as es
            from quantify_scheduler.backends.qblox import (
                instrument_compilers, compiler_container
            )

            esp = inspect.getfile(es)
            map_f = os.path.abspath(os.path.join(esp, "..", "qblox_test_mapping.json"))
            with open(map_f, "r") as f:
                HARDWARE_MAPPING = json.load(f)

            sched = Schedule("example")
            container = compiler_container.CompilerContainer(sched)
            qcm = instrument_compilers.Pulsar_QCM(
                container,
                "qcm0",
                total_play_time=10,
                hw_mapping=HARDWARE_MAPPING["qcm0"]
            )
            qasm = QASMProgram(qcm.sequencers["seq0"])

            with qasm.loop(register='R0', label='repeat', repetitions=10):
                qasm.auto_wait(100)

            qasm.instructions
        """  # FIXME replace json.load() quantify-scheduler#132 #pylint: disable=fixme
        comment = f"iterator for loop with label {label}"

        def gen_start():
            self.emit(q1asm_instructions.MOVE, repetitions, register, comment=comment)
            self.emit(q1asm_instructions.NEW_LINE, label=label)

        try:
            yield gen_start()
        finally:
            self.emit(q1asm_instructions.LOOP, register, f"@{label}")
