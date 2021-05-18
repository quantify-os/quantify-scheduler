# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
from __future__ import annotations

import logging
import textwrap
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from quantify.scheduler.backends.types import zhinst
from quantify.scheduler.helpers import schedule as schedule_helpers

logger = logging.getLogger()


class SeqcInstructions(Enum):
    """The sequencer enum type."""

    NONE = ""
    PLAY_WAVE = "playWave"
    SET_TRIGGER = "setTrigger"
    WAIT = "wait"
    WAIT_WAVE = "waitWave"
    # addi + addiu + st
    ARM_INTEGRATION = (
        "AWG_INTEGRATION_ARM + AWG_INTEGRATION_TRIGGER + AWG_MONITOR_TRIGGER"
    )
    EXECUTE_TABLE_ENTRY = "executeTableEntry"


SEQC_INSTR_CLOCKS: Dict[zhinst.DeviceType, Dict[SeqcInstructions, int]] = {
    zhinst.DeviceType.HDAWG: {
        SeqcInstructions.PLAY_WAVE: 3,
        SeqcInstructions.SET_TRIGGER: 1,
        SeqcInstructions.WAIT: 3,
        SeqcInstructions.WAIT_WAVE: 3,
        SeqcInstructions.EXECUTE_TABLE_ENTRY: 0,
    },
    zhinst.DeviceType.UHFQA: {
        SeqcInstructions.WAIT: 0,
        SeqcInstructions.PLAY_WAVE: 0,
        SeqcInstructions.SET_TRIGGER: 1,
        SeqcInstructions.ARM_INTEGRATION: 3,
    },
}


class SeqcInfo:
    """
    The Sequencer information class containing durations,
    offsets, pulses and clocks.
    """

    sequencer_clock: float

    def __init__(
        self,
        cached_schedule: schedule_helpers.CachedSchedule,
        output: zhinst.Output,
        low_res_clock: float,
    ) -> None:
        self.sequencer_clock = low_res_clock
        self._line_trigger_delay_in_seconds = output.line_trigger_delay

        self._schedule_offset_in_seconds = cached_schedule.start_offset_in_seconds
        self._schedule_duration_in_seconds: float = (
            cached_schedule.total_duration_in_seconds - self._schedule_offset_in_seconds
        )
        timeslot_indexes: List[int] = list(
            cached_schedule.port_timeline_dict[output.port].keys()
        )
        self._timeline_start_in_seconds: float = (
            schedule_helpers.get_operation_start(
                cached_schedule.schedule,
                timeslot_index=timeslot_indexes[0],
            )
            - self.schedule_offset_in_seconds
        )
        self._timeline_end_in_seconds: float = (
            schedule_helpers.get_operation_end(
                cached_schedule.schedule, timeslot_index=timeslot_indexes[-1]
            )
            - self.schedule_offset_in_seconds
        )

    def to_clocks(self, seconds: float) -> int:
        """
        Returns the converted value in clocks.

        Parameters
        ----------
        seconds : float

        Returns
        -------
        int
        """
        if seconds <= 0:
            return 0
        return round(seconds / self.sequencer_clock)

    @property
    def schedule_offset_in_seconds(self) -> float:
        """
        Returns the schedule start offset in seconds.
        The offset is determined by the Reset Operation.

        Returns
        -------
        float
        """
        return self._schedule_offset_in_seconds

    @property
    def schedule_offset_in_clocks(self) -> int:
        """
        Returns the schedule start offset in clocks.
        The offset is determined by the Reset Operation.

        Returns
        -------
        int
        """
        return self.to_clocks(self.schedule_offset_in_seconds)

    @property
    def schedule_duration_in_seconds(self) -> float:
        """
        Returns the total schedule duration in seconds.

        Returns
        -------
        float
        """
        return self._schedule_duration_in_seconds

    @property
    def schedule_duration_in_clocks(self) -> int:
        """
        Returns the total schedule duration in clocks.

        Returns
        -------
        int
        """
        return self.to_clocks(self.schedule_duration_in_seconds)

    @property
    def timeline_start_in_seconds(self) -> float:
        """
        Returns the port timeline start in seconds.

        Returns
        -------
        float
        """
        return self._timeline_start_in_seconds

    @property
    def timeline_start_in_clocks(self) -> int:
        """
        Returns the port timeline start in clocks.

        Returns
        -------
        float
        """
        return self.to_clocks(self.timeline_start_in_seconds)

    @property
    def timeline_end_in_seconds(self) -> float:
        """
        Returns the port timeline end in seconds.

        Returns
        -------
        float
        """
        return self._timeline_end_in_seconds

    @property
    def timeline_end_in_clocks(self) -> int:
        """
        Returns the port timeline start in clocks.

        Returns
        -------
        float
        """
        return self.to_clocks(self.timeline_end_in_seconds)

    @property
    def line_trigger_delay_in_seconds(self) -> float:
        """
        Returns the configured line delay when using
        triggers in seconds.

        Returns
        -------
        float
        """
        return self._line_trigger_delay_in_seconds

    @property
    def line_trigger_delay_in_clocks(self) -> int:
        """
        Returns the configured line delay when using
        triggers in clocks.

        Returns
        -------
        int
        """
        return self.to_clocks(self.line_trigger_delay_in_seconds)


class SeqcILGenerator(object):
    """
    The Intermediate Sequencer Language Generator.

    This class acts a an assembler for the seqc programs
    which can be executed on the ZI AWG(s).
    """

    _level: int
    _variables: Dict[str, Tuple[str, Optional[str]]]
    _program: List[Tuple[int, str]]

    def __init__(self) -> None:
        """
        Creates a new instance of SeqcILGenerator.
        """
        self._level = 0
        self._variables = dict()
        self._program = list()

    def _declare_local(self, type_def: str, name: str) -> None:
        """
        Creates a new local variable.

        Parameters
        ----------
            type_def :
                The variable type definition.
            name :
                The variable name.

        Raises
        ------
            ValueError
                Duplicate local variable error.
        """
        if name in self._variables:
            raise ValueError(f"Duplicate local variable '{name}'!")
        self._variables[name] = (f"{type_def} {name}", None)

    def _assign_local(self, name: str, value: str) -> None:
        """
        Assign a value to a local variable.

        Parameters
        ----------
            name :
                The variable name.
            value :
                The new variable value.

        Raises
        ------
            ValueError
                Undefined reference error.
        """
        if name not in self._variables:
            raise ValueError(f"Undefined reference '{name}'!")

        (declaration, _) = self._variables[name]
        self._variables[name] = (declaration, value)

    def _emit(self, operation: str) -> None:
        """
        Emits a new operation to the program.

        Parameters
        ----------
            operation :
                The operation to append.
        """
        self._program.append((self._level, operation))

    def _begin_scope(self) -> None:
        """
        Indent a new scope level.
        """
        self._level += 1

    def _end_scope(self) -> None:
        """
        Dedent the scope level.

        Raises
        ------
            ValueError
                Scope level error.
        """
        self._level -= 1
        if self._level < 0:
            raise ValueError("SeqcILGenerator scope level is to low!")

    def declare_var(self, name: str, value: Optional[Union[str, int]] = None) -> None:
        """
        Creates a new variable of type `var` with a name and
        optionally its value.

        Parameters
        ----------
            name :
                The variable name.
            value: Optional[str]
                The variable value. (optional)
        """
        self._declare_local("var", name)
        if value is not None:
            if isinstance(value, list):
                values = " + ".join(value)
                self._assign_local(name, f"{values};")
            else:
                self.assign_var(name, value)

    def declare_wave(self, name: str, value: Optional[str] = None) -> None:
        """
        Creates a new variable of type `wave` with a name and
        optionally its value.

        Parameters
        ----------
            name :
                The variable name.
            value:
                The variable value. (optional)
        """
        self._declare_local("wave", name)
        if value is not None:
            self.assign_var(name, value)

    def assign_get_user_reg(self, name: str, index: int) -> None:
        """
        Assign the getUserReg function to a variable by name with a specific index.

        Parameters
        ----------
        name :
            The variable name.
        index :
            The register index.
        """
        self._assign_local(name, f"getUserReg({index});")

    def assign_placeholder(self, name: str, size: int) -> None:
        """
        Assign a placeholder to a variable by name with a specific size.

        Parameters
        ----------
            name :
                The variable name.
            size :
                The size of the placeholder.
        """
        self._assign_local(name, f"placeholder({size});")

    def assign_var(self, name: str, value: Union[str, int, List[Any]]) -> None:
        """
        Assign a value to a variable by name.

        This method requires the variable to be declared
        before allowing a new value to be assigned to it.

        Translates to:
        ```
        wave w0;
        wave w0 = "dev1234_awg0_0"; # <--
        ```

        Parameters
        ----------
            name :
                The variable name.
            value :
                The new value.
        """
        if isinstance(value, (int, list)):
            self._assign_local(name, f"{value};")
        else:
            self._assign_local(name, f'"{value}";')

    def emit_comment(self, text: str) -> None:
        """
        Emit a comment to the program.

        Parameters
        ----------
        text :
        """
        self._emit(f"// {text}")

    def emit_assign_wave_index(self, *args: str, index: int) -> None:
        """
        Emit assignWaveIndex to the program which
        assigns a wave variable to an index in the
        waveform table (just like a pointer).

        Parameters
        ----------
            index :
                The waveform table index.
        """
        variables: str = ", ".join(args)
        self._emit(f"assignWaveIndex({variables}, {index});")

    def emit_execute_table_entry(self, index: int, comment: str = "") -> None:
        """
        Emit executeTableEntry to the program.
        Executes a command table waveform.

        ::note `executeTableEntry` is not blocking
        so if you want to await the wave use `waitWave`
        directly after it.

        Parameters
        ----------
            index :
                The wave index to execute.
        """
        self._emit(f"executeTableEntry({index});{comment}")

    def emit_play_wave(self, *names: str, comment: str = "") -> None:
        """
        Emit playWave to the program.

        Parameters
        ----------
            name :
                The variable name.
        """
        _names = ", ".join(names)
        self._emit(f"playWave({_names});{comment}")

    def emit_wait_wave(self, comment: str = "") -> None:
        """
        Emit waitWave to the program.
        """
        self._emit(f"waitWave();{comment}")

    def emit_wait(self, cycles: int, comment: str = "") -> None:
        """
        Emits a wait instruction to the sequencer program.

        Parameters
        ----------
            cycles :
                The number of cycles to wait.
        """
        self._emit(f"wait({cycles});{comment}")

    def emit_set_trigger(self, index: Union[int, str], comment: str = "") -> None:
        """
        Emit setTrigger to the program.

        Parameters
        ----------
        index :
            The number or string of a trigger to set.
        """
        self._emit(f"setTrigger({index});{comment}")

    def emit_wait_dig_trigger(self, index: int = 0, comment: str = "") -> None:
        """
        Emit waitDigTrigger to the program.

        Parameters
        ----------
            index :
                The trigger to wait on, by default 0
        """
        trigger: str = None
        if index == 0:
            trigger = f"waitDigTrigger(1);{comment}"
        else:
            trigger = f"waitDigTrigger({index}, 1);{comment}"
        self._emit(trigger)

    def emit_start_qa_monitor(self) -> None:
        """
        Starts the Quantum Analysis Monitor unit
        by setting and clearing appropriate AWG
        trigger output signals.
        """
        self._emit("startQAMonitor();")

    def emit_start_qa_result(
        self, bitmask: Optional[str] = "", trigger: Optional[str] = ""
    ) -> None:
        """
        Starts the Quantum Analysis Result unit by setting
        and clearing appropriate AWG trigger output signals.

        .. code-block:: python

            // Start Quantum Analysis Result unit for
            // channel=1 on trigger=AWG_INTEGRATION_TRIGGER
            startQAResult(0b0000000001, AWG_INTEGRATION_TRIGGER);

            // Reset and clear triggers
            startQAResult();

        Parameters
        ----------
        bitmask :
            The bitmake to select explicitly which of the
            ten possible qubit results should be read, by default ""
        trigger :
            The trigger to start the Quantum Analysis Result
            unit. If no trigger is specified it will clear
            the triggers, by default ""
        """
        params = ", ".join([bitmask, trigger])
        self._emit(f"startQAResult({params});")

    def emit_begin_while(self, predicate: str = "true") -> None:
        """
        Emit while loop to the program.

        Parameters
        ----------
            predicate :
                The while condition, by default "true"
        """
        self._emit(f"while({predicate})")
        self._begin_scope()

    def emit_end_while(self) -> None:
        """
        Emit ending the while loop.
        """
        self._end_scope()

    def emit_begin_repeat(self, repetitions: Union[int, str] = 1) -> None:
        """
        Emit repeat loop to the program.

        Parameters
        ----------
            repetitions :
                The repeat condition, by default 1
        """
        self._emit(f"repeat({repetitions})")
        self._begin_scope()

    def emit_end_repeat(self) -> None:
        """
        Emit ending the repeat loop.
        """
        self._end_scope()

    def generate(self) -> str:
        """
        Returns the generated seqc program.

        This program can be run on ZI AWG(s).

        Returns
        -------
            str
                The seqc program.
        """
        program: str = ""
        program += "// Generated by quantify-scheduler.\n"
        program += "// Variables\n"
        for (name, operation) in self._variables.values():
            if operation is not None:
                program += f"{name} = {operation}\n"
            else:
                program += f"{name};\n"

        if len(self._program) == 0:
            return program

        program += "\n// Operations\n"
        current_level = 0
        previous_level = 0
        for (level, operation) in self._program:
            current_level = level
            if current_level > previous_level:
                # indent
                program += textwrap.indent("{\n", "  " * (level - 1))
            elif current_level < previous_level:
                # dedent
                program += textwrap.indent("}\n", "  " * (level - 1))

            program += textwrap.indent(operation + "\n", "  " * level)
            previous_level = level

        if current_level == 1:
            # dedent
            program += "}\n"

        return program


def add_wait(
    seqc_gen: SeqcILGenerator,
    delay: int,
    device_type: zhinst.DeviceType,
    comment: str = "",
) -> int:
    """
    Add a wait instruction to the SeqcILGenerator with the specified delay.

    Parameters
    ----------
    seqc_gen :
    delay :
        The delay in clocks.
    device_type :
    comment :
        An optional comment to the instruction, by default ""

    Returns
    -------
    int
        The number of clocks waited.
    """
    assert delay >= 0

    elapsed_clocks: int = 0

    # Add the timeline start offset. Consider the trigger as clock=0.
    if device_type == zhinst.DeviceType.UHFQA:
        n_assembly_instructions = delay
        cycles_to_wait = delay
    elif device_type == zhinst.DeviceType.HDAWG:
        n_assembly_instructions = SEQC_INSTR_CLOCKS[device_type][SeqcInstructions.WAIT]
        cycles_to_wait = delay - n_assembly_instructions

    if cycles_to_wait < 0:
        logger.warning("Minimum number of clocks to wait must at least be 3!")
        seqc_gen.emit_wait(
            0, comment=f"\t// {comment} n_instr={n_assembly_instructions} <--"
        )
        elapsed_clocks = n_assembly_instructions
    else:
        seqc_gen.emit_wait(
            cycles_to_wait,
            comment=f"\t// {comment} n_instr={n_assembly_instructions}",
        )
        elapsed_clocks = delay

    return elapsed_clocks


def add_play_wave(
    seqc_gen: SeqcILGenerator,
    variable: str,
    device_type: zhinst.DeviceType,
    comment: str = "",
) -> int:
    """
    Adds a playWave instruction to the
    seqc program.

    Parameters
    ----------
    seqc_gen :
    variable :
    device_type :
    comment :

    Returns
    -------
    int
        Elapsed number of clock cycles.
    """
    n_assembly_instructions = SEQC_INSTR_CLOCKS[device_type][SeqcInstructions.PLAY_WAVE]
    seqc_gen.emit_play_wave(
        variable,
        comment=f"\t// {comment} n_instr={n_assembly_instructions}",
    )
    return n_assembly_instructions


def add_execute_table_entry(
    seqc_gen: SeqcILGenerator,
    index: int,
    device_type: zhinst.DeviceType,
    comment: str = "",
) -> int:
    """
    Adds an executeTableEntry instruction to
    seqc program.

    Parameters
    ----------
    seqc_gen :
    index :
    device_type :
    comment :

    Returns
    -------
    int
        Elapsed number of clock cycles.

    Raises
    ------
    AttributeError
        Raised when the DeviceType not equals HDAWG.
    """
    if device_type != zhinst.DeviceType.HDAWG:
        raise AttributeError(
            "Unsupported sequencer instruction "
            + f"'{SeqcInstructions.EXECUTE_TABLE_ENTRY}'"
        )

    n_assembly_instructions = SEQC_INSTR_CLOCKS[device_type][
        SeqcInstructions.EXECUTE_TABLE_ENTRY
    ]

    seqc_gen.emit_execute_table_entry(
        index,
        comment=f"\t// {comment} pulse={index} n_instr={n_assembly_instructions}",
    )

    return n_assembly_instructions


def add_set_trigger(
    seqc_gen: SeqcILGenerator,
    value: Union[List[str], int, str],
    device_type: zhinst.DeviceType,
    comment: str = "",
) -> int:
    """
    Adds a setTrigger instruction to the seqc
    program.

    Parameters
    ----------
    seqc_gen :
    value :
    device_type :
    comment :

    Returns
    -------
    int
        Elapsed number of clock cycles.
    """
    n_assembly_instructions = SEQC_INSTR_CLOCKS[device_type][
        SeqcInstructions.SET_TRIGGER
    ]

    if isinstance(value, list):
        trigger = " + ".join(value)
        n_assembly_instructions += min(len(value) - 1, 2)
    else:
        trigger = value

    seqc_gen.emit_set_trigger(
        trigger, comment=f"\t// {comment} n_instr={n_assembly_instructions}"
    )

    return n_assembly_instructions


def add_seqc_info(seqc_gen: SeqcILGenerator, seqc_info: SeqcInfo):
    """
    Add Sequence Information to the SeqcILGenerator using comments.

    Parameters
    ----------
    seqc_gen :
    seqc_info :
    """
    seqc_gen.emit_comment(
        f"Schedule offset: {seqc_info.schedule_offset_in_seconds:.9f}s "
        + f"{seqc_info.schedule_offset_in_clocks:d} clocks"
    )
    seqc_gen.emit_comment(
        f"Schedule duration: {seqc_info.schedule_duration_in_seconds:.9f}s "
        + f"{seqc_info.schedule_duration_in_clocks:d} clocks"
    )
    seqc_gen.emit_comment(
        f"Sequence start: {seqc_info.timeline_start_in_seconds:.9f}s "
        + f"{seqc_info.timeline_start_in_clocks:d} clocks"
    )
    seq_duration_in_seconds = (
        seqc_info.timeline_end_in_seconds - seqc_info.timeline_start_in_seconds
    )
    seq_duration_in_clocks = (
        seqc_info.timeline_end_in_clocks - seqc_info.timeline_start_in_clocks
    )
    seqc_gen.emit_comment(
        f"Sequence duration: {seq_duration_in_seconds:.9f}s "
        + f"{seq_duration_in_clocks:d} clocks"
    )
    seqc_gen.emit_comment(
        f"Sequence end: {seqc_info.timeline_end_in_seconds:.9f}s "
        + f"{seqc_info.timeline_end_in_clocks:d} clocks"
    )
    seqc_gen.emit_comment(
        f"Line delay: {seqc_info.line_trigger_delay_in_seconds:.9f}s "
        + f"{seqc_info.line_trigger_delay_in_clocks:d} clocks"
    )


def add_csv_waveform_variables(
    seqc_gen: SeqcILGenerator,
    device_serial: str,
    commandtable_map: Dict[int, int],
):
    """
    Adds wave variables in form of a
    CSV filename to the seqc file.

    Parameters
    ----------
    seqc_gen :
    device_serial :
    commandtable_map :
    """
    for waveform_index in commandtable_map.values():
        # Declare new placeholder and assign wave index
        name: str = f"w{waveform_index:d}"
        seqc_gen.declare_wave(name, f"{device_serial}_wave{waveform_index:d}")
