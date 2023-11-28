# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
from __future__ import annotations

import logging
import textwrap
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from quantify_scheduler.backends.types import zhinst

logger = logging.getLogger(__name__)


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
    START_QA = "startQA"


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
        SeqcInstructions.START_QA: 7,
    },
}


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
        """Indent a new scope level."""
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
        Creates a new variable of type ``var`` with a name and
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
        Creates a new variable of type ``wave`` with a name and
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

    def emit_blankline(self) -> None:
        """
        Emits a blank line to the program.

        This is typically used to create a visual separation for readability.
        """
        self._emit("")

    def emit_comment(self, text: str) -> None:
        """
        Emit a comment to the program.

        Parameters
        ----------
        text :
            The comment text.
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

        ::note ``executeTableEntry`` is not blocking
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
        names :
            The wave names to be played. This should refer to the wave variable name
            as defined in the seqc, or the wave index in the commandtable
            to be played.
        comment :
            The inline comment to be emitted in the seqc.

        Examples
        --------
        An example for the use of the playWave instruction from the LabOne Programming
        Manual.

        Ensure that the "wave_file" variable (the name argument) corresponds to a
        filename that was declared using the declareWave

        .. code-block::

            //Definition inline with playWave
            playWave("wave_file");
            //Assign first to a wave data type, then use
            wave w = "wave_file";
            playWave(w)

        """
        if comment:
            comment = f"\t// {comment}"

        _names = ", ".join(names)
        self._emit(f"playWave({_names});{comment}")

    def emit_wait_wave(self, comment: str = "") -> None:
        """Emit waitWave to the program."""
        self._emit(f"waitWave();{comment}")

    def emit_start_qa(self, comment: str = "") -> None:
        """
        Starts the Quantum Analysis Result and Input units by setting and clearing
        appropriate AWG trigger output signals. The choice of whether to start one or
        the other or both  units can be controlled using the command argument. An
        bitmask may be used to select explicitly which of the ten possible qubit
        results should be read. If no qubit results are enabled, then the Quantum
        Analysis Result unit will not be triggered. An optional value may be used to
        set the normal trigger outputs of the AWG together with starting the
        Quantum Analysis Result and input units. If the
        value is not used, then the trigger signals will be cleared.

        Parameter

        - monitor: Enable for QA monitor, default: false
        - result_address: Set address associated with result, default: 0x0
        - trigger: Trigger value, default: 0x0
        - weighted_integrator_mask: Integration unit enable mask, default: QA_INT_ALL

        """
        # using default arguments to start all channels for acquisition.
        # example based on the UHFQA manual.

        if comment:
            comment = f"\t// {comment}"

        self._emit(f"startQA(QA_INT_ALL, true);{comment}")

    def emit_wait(self, cycles: int, comment: str = "") -> None:
        """
        Emits a wait instruction to the sequencer program.

        Parameters
        ----------
        cycles :
            The number of cycles to wait.
        comment :
            The inline comment to be emitted in the seqc.
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

    def emit_wait_dig_trigger(
        self,
        index: int = 0,
        comment: str = "",
        device_type: Optional[zhinst.DeviceType] = None,
    ) -> None:
        """
        Emit waitDigTrigger to the program.

        Parameters
        ----------
        index :
            The trigger to wait on, by default 0
        """
        if comment:
            comment = f"\t// {comment}"

        if index == 0:
            trigger = f"waitDigTrigger(1);{comment}"
        elif device_type == zhinst.DeviceType.HDAWG:
            trigger = f"waitDigTrigger({index});{comment}"
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
        self, bitmask: Optional[str] = None, trigger: Optional[str] = None
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
        if bitmask is None and trigger is None:
            self._emit("startQAResult();")
        else:
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
        """Emit ending the while loop."""
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
        """Emit ending the repeat loop."""
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
        for name, operation in self._variables.values():
            if operation is not None:
                program += f"{name} = {operation}\n"
            else:
                program += f"{name};\n"

        if len(self._program) == 0:
            return program

        program += "\n// Operations\n"
        current_level = 0
        previous_level = 0
        for level, operation in self._program:
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
        The SeqcILGenerator to add the wait instruction to.
    delay :
        The delay in clocks.
    device_type :
        The device type.
    comment :
        An optional comment to the instruction, by default ""

    Returns
    -------
    int
        The number of clocks waited.

    Raises
    ------
    ValueError
    """
    if comment:
        comment = f"{comment}\t"

    assert delay >= 0

    elapsed_clocks: int = 0

    # Add the timeline start offset. Consider the trigger as clock=0.
    if device_type == zhinst.DeviceType.UHFQA:
        n_assembly_instructions = delay
        cycles_to_wait = delay
    elif device_type == zhinst.DeviceType.HDAWG:
        n_assembly_instructions = SEQC_INSTR_CLOCKS[device_type][SeqcInstructions.WAIT]
        cycles_to_wait = delay - n_assembly_instructions

    # cycles to wait checks for
    if cycles_to_wait < 0:
        raise ValueError(
            "Minimum number of clocks to wait must be at least 3 (HDAWG) or 0 (UHFQA)!"
        )

    seqc_gen.emit_wait(
        cycles_to_wait,
        comment=f"\t\t// {comment} n_instr={n_assembly_instructions}",
    )
    elapsed_clocks = delay

    return elapsed_clocks


def add_play_wave(
    seqc_gen: SeqcILGenerator,
    *variable: str,
    device_type: zhinst.DeviceType,
    comment: str = "",
) -> int:
    """
    Adds a playWave instruction to the
    seqc program.

    Parameters
    ----------
    seqc_gen :
        The SeqcILGenerator to add the playWave instruction to.
    variable :
        The variable to play.
    device_type :
        The device type.
    comment :
        An optional comment to the instruction, by default "".

    Returns
    -------
    int
        Elapsed number of clock cycles.
    """
    if comment:
        comment = f"{comment}\t"

    n_assembly_instructions = SEQC_INSTR_CLOCKS[device_type][SeqcInstructions.PLAY_WAVE]
    seqc_gen.emit_play_wave(
        *variable,
        comment=f"\t// {comment} n_instr={n_assembly_instructions}",
    )
    return n_assembly_instructions


def add_start_qa(
    seqc_gen: SeqcILGenerator,
    device_type: zhinst.DeviceType,
    comment: str = "",
) -> int:
    """
    Adds a startQA instruction to the
    seqc program.
    See :func:`~quantify_scheduler.backends.zhinst.seqc_il_generator.SeqcILGenerator.emit_start_qa`
    for more details.

    Parameters
    ----------
    seqc_gen :
        The SeqcILGenerator to add the startQA instruction to.
    device_type :
        The device type.
    comment :
        An optional comment to the instruction, by default "".

    Returns
    -------
    int
        Elapsed number of clock cycles.
    """
    n_assembly_instructions = SEQC_INSTR_CLOCKS[device_type][SeqcInstructions.START_QA]

    seqc_gen.emit_start_qa(
        comment=f"{comment} n_instr={n_assembly_instructions}",
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
        The SeqcILGenerator to add the executeTableEntry instruction to.
    index :
        The index of the table entry to execute.
    device_type :
        The device type.
    comment :
        An optional comment to the instruction, by default "".


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
        The SeqcILGenerator to add the setTrigger instruction to.
    value :
        The trigger to set.
    device_type :
        The device type.
    comment :
        An optional comment to the instruction, by default "".

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


def declare_csv_waveform_variables(
    seqc_gen: SeqcILGenerator,
    device_name: str,
    waveform_indices: List[int],
    awg_index: int = 0,
):
    """
    Declares waveforms and links them to filenames of .csv files.

    e.g. `wave w0 = `uhfqa1234_awg0_wave0`
    """
    for waveform_index in waveform_indices:
        name: str = f"w{waveform_index:d}"
        seqc_gen.declare_wave(
            name, f"{device_name}_awg{awg_index}_wave{waveform_index:d}"
        )


def add_csv_waveform_variables(
    seqc_gen: SeqcILGenerator,
    device_serial: str,
    awg_index: int,
    commandtable_map: Dict[int, int],
):
    """
    Adds wave variables in form of a
    CSV filename to the seqc file.

    Parameters
    ----------
    seqc_gen :
        The SeqcILGenerator to add the setTrigger instruction to.
    device_serial :
        The device serial number.
    awg_index :
        The AWG index.
    commandtable_map :
        The commandtable map.
    """
    for waveform_index in commandtable_map.values():
        # Declare new placeholder and assign wave index
        name: str = f"w{waveform_index:d}"
        seqc_gen.declare_wave(
            name, f"{device_serial}_awg{awg_index}_wave{waveform_index:d}"
        )
