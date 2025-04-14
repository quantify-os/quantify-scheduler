# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Qblox specific operation which can be used to inject Q1ASM code directly into a Schedule."""

from __future__ import annotations

import inspect
from copy import deepcopy

from quantify_scheduler.backends.types.qblox import OpInfo
from quantify_scheduler.helpers.collections import make_hash
from quantify_scheduler.operations.operation import Operation


class InlineQ1ASM(Operation):
    """
    Initialize an InlineQ1ASM operation.

    This method sets up an operation that contains inline Q1ASM code
     to be injected directly into a Schedule.

    All comments in the program will be prefixed with an '[inline]' prefix
    to help identify the inline assembly within the sequencer program.


    When using safe labels, then all labels included in the input program
    will get a prefix of 'inj<digits>_'.
    By default, safe labels are always used.
    Labels in comments will not be modified.

    Parameters
    ----------
    program
        The Q1ASM program to be injected.
    duration
        The duration of the operation in seconds.
    port
        The port on which the operation is to be executed.
    clock
        The clock associated with the operation.
    waveforms
        Dictionary containing waveform information, by default None.
    safe_labels
        Flag to indicate if safe labels should be used, by default True.

    Returns
    -------
    None

    Notes
    -----
    .. warning::

        When using safe_labels=False then all labels in the sequencer program are accessible from
        inside the inline Q1ASM injection, and so can be jumped to or overwritten.  Disabling this
        feature is available for debugging and advanced compilation strategies only.

    """

    def __init__(
        self,
        program: str,
        duration: float,
        port: str,
        clock: str,
        *,
        waveforms: dict | None = None,
        safe_labels: bool = True,
    ) -> None:
        waveforms = {} if waveforms is None else waveforms

        super().__init__(name=str(self.__class__.__name__))
        self._name = self.__class__.__name__
        self.program = program
        self._duration = duration
        self.port = port
        self.clock = clock
        self.waveforms = waveforms
        self.safe_labels = safe_labels
        self._update()

    @property
    def name(self) -> str:
        """Return the name of the operation."""
        return self._name

    def _update(self) -> None:
        """Update the Operation's internals."""
        # No need in this class
        pass

    def get_used_port_clocks(self) -> set[tuple[str, str]]:
        """
        Extracts which port-clock combinations are used in this operation.

        Returns
        -------
        :
            All (port, clock) combinations this operation uses.

        """
        return {(self.port, self.clock)}

    def __str__(self) -> str:
        """
        Return a string representation of the InlineQ1ASM operation.

        This method generates a human-readable string that describes the InlineQ1ASM operation.

        Returns
        -------
        str
            A string representation of the InlineQ1ASM operation.

        """
        x = inspect.signature(self.__init__)
        parameter_list = []
        for parameter, _value in x.parameters.items():
            cur_atr = self.__getattribute__(parameter)
            if isinstance(cur_atr, str):
                cur_atr = f"'{cur_atr}'"
            parameter_list.append(f"{parameter}={cur_atr}")
        return f"{self._name}({', '.join(parameter_list)})"

    # TODO remove once UserDict is removed as superclass
    def __deepcopy__(self, memo: dict) -> InlineQ1ASM:
        """Make a deepcopy of this object."""
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result

    def __hash__(self) -> int:
        return make_hash(
            [
                self._name,
                self.program,
                self.duration,
                self.port,
                self.clock,
                self.waveforms,
                self.safe_labels,
            ]
        )


class Q1ASMOpInfo(OpInfo):
    """
    Structure describing an inline Q1ASM operation and containing all the information
    required to play it.
    """

    def __init__(self, inline_q1asm: InlineQ1ASM, operation_start_time: float) -> None:
        # TODO, make proper attributes when OpInfo is no longer a DataClassJsonMixin
        info = {
            "name": inline_q1asm.name,
            "program": inline_q1asm.program,
            "duration": inline_q1asm.duration,
            "port": inline_q1asm.port,
            "clock": inline_q1asm.clock,
            "waveforms": inline_q1asm.waveforms,
            "safe_labels": inline_q1asm.safe_labels,
        }
        super().__init__(
            name=inline_q1asm.name,
            data=info,
            timing=operation_start_time,
        )

    def __str__(self) -> str:
        return f'Assembly "{self.name}" (t0={self.timing}, duration={self.duration})'
