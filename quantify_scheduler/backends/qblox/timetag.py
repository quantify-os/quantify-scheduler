# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Utilty classes for Qblox timetag module."""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from quantify_scheduler.backends.qblox import constants, q1asm_instructions
from quantify_scheduler.backends.qblox.compiler_abc import SequencerCompiler

if TYPE_CHECKING:
    from quantify_scheduler.backends.qblox.instrument_compilers import (
        TimetagModuleCompiler,
    )
    from quantify_scheduler.backends.qblox.operation_handling.base import (
        IOperationStrategy,
    )
    from quantify_scheduler.backends.qblox.qasm_program import QASMProgram
    from quantify_scheduler.backends.types.qblox import (
        StaticHardwareProperties,
        TimetagSequencerSettings,
    )
    from quantify_scheduler.schedules.schedule import AcquisitionMetadata


class TimetagSequencerCompiler(SequencerCompiler):
    """
    Class that performs the compilation steps on the sequencer level, for the QTM.

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
        parent: TimetagModuleCompiler,
        index: int,
        portclock: tuple[str, str],
        static_hw_properties: StaticHardwareProperties,
        settings: TimetagSequencerSettings,
        latency_corrections: dict[str, float],
        qasm_hook_func: Callable | None = None,
    ) -> None:
        super().__init__(
            parent=parent,
            index=index,
            portclock=portclock,
            static_hw_properties=static_hw_properties,
            settings=settings,
            latency_corrections=latency_corrections,
            qasm_hook_func=qasm_hook_func,
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
        # Work in progress: there are no acquisitions for the QTM yet.

    def _write_pre_wait_sync_instructions(self, qasm: QASMProgram) -> None:
        """
        Write instructions to the QASM program that must come before the first wait_sync.

        The duration must be equal for all module types.
        """
        # No pre-wait_sync instructions.

    def _write_repetition_loop_header(self, qasm: QASMProgram) -> None:
        """
        Write the Q1ASM that should appear at the start of the repetition loop.

        The duration must be equal for all module types.
        """
        qasm.emit(q1asm_instructions.WAIT, constants.MIN_TIME_BETWEEN_OPERATIONS)

    def _insert_qasm(
        self, op_strategy: IOperationStrategy, qasm_program: QASMProgram
    ) -> None:
        """Get Q1ASM instruction(s) from ``op_strategy`` and insert them into ``qasm_program``."""
        op_strategy.insert_qasm(qasm_program)
