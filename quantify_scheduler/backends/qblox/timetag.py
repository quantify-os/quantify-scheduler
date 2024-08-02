# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Utilty classes for Qblox timetag module."""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

from quantify_scheduler.backends.qblox import constants, q1asm_instructions
from quantify_scheduler.backends.qblox.compiler_abc import SequencerCompiler
from quantify_scheduler.backends.qblox.enums import TimetagTraceType
from quantify_scheduler.backends.qblox.operation_handling.acquisitions import (
    TimetagAcquisitionStrategy,
)
from quantify_scheduler.backends.qblox.operation_handling.factory_timetag import (
    get_operation_strategy,
)
from quantify_scheduler.backends.qblox.operation_handling.virtual import (
    TimestampStrategy,
)
from quantify_scheduler.enums import TimeRef

if TYPE_CHECKING:
    from quantify_scheduler.backends.qblox.instrument_compilers import (
        QTMCompiler,
    )
    from quantify_scheduler.backends.qblox.operation_handling.base import (
        IOperationStrategy,
    )
    from quantify_scheduler.backends.qblox.qasm_program import QASMProgram
    from quantify_scheduler.backends.qblox_backend import _SequencerCompilationConfig
    from quantify_scheduler.backends.types.qblox import (
        OpInfo,
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
    static_hw_properties
        The static properties of the hardware. This effectively gathers all the
        differences between the different modules.
    settings
        The settings set to this sequencer.
    sequencer_cfg
        The instrument compiler config associated to this device.
    """

    def __init__(
        self,
        parent: QTMCompiler,
        index: int,
        static_hw_properties: StaticHardwareProperties,
        settings: TimetagSequencerSettings,
        sequencer_cfg: _SequencerCompilationConfig,
    ) -> None:
        self._settings: TimetagSequencerSettings  # Help the type checker
        super().__init__(
            parent=parent,
            index=index,
            static_hw_properties=static_hw_properties,
            settings=settings,
            sequencer_cfg=sequencer_cfg,
        )

    def prepare(self) -> None:
        """
        Perform necessary operations on this sequencer's data before
        :meth:`~quantify_scheduler.backends.qblox.compiler_abc.SequencerCompiler.compile`
        is called.
        """
        super().prepare()
        self._assert_correct_time_ref_used_with_timestamp()

    def _assert_correct_time_ref_used_with_timestamp(self) -> None:
        """
        Assert that the Timestamp operation is present if the user specified the
        appropriate argument for the Timetag acquisition, or vice-versa that there is no
        Timestamp operation present if the user specified another time reference.

        Warn if this is not the case.
        """
        # It is enough to check the first acquisition, since it is enforced in
        # _prepare_acq_settings that all timetag acquisitions have the same time
        # reference.
        try:
            first_timetag = next(
                op
                for op in self.op_strategies
                if isinstance(op, TimetagAcquisitionStrategy)
                and op.operation_info.data["protocol"] == "Timetag"
            )
        except StopIteration:
            # Early return because Timetag is not used.
            return

        timestamp_in_arg = (
            first_timetag.operation_info.data["time_ref"] == TimeRef.TIMESTAMP
        )
        timestamp_operation_present = any(
            op for op in self.op_strategies if isinstance(op, TimestampStrategy)
        )

        if timestamp_in_arg and not timestamp_operation_present:
            warnings.warn(
                "A Timetag acquisition was scheduled with argument 'time_ref="
                f"TimeRef.TIMESTAMP' on port '{self.port}' and clock '{self.clock}', "
                "but no Timestamp operation was found with the same port and clock.",
                UserWarning,
            )
        if not timestamp_in_arg and timestamp_operation_present:
            warnings.warn(
                f"A Timestamp operation was found on port '{self.port}' and clock "
                f"'{self.clock}', but no Timetag acquisition was scheduled with "
                "argument 'time_ref=TimeRef.TIMESTAMP'.",
                UserWarning,
            )

    def get_operation_strategy(
        self,
        operation_info: OpInfo,
    ) -> IOperationStrategy:
        """
        Determines and instantiates the correct strategy object.

        Parameters
        ----------
        operation_info
            The operation we are building the strategy for.

        Returns
        -------
        :
            The instantiated strategy object.
        """
        return get_operation_strategy(operation_info, self.settings.channel_name)

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

        def assert_all_op_info_values_equal(key: str) -> None:
            unique_op_infos = set(acq.operation_info.data[key] for acq in acquisitions)
            if len(unique_op_infos) != 1:
                raise ValueError(
                    f"{key} must be the same for all acquisitions on a port-clock "
                    "combination."
                )

        if acq_metadata.acq_protocol == "Trace":
            self._settings.scope_trace_type = TimetagTraceType.SCOPE
            # Trace acquisitions have bin mode FIRST, meaning only the first acquisition
            # has any effect. Therefore we take the duration from the first acquisition.
            self._settings.trace_acq_duration = round(
                acquisitions[0].operation_info.duration * 1e9
            )
        elif acq_metadata.acq_protocol == "TimetagTrace":
            self._settings.scope_trace_type = TimetagTraceType.TIMETAG

        if acq_metadata.acq_protocol in ("Timetag", "TimetagTrace"):
            assert_all_op_info_values_equal("time_source")
            assert_all_op_info_values_equal("time_ref")
            self._settings.time_source = acquisitions[0].operation_info.data[
                "time_source"
            ]
            self._settings.time_ref = acquisitions[0].operation_info.data["time_ref"]

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

    def _generate_acq_declaration_dict(
        self,
        repetitions: int,
        acq_metadata: AcquisitionMetadata,
    ) -> dict[str, Any]:
        """
        Generates the "acquisitions" entry of the program json. It contains declaration
        of the acquisitions along with the number of bins and the corresponding index.

        Overrides the superclass implementation to check additionally that only one
        acquisition channel is used if Trace or TimetagTrace acquisitions are present.

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
        # This restriction is necessary because there will be only one set of trace data
        # per sequencer, regardless of acquisition channels.
        if (
            acq_metadata.acq_protocol in ("Trace", "TimetagTrace")
            and len(acq_metadata.acq_channels_metadata) > 1
        ):
            raise RuntimeError(
                "Only one acquisition channel per port-clock can be specified, if the "
                f"{acq_metadata.acq_protocol} acquisition protocol is used.\n"
                "Acquisition channels "
                f"{list(acq_metadata.acq_channels_metadata.keys())} were "
                f"found on port-clock {self.port}-{self.clock}."
            )
        return super()._generate_acq_declaration_dict(repetitions, acq_metadata)
