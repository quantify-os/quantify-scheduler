# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Functions for producing operation handling strategies."""

from __future__ import annotations

from quantify_scheduler.enums import BinMode

from quantify_scheduler.backends.types.qblox import OpInfo
from quantify_scheduler.backends.qblox.operation_handling import (
    base,
    pulses,
    acquisitions,
    virtual,
)


def get_operation_strategy(
    operation: OpInfo, instruction_generated_pulses_enabled: bool, output_mode: str
) -> base.IOperationStrategy:
    """
    Determines and instantiates the correct strategy object.

    Parameters
    ----------
    operation
        The operation we are building the strategy for.
    instruction_generated_pulses_enabled
        Specifies if instruction generated pulses (e.g. staircase through offsets) are
        allowed. If set to False, only generically treated pulses are allowed.
    output_mode
        Either "real", "imag" or complex depending on whether the signal affects only
        path0, path1 or both.

    Returns
    -------
    :
        The instantiated strategy object.
    """
    if operation.data["port"] is None:
        if operation.name == "ShiftClockPhase":
            return virtual.NcoPhaseShiftStrategy(operation)
        return virtual.IdleStrategy(operation)

    if operation.is_acquisition:
        return _get_acquisition_strategy(operation)
    return _get_pulse_strategy(
        operation, instruction_generated_pulses_enabled, output_mode
    )


def _get_acquisition_strategy(
    operation: OpInfo,
) -> acquisitions.AcquisitionStrategyPartial:
    """Handles the logic for determining the correct acquisition type."""
    protocol = operation.data["protocol"]
    if protocol in ("trace", "ssb_integration_complex"):
        if protocol == "trace" and operation.data["bin_mode"] == BinMode.APPEND.value:
            raise ValueError(
                f"Trace acquisition does not support APPEND bin mode.\n\n"
                f"{repr(operation)} caused this exception to occur."
            )
        return acquisitions.SquareAcquisitionStrategy(operation)
    if protocol == "weighted_integrated_complex":
        return acquisitions.WeightedAcquisitionStrategy(operation)
    raise ValueError(
        f'Unknown acquisition protocol "{protocol}" encountered in '
        f"Qblox backend when processing acquisition {repr(operation)}."
    )


def _get_pulse_strategy(
    operation: OpInfo, instruction_generated_pulses_enabled: bool, output_mode: str
) -> base.IOperationStrategy:
    """Handles the logic for determining the correct pulse type."""
    if instruction_generated_pulses_enabled:
        wf_func = operation.data["wf_func"]
        if wf_func == "quantify_scheduler.waveforms.square":
            return pulses.StitchedSquarePulseStrategy(operation, output_mode)
        if wf_func == "quantify_scheduler.waveforms.staircase":
            return pulses.StaircasePulseStrategy(operation, output_mode)
    return pulses.GenericPulseStrategy(operation, output_mode)
