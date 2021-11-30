from __future__ import annotations

from typing import Dict, Any

from quantify_scheduler.enums import BinMode

from quantify_scheduler.backends.types.qblox import OpInfo
from quantify_scheduler.backends.qblox.operation_handling import (
    base,
    pulses,
    acquisitions,
)


def get_operation_strategy(
    operation: OpInfo, force_generic: bool, output_mode: str
) -> base.IOperationStrategy:
    if operation.is_acquisition:
        return _get_acquisition_strategy(operation)
    return _get_pulse_strategy(operation, force_generic, output_mode)


def _get_acquisition_strategy(
    operation: OpInfo,
) -> acquisitions.AcquisitionStrategyPartial:
    protocol = operation.data["protocol"]
    if protocol in ("trace", "ssb_integration_complex"):
        if protocol == "trace" and operation.data["bin_mode"] == BinMode.APPEND.value:
            raise ValueError(
                f"Trace acquisition does not support APPEND bin mode.\n\n"
                f"Acquisition {repr(operation)} caused this exception to occur."
            )
        return acquisitions.SquareAcquisitionStrategy(operation)
    elif protocol == "weighted_integrated_complex":
        return acquisitions.WeightedAcquisitionStrategy(operation)
    raise ValueError(
        f'Unknown acquisition protocol "{protocol}" encountered in '
        f"Qblox backend when processing acquisition {repr(operation)}."
    )


def _get_pulse_strategy(
    operation: OpInfo, instruction_generated_pulses_enabled: bool, output_mode: str
) -> base.IOperationStrategy:
    if instruction_generated_pulses_enabled:
        wf_func = operation.data["wf_func"]
        if wf_func == "quantify_scheduler.waveforms.square":
            return pulses.StitchedSquarePulseStrategy(operation, output_mode)
        elif wf_func == "quantify_scheduler.waveforms.staircase":
            return pulses.StaircasePulseStrategy(operation, output_mode)
    return pulses.GenericPulseStrategy(operation, output_mode)
