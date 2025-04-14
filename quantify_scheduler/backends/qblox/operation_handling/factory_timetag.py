# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Functions for producing operation handling strategies for the QTM."""

from __future__ import annotations

from typing import TYPE_CHECKING

from quantify_scheduler.backends.qblox.operation_handling import (
    acquisitions,
    base,
    pulses,
    q1asm_injection_strategy,
    virtual,
)
from quantify_scheduler.backends.qblox.operation_handling.bin_mode_compat import (
    QTM_COMPATIBLE_BIN_MODES,
)
from quantify_scheduler.backends.qblox.operation_handling.factory_analog import (
    IncompatibleBinModeError,
)
from quantify_scheduler.backends.qblox.operation_handling.factory_common import (
    try_get_pulse_strategy_common,
)
from quantify_scheduler.backends.qblox.operations.inline_q1asm import Q1ASMOpInfo

if TYPE_CHECKING:
    from quantify_scheduler.backends.types.qblox import OpInfo


def get_operation_strategy(
    operation_info: OpInfo,
    channel_name: str,
) -> base.IOperationStrategy:
    """
    Determine and instantiate the correct operation strategy object.

    Parameters
    ----------
    operation_info
        The operation for which we are building the strategy. This object
        contains all the necessary information about the operation.
    channel_name
        Specifies the channel identifier of the hardware config (e.g. 'complex_output_0').

    Returns
    -------
    :
        The instantiated strategy object that implements the IOperationStrategy interface.
        This could be a Q1ASMInjectionStrategy, an acquisition strategy, a pulse strategy,
        or other specialized strategies depending on the operation type.

    Raises
    ------
    ValueError
        If the operation cannot be compiled for the target hardware
        or if an unsupported operation type is encountered.

    """
    if isinstance(operation_info, Q1ASMOpInfo):
        return q1asm_injection_strategy.Q1ASMInjectionStrategy(operation_info)

    if operation_info.is_acquisition:
        return _get_acquisition_strategy(operation_info)

    return _get_pulse_strategy(
        operation_info=operation_info,
        channel_name=channel_name,
    )


def _get_acquisition_strategy(
    operation_info: OpInfo,
) -> acquisitions.AcquisitionStrategyPartial:
    """Handles the logic for determining the correct acquisition type."""
    protocol = operation_info.data["protocol"]
    try:
        compatible_bin_modes = QTM_COMPATIBLE_BIN_MODES[protocol]
    except KeyError as err:
        raise ValueError(f"Operation info {operation_info} cannot be compiled for a QTM.") from err
    if operation_info.data["bin_mode"] not in compatible_bin_modes:
        raise IncompatibleBinModeError(
            module_type="QTM",
            protocol=protocol,
            bin_mode=operation_info.data["bin_mode"],
            operation_info=operation_info,
        )

    if protocol in (
        "DualThresholdedTriggerCount",
        "TriggerCount",
        "ThresholdedTriggerCount",
        "Timetag",
    ):
        return acquisitions.TimetagAcquisitionStrategy(operation_info)

    if protocol in ("Trace", "TimetagTrace"):
        return acquisitions.ScopedTimetagAcquisitionStrategy(operation_info)

    raise AssertionError("This should not be reachable due to the bin mode check above.")


def _get_pulse_strategy(
    operation_info: OpInfo,
    channel_name: str,
) -> base.IOperationStrategy:
    """Handles the logic for determining the correct pulse type."""
    if (strategy := try_get_pulse_strategy_common(operation_info)) is not None:
        return strategy
    elif operation_info.data["port"] is None:
        return virtual.IdleStrategy(operation_info)

    elif operation_info.data.get("marker_pulse", False):
        return pulses.DigitalPulseStrategy(
            operation_info=operation_info,
            channel_name=channel_name,
        )
    elif operation_info.data.get("timestamp", False):
        return virtual.TimestampStrategy(
            operation_info=operation_info,
        )

    raise ValueError(f"Operation info {operation_info} cannot be compiled for a QTM.")
