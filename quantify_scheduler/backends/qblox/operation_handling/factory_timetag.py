# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Functions for producing operation handling strategies for the QTM."""
from __future__ import annotations

from typing import TYPE_CHECKING

from quantify_scheduler.backends.qblox.operation_handling import (
    acquisitions,
    base,
    pulses,
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

if TYPE_CHECKING:
    from quantify_scheduler.backends.types.qblox import OpInfo


def get_operation_strategy(
    operation_info: OpInfo,
    channel_name: str,
) -> base.IOperationStrategy:
    """
    Determines and instantiates the correct strategy object.

    Parameters
    ----------
    operation_info
        The operation we are building the strategy for.
    channel_name
        Specifies the channel identifier of the hardware config (e.g. `complex_output_0`).

    Returns
    -------
    :
        The instantiated strategy object.

    """
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

    if protocol in ("TriggerCount", "ThresholdedTriggerCount", "Timetag"):
        return acquisitions.TimetagAcquisitionStrategy(operation_info)

    if protocol in ("Trace", "TimetagTrace"):
        return acquisitions.ScopedTimetagAcquisitionStrategy(operation_info)

    assert False, "This should not be reachable due to the bin mode check above."


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
