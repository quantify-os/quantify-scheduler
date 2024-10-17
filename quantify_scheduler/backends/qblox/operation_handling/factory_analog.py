# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Functions for producing operation handling strategies for QCM/QRM modules."""

from __future__ import annotations

from typing import TYPE_CHECKING

from quantify_scheduler.backends.qblox.operation_handling import (
    acquisitions,
    base,
    pulses,
    virtual,
)
from quantify_scheduler.backends.qblox.operation_handling.factory_common import (
    try_get_pulse_strategy_common,
)
from quantify_scheduler.enums import BinMode

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
    if protocol in ("Trace", "SSBIntegrationComplex", "ThresholdedAcquisition"):
        if protocol == "Trace" and operation_info.data["bin_mode"] != BinMode.AVERAGE:
            raise ValueError(
                f"{protocol} acquisition does not support bin mode "
                f"{operation_info.data['bin_mode']}.\n\n{repr(operation_info)} caused "
                "this exception to occur."
            )
        return acquisitions.SquareAcquisitionStrategy(operation_info)

    elif protocol in (
        "WeightedIntegratedSeparated",
        "NumericalSeparatedWeightedIntegration",
        "NumericalWeightedIntegration",
    ):
        return acquisitions.WeightedAcquisitionStrategy(operation_info)

    elif protocol == "TriggerCount":
        return acquisitions.TriggerCountAcquisitionStrategy(operation_info)

    raise ValueError(
        f'Unknown acquisition protocol "{protocol}" encountered in '
        f"Qblox backend when processing acquisition {repr(operation_info)}."
    )


def _get_pulse_strategy(  # noqa: PLR0911  # too many return statements
    operation_info: OpInfo,
    channel_name: str,
) -> base.IOperationStrategy:
    """Handles the logic for determining the correct pulse type."""
    if operation_info.is_offset_instruction:
        return virtual.AwgOffsetStrategy(operation_info)
    elif (strategy := try_get_pulse_strategy_common(operation_info)) is not None:
        return strategy
    elif operation_info.data["port"] is None:
        if "phase_shift" in operation_info.data:
            return virtual.NcoPhaseShiftStrategy(operation_info)
        elif "reset_clock_phase" in operation_info.data:
            return virtual.NcoResetClockPhaseStrategy(operation_info)
        elif "clock_freq_new" in operation_info.data:
            return virtual.NcoSetClockFrequencyStrategy(operation_info)
        else:
            return virtual.IdleStrategy(operation_info)

    elif operation_info.data.get("marker_pulse", False):
        return pulses.MarkerPulseStrategy(
            operation_info=operation_info,
            channel_name=channel_name,
        )

    return pulses.GenericPulseStrategy(
        operation_info=operation_info,
        channel_name=channel_name,
    )
