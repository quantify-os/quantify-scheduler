# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Functions for producing operation handling strategies for QCM/QRM modules."""

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
    QRM_COMPATIBLE_BIN_MODES,
    IncompatibleBinModeError,
)
from quantify_scheduler.backends.qblox.operation_handling.factory_common import (
    try_get_pulse_strategy_common,
)
from quantify_scheduler.backends.qblox.operations.inline_q1asm import Q1ASMOpInfo

if TYPE_CHECKING:
    from quantify_scheduler.backends.types.qblox import ClusterModuleDescription, OpInfo


def get_operation_strategy(
    operation_info: OpInfo, channel_name: str, module_options: ClusterModuleDescription
) -> base.IOperationStrategy:
    """
    Determine and instantiate the correct operation strategy object.

    Parameters
    ----------
    operation_info
        The operation for which we are building the strategy.
        This object contains all the necessary information about the operation.
    channel_name
        Specifies the channel identifier of the hardware config (e.g. `complex_output_0`).
    module_options
        The module description the operation will run on

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
        operation_info=operation_info, channel_name=channel_name, module_options=module_options
    )


def _get_acquisition_strategy(
    operation_info: OpInfo,
) -> acquisitions.AcquisitionStrategyPartial:
    """Handles the logic for determining the correct acquisition type."""
    protocol = operation_info.data["protocol"]
    try:
        compatible_bin_modes = QRM_COMPATIBLE_BIN_MODES[protocol]
    except KeyError as err:
        raise ValueError(
            f'Unknown acquisition protocol "{protocol}" encountered in '
            f"Qblox backend when processing acquisition {operation_info!r}."
        ) from err
    if operation_info.data["bin_mode"] not in compatible_bin_modes:
        raise IncompatibleBinModeError(
            module_type="QRM",
            protocol=protocol,
            bin_mode=operation_info.data["bin_mode"],
            operation_info=operation_info,
        )

    if protocol in ("Trace", "SSBIntegrationComplex", "ThresholdedAcquisition"):
        return acquisitions.SquareAcquisitionStrategy(operation_info)

    elif protocol in (
        "WeightedIntegratedSeparated",
        "NumericalSeparatedWeightedIntegration",
        "NumericalWeightedIntegration",
        "WeightedThresholdedAcquisition",
    ):
        return acquisitions.WeightedAcquisitionStrategy(operation_info)

    elif protocol in ("TriggerCount", "ThresholdedTriggerCount"):
        return acquisitions.TriggerCountAcquisitionStrategy(operation_info)

    raise AssertionError("This should not be reachable due to the bin mode check above.")


def _get_pulse_strategy(  # noqa: PLR0911  # too many return statements
    operation_info: OpInfo, channel_name: str, module_options: ClusterModuleDescription
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
            operation_info=operation_info, channel_name=channel_name, module_options=module_options
        )

    return pulses.GenericPulseStrategy(
        operation_info=operation_info,
        channel_name=channel_name,
    )
