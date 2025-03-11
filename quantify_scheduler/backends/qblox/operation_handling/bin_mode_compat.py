# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Functionality to determine if the bin mode is compatible with the acquisition protocol."""
from __future__ import annotations

from typing import TYPE_CHECKING

from quantify_scheduler.enums import BinMode

if TYPE_CHECKING:
    from quantify_scheduler.backends.types.qblox import OpInfo

QRM_COMPATIBLE_BIN_MODES = {
    "SSBIntegrationComplex": {BinMode.APPEND, BinMode.AVERAGE},
    "Trace": {BinMode.AVERAGE},
    "ThresholdedAcquisition": {BinMode.APPEND, BinMode.AVERAGE},
    "WeightedThresholdedAcquisition": {BinMode.APPEND, BinMode.AVERAGE},
    "TriggerCount": {BinMode.APPEND, BinMode.SUM, BinMode.DISTRIBUTION},
    "ThresholdedTriggerCount": {BinMode.APPEND},
    "WeightedIntegratedSeparated": {BinMode.APPEND, BinMode.AVERAGE},
    "NumericalSeparatedWeightedIntegration": {BinMode.APPEND, BinMode.AVERAGE},
    "NumericalWeightedIntegration": {BinMode.APPEND, BinMode.AVERAGE},
}

QTM_COMPATIBLE_BIN_MODES = {
    "TriggerCount": {BinMode.APPEND, BinMode.SUM},
    "ThresholdedTriggerCount": {BinMode.APPEND},
    "Timetag": {BinMode.APPEND, BinMode.AVERAGE},
    "Trace": {BinMode.FIRST},
    "TimetagTrace": {BinMode.APPEND},
}


class IncompatibleBinModeError(Exception):
    """
    Compiler exception to be raised when a bin mode is incomatible with the acquisition protocol for
    the module type.
    """

    def __init__(
        self,
        module_type: str,
        protocol: str,
        bin_mode: BinMode,
        operation_info: OpInfo | None = None,
    ) -> None:
        err_msg = (
            f"{protocol} acquisition on the {module_type} does not support bin mode {bin_mode}."
        )
        if operation_info:
            err_msg += f"\n\n{repr(operation_info)} caused this exception to occur."
        super().__init__(err_msg)
