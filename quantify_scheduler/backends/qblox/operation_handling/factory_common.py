# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Functions for producing common operation handling strategies."""

from __future__ import annotations

from typing import TYPE_CHECKING

from quantify_scheduler.backends.qblox.conditional import (
    FeedbackTriggerCondition,
    FeedbackTriggerOperator,
)
from quantify_scheduler.backends.qblox.operation_handling import base, virtual

if TYPE_CHECKING:
    from quantify_scheduler.backends.types.qblox import OpInfo


def try_get_pulse_strategy_common(
    operation_info: OpInfo,
) -> base.IOperationStrategy | None:
    """
    Handles the logic for determining the correct pulse type.

    Returns ``None`` if no matching strategy class is found.
    """
    if operation_info.is_parameter_update:
        return virtual.UpdateParameterStrategy(operation_info)
    elif operation_info.is_loop:
        return virtual.LoopStrategy(operation_info)
    elif (
        feedback_trigger_address := operation_info.data.get("feedback_trigger_address")
    ) is not None:
        trigger_condition = FeedbackTriggerCondition(
            enable=True,
            operator=FeedbackTriggerOperator.OR,
            addresses=[feedback_trigger_address],
        )
        return virtual.ConditionalStrategy(
            operation_info=operation_info, trigger_condition=trigger_condition
        )
    elif operation_info.is_control_flow_end:
        return virtual.ControlFlowReturnStrategy(operation_info)
    elif operation_info.data.get("name") == "LatchReset":
        return virtual.ResetFeedbackTriggersStrategy(operation_info=operation_info)

    return None
