# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Exceptions used by Qblox backend."""


from typing import Literal

from quantify_scheduler.backends.types.qblox import OpInfo


class NcoOperationTimingError(ValueError):
    """Exception thrown if there are timing errors for NCO operations."""


class FineDelayTimingError(ValueError):
    """
    Exception thrown if there are timing errors for fine delays.

    Note that all units are in picoseconds.
    """

    def __init__(
        self,
        error_type: Literal["within_op", "between_op"],
        operation_info: OpInfo,
        fine_start_delay: int,
        fine_end_delay: int,
        operation_start_time: int,
        operation_duration: int,
        last_digital_pulse_end_ps: int,
    ) -> None:
        if error_type == "within_op":
            message = (
                f"Operation {operation_info} has fine delay specifications "
                f"that are not supported by the hardware: {fine_start_delay=} "
                f"ps, {fine_end_delay=} ps and the duration of the operation "
                f"is {operation_duration//1000} ns. To avoid undefined "
                "behaviour, there must be at least 7ns between the start and "
                "end of the operation including the fine delay, OR the time "
                "between the start and end must be an integer number of "
                "nanoseconds."
            )
        else:
            message = (
                f"Operation {operation_info} is started too soon after "
                "another operation with fine delay. This operation starts at "
                f"{operation_start_time + fine_start_delay} ps including fine "
                "delay, while the previous operation with fine delay ended at "
                f"{last_digital_pulse_end_ps} ps. To avoid undefined "
                "behaviour, there must be at least 7ns between the end of the "
                "previous operation and the start of this one including the "
                "fine delay, OR the time between the end and start must be an "
                "integer number of nanoseconds."
            )
        super().__init__(message)
