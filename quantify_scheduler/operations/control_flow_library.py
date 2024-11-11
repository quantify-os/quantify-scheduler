# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Standard control flow operations for use with the quantify_scheduler."""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING

from quantify_scheduler.operations.operation import Operation

if TYPE_CHECKING:
    from quantify_scheduler.schedules.schedule import Schedule


class ControlFlowOperation(Operation, metaclass=ABCMeta):
    """
    Control flow operation that can be used as an ``Operation`` in ``Schedule``.

    This is an abstract class. Each concrete implementation
    of the control flow operation decides how and when
    their ``body`` operation is executed.
    """

    @property
    @abstractmethod
    def body(self) -> Operation | Schedule:
        """Body of a control flow."""
        pass

    @body.setter
    @abstractmethod
    def body(self, value: Operation | Schedule) -> None:
        """Body of a control flow."""
        pass

    def __str__(self) -> str:
        """
        Represent the Operation as a string.

        Returns
        -------
        str
            description

        """
        return self._get_signature(self.data["control_flow_info"])


class LoopOperation(ControlFlowOperation):
    """
    Loop over another operation predefined times.

    Repeats the operation defined in ``body`` ``repetitions`` times.
    The actual implementation depends on the backend.

    Parameters
    ----------
    body
        Operation to be repeated
    repetitions
        Number of repetitions
    t0
        Time offset, by default 0

    """

    def __init__(self, body: Operation | Schedule, repetitions: int, t0: float = 0.0) -> None:
        super().__init__(name="LoopOperation")
        self.data.update(
            {
                "control_flow_info": {
                    "body": body,
                    "repetitions": repetitions,
                    "t0": t0,
                },
            }
        )
        self._update()

    @property
    def body(self) -> Operation | Schedule:
        """Body of a control flow."""
        return self.data["control_flow_info"]["body"]

    @body.setter
    def body(self, value: Operation | Schedule) -> None:
        """Body of a control flow."""
        self.data["control_flow_info"]["body"] = value

    @property
    def duration(self) -> float:
        """Duration of a control flow."""
        return (
            self.data["control_flow_info"]["repetitions"]
            * self.data["control_flow_info"]["body"].duration
        )


class ConditionalOperation(ControlFlowOperation):
    """
    Conditional over another operation.

    If a preceding thresholded acquisition on ``qubit_name`` results in a "1", the
    body will be executed, otherwise it will generate a wait time that is
    equal to the time of the subschedule, to ensure the absolute timing of later
    operations remains consistent.

    Parameters
    ----------
    body
        Operation to be conditionally played
    qubit_name
        Name of the qubit on which the body will be conditioned
    t0
        Time offset, by default 0
    hardware_buffer_time
        Time buffer, by default 0

    Example
    -------

    A conditional reset can be implemented as follows:

    .. admonition:: example

        .. jupyter-execute

            # relevant imports
            from quantify_scheduler import Schedule
            from quantify_scheduler.operations.control_flow_library import Conditional
            from quantify_scheduler.operations.gate_library import Measure, X

            # define conditional reset as a Schedule
            conditional_reset = Schedule("conditional reset")
            conditional_reset.add(Measure("q0", feedback_trigger_label="q0"))
            conditional_reset.add(
                ConditionalOperation(body=X("q0"), qubit_name="q0"),
                rel_time=364e-9,
            )

    .. versionadded:: 0.22.0

        For some hardware specific implementations, a ``hardware_buffer_time``
        might be required to ensure the correct timing of the operations. This will
        be added to the duration of the ``body`` to prevent overlap with other
        operations.

    """

    def __init__(
        self,
        body: Operation | Schedule,
        qubit_name: str,
        t0: float = 0.0,
        hardware_buffer_time: float = 0.0,
    ) -> None:
        super().__init__(name="ConditionalOperation")
        self.data.update(
            {
                "control_flow_info": {
                    "body": body,
                    "qubit_name": qubit_name,
                    "t0": t0,
                    "feedback_trigger_label": qubit_name,
                    "feedback_trigger_address": None,  # Filled in at compilation.
                    "hardware_buffer_time": hardware_buffer_time,
                },
            }
        )
        self._update()

    @property
    def body(self) -> Operation | Schedule:
        """Body of a control flow."""
        return self.data["control_flow_info"]["body"]

    @body.setter
    def body(self, value: Operation | Schedule) -> None:
        """Body of a control flow."""
        self.data["control_flow_info"]["body"] = value

    @property
    def duration(self) -> float:
        """Duration of a control flow."""
        return (
            self.data["control_flow_info"]["body"].duration
            + self.data["control_flow_info"]["hardware_buffer_time"]
        )


class ControlFlowSpec(metaclass=ABCMeta):
    """
    Control flow specification to be used at ``Schedule.add``.

    The users can specify any concrete control flow with
    the ``control_flow`` argument to ``Schedule.add``.
    The ``ControlFlowSpec`` is only a type which by itself
    cannot be used for the ``control_flow`` argument,
    use any concrete control flow derived from it.
    """

    @abstractmethod
    def create_operation(self, body: Operation | Schedule) -> Operation | Schedule:
        """Transform the control flow specification to an operation or schedule."""
        pass


class Loop(ControlFlowSpec):
    """
    Loop control flow specification to be used at ``Schedule.add``.

    For more information, see ``LoopOperation``.

    Parameters
    ----------
    repetitions
        Number of repetitions
    t0
        Time offset, by default 0

    """

    def __init__(self, repetitions: int, t0: float = 0.0) -> None:
        self.repetitions = repetitions
        self.t0 = t0

    def create_operation(self, body: Operation | Schedule) -> LoopOperation:
        """Transform the control flow specification to an operation or schedule."""
        return LoopOperation(body, self.repetitions, self.t0)


class Conditional(ControlFlowSpec):
    """
    Conditional control flow specification to be used at ``Schedule.add``.

    For more information, see ``ConditionalOperation``.

    Parameters
    ----------
    qubit_name
        Number of repetitions
    t0
        Time offset, by default 0

    """

    def __init__(self, qubit_name: str, t0: float = 0.0) -> None:
        self.qubit_name = qubit_name
        self.t0 = t0

    def create_operation(self, body: Operation | Schedule) -> ConditionalOperation:
        """Transform the control flow specification to an operation or schedule."""
        return ConditionalOperation(body, self.qubit_name, self.t0)
