# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Standard control flow operations for use with the quantify_scheduler."""

from __future__ import annotations

from quantify_scheduler.operations.operation import Operation


class Loop(Operation):
    """
    Loop over another operation.

    Cannot be added to Schedule manually, to be used with the ``control_flow`` arg of
    Schedule.add

    Parameters
    ----------
    repetitions : int
        number of repetitions
    t0 : float, optional
        time offset, by default 0
    """

    def __init__(self, repetitions: int, t0: float = 0) -> None:
        super().__init__(name="Loop")
        self.data.update(
            {
                "name": "Loop",
                "control_flow_info": {
                    "t0": t0,
                    "repetitions": repetitions,
                },
            }
        )
        self._update()

    def __str__(self) -> str:
        """
        Represent the Operation as string.

        Returns
        -------
        str
            description

        """
        return self._get_signature(self.data["control_flow_info"])


class Conditional(Operation):
    """
    Conditional operation.

    Cannot be added to Schedule manually, to be used with the `control_flow` arg
    of :meth:`~quantify_scheduler.schedules.schedule.Schedule.add`.

    When passing ``control_flow=Conditional(<qubit_name>)`` to ``Schedule.add``,
    the subschedule will be *conditional*  on ``qubit_name``. In other
    words, if a preceding thresholded acquisition on ``qubit_name`` results in a "1", the
    subschedule will be executed, otherwise it will generate a wait time that is
    equal to the time of the subschedule, to ensure the absolute timing of later
    operations remains consistent.

    Parameters
    ----------
    qubit_name: str
        The name of the qubit to condition on.
    t0 : float, optional
        Time offset, by default 0

    Example
    -------

    A conditional reset can be implemented as follows:

    .. admonition:: example

        # relevant imports
        from quantify_scheduler import Schedule
        from quantify_scheduler.operations.control_flow_library import Conditional
        from quantify_scheduler.operations.gate_library import Measure, X


        # define schedule
        schedule = Schedule("main schedule")

        # define a subschedule containing conditional reset
        conditional_reset = Schedule("conditional reset")
        conditional_reset.add(Measure("q0", feedback_trigger_label="q0"))
        sub_schedule = Schedule("conditional x")
        sub_schedule.add(X("q0"))
        conditional_reset.add(sub_schedule, control_flow=Conditional("q0"))

        # add conditional reset as if it were a gate
        schedule.add(conditional_reset)

    """

    def __init__(self, qubit_name: str, t0: float = 0) -> None:
        class_name = self.__class__.__name__
        super().__init__(name=class_name)
        self.data.update(
            {
                "name": class_name,
                "control_flow_info": {
                    "qubit_name": qubit_name,
                    "t0": t0,
                    "feedback_trigger_label": qubit_name,
                },
            }
        )
        self._update()

    def __str__(self) -> str:
        """
        Represent the Operation as string.

        Returns
        -------
        str
            The string representation of this operation.

        """
        return self._get_signature(self.data["control_flow_info"])
