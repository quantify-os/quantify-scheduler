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
