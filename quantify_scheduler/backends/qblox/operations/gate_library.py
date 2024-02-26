# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Contains the gate library for the Qblox backend."""
from __future__ import annotations

import warnings

from quantify_scheduler.backends.qblox.constants import TRIGGER_DELAY
from quantify_scheduler.operations.control_flow_library import Conditional
from quantify_scheduler.operations.gate_library import Measure, X
from quantify_scheduler.schedules.schedule import Schedule


class ConditionalReset(Schedule):
    r"""
    Reset a qubit to the :math:`|0\rangle` state.

    The
    :class:`~quantify_scheduler.backends.qblox.operations.gate_library.ConditionalReset`
    gate is a conditional gate that first measures the state of the qubit using
    an
    :class:`~quantify_scheduler.operations.acquisition_library.ThresholdedAcquisition`
    operation and then performs a :math:`\pi` rotation on the condition that the
    measured state is :math:`|1\rangle`. If the measured state is in
    :math:`|0\rangle`, the hardware will wait the same amount of time the
    :math:`\pi` rotation would've taken to ensure that total execution time of
    :class:`~quantify_scheduler.backends.qblox.operations.gate_library.ConditionalReset`
    is the same regardless of the measured state.

    .. note::

        The total time of the the ConditionalReset is the sum of

         1) integration time (<qubit>.measure.integration_time)
         2) acquisition delay (<qubit>.measure.acq_delay)
         3) trigger delay (364ns)
         4) pi-pulse duration (<qubit>.rxy.duration)
         5) idle time (4ns)

    .. note::

        Due to current hardware limitations, overlapping conditional resets
        might not work correctly if multiple triggers are sent within a 364ns
        window. See :ref:`sec-qblox-conditional-playback` for more information.

    .. note::

        :class:`~quantify_scheduler.backends.qblox.operations.gate_library.ConditionalReset`
        is currently implemented as a subschedule, but can be added to an
        existing schedule as if it were a gate. See examples below.

    Parameters
    ----------
    name : str
        The name of the conditional subschedule, by default "conditional_reset".
    qubit_name : str
        The name of the qubit to reset to the :math:`|0\rangle` state.
    **kwargs:
        Additional keyword arguments are passed to
        :class:`~quantify_scheduler.operations.gate_library.Measure`. e.g.
        ``acq_channel``, ``acq_index``, and ``bin_mode``.

    Examples
    --------
    .. admonition:: Examples

        .. jupyter-execute::
            :hide-output:

            from quantify_scheduler.backends.qblox.operations.gate_library import ConditionalReset
            from quantify_scheduler.schedules.schedule import Schedule

            schedule = Schedule("example schedule")
            schedule.add(ConditionalReset("q0"))

    """

    def __init__(
        self,
        qubit_name: str,
        name: str = "conditional_reset",
        **kwargs,  # noqa: ANN003 (kwargs not annotated)
    ) -> None:
        super().__init__(name)
        # `control_flow` operations warn the user about it being an experimental
        # features and subject to potential interface changes. Silencing here,
        # because the control flow interface is hidden from the user.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", ".*experimental feature.*", UserWarning)
            self.add(
                Measure(
                    qubit_name,
                    acq_protocol="ThresholdedAcquisition",
                    feedback_trigger_label=qubit_name,
                    **kwargs,
                )
            )
            sub_schedule = Schedule("")
            sub_schedule.add(X(qubit_name))
            self.add(
                sub_schedule,
                control_flow=Conditional(qubit_name),
                rel_time=TRIGGER_DELAY,
            )
