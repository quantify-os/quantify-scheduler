# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Contains the gate library for the Qblox backend."""

from __future__ import annotations

from quantify_scheduler.backends.qblox.constants import TRIGGER_DELAY
from quantify_scheduler.backends.qblox.operations.control_flow_library import (
    ConditionalOperation,
)
from quantify_scheduler.operations.gate_library import Measure, X
from quantify_scheduler.schedules.schedule import Schedule


class ConditionalReset(Schedule):
    r"""
    Reset a qubit to the :math:`|0\rangle` state.

    The
    :class:`~quantify_scheduler.backends.qblox.operations.gate_library.ConditionalReset`
    gate is a conditional gate that first measures the state of the device element using
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

         1) integration time (<device_element>.measure.integration_time)
         2) acquisition delay (<device_element>.measure.acq_delay)
         3) trigger delay (364ns)
         4) pi-pulse duration (<device_element>.rxy.duration)
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
        The name of the device element to reset to the :math:`|0\rangle` state.
    **kwargs:
        Additional keyword arguments are passed to
        :class:`~quantify_scheduler.operations.gate_library.Measure`. e.g.
        ``acq_channel``, ``acq_index``, and ``bin_mode``.

    Examples
    --------

    .. jupyter-execute::
        :hide-output:

        from quantify_scheduler import Schedule
        from quantify_scheduler.qblox.operations import ConditionalReset

        schedule = Schedule("example schedule")
        schedule.add(ConditionalReset("q0"))

    """

    def __init__(
        self,
        qubit_name: str,
        name: str = "conditional_reset",
        **kwargs,
    ) -> None:
        device_element_name = qubit_name
        super().__init__(name)
        self.add(
            Measure(
                device_element_name,
                acq_protocol="ThresholdedAcquisition",
                feedback_trigger_label=device_element_name,
                **kwargs,
            )
        )
        self.add(
            ConditionalOperation(body=X(device_element_name), qubit_name=device_element_name),
            rel_time=TRIGGER_DELAY,
        )
