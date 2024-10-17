# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Module containing schedules for common two qubit experiments (transmon)."""
from __future__ import annotations

import numpy as np

from quantify_scheduler.operations.gate_library import Measure, Reset, X
from quantify_scheduler.operations.pulse_library import SquarePulse
from quantify_scheduler.schedules.schedule import Schedule


def chevron_cz_sched(
    lf_qubit: str,
    hf_qubit: str,
    amplitudes: float | np.ndarray,
    duration: float,
    flux_port: str | None = None,
    repetitions: int = 1,
) -> Schedule:
    r"""
    Chevron CZ calibration schedule that measures coupling of a qubit pair.

    This experiment provides information about the location
    of the :math:`|11\rangle \leftrightarrow |02\rangle` avoided crossing and
    distortions in the flux-control line.

    .. admonition:: Schedule sequence
        :class: tip

        .. jupyter-execute::

                from quantify_scheduler.schedules.two_qubit_transmon_schedules import (
                    chevron_cz_sched
                )

                sched = chevron_cz_sched(
                    lf_qubit="q0",
                    hf_qubit="q1",
                    amplitudes=0.5,
                    duration=20e-9,
                )

                sched.plot_circuit_diagram();

    .. note::
        This schedule uses a unipolar square flux pulse, which will cause
        distortions and leakage. For a high quality CZ
        gate, distortions should be corrected for by modelling and
        subsequently inverting the transfer function of the
        flux-control line.
        See e.g. :cite:t:`Jerger2019` or :cite:t:`Rol2020`
        for more information.

    Parameters
    ----------
    lf_qubit
        The name of a qubit, e.g., "q0", the qubit with lower frequency.
    hf_qubit
        The name of coupled qubit, the qubit with the higher frequency.
    amplitudes
        An array (or scalar) of the flux pulse amplitude(s) in V.
    duration
        A scalar specifying the flux pulse duration in s.
    flux_port
        An optional string for a flux port. If ``None``, this will default to
        the ``hf_qubit`` flux port (``"{hf_qubit}:fl"``).
    repetitions
        The amount of times the Schedule will be repeated.

    Returns
    -------
    :
        An experiment schedule.

    """
    sched = Schedule("Two-qubit Chevron CZ schedule", repetitions)

    # Ensure amplitudes is an iterable when passing a float
    amplitudes = np.asarray(amplitudes)
    amplitudes = amplitudes.reshape(amplitudes.shape or (1,))

    # Set flux port
    flux_port = flux_port if flux_port is not None else f"{hf_qubit}:fl"

    for acq_index, amp in enumerate(amplitudes):
        # Reset to |00>
        sched.add(Reset(lf_qubit, hf_qubit), label=f"Reset {acq_index}")

        # Prepare |11>
        excite_lf = sched.add(X(lf_qubit), label=f"X({lf_qubit}) {acq_index}")
        sched.add(
            X(hf_qubit),
            ref_op=excite_lf,
            ref_pt="start",
            label=f"X({hf_qubit}) {acq_index}",
        )

        # Go to |11> <=> |02> avoided crossing and come back
        sched.add(
            SquarePulse(
                amp=amp,
                duration=duration,
                port=flux_port,
                clock="cl0.baseband",
            ),
            label=f"SquarePulse({flux_port}) {acq_index}",
        )

        # Measure system
        sched.add(
            Measure(lf_qubit, hf_qubit, acq_index=acq_index),
            label=f"Measure({lf_qubit},{hf_qubit}) {acq_index}",
        )

    return sched
