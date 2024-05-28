# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""
A set of standard schedules used to test compilation of backends.
Note that these are *not* fixtures as we need to be able to pass them to a
pytest.mark.parametrize

The schedules are chosen to cover a range of use-cases that the backends should support.

These are mostly used to assert that the compilers run without errors. These are not
used to pin specific outcomes of the compiler.
"""

import numpy as np

from quantify_scheduler import Schedule
from quantify_scheduler.operations.acquisition_library import SSBIntegrationComplex
from quantify_scheduler.operations.gate_library import (
    CZ,
    Measure,
    Reset,
    Rxy,
    X,
    Y,
)
from quantify_scheduler.operations.pulse_library import (
    DRAGPulse,
    IdlePulse,
    SquarePulse,
)


def single_qubit_schedule_circuit_level() -> Schedule:
    """
    A trivial schedule containing a reset, a gate and a measurement.
    """
    qubit = "q0"
    sched = Schedule("single_qubit_schedule_circuit_level")

    sched.add(Reset(qubit))
    sched.add(X(qubit))
    sched.add(Measure(qubit, acq_index=0))

    sched.add(Reset(qubit))
    sched.add(Y(qubit))
    sched.add(Measure(qubit, acq_index=1))

    return sched


def two_qubit_t1_schedule() -> Schedule:
    q0, q1 = ("q0", "q1")
    repetitions = 1024
    sched = Schedule("two_qubit_t1_schedule", repetitions)

    times = np.arange(0, 60e-6, 3e-6)

    for i, tau in enumerate(times):
        sched.add(Reset(q0, q1), label=f"Reset {i}")
        sched.add(X(q0), label=f"pi {i} {q0}")
        sched.add(X(q1), label=f"pi {i} {q1}", ref_pt="start")

        sched.add(
            Measure(q0, acq_index=i),
            ref_pt="start",
            rel_time=tau,
            label=f"Measurement {q0}{i}",
        )
        sched.add(
            Measure(q1, acq_index=i),
            ref_pt="start",
            rel_time=tau,
            label=f"Measurement {q1}{i}",
        )
    return sched


def two_qubit_schedule_with_edge() -> Schedule:
    q0, q2 = ("q0", "q2")
    repetitions = 1024
    sched = Schedule("two_qubit_schedule_with_edge", repetitions)

    times = np.arange(0, 60e-6, 3e-6)

    for i, tau in enumerate(times):
        sched.add(Reset(q0, q2), label=f"Reset {i}")
        sched.add(X(q0), label=f"pi {i} {q0}")
        sched.add(X(q2), label=f"pi {i} {q2}", ref_pt="start")
        sched.add(CZ(q0, q2))

        sched.add(
            Measure(q0, acq_index=i),
            ref_pt="start",
            rel_time=tau,
            label=f"Measurement {q0}{i}",
        )
        sched.add(
            Measure(q2, acq_index=i),
            ref_pt="start",
            rel_time=tau,
            label=f"Measurement {q2}{i}",
        )
    return sched


def pulse_only_schedule() -> Schedule:
    sched = Schedule(name="pulse_only_schedule", repetitions=1024)

    # these are kind of magic names that are known to exist in the default hardware
    # config.
    port = "q0:res"
    clock = "q0.ro"

    for acq_index in [0, 1, 2]:
        sched.add(IdlePulse(duration=200e-6))
        sched.add(
            SquarePulse(
                duration=20e-9,
                amp=0.3,
                port=port,
                clock=clock,
            ),
        )

        sched.add(
            SSBIntegrationComplex(
                duration=2e-6,
                port=port,
                clock=clock,
                acq_index=acq_index,
                acq_channel=0,
            ),
        )
    return sched


def parametrized_operation_schedule() -> Schedule:
    sched = Schedule("parametrized_operation_schedule")

    qubit = "q0"
    for i, theta in enumerate([0, 45, 90, 1723.435]):
        sched.add(Reset(qubit))
        sched.add(Rxy(theta=theta, phi=0, qubit=qubit))
        sched.add(Measure(qubit, acq_index=i))

    return sched


def hybrid_schedule_rabi() -> Schedule:
    schedule = Schedule("hybrid_schedule_rabi", 8192)

    port = "q0:mw"
    clock = "q0.01"

    for i, amp in enumerate(np.linspace(-1, 1, 11)):
        schedule.add(Reset("q0"), label=f"Reset {i}")
        schedule.add(
            DRAGPulse(
                duration=20e-9,
                G_amp=amp,
                D_amp=0,
                port=port,
                clock=clock,
                phase=0,
            ),
            label=f"Rabi_pulse {i}",
        )
        # N.B. acq_channel is not specified
        schedule.add(Measure("q0", acq_index=i), label=f"Measurement {i}")

    return schedule
