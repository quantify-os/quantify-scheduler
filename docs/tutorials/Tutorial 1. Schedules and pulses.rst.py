# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.12.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [raw]
#  .. _sec-tutorial1:
#
# Tutorial 1. Schedules and pulses
# ================================
#
#  .. jupyter-kernel::
#    :id: Tutorial 1. Schedules and pulses
#
# .. seealso::
#
#     The complete source code of this tutorial can be found in
#
#     :jupyter-download:notebook:`Tutorial 1. Schedules and pulses`
#
#     :jupyter-download:script:`Tutorial 1. Schedules and pulses`

# %% [raw]
# The Schedule
# ------------
#
# The main data structure that describes an experiment in the `quantify_scheduler` is the Schedule. We will show how the Schedule works through an example.

# %%
from quantify_scheduler import Schedule

sched = Schedule("Hello quantum world!")

sched

# %% [raw]
# As we can see, our newly created schedule is still empty. We need to manually add operations to it. In `quantify_scheduler` there are three types of operations: pulses, acquisitions and gates. All of these have explicit timing control. In this tutorial we will only cover pulses. The goal will not be to make a schedule that is physically meaningful, but to demonstrate the control over the scheduling to its fullest.
#
# While it is possible to define a pulse completely from scratch, we will be using some of the pulse definitions provided with the `quantify_scheduler`. These pulses are described in the `pulse_library`. It's worth noting that no sampling of the data yet occurs at this stage, but the pulse is kept in a parameterized form.
#
# We will add a square pulse from the pulse library to the schedule.

# %%
from quantify_scheduler.operations import pulse_library

square_pulse = sched.add(
    pulse_library.SquarePulse(amp=1, duration=1e-6, port="q0:res", clock="q0.ro")
)

sched

# %% [raw]
# You may have noticed that we passed a port and a clock to the pulse. The port specifies the physical location on the quantum chip to which we are sending the pulses, whilst the clock tracks the frequency of the signal. This clock frequency has not yet been defined, so prior to any compilation step this clock needs to be added to the schedule as a resource.

# %%
from quantify_scheduler.resources import ClockResource

readout_clock = ClockResource(name="q0.ro", freq=7e9)
sched.add_resource(readout_clock)

sched

# %% [raw]
# `quantify_scheduler` provides several visualization tools to show a visual representation of the schedule we made. First, however, we need to instruct the scheduler to calculate the pulse timings. We can accomplish this using the `determine_absolute_timing` function. In the cell below we call this function and draw the schedule.
#
# Note that these plots are interactive and modulation is not shown by default.

# %%
from quantify_scheduler import compilation
from quantify_scheduler.visualization.pulse_diagram import pulse_diagram_plotly

compilation.determine_absolute_timing(sched)
pulse_diagram_plotly(sched)

# %% [raw]
# Explicit timing control
# -----------------------
#
# What we see in the pulse diagram is only a flat line, corresponding to our single square pulse. To make our schedule more interesting, we should add more pulses to it. We will add another square pulse, but with a 500 ns delay.

# %%
sched.add(
    pulse_library.SquarePulse(amp=1, duration=1e-6, port="q0:res", clock="q0.ro"),
    ref_op=square_pulse,
    rel_time=500e-9,
)

compilation.determine_absolute_timing(sched)
pulse_diagram_plotly(sched)

# %% [raw]
# We can see that `rel_time=500e-9` schedules the pulse 500 ns shifted relative to the end of the `ref_op`. If no additional arguments are passed, operations are added directly after the operation that was added last.
#
# Let's now instead align a pulse to start at the same time as the first square pulse. Before, we specified the timing relative to the end of a different pulse, but we can choose to instead specify it relative to the beginning. This is done by passing `ref_pt='start'`.

# %%
sched.add(
    pulse_library.DRAGPulse(
        G_amp=0.5, D_amp=0.5, duration=1e-6, phase=0, port="q0:mw", clock="q0.01"
    ),
    ref_op=square_pulse,
    ref_pt="start",
)
sched.add_resource(ClockResource(name="q0.01", freq=7e9))

compilation.determine_absolute_timing(sched)
pulse_diagram_plotly(sched)

# %% [raw]
# We see that we added a DRAG pulse to the schedule. Two things stand out:
#
# 1. The DRAG pulse is plotted seperately from the square pulse, this is because the we specified a different port for this pulse than we did for the square pulse.
# 2. The DRAG pulse shows two lines instead of one. This is because a DRAG pulse is specified as a complex-valued pulse, so we have to plot both the I and Q components of the signal. The real part of the waveform is shown in color, whereas the imaginary component is shown in greyscale.

# %% [raw]
#
# Parameterized schedules
# -----------------------
#
# In an experiment, often the need arises to vary one of the parameters of a schedule programmatically. Currently, the canonical way of achieving this is by defining a function that returns a generated schedule. We will use this to generate a pulse train, where we can specify the timing parameters separately.

# %%
from quantify_scheduler.resources import BasebandClockResource


def pulse_train_schedule(
    amp: float, time_high: float, time_low: float, amount_of_pulses: int
) -> Schedule:
    sched = Schedule("Pulse train schedule")
    square_pulse = sched.add(
        pulse_library.SquarePulse(
            amp=amp,
            duration=time_high,
            port="q0:fl",
            clock=BasebandClockResource.IDENTITY,
        ),
    )
    for _ in range(amount_of_pulses - 1):
        square_pulse = sched.add(
            pulse_library.SquarePulse(
                amp=amp,
                duration=time_high,
                port="q0:fl",
                clock=BasebandClockResource.IDENTITY,
            ),
            rel_time=time_low,
            ref_op=square_pulse,
        )
    return sched


sched = pulse_train_schedule(1, 200e-9, 300e-9, 5)
compilation.determine_absolute_timing(sched)
pulse_diagram_plotly(sched)

# %% [raw]
# Note that we used the `BasebandClockResource` as clock, which is always at 0 Hz and added automatically to the schedule for convenience. We can see that the pulses start every 500 ns and are 200 ns long.
