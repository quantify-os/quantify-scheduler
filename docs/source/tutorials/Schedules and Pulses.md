---
file_format: mystnb
kernelspec:
    name: python3

---
(sec-tutorial-sched-pulse)=

# Tutorial: Schedules and Pulses

```{seealso}
The complete source code of this tutorial can be found in

{nb-download}`Schedules and Pulses.ipynb`
```

## The Schedule

The main data structure that describes an experiment in the `quantify-scheduler` is the Schedule. We will show how the Schedule works through an example.

```{code-cell} ipython3

from quantify_scheduler import Schedule

sched = Schedule("Hello quantum world!")

sched


```

As we can see, our newly created schedule is still empty. We need to manually add operations to it. In `quantify-scheduler` there are three types of operations: pulses, acquisitions and gates. All of these have explicit timing control. In this tutorial, we will only cover pulses. The goal will not be to make a schedule that is physically meaningful, but to demonstrate the control over the scheduling to its fullest.

While it is possible to define a pulse completely from scratch, we will be using some of the pulse definitions provided with the `quantify-scheduler`. These pulses are described in the {mod}`quantify_scheduler.operations.pulse_library` submodule. It's worth noting that no sampling of the data yet occurs at this stage, but the pulse is kept in a parameterized form.

We will add a square pulse from the pulse library to the schedule.

```{code-cell} ipython3

from quantify_scheduler.operations import pulse_library

square_pulse = sched.add(
    pulse_library.SquarePulse(amp=1, duration=1e-6, port="q0:res", clock="q0.ro")
)

sched


```

You may have noticed that we passed a {code}`port` and a {code}`clock` to the pulse. The {code}`port` specifies the physical location on the quantum chip to which we are sending the pulses, whilst the {code}`clock` tracks the frequency of the signal (see {ref}`sec-user-guide-ports-clocks`). This clock frequency has not yet been defined, so prior to any compilation step this clock needs to be added to the schedule as a resource.

```{code-cell} ipython3

from quantify_scheduler.resources import ClockResource

readout_clock = ClockResource(name="q0.ro", freq=7e9)
sched.add_resource(readout_clock)

sched


```

`quantify-scheduler` provides several visualization tools to show a visual representation of the schedule we made. First, however, we need to instruct the scheduler to calculate the pulse timings. We can accomplish this using the {func}`~quantify_scheduler.compilation.determine_absolute_timing` function. In the cell below we call this function, and draw the schedule using a {meth}`pulse diagram <.plot_pulse_diagram>`.

Note that these plots are interactive and modulation is not shown by default.

```{code-cell} ipython3

from quantify_scheduler import compilation

timed_sched = compilation.determine_absolute_timing(sched)
timed_sched.plot_pulse_diagram(plot_backend="plotly")


```

## Explicit timing control

What we see in the pulse diagram is only a flat line, corresponding to our single square pulse. To make our schedule more interesting, we should add more pulses to it. We will add another square pulse, but with a 500 ns delay.

```{code-cell} ipython3

sched.add(
    pulse_library.SquarePulse(amp=1, duration=1e-6, port="q0:res", clock="q0.ro"),
    ref_op=square_pulse,
    rel_time=500e-9,
)

timed_sched = compilation.determine_absolute_timing(sched)
timed_sched.plot_pulse_diagram(plot_backend="plotly")


```

We can see that {code}`rel_time=500e-9` schedules the pulse 500 ns shifted relative to the end of the {code}`ref_op`. If no additional arguments are passed, operations are added directly after the operation that was added last.

Let's now instead align a pulse to start at the same time as the first square pulse. Before, we specified the timing relative to the end of a different pulse, but we can choose to instead specify it relative to the beginning. This is done by passing {code}`ref_pt="start"`.

```{code-cell} ipython3

sched.add(
    pulse_library.DRAGPulse(
        G_amp=0.5, D_amp=0.5, duration=1e-6, phase=0, port="q0:mw", clock="q0.01"
    ),
    ref_op=square_pulse,
    ref_pt="start",
)
sched.add_resource(ClockResource(name="q0.01", freq=7e9))

timed_sched = compilation.determine_absolute_timing(sched)
timed_sched.plot_pulse_diagram(plot_backend="plotly")


```

We see that we added a DRAG pulse to the schedule. Two things stand out:

1. The DRAG pulse is plotted separately from the square pulse, this is because we specified a different {code}`port` for this pulse than we did for the square pulse.
2. The DRAG pulse shows two lines instead of one. This is because a DRAG pulse is specified as a complex-valued pulse, so we have to plot both the I and Q components of the signal. The real part of the waveform is shown in color, whereas the imaginary component is shown in grayscale.

## Parameterized schedules

In an experiment, often the need arises to vary one of the parameters of a schedule programmatically. Currently, the canonical way of achieving this is by defining a function that returns a generated schedule. We will use this to generate a pulse train, where we can specify the timing parameters separately.

```{code-cell} ipython3

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
timed_sched = compilation.determine_absolute_timing(sched)
timed_sched.plot_pulse_diagram(plot_backend="plotly")


```

Note that we used the {class}`~quantify_scheduler.resources.BasebandClockResource` as a clock, which is always at 0 Hz and was added automatically to the schedule for convenience. We can see that the pulses start every 500 ns and are 200 ns long.

(sec-long-waveforms-via-stitchedpulse)=

## Long waveforms via StitchedPulse

```{note}
This feature is only available for Qblox hardware. More information about this feature can also be found in section {ref}`Long waveform support <sec-qblox-cluster-long-waveform-support-new>`.
```

The sequencers in Qblox modules have a waveform sample limit of {class}`~quantify_scheduler.backends.qblox.constants.MAX_SAMPLE_SIZE_WAVEFORMS`. Trying to play many (long) waveforms might cause you to exceed this limit. For certain waveforms, however, it is possible to use the available memory more efficiently. This section explains how to do this with the {class}`~quantify_scheduler.operations.stitched_pulse.StitchedPulse`.

For convenience, `quantify-scheduler` provides helper functions for the `square` ({func}`~quantify_scheduler.operations.pulse_factories.long_square_pulse`), `ramp` ({func}`~quantify_scheduler.operations.pulse_factories.long_ramp_pulse`) and `staircase` ({func}`~quantify_scheduler.operations.pulse_factories.staircase_pulse`) waveforms for when they become too long to fit into the waveform memory of the hardware.

```{code-cell} ipython3
from quantify_scheduler.operations.pulse_factories import (
    long_ramp_pulse,
    long_square_pulse,
    staircase_pulse,
)


sched = Schedule("Basic long pulses")
sched.add(
    long_square_pulse(
        amp=0.5,
        duration=10e-6,
        port="q0:fl",
        clock=BasebandClockResource.IDENTITY,
    ),
)
sched.add(
    long_ramp_pulse(
        amp=1.0,
        duration=10e-6,
        port="q0:fl",
        offset=-0.5,
        clock=BasebandClockResource.IDENTITY,
    ),
    rel_time=5e-7,
)
sched.add(
    staircase_pulse(
        start_amp=-0.5,
        final_amp=0.5,
        num_steps=20,
        duration=10e-6,
        port="q0:fl",
        clock=BasebandClockResource.IDENTITY,
    ),
    rel_time=5e-7,
)

timed_sched = compilation.determine_absolute_timing(sched)
timed_sched.plot_pulse_diagram(plot_backend="plotly")
```

Using these factory functions, the resulting square and staircase pulses use no waveform memory at all. The ramp pulse uses waveform memory for a short section of the waveform, which is repeated multiple times.

For more complicated shapes, the {class}`~quantify_scheduler.operations.stitched_pulse.StitchedPulseBuilder` makes it possible to stitch together pulse shapes yourself. In the following example, we create a long soft square pulse where the constant-voltage middle part is created with a voltage offset instruction, using no waveform memory.

```{code-cell} ipython3
import numpy as np

from quantify_scheduler.operations.pulse_library import NumericalPulse
from quantify_scheduler.operations.stitched_pulse import StitchedPulseBuilder


# Define a few constants
port = "q0:fl"
clock = BasebandClockResource.IDENTITY
ramp_duration = 4e-6
constant_duration = 8e-6

ramp_t = np.arange(0, round(ramp_duration * 1e9) + 1) * 1e-9

# Define the waveforms for the up and down ramps
hann_up = ramp_t / ramp_duration - 1 / 2 / np.pi * np.sin(
    2 * np.pi / ramp_duration * ramp_t
)
hann_down = 1 - hann_up

# Make the stitched pulse
builder = StitchedPulseBuilder(port=port, clock=BasebandClockResource.IDENTITY)
builder.add_pulse(
    NumericalPulse(samples=hann_up, t_samples=ramp_t, port=port, clock=clock)
)
builder.add_voltage_offset(path_0=1.0, path_1=0.0, duration=constant_duration)
builder.add_pulse(
    NumericalPulse(samples=hann_down, t_samples=ramp_t, port=port, clock=clock)
)
pulse = builder.build()

sched = Schedule("Long soft square pulse")
sched.add(pulse)

timed_sched = compilation.determine_absolute_timing(sched)
timed_sched.plot_pulse_diagram(plot_backend="plotly")
```

Alternatively, the building methods of the {class}`~quantify_scheduler.operations.stitched_pulse.StitchedPulseBuilder` can be conveniently **chained** to create a {class}`~quantify_scheduler.operations.stitched_pulse.StitchedPulse` via more elegant syntax:

```{code-cell} ipython3
pulse = (
    StitchedPulseBuilder(port=port, clock=BasebandClockResource.IDENTITY)
    .add_pulse(
        NumericalPulse(samples=hann_up, t_samples=ramp_t, port=port, clock=clock)
    )
    .add_voltage_offset(path_0=1.0, path_1=0.0, duration=constant_duration)
    .add_pulse(
        NumericalPulse(samples=hann_up, t_samples=ramp_t, port=port, clock=clock)
    )
    .build()
)
```
