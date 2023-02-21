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

The main data structure that describes an experiment in the `quantify_scheduler` is the Schedule. We will show how the Schedule works through an example.

```{code-cell} ipython3

from quantify_scheduler import Schedule

sched = Schedule("Hello quantum world!")

sched


```

As we can see, our newly created schedule is still empty. We need to manually add operations to it. In `quantify_scheduler` there are three types of operations: pulses, acquisitions and gates. All of these have explicit timing control. In this tutorial we will only cover pulses. The goal will not be to make a schedule that is physically meaningful, but to demonstrate the control over the scheduling to its fullest.

While it is possible to define a pulse completely from scratch, we will be using some of the pulse definitions provided with the `quantify_scheduler`. These pulses are described in the {mod}`quantify_scheduler.operations.pulse_library` submodule. It's worth noting that no sampling of the data yet occurs at this stage, but the pulse is kept in a parameterized form.

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

`quantify_scheduler` provides several visualization tools to show a visual representation of the schedule we made. First, however, we need to instruct the scheduler to calculate the pulse timings. We can accomplish this using the {func}`~quantify_scheduler.compilation.determine_absolute_timing` function. In the cell below we call this function and draw the schedule.

Note that these plots are interactive and modulation is not shown by default.

```{code-cell} ipython3

from quantify_scheduler import compilation

compilation.determine_absolute_timing(sched)
sched.plot_pulse_diagram(plot_backend='plotly')


```

## Explicit timing control

What we see in the pulse diagram is only a flat line, corresponding to our single square pulse. To make our schedule more interesting, we should add more pulses to it. We will add another square pulse, but with a 500 ns delay.

```{code-cell} ipython3

sched.add(
    pulse_library.SquarePulse(amp=1, duration=1e-6, port="q0:res", clock="q0.ro"),
    ref_op=square_pulse,
    rel_time=500e-9,
)

compilation.determine_absolute_timing(sched)
sched.plot_pulse_diagram(plot_backend='plotly')


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

compilation.determine_absolute_timing(sched)
sched.plot_pulse_diagram(plot_backend='plotly')


```

We see that we added a DRAG pulse to the schedule. Two things stand out:

1. The DRAG pulse is plotted separately from the square pulse, this is because we specified a different {code}`port` for this pulse than we did for the square pulse.
2. The DRAG pulse shows two lines instead of one. This is because a DRAG pulse is specified as a complex-valued pulse, so we have to plot both the I and Q components of the signal. The real part of the waveform is shown in color, whereas the imaginary component is shown in greyscale.

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
compilation.determine_absolute_timing(sched)
sched.plot_pulse_diagram(plot_backend='plotly')


```

Note that we used the {class}`~quantify_scheduler.resources.BasebandClockResource` as clock, which is always at 0 Hz and added automatically to the schedule for convenience. We can see that the pulses start every 500 ns and are 200 ns long.
