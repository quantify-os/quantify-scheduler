---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.6
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

(sec-control-flow)=
# Control flow

Complex schedules can be constructed from pulses, gates and schedules using control flow. The {class}`~.operations.control_flow_library.ConditionalOperation` can be added to a Schedule, or to another control flow operation. Note: {class}`~.operations.control_flow_library.ConditionalOperation` cannot be added to another conditional, because currently nested conditionals are not supported.

(sec-control-flow-subschedule)=
## Adding schedules to a schedule ("subschedules")

- Supported by {mod}`Qblox <quantify_scheduler.backends.qblox>` and
  {mod}`Zurich Instruments <quantify_scheduler.backends.zhinst>` backends.

A schedule can be added to a schedule just like an operation.

This is useful e.g. to define a custom composite gate:
```{code-cell} ipython3
from quantify_scheduler.operations.gate_library import X, Y90
from quantify_scheduler import Schedule

def hadamard(qubit: str) -> Schedule:
    hadamard_sched = Schedule("hadamard")
    hadamard_sched.add(X(qubit))
    hadamard_sched.add(Y90(qubit))
    return hadamard_sched

my_schedule = Schedule("nice_experiment")
my_schedule.add(X("q1"))
my_schedule.add(hadamard("q1"))
```

Note: The `repetitions` argument of all but the outermost Schedules is ignored. Schedules can be nested arbitrarily. Timing constraints relative to an inner schedule interpret the inner schedule as one continuous operation. It is not possible to use an operation within a subschedule from outside as reference operation.

(sec-control-flow-loops)=
## Repetition loops

- Supported by {mod}`Qblox <quantify_scheduler.backends.qblox>` backend.

The `body` of a {class}`~.operations.control_flow_library.LoopOperation` will be repeated `repetitions` times.

This can be used to efficiently implement sequential averaging without running over the instruction limit of the hardware:
```{code-cell} ipython3
import numpy as np
from typing import Union
from quantify_scheduler.operations.control_flow_library import LoopOperation
from quantify_scheduler.operations.gate_library import Reset, Measure

def t1_sched_sequential(
    times: Union[np.ndarray, float],
    qubit: str,
    repetitions: int = 1,
) -> Schedule:
    times = np.asarray(times)
    times = times.reshape(times.shape or (1,))

    schedule = Schedule("T1")
    for i, tau in enumerate(times):
        inner = Schedule(f"inner_{i}")
        inner.add(Reset(qubit), label=f"Reset {i}")
        inner.add(X(qubit), label=f"pi {i}")
        inner.add(
            Measure(qubit, acq_index=i),
            ref_pt="start",
            rel_time=tau,
            label=f"Measurement {i}",
        )
        schedule.add(LoopOperation(body=inner, repetitions=repetitions))
    return schedule
```
Hardware averaging works as expected. In `BinMode.APPEND` binning mode, the data is returned in chronological order.

```{note}
Loops are an experimental feature and come with several limitations at this time, see below.
```

### Limitations
1. The time order for zero-duration assembly instructions with the same timing may be incorrect, so verify the compiled schedule (via the generated assembly code). Using loops to implement sequential averaging for qubit spectroscopy is verified to work as expected. Known issues occur in using `SetClockFrequency` and `SquarePulse` with duration > 1us at the beginning or end of a loop, for example:
```{code-cell} ipython3
from quantify_scheduler.operations.pulse_library import SquarePulse

schedule = Schedule("T1")
schedule.add(
    LoopOperation(
        body=SquarePulse(
            amp=0.3,
            port="q0:res",
            duration=2e-6,
            clock="q0.ro",
        ),
        repetitions=3
    )
)
```
2. Repetition loops act on all port-clock combinations present in the circuit. This means that both `X("q0")` and `Y90("q1")` in the following circuit are repeated three times:
```{code-cell} ipython3
schedule = Schedule("T1")
x = schedule.add(LoopOperation(body=X("q0"), repetitions=3))
schedule.add(Y90("q1"), ref_op=x, ref_pt="start", rel_time=0)
```
### Safe use with the limitations
To avoid the limitations mentioned above, it is strongly recommended to use loops only with subschedules, with no operations overlapping with the subschedule. Adding wait times before and after loops ensures that everything works as expected:
```{code-cell} ipython3
from quantify_scheduler.operations.pulse_library import IdlePulse, SquarePulse

inner_schedule = Schedule("inner")
inner_schedule.add(IdlePulse(16e-9))
# anything can go here
inner_schedule.add(
    SquarePulse(
        amp=0.3,
        port="q0:res",
        duration=2e-6,
        clock="q0.ro",
    )
)
# End the inner schedule with a wait time
inner_schedule.add(IdlePulse(16e-9))

outer_schedule = Schedule("outer")
# anything can go here
outer_schedule.add(IdlePulse(16e-9))
outer_schedule.add(LoopOperation(body=inner_schedule, repetitions=5))
outer_schedule.add(IdlePulse(16e-9))
# anything can go here
```
