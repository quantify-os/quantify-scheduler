---
file_format: mystnb
kernelspec:
    name: python3
myst:
  substitutions:
    MeasurementControl: "{class}`~quantify_core.measurement.control.MeasurementControl`"
    ConditionalReset: "{class}`~quantify_scheduler.qblox.operations.ConditionalReset`"
    ConditionalOperation: "{class}`~quantify_scheduler.qblox.operations.ConditionalOperation`"
    QuantumDevice: "{class}`~quantify_scheduler.QuantumDevice`"
    Measure: "{class}`~quantify_scheduler.operations.Measure`"
    Reset: "{class}`~quantify_scheduler.operations.Reset`"
    Schedule: "{class}`~quantify_scheduler.Schedule`"
    ScheduleGettable: "{class}`~quantify_scheduler.ScheduleGettable`"
    BasicTransmonElement: "{class}`~quantify_scheduler.BasicTransmonElement`"
    ThresholdedAcquisition: "{class}`~quantify_scheduler.operations.ThresholdedAcquisition`"
    X: "{class}`~quantify_scheduler.operations.X`"
    TRIGGER_DELAY: "{class}`~quantify_scheduler.backends.qblox.constants.TRIGGER_DELAY`" 
---
(sec-qblox-conditional-playback)=

# Conditional Playback

The conditional playback feature introduces a method for dynamic qubit state management.

## Example: Conditional Reset

### Overview

This feature is centered around the
{{ ConditionalReset }} operation, which
measures the state of a qubit and applies a corrective action based on the
measurement outcome.

### Conditional Reset Operation

The {{ ConditionalReset }} operation functions as follows:
- It first measures the state of a qubit, using a {{ ThresholdedAcquisition }}.
- If the qubit is in an excited state, an {{ X }} gate (e.g. a DRAG pulse for transmons) is applied to
  bring the qubit back to the ground state.
- Conversely, if the qubit is already in the ground state, the operation simply
  waits for the duration of the X gate.
- This ensures that the total duration of the {{ ConditionalReset }} operation
  remains consistent, regardless of the qubit's initial state.

### Usage

The {{ ConditionalReset }} is used by adding it to a schedule:

```python
from quantify_scheduler import Schedule
from quantify_scheduler.qblox.operations import ConditionalReset

schedule = Schedule("")
schedule.add(ConditionalReset("q0"))
schedule.add(...)
```

Internally, {{ ConditionalReset }} performs a {{ ThresholdedAcquisition }}.
Since the {{ ConditionalReset }} performs a {{ ThresholdedAcquisition }}, the measured
state of the qubit (0 or 1) will be saved in the acquisition dataset.


### Qblox backend implementation

The {{ ConditionalReset }} is implemented as a new subschedule in the operations library:

```python
class ConditionalReset(Schedule):
    def __init__(self, qubit_name):
        super().__init__("conditional reset")
        self.add(
            Measure(
                qubit_name,
                acq_protocol="ThresholdedAcquisition",
                feedback_trigger_label=qubit_name,
            )
        )
        cond_schedule = Schedule("conditional subschedule")
        cond_schedule.add(X(qubit_name))
        self.add(
            ConditionalOperation(body=cond_schedule, qubit_name=qubit_name),
            rel_time=TRIGGER_DELAY,
        )
```

where the Measure operation is given an additional trigger label
`feedback_trigger_label=qubit_name`. The label needs to match the label given to
{{ ConditionalOperation }} to match relevant triggers with the conditional schedule. It is
set here to be equal to `qubit_name`, but could be any string. 


In the Qblox hardware, a finite delay is added between the
end of the acquisition and the start of the conditional operation. This delay
is specified in {{ TRIGGER_DELAY }}. Furthermore, the {{ ConditionalReset }} operation adds an additional 4 ns of idle time at the end the operation that is passed to `body` (in this case, the {{ X }} gate). The total duration of the {{ ConditionalOperation }} is then the duration of the body ( {{ X }} in this case ) plus 4 ns.

The total duration (`t`) of the {{ ConditionalReset }} is calculated as the sum of
various time components, typically shorter than the standard idle reset duration
(`qubit.reset.duration`). For example, in our test suite's
`mock_setup_basic_transmon_with_standard_params` fixture, the following values
are used:

```{code-cell} ipython3
:tags: [remove-cell]

from quantify_scheduler.backends.qblox.constants import TRIGGER_DELAY, IMMEDIATE_MAX_WAIT_TIME

# Example values
measure_acq_delay = 100e-9
measure_integration_time = 1000e-9
rxy_duration = 20e-9
trigger_delay = TRIGGER_DELAY
reset_duration = 200000e-9
buffer_time = 4e-9

# Convert to ns for display
def to_ns(s): return int(s*1e9)

# Glue the values
from myst_nb import glue
glue("measure_acq_delay", to_ns(measure_acq_delay))
glue("measure_integration_time", to_ns(measure_integration_time))
glue("rxy_duration", to_ns(rxy_duration))
glue("trigger_delay", to_ns(trigger_delay))
glue("reset_duration", to_ns(reset_duration))
glue("buffer_time", to_ns(buffer_time))
glue("max_wait_time", IMMEDIATE_MAX_WAIT_TIME)
glue("total_cond_reset_duration", to_ns(measure_acq_delay + measure_integration_time + rxy_duration + trigger_delay + buffer_time))
```

- `q0.measure.acq_delay()` = {glue:}`measure_acq_delay` ns
- `q0.measure.integration_time()` = {glue:}`measure_integration_time` ns
- `q0.rxy.duration()` = {glue:}`rxy_duration` ns
- `TRIGGER_DELAY` = {glue:}`trigger_delay` ns
- `buffer_time` = {glue:}`buffer_time` ns
- `q0.reset.duration()` = {glue:}`reset_duration` ns

So that the total duration of the {{ ConditionalReset }} is {glue:}`total_cond_reset_duration` ns, compared to the standard idle reset duration of {glue:}`reset_duration` ns.

## Conditional Playback

Conditionally playing a subschedule is implemented using the {{ ConditionalOperation }} operation. The previous example on implementing a {{ ConditionalReset }} can be extended by simply replacing the {{ X }} gate with the subschedule to be conditionally played:

```python
cond_schedule = Schedule("conditional subschedule")
cond_schedule.add(X(qubit_name))
cond_schedule.add(...)

schedule.add(
    Measure(qubit_name, 
    acq_protocol="ThresholdedAcquisition", 
    feedback_trigger_label=qubit_name))
schedule.add(
    ConditionalOperation(body=cond_schedule, qubit_name=qubit_name),
    rel_time=TRIGGER_DELAY,
)
```


There are a few rules to follow for the `cond_schedule`:

1. The `cond_schedule` must have a duration of at least 4 ns.
2. The `cond_schedule` must have a duration of max {glue:}`max_wait_time` ns (per repetition)
3. Nested conditional operations are not allowed.


## Limitations

- Currently only implemented for the Qblox backend.
- Triggers cannot be sent more frequently than once every 252 ns.
- The interval between the end of an acquisition and the start of a conditional
  operation must be at least {glue:}`trigger_delay` ns, as specified by 
  {{ TRIGGER_DELAY }}.
  See [the qblox documentation](https://docs.qblox.com/en/main/cluster/feedback.html) for details.
- The measurement result of {{ ConditionalReset }} is saved inside the dataset as well, potentially obscuring interesting data. 
- Currently, it is not possible to use
  {{ ConditionalReset }}
  with {{ MeasurementControl }} class, when using
  `BinMode.APPEND`, but it works correctly when using `BinMode.AVERAGE`.

### Example: Obscured measurement data

For example, including the {{ ConditionalReset }} inside a basic Rabi schedule, we could have the following implementation:

```{code-cell} ipython3
---
tags: [remove-cell]
---
import numpy as np
from quantify_scheduler import BasicTransmonElement, ClockResource, Schedule
from quantify_scheduler.operations import Measure, DRAGPulse
from quantify_scheduler.qblox.operations import ConditionalReset, ConditionalOperation

repetitions = 1024
clock = "q0.r0"
port = "q0:mw"
frequency = "100e6"
amps = np.linspace(0,1,10)
durations = np.linspace(4e-9,84e-9,10)
```

```{code-cell} ipython3
---
tags: [remove-output]
---

schedule = Schedule("Rabi", repetitions)
schedule.add_resource(ClockResource(name=clock, freq=frequency))

for i, (amp, duration) in enumerate(zip(amps, durations)):
    schedule.add(ConditionalReset("q0"), label=f"Reset {i}")
    schedule.add(
        DRAGPulse(
            duration=duration,
            G_amp=amp,
            D_amp=0,
            port=port,
            clock=clock,
            phase=0,
        ),
        label=f"Rabi_pulse {i}",
    )
    schedule.add(Measure("q0"), label=f"Measurement {i}")
```

where the relevant data is saved in the *odd* (`2*i+1`) acquisition indices. Currently, it is not possible to disregard the measurement data related to {{ ConditionalReset }}. 

A possible workaround is to introduce an extra port-clock combination to the hardware configuration, and an extra qubit to the device configuration to separate "conditional" qubits, from regular ones. For example:

```{code-cell} ipython3
hardware_config = {
    "portclock_configs": [
        {"port": "q0:res", "clock": "q0.ro", "interm_freq": 200000000.0},
        {"port": "q0c:res", "clock": "q0.ro", "interm_freq": 200000000.0},
    ],
    ... : ...
}

# define qubit parameters
q0c = BasicTransmonElement("q0c")
...

schedule = Schedule("Rabi", repetitions)
schedule.add_resource(ClockResource(name=clock, freq=frequency))

for i, (amp, duration) in enumerate(zip(amps, durations)):
    schedule.add(ConditionalReset("q0c"), label=f"Reset {i}")
    schedule.add(
        DRAGPulse(
            duration=duration,
            G_amp=amp,
            D_amp=0,
            port=port,
            clock=clock,
            phase=0,
        ),
        label=f"Rabi_pulse {i}",
    )
    schedule.add(Measure("q0"), label=f"Measurement {i}")
```

Here, `qubit_c` and `qubit` will correspond to the same physical qubit and are controlled by the same port on the hardware, but the measurements will use two different sequencers and the data will be stored in two different acquisition channels. Note that this will limit your ability to do multiplexed readout. 

### Example: Parallel conditional operations and acquisitions

