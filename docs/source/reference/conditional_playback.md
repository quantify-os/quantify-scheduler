---
file_format: mystnb
kernelspec:
    name: python3
---
(sec-qblox-conditional-playback)=

# Conditional Playback

The conditional playback feature introduces a method for dynamic qubit state management.

## Conditional Reset

### Overview

This feature is centered around the {class}`~quantify_scheduler.backends.qblox.operations.gate_library.ConditionalReset` operation, which measures
the state of a qubit and applies a corrective action based on the measurement
outcome.

### Conditional Reset Operation

The {class}`~quantify_scheduler.backends.qblox.operations.gate_library.ConditionalReset` operation functions as follows:
- It first measures the state of a qubit, using a {class}`~quantify_scheduler.operations.acquisition_library.ThresholdedAcquisition`.
- If the qubit is in an excited state, an `X` gate (DRAG pulse for transmons) is applied to
  bring the qubit back to the ground state.
- Conversely, if the qubit is already in the ground state, the operation simply
  waits.
- This ensures that the total duration of the {class}`~quantify_scheduler.backends.qblox.operations.gate_library.ConditionalReset` operation
  remains consistent, regardless of the qubit's initial state.

#### Usage

The {class}`~quantify_scheduler.backends.qblox.operations.gate_library.ConditionalReset` is used by adding it to a schedule:

```python
schedule = Schedule("")
schedule.add(ConditionalReset("q0"))
schedule.add(...)
```

Internally, {class}`~quantify_scheduler.backends.qblox.operations.gate_library.ConditionalReset` performs a {class}`~quantify_scheduler.operations.acquisition_library.ThresholdedAcquisition`. If a
schedule includes multiple acquisitions on the same qubit, each
{class}`~quantify_scheduler.backends.qblox.operations.gate_library.ConditionalReset` and {class}`~quantify_scheduler.operations.acquisition_library.ThresholdedAcquisition` must have a unique `acq_index`.

For example:

```python
schedule = Schedule("conditional")
schedule.add(ConditionalReset("q0", acq_index=0))
schedule.add(Measure("q0", acq_index=1, acq_protocol="ThresholdedAcquisition"))
```

When using multiple consecutive {class}`~quantify_scheduler.backends.qblox.operations.gate_library.ConditionalReset` on the same qubit, increment the `acq_index` for each:

```python
schedule = Schedule()
schedule.add(ConditionalReset("q0"))
schedule.add(...)
schedule.add(ConditionalReset("q0", acq_index=1))
schedule.add(...)
schedule.add(ConditionalReset("q0", acq_index=2))
```

Since the {class}`~quantify_scheduler.backends.qblox.operations.gate_library.ConditionalReset` performs a {class}`~quantify_scheduler.operations.acquisition_library.ThresholdedAcquisition`, the measured
state of the qubit (0 or 1) will be saved in the acquisition dataset.


### Qblox backend implementation

The {class}`~quantify_scheduler.backends.qblox.operations.gate_library.ConditionalReset` is implemented as a new subschedule in the gate library:

```python
class ConditionalReset(Schedule):
    def __init__(self, qubit_name):
        super().__init__("conditional reset")
        self.add(Measure(qubit_name, acq_protocol="ThresholdedAcquisition", feedback_trigger_label=qubit_name))
        cond_schedule = Schedule("conditional subschedule")
        cond_schedule.add(X(qubit_name))
        self.add(ConditionalOperation(body=cond_schedule, qubit_name=qubit_name))
```

where the Measure operation is given an additional trigger label
`feedback_trigger_label=qubit_name`. The label needs to match the label given to
`Conditional` to match relevant triggers with the conditional schedule. It is
set here to be equal to `qubit_name`, but could be any string.

The total duration (`t`) of the {class}`~quantify_scheduler.backends.qblox.operations.gate_library.ConditionalReset` is calculated as the sum of
various time components, typically shorter than the standard idle reset duration
(`reset.duration`). For example, in our test suite's
`mock_setup_basic_transmon_with_standard_params` fixture, the following values
are used:

- `q0.measure.trigger_delay() = 340 ns`
- `q0.measure.acq_delay() = 100 ns`
- `q0.measure.integration_time() = 1,000 ns`
- `q0.rxy.duration() = 20 ns`
- `q0.reset.duration() = 200,000 ns`

### Limitations

- Currently only implemented for the Qblox backend.
- Triggers cannot be sent more frequently than once every 252 ns.
- The interval between the end of an acquisition and the start of a conditional
  operation must be at least 364 ns, as specified by 
  {class}`~quantify_scheduler.backends.qblox.constants.TRIGGER_DELAY`.
- The measurement result of {class}`~quantify_scheduler.backends.qblox.operations.gate_library.ConditionalReset` is saved inside the dataset as well, potentially obscuring interesting data. For example, including the {class}`~quantify_scheduler.backends.qblox.operations.gate_library.ConditionalReset` inside a basic Rabi schedule, we could have the following implementation:

```{code-cell} ipython3
---
tags: [remove-cell]
---
import numpy as np
from quantify_scheduler.backends.qblox.operations.gate_library import ConditionalReset
from quantify_scheduler.device_under_test.transmon_element import BasicTransmonElement
from quantify_scheduler.operations.gate_library import Measure
from quantify_scheduler.operations.pulse_library import DRAGPulse
from quantify_scheduler.resources import ClockResource
from quantify_scheduler.schedules.schedule import Schedule
from quantify_scheduler.operations.control_flow_library import ConditionalOperation

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
    schedule.add(ConditionalReset("q0", acq_index = 2*i), label=f"Reset {i}")
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
    schedule.add(Measure("q0", acq_index=2*i+1), label=f"Measurement {i}")
```

where the relevant data is saved in the *odd* (`2*i+1`) acquisition indices. Currently, it is not possible to not save the measurement data related to {class}`~quantify_scheduler.backends.qblox.operations.gate_library.ConditionalReset`. 

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
    schedule.add(ConditionalReset("q0c", acq_index=i), label=f"Reset {i}")
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
    schedule.add(Measure("q0", acq_index=i), label=f"Measurement {i}")
```

Here, `qubit_c` and `qubit` will correspond to the same physical qubit and are controlled by the same port on the hardware, but the measurements will use two different sequencers and the data will be stored in two different acquisition channels. Note that this will limit your ability to do multiplexed readout. 
- Currently, it is not possible to use
  {class}`~quantify_scheduler.backends.qblox.operations.gate_library.ConditionalReset`
  with {class}`~quantify_core.measurement.control.MeasurementControl` class, when using
  `BinMode.APPEND`, but it works correctly when using `BinMode.AVERAGE`.
