---
file_format: mystnb
kernelspec:
    name: python3

---
(sec-qblox-conditional-playback)=

# Conditional Playback

The Conditional Playback feature introduces a method for dynamic qubit state management.

## Conditional Reset

### Overview

This feature is centered around the new `ConditionalReset` operation, which measures the state of a qubit and applies a corrective action based on the measurement outcome.

### Conditional Reset Operation

The `ConditionalReset` operation functions as follows:
- It first measures the state of a qubit, using a `ThresholdedAcquisition`.
- If the qubit is in an excited state, a DRAG pulse (`X` gate) is applied to bring it back to the ground state.
- Conversely, if the qubit is already in the ground state, the operation simply waits.
- This ensures that the total duration of the `ConditionalReset` operation remains consistent, regardless of the qubit's initial state.

#### Usage

The `ConditionalReset` is used as part of a schedule:

```python
schedule = Schedule("conditional")
schedule.add(ConditionalReset("q0"))
schedule.add(...)
```

Internally, `ConditionalReset` performs a `ThresholdedAcquisition`. If a schedule includes multiple acquisitions on the same qubit, each `ConditionalReset` and `ThresholdedAcquisition` must have a unique `acq_index`.

For example:

```python
schedule = Schedule("conditional")
schedule.add(ConditionalReset("q0", acq_index=0))
schedule.add(Measure("q0", acq_index=1, acq_protocol="ThresholdedAcquisition"))
```

When using multiple consecutive `ConditionalResets` on the same qubit, increment the `acq_index` for each:

```python
schedule = Schedule()
schedule.add(ConditionalReset("q0"))
schedule.add(...)
schedule.add(ConditionalReset("q0", acq_index=1))
schedule.add(...)
schedule.add(ConditionalReset("q0", acq_index=2))
```



### Implementation

The `ConditionalReset` is implemented as a new subschedule in the gate library:

```python
class ConditionalReset(Schedule):
    def __init__(self, qubit_name):
        super().__init__("conditional reset")
        self.add(Measure(qubit_name, acq_protocol="ThresholdedAcquisition"))
        cond_schedule = Schedule("conditional subschedule")
        cond_schedule.add(X(qubit_name))
        self.add(cond_schedule, control_flow=Conditional("q0"))
```

The total duration (`t`) of the `ConditionalReset` is calculated as the sum of various time components, typically shorter than the standard idle reset duration (`reset.duration`). For example, in our test suite's `mock_setup_basic_transmon_with_standard_params` fixture, we use:

- `q0.measure.trigger_delay() = 340ns`
- `q0.measure.acq_delay() = 100ns`
- `q0.measure.integration_time() = 1,000ns`
- `q0.rxy.duration() = 20ns`
- `q0.reset.duration() = 200,000ns`

### Limitations

- This feature is currently only compatible with Qblox hardware.
- Triggers cannot be sent more frequently than once every 252ns.
- The interval between the end of an acquisition and the start of a conditional operation must be at least 364ns, as determined by `q0.measure.trigger_delay`.
- The conditionality of the `ConditionalReset` extends to all sequencers. If an operation is executed on another sequencer during the `ConditionalReset` block, it will also be conditional.

For example:

```python
schedule = Schedule()
schedule.add(ConditionalReset("q0"))
schedule.add(X("q1"), ref_pt_new="end", rel_time=-4e-9)
```

In this case, a pi pulse will be applied to both q1 and q2 if q1 is in state "1". If q1 is in state "0", no pulses will be applied.
