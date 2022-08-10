---
file_format: mystnb
kernelspec:
    name: python3

---

```{code-cell}
:tags: [hide-input]

# Make output easier to read
from rich import pretty
pretty.install()

```

```{jupyter-execute}
:hide-code:

# pylint: disable=line-too-long
# pylint: disable=wrong-import-order
# pylint: disable=wrong-import-position
# pylint: disable=pointless-string-statement
# pylint: disable=pointless-statement
# pylint: disable=invalid-name
# pylint: disable=duplicate-code
# pylint: disable=missing-function-docstring
```

```{caution}
This is intended functionality that is being developed. It is not (fully) available yet.
```

# Tutorial: Dark ESR Experiment NV Centers

This tutorial guides through the dark electron spin resonance (ESR) experiment on electronic qubits in an NV center. This tutorial assumes basic familiarity the experiment; more information can be found in literature, for example :cite:t:`PhysRevLett.101.117601`. Knoweldge of the basics of quantify-scheduler are also useful as reviewed in the {ref}`User guide`.

## Introduction

The Dark ESR experiment is a spectroscopy measurement to determine the frequency of the transition between states forming a computational basis $|0\rangle$ and $|1\rangle$. The transition is in the microwave range and the central part of the experiment is a spectroscopy pulse. Before the pulse, a Reset operation is applied and afterwards a Measure operation.

To prepare the NV center in the NV$^-$ state, a green laser pulse is used to excite the entire system. In the following relaxation, the NV center settles into the NV$^-$ state with a certain probability. Whether this is the case is determined by measuring and interpreting the response to excitation with red lasers. The same method is also used after the spectroscopy experiment to check whether the NV center is in the same state as initially.

The procedure outlined above is repeated multiple times to collect sufficient statistical precision. It is also repeated for different spectroscopy frequencies.

## Parameters of the NV Center

In quantify-scheduler, the {class}`.QuantumDevice` keeps track of the device-specific parameters of all qubits. Let's set up an NV center with one electronic qubit and give it default parameters.

```{code-block} ipython3
import quantify_scheduler.device_under_test.mock as mock
quantum_device = mock.get_bare_nv_center()
mock.dress_nv_center(quantum_device)
```

For NV Centers, the electronic qubit is described with the class {class}`.BasicElectronicNVElement`. We can retrieve the electronic qubit with name "qe0" and look at its parameters with

```{code-block} ipython3
qubit_name = "qe0"
qubit = quantum_device.get_element(qubit_name)
print(qubit)
```

## Investigate the ESR operations

This section introduces some of the building blocks of the ESR experiment. All of these components start with a general circuit-layer definition that is translated into NV center-specific pulses and hardware instructions. We will start with the spectroscopy pulse.

### Spectroscopy pulse

Let's say we want to apply a spectroscopy pulse to our test qubit. We can use the operation {class}`.SpectroscopyPulse` for that. We need to give it the name of our qubit as argument.

```{note}
We pass the name of the qubit instead of the qubit instance itself, because this name is used later in the device config to look up the specifics of this qubit.
```

```{warning}
How to specify a parameter? Describe. Current version not intuitive. Would be more intuitive if we could pass a Parameter.
```

```{code-block} ipython3
from quantify_scheduler.operations.gate_library import SpectroscopyPulse
spec_pulse = SpectroscopyPulse("qe0")
```

In order to compile the pulse to the device layer, we will make it part of a {class}`.Schedule`

```{code-block} ipython3
schedule = Schedule(name="Single Spectroscopy Pulse")
_ = schedule.add(SpectroscopyPulse(qubit_name), label="Spectroscopy pulse")
```

When we compile the schedule to the device layer, we translate the general circuit-level operation into a NV-center specific pulse. The information about our device is encoded in the device config and comes from our {class}`.BasicElectronicNVElement`.


```{code-block} ipython3
device_config = qubit.generate_device_config().dict()
schedule_device = device_compile(schedule, dev_cfg)
```

```{warning}
Why is there a `.dict()` in the line in which we retrieve the device config?
```

When we look into the device config, we can see that the quantum device specifies that the operation type `"spectroscopy_pulse"` should be translated using the function {func}`.nv_spec_pulse_mw`.

```{code-block} ipython3
device_config["elements"][qubit_name]["spectroscopy_pulse"]
```

This function returns a {class}`.SkewedHermitePulse`. This class can now be found in the compiled schedule:

```{code-block} ipython3
schedule.schedulables["Spectroscopy pulse"]
```

### Other Operations

TODO: maybe one acquisition?

## Combining all Operations to a Schedule

```{warning}
Idea: high-level operation `CRCheck` will translate into `ChargeReset` and `CRCount`. This `CRCheck` operation could in principle support a repeat-until-success loop (not currently possible). For now, specify directly the `ChargeReset` and `CRCount` operations. Need to define if they are circuit level or device level. Are there equivalent operations for non-NV centers?
```

```{warning}
Add description. Move function in schedules module and reference it from here. Add arguments.
```

```{code-block} ipython3
def dark_esr_schedule(qubit_name: str, repetitions: int = 1):
    schedule = Schedule(name="Dark ESR", repetitions=repetitions)
    freq_clock_name = f"{qubit_name}.01"
    schedule.add_resource(ClockResource(name=freq_clock_name, freq=spec_frequency))

    op = schedule.add(ChargeReset(qubit_name=qubit_name))
    op = schedule.add(CRCount(qubit_name=qubit_name, acquisition_type="photon count"), rel_time=20e-6, ref_op=last_op, ref_pt="end", label="CR Count before")
    op = schedule.add(Reset(qubit_name=qubit_name), rel_time=10e-6, ref_op=op, ref_pt="end", label="Reset")
    op = schedule.add(SpectroscopyPulse(qubit_name=qubit_name, clock=freq_clock_name), label="MW pi pulse")
    op = schedule.add(Measure(qubit_name=qubit_name, acq_index=1), label="Measurement in computing bassis")
    op = schedule.add(CRCount(qubit_name=qubit_name, acq_index=2), label="CR Count after ESR")

schedule = dark_esr_schedule(qubit_name=qubit_name)
```

The schedule defines one iteration of the Dark ESR experiment. But now we want to execute it for many different frequencies. We will loop over a frequency parameter in a measurement function.

## Measurement function

```{warning}
Write some text and break up block into smaller parts. Or move into measurement module. And display here.
```

```{code-block}
meas_ctrl = quantum_device.instr_measurement_control.get_instr()

gettable = ScheduleGettable(
    quantum_device=quantum_device,
    schedule_function=dark_esr_sched,
    schedule_kwargs=dict(
        qubit_name=qubit.name,
        repetitions=10,
    )
)

meas_ctrl.settables(qubit.clock_freqs.spec)
meas_ctrl.setpoints(frequencies)
meas_ctrl.gettables([gettable])

dataset = meas_ctrl.run()

analysis = DarkESRAnalysis().run()
```
