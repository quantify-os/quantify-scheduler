# Release Notes

## Release v0.26.0 (2025-11-04)

A new release that includes a major overhaul of the acquisition framework! 

### New acquisition framework

#### coords parameter

We introduced a `coords` parameter to `Measure` and `Acquisition` operations. This allows the user to add meta information to the acquisition data that will be added to the returned dataset.

for example:

```
for freq in freqs:
    schedule.add(Measure("q0", coords={"freq": freq}, freq=freq))
```

which will result in a dataset containing "freq" as an additional coordinate.

#### unified return data

The return data from ScheduleGettable, InstrumentCoordinator, MeasurementControl is now the same: an `xarray.DataSet` object.

#### More flexibility for acquisitions

1. It is now possible to use the same acquisition channel for multiple acquisitions
2. It is not possible to use different acquisition protocols and binning modes on the same acquisition channel.


## Release v0.25.0 (2025-07-23)

Support for ALAP scheduling, improved pulse compensation and a few bug fixes.

- [ALAP scheduling](#alap-scheduling)
- [Improved pulse compensation](#improved-pulse-compensation)
- [Fix `ThresholdedAcquisition` with `binMode.AVERAGE` returns 0](#fix-thresholdedacquisition-with-binmodeaverage-returns-0)

see the [changelog](https://gitlab.com/quantify-os/quantify-scheduler/-/blob/main/CHANGELOG.md) for more details.

### ALAP scheduling

Users can now use `ALAP` to construct their schedules. E.g.

```{code-block} python
# Some minimal schedule
schedule = Schedule("test")
schedule.add(
    gate_library.Reset("q0"),
    label="Reset",
)
schedule.add(
    gate_library.X("q0"),
    label="X",
)
schedule.add(
    gate_library.Measure("q0"),
    label="Measure"
)

# Update timing constraints for ALAP scheduling
schedule.schedulables["Reset"]["timing_constraints"][0].update({
    "ref_schedulable": "X",
    "ref_pt_new": "end",
    "ref_pt": "start"
})
schedule.schedulables["X"]["timing_constraints"][0].update({
    "ref_schedulable": "Measure",
    "ref_pt_new": "end",
    "ref_pt": "start"
})
```

In this example, even though the gates were added in the order `Reset`, `X` and `Measure`, the `Reset` gate will be scheduled first, followed by the `X` gate, and finally the `Measure` gate.


### Improved pulse compensation

Pulse compensation now supports conditionals, and is more robust against running out of waveform memory.

e.g.

```{code-block} python
ref_schedule = Schedule()
ref_schedule.add(X("q0"))
ref_schedule.add(ConditionalOperation(X("q0"), "q0"))

schedule.add(PulseCompensation(body=ref_schedule))
```

will compile successfully. Additionally, we improved the compiler to prevent running out of waveform memory. 



### Fix `ThresholdedAcquisition` with `binMode.AVERAGE` returns 0

When using the `ThresholdedAcquisition` protocol together with `binMode.AVERAGE`, the returned data would be always mapped to 0, instead of return the actual averaged values. This has now been fixed.


## Release v0.23.0 (2025-04-01)

Bugfixes, new features, and more! Some of the highlights include:

- [Fix `MarkerPulse` raises "NCO frequency cannot be 'None' it must be int or float"](#fix-markerpulse-raises-nco-frequency-cannot-be-none-it-must-be-int-or-float)
- [Fix `Schedule.to_json/to_json_file()` raises "TypeError: Object of type 'ReferenceMagnitude' is not JSON serializable"](#fix-scheduleto_jsonto_json_file-raises-typeerror-object-of-type-referencemagnitude-is-not-json-serializable)
- [Fix incorrect acquisition values when using `TriggerCount` acquisition with `SUM` bin mode](#fix-incorrect-acquisition-values-when-using-triggercount-acquisition-with-sum-bin-mode)
- [Fix plotting a `Schedule` containing `LoopOperation` raises "AttributeError: 'LoopOperation' object has no attribute 'repetitions'"](#fix-plotting-a-schedule-containing-loopoperation-raises-attributeerror-loopoperation-object-has-no-attribute-repetitions)  
- [Breaking change: drop support for python 3.8](#breaking-change-drop-support-for-python-38)
- [Synchronize cluster on external triggers](#synchronize-cluster-on-external-triggers)
- [CZ gate for spin qubits](#cz-gate-for-spin-qubits)
- [DualThresholdedTriggerCount acquisition](#dualthresholdedtriggercount-acquisition)
- [Qblox NCO-related operations can now be scheduled on a 1 ns time grid](#qblox-nco-related-operations-can-now-be-scheduled-on-a-1-ns-time-grid)
- [WeightedThresholdedAcquisition](#weightedthresholdedacquisition)
- [Add RFSwitchToggle operation](#add-rfswitchtoggle-operation)
- [Interactive view on compiled instructions](#interactive-view-on-compiled-instructions)
- [Other](#other)

### Fix `MarkerPulse` raises "NCO frequency cannot be 'None' it must be int or float"

In the development version of quantify-scheduler, compiling schedules containing the `MarkerPulse` operation would raise the following error:

```
NCO frequency cannot be 'None' it must be int or float
```

this has been fixed.

### Fix `Schedule.to_json/to_json_file()` raises "TypeError: Object of type 'ReferenceMagnitude' is not JSON serializable"

Schedules that contain operations with a {class}`~quantify_scheduler.operations.ReferenceMagnitude` parameter would raise the following error when calling `Schedule.to_json()` and `Schedule.to_json_file()`:

```
TypeError: Object of type 'ReferenceMagnitude' is not JSON serializable
```

This has been fixed.

### Fix incorrect acquisition values when using `TriggerCount` acquisition with `SUM` bin mode

When using the {class}`~quantify_scheduler.operations.TriggerCount` acquisition with the `SUM` bin mode, the acquisition values were incorrect, where the number of repetitions was not taken into account. This has now been fixed.

### Fix plotting a `Schedule` containing `LoopOperation` raises "AttributeError: 'LoopOperation' object has no attribute 'repetitions'"

Plotting a pulse diagram of a {class}`~quantify_scheduler.Schedule` containing a {class}`~quantify_scheduler.operations.LoopOperation` would raise the following error:

```
AttributeError: 'LoopOperation' object has no attribute 'repetitions'
```

This has now been fixed.

### Breaking change: drop support for python 3.8

With this release, we drop support for python 3.8. 

### Synchronize cluster on external triggers

The Qblox Cluster can now be synchronized with an external trigger. This is useful when recording time tags using the {class}`~quantify_scheduler.operations.Timetag` and {class}`~quantify_scheduler.operations.TimetagTrace` acquisition protocols, where the time tags will be relative to the time of the external trigger.

This can be configured in the hardware description of the hardware configuration:

```{code-block} python
    "hardware_description": {
        "cluster0": {
            "instrument_type": "Cluster",
            "modules": {
                "2": {
                    "instrument_type": "QTM"
                }
            },
            "sequence_to_file": false,
            "ref": "internal",
            "sync_on_external_trigger": {
                "slot": 2,
                "channel": 1,
            }
        }
    }
```

For all the available options, see {class}`quantify_scheduler.backends.types.qblox.ExternalTriggerSyncSettings`.

### CZ gate for spin qubits

The CZ gate is now supported for {class}`~quantify_scheduler.BasicSpinElement`:

```{code-block} python
q1 = BasicSpinElement("spin1")
q2 = BasicSpinElement("spin2")
edge_q1_q2 = SpinEdge(parent_element_name=q1.name, child_element_name=q2.name)

schedule = Schedule("CZ gate")
schedule.add(CZ("spin1", "spin2"))
```

### DualThresholdedTriggerCount acquisition

Add a new operation for QTM modules. {class}`quantify_scheduler.operations.DualThresholdedTriggerCount` can take two count-thresholds and can send triggers on 4 different addresses depending on whether:

- counts < threshold_low
- threshold_low <= counts < threshold_high
- counts >= threshold_high
- counts invalid

Example:

```{code-block} python
schedule = Schedule("test")

trigcnt = schedule.add(
    DualThresholdedTriggerCount(
        port="q0:example",
        clock="digital",
        duration=10e-6,
        threshold_low=10,
        threshold_high=20,
        label_low="q0_low",
        label_mid="q0_mid",
        label_high="q0_high",
    ),
    ref_pt="start",
)

cond_body = Schedule("cond_body")
cond_body.add(X("q0"))

schedule.add(
    ConditionalOperation(body=cond_body, qubit_name="q0_low"),
    ref_op=trigcnt,
    rel_time=1000e-9,
)
schedule.add(IdlePulse(4e-9))
schedule.add(
    ConditionalOperation(body=cond_body, qubit_name="q0_mid"),
)
schedule.add(IdlePulse(4e-9))
schedule.add(
    ConditionalOperation(body=cond_body, qubit_name="q0_high"),
)
schedule.add(IdlePulse(4e-9))
```

### Qblox NCO-related operations can now be scheduled on a 1 ns time grid

With the new Qblox Cluster firmware, NCO-related operations no longer need to be scheduled on a 4 ns time grid, and can be scheduled at any time step (the minimum of 4 ns still applies). This concerns the following operations:

- {class}`~quantify_scheduler.operations.ResetClockPhase`,
- {class}`~quantify_scheduler.operations.SetClockFrequency`,
- {class}`~quantify_scheduler.operations.ShiftClockPhase`

### WeightedThresholdedAcquisition

We added a new protocol for acquisitions: {class}`~quantify_scheduler.operations.WeightedThresholdedAcquisition` that performs the same operation as {class}`~quantify_scheduler.operations.ThresholdedAcquisition` but instead of just summing all values you can supply weights.

#### Examples:

use as an Acquisition Operation

```{code-block} python
schedule.add(
    WeightedThresholdedAcquisition(
        weights_a=np.zeros(3, dtype=complex),
        weights_b=np.ones(3, dtype=complex),
        acq_rotation = 90,
        acq_threshold = 0.5,
        port="q0:res",
        clock="q0.ro",
    )
)
```


or, use qubit parameters together with the Measure gate


```{code-block} python
q0.measure.acq_weights_a(np.zeros(3, dtype=complex))
q0.measure.acq_weights_b(np.ones(3, dtype=complex))
q0.measure.acq_rotation(90)
q0.measure.acq_threshold(0.5)
schedule.add(Measure("q0", acq_protocol="WeightedThresholdedAcquisition"))
```


or, pass weights and thresholds as override parameters to the Measure gate

```{code-block} python
schedule.add(
    Measure(
        "q0",
        acq_protocol="WeightedThresholdedAcquisition",
        acq_weights_a=np.zeros(3, dtype=complex),
        acq_weights_b=np.ones(3, dtype=complex),
        acq_rotation=90,
        acq_threshold=0.5
    )
)
```


more info: WeightedThresholdedAcquisition

### Add RFSwitchToggle operation
In combination with the `rf_output_on=False` flag in the hardware config, the {class}`~~.quantify_scheduler.backends.qblox.operations.rf_switch_toggle.RFSwitchToggle` operation can be used to temporarily turn on the RF output of a module:

```{code-block} python
hardware_config = {
    "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
    "hardware_description": {
        "cluster0": {
            "instrument_type": "Cluster",
            "modules": {
                ...
                "8": {"instrument_type": "QCM_RF", "rf_output_on": False},
                ...
            },
            "sequence_to_file": False,
            "ref": "internal",
        }
    },
    "hardware_options": {
        ...
    },
    "connectivity": {
        ...
    },
}

schedule = Schedule("")
schedule.add(RFSwitchToggle(duration=100e-9, port=f"{qubit}:mw", clock=f"{qubit}.01"))
schedule.add(X(qubit.name), rel_time=30e-9)
schedule.add(IdlePulse(duration=200e-9))
```

### Interactive view on compiled instructions

When compiling a schedule to Qblox hardware, the compiled instructions will now display an interactive widget in jupyter notebooks:

```{code-block} python
schedule = Schedule("demo compiled instructions")
schedule.add(Reset("q0", "q4"))
schedule.add(X("q0"))
schedule.add(Y("q4"))
schedule.add(Measure("q0", acq_channel=0, acq_protocol='ThresholdedAcquisition'))
schedule.add(Measure("q4", acq_channel=1, acq_protocol='ThresholdedAcquisition'))

comp_schedule = compiler.compile(schedule)
comp_schedule.compiled_instructions
```

will show for example

```{image} ../images/compiled_instructions.png
:alt: Interactive compiled instructions
:width: 100%
```

### Other
- Add S, SDagger, T and TDagger gates to gate library ([!1191](https://gitlab.com/quantify-os/quantify-scheduler/-/merge_requests/1191) by [@Tim Vroomans](https://gitlab.com/TimVroomans))
- Adding A charge sensor to the spin back-end. ([!1171](https://gitlab.com/quantify-os/quantify-scheduler/-/merge_requests/1171) by [@Nicolas Piot](https://gitlab.com/npiot))
- Nv pulse shape change ([!1150](https://gitlab.com/quantify-os/quantify-scheduler/-/merge_requests/1150) by [@Vatshal Srivastav](https://gitlab.com/vsrivastav1))
- TimeRef relative to other channel of same module (SE-636) ([!1151](https://gitlab.com/quantify-os/quantify-scheduler/-/merge_requests/1151) by [@Thomas Middelburg](https://gitlab.com/ThomasMiddelburg))
- name parameter for Schedule and SerialCompiler are no longer required ([!1210](https://gitlab.com/quantify-os/quantify-scheduler/-/merge_requests/1210) by [@Leon Wubben](https://gitlab.com/LeonQblox))
- Feat: Make ScheduleGettable.compile public ([!1218](https://gitlab.com/quantify-os/quantify-scheduler/-/merge_requests/1218) by [@Leon Wubben](https://gitlab.com/LeonQblox))


## Release v0.22.2 (2025-01-17)

Bugfixes, compatibility with qblox-instruments v0.15.0, and more nv center element features. Some of the highlights include:

- [Fix v0.2 hardware config does not support multiplexing](#fix-v02-hardware-config-does-not-support-multiplexing)
- [ Fix gate-level pulse compensation operation compiling](#fix-gate-level-pulse-compensation-operation-compiling)
- [ Fix v0.2 hardware config does not support multiplexing](#fix-v02-hardware-config-does-not-support-multiplexing)
- [Fix low amplitude long ramp pulses do not play](#fix-low-amplitude-long-ramp-pulses-do-not-play)
- [Fix `BasicElectronicNVElement` misses info when serialized](#fix-basicelectronicnvelement-misses-info-when-serialized)
- [New acquisition protocol: `ThresholdedTriggerCount`](#new-acquisition-protocol-thresholdedtriggercount)
- [New bin mode for trigger count: `SUM`](#new-bin-mode-sum-for-triggercount)
- [Compatibility with qblox-instruments v0.15.0](#compatibility-with-qblox-instruments-v0150)

## Fix gate-level pulse compensation operation compiling

There was a bug with the `PulseCompensation` operation where the body argument wasn't compiled recursively. This has now been fixed.

## Fix v0.2 hardware config does not support multiplexing

There was a bug where multiplexed readout was no longer supported in the v0.2 hardware config format. This has now been fixed.

## Fix low amplitude long ramp pulses do not play

There was a bug where long ramp pulses with a small slope the amplitude would be rounded to zero. This has now been fixed. 

## Fix `BasicElectronicNVElement` misses info when serialized

When (de)serializing `BasicElectronicNVElement` objects using `to_json` and `from_json`, some of the qubit parameter values were missing. This has now been fixed. 

## New acquisition protocol: `ThresholdedTriggerCount`

A new AcquisitionProtocol has been added which the user to threshold trigger counts. This can be used in combination with conditional playback to play instructions only when the trigger count is above a certain threshold. Example:


```{code-block} python
schedule = Schedule("example")

schedule.add(ThresholdedTriggerCount(
            port="qe0:optical_readout",
            clock="qe0.ge0",
            duration=10e-6,
            threshold=10,
            feedback_trigger_label="qe0",
            feedback_trigger_condition=TriggerCondition.GREATER_EQUAL,
        )
)

sub_schedule = ...

schedule.add(ConditionalOperation(body=sub_schedule, qubit_name="qe0"), rel_time=...)

```

## New bin mode SUM for TriggerCount

A new bin mode `SUM` for `TriggerCount` has been added. This mode will sum the trigger counts over all runs. For example, a schedule that contains a single `TriggerCount` operation, while `APPEND` mode would return the trigger counts for each repetition of the schedule, the `SUM` mode will simply return the total number of times (across all repetitions) a trigger was counted. When using multiple `TriggerCount` operations, using multiple acquisition indices, the `SUM` mode will only sum the triggers specific to the acquisition index. For more information, please refer to the [trigger count acquisition tutorial](https://quantify-os.org/docs/quantify-scheduler/tutorials/Acquisitions.html#trigger-count-acquisition)

## Compatibility with qblox-instruments v0.15.0

This release adds compatibility with qblox-instruments v0.15.0. For more information, please refer to the [qblox-instruments release notes](https://docs.qblox.com/en/main/getting_started/whats_new.html#new-features).

## Release v0.22.1 (2024-11-21)

A small bug fix to make the package compatible with the latest version of pydantic. 

## Release v0.22.0 (2024-11-20)

Many updates and improvements in this version! Some of the highlights include:

- [Friendlier import paths for common quantify operations and other classes](#friendly-imports)
- [More spin operations](#more-spin-operations)
- [Crosstalk compensation](#crosstalk-compensation)
- [Improve conditional playback](#improve-conditional-playback)
- [Version 0.2 of the Qblox hardware compilation config](#version-02-of-the-qblox-hardware-compilation-config)
- [`ScheduleGettable` optionally returns `xarray.DataSet`](#schedulegettable-optionally-returns-xarray-dataset)
- [Fine delay support in QTM instructions](#fine-delay-support-in-qtm-instructions)
- [New bin mode for trigger count](#new-bin-mode-for-trigger-count)
- [Improve plotting of final data point](#improve-plotting-of-final-data-point)
- [Allow user to change `rel_tolerance` in `to_grid_time` function](#allow-user-to-change-rel_tolerance-in-to_grid_time-function)
- [More serialization features](#more-serialization-features)
- [Reloading hardware config](#reloading-hardware-config)

We also fixed a few bugs:

- [Fix unused `digitization_thresholds`](#fix-unused-digitization_thresholds)
- [Fix thresholded NaN values in Qblox backend](#fix-thresholded-nan-values-in-qblox-backend)

### Friendly imports

To reduce the number of imports in your code, we have added friendly import paths for common quantify operations and other classes:

```{code-block} python
from quantify_scheduler import Schedule, QuantumDevice, ScheduleGettable, ...
from quantify_scheduler.operations import Measure, DRAGPulse, ConditionalOperation, Trace, ...
```

for Qblox users, similar convenience imports are

```{code-block} python
from quantify_scheduler.qblox import ClusterComponent, ...
from quantify_scheduler.qblox.operations import ConditionalReset, SimpleNumericalPulse, ...
```

To see a complete list of available imports, please refer to the [API reference](https://quantify-os.org/docs/quantify-scheduler/dev/autoapi/quantify_scheduler/index.html). e.g.:

- [quantify_scheduler](https://quantify-os.org/docs/quantify-scheduler/dev/autoapi/quantify_scheduler/index.html)
- [quantify_scheduler.operations](https://quantify-os.org/docs/quantify-scheduler/dev/autoapi/quantify_scheduler/operations/index.html)
- [quantify_scheduler.qblox](https://quantify-os.org/docs/quantify-scheduler/dev/autoapi/quantify_scheduler/qblox/index.html)
- [quantify_scheduler.qblox.operations](https://quantify-os.org/docs/quantify-scheduler/dev/autoapi/quantify_scheduler/qblox/operations/index.html)

### More spin operations

We have added a few more operations specific for spin-based elements. These include:

- [`quantify_scheduler.operations.SpinEdge`](https://quantify-os.org/docs/quantify-scheduler/dev/autoapi/quantify_scheduler/device_under_test/spin_edge/index.html#quantify_scheduler.device_under_test.spin_edge.SpinEdge)
- [`quantify_scheduler.operations.SpinInit`](https://quantify-os.org/docs/quantify-scheduler/dev/autoapi/quantify_scheduler/operations/spin_library/index.html#quantify_scheduler.operations.spin_library.SpinInit)
- [`quantify_scheduler.operations.SpinPSB`](https://quantify-os.org/docs/quantify-scheduler/dev/autoapi/quantify_scheduler/operations/spin_operations/index.html)

### Improve conditional playback

We have added support for generic subschedules in conditional playback. This allows for more flexible and reusable conditional logic in your schedules. For more details, please refer to the [Conditional playback](https://quantify-os.org/docs/quantify-scheduler/dev/tutorials/Conditional%20Reset.html) tutorial and the [reference guide](https://quantify-os.org/docs/quantify-scheduler/dev/reference/conditional_playback.html).

````{admonition} Breaking change
:class: warning
For Qblox users, this is a breaking change. For conditional operations please import from `quantify_scheduler.qblox.operations`:

```python
from quantify_scheduler.qblox.operations import ConditionalOperation, ConditionalReset
```
````

### Crosstalk compensation

For Qblox users we have added support for cross talk compensation to remove unwanted interference and interactions between qubits. For example, if we have a 2 qubit device with qubits `q0` and `q1`, we can define a cross talk matrix in the hardware configuration:

```{code-block} python
hardware_comp_cfg = {
    ...
    "hardware_options": {
        "modulation_frequencies": {
            "q0:gt-q0.01": {"interm_freq": 7e9},
            "q1:gt-q1.01": {"interm_freq": 7e9},
        },
        "crosstalk_compensation_enable": True,
        "crosstalk": {
            "q0:gt-q0.01": {
                "q1:gt-q1.01": 0.5
            },
            "q1:gt-q1.01": {
                "q0:gt-q0.01": 0.5
            }
        },
    },
    "connectivity": {
        "graph": [
            ("cluster0.module2.real_output_0", "q0:gt"),
            ("cluster0.module2.real_output_1", "q1:gt"),
        ]
    },
}
```

Which will compensate pulses using a linear combination of weights:

$$
\begin{pmatrix} 
1 & 0.5 \\ 
0.5 & 1 
\end{pmatrix}^{-1}
\begin{pmatrix} 
f(t)\\
g(t)
\end{pmatrix} 
\Rightarrow \frac{1}{3}
\begin{pmatrix} 
4\,f(t) -2\,g(t)\\
-2\,f(t) + 4\,g(t)
\end{pmatrix}
$$

where $f(t)$ and $g(t)$ are the original pulses that were scheduled on qubits `q0` and `q1` respectively.


### Version 0.2 of the Qblox hardware compilation config

The Qblox hardware compilation configuration is now versioned to make it easier for us to make changes without breaking existing code. v0.2 brings a behavioral change compared to v0.1 (and unversioned), where `Measure` operations will require to have both input and output channels specified. 

For example, previously:

```{code-block} python
{
    "version": "0.1", #<-- this field is optional
    "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
    "hardware_description": {
        "cluster0": {
            "instrument_type": "Cluster",
            "ref": "internal",
            "ip": None,
            "modules": {"1": {"instrument_type": "QCM"}},
        }
    },
    "hardware_options": {},
    "connectivity": {
        "graph": [["cluster0.module1.complex_output_0", "q0:mw"]]
    }
}
```

`complex_output_0` in the connectivity would imply both input and output channels. In v0.2, this will raise an error, as the compiler will not know which channel to use as input and which as output. The user will have to specify this explicitly:

```{code-block} python
{
    "version": "0.2",
    "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
    "hardware_description": {
        "cluster0": {
            "instrument_type": "Cluster",
            "ref": "internal",
            "ip": None,
            "modules": {
                "1": {"instrument_type": "QRM"},
                "2": {"instrument_type": "QRM_RF"},
                },
        }
    },
    "hardware_options": {},
    "connectivity": {
        "graph": [
            ["cluster0.module1.real_output_0", "q0:res"],
            ["cluster0.module1.real_input_1", "q0:res"],
            ["cluster0.module2.complex_output_0", "q1:res"],
            ["cluster0.module2.complex_input_0", "q1:res"],
        ]
    }
}
```

See the [Qblox hardware compilation config](https://quantify-os.org/docs/quantify-scheduler/dev/reference/qblox/Hardware%20config%20versioning.html) reference guide for more details.

### `ScheduleGettable` optionally returns `xarray.DataSet`

We introduced a new boolean parameter `return_xarray` for the `ScheduleGettable` class. When set to `True`, the `ScheduleGettable` will return an `xarray.DataSet` object instead of a list of numpy arrays:

```{code-block} python
gettable = ScheduleGettable(
    single_qubit_device,
    schedule_function=schedule_function,
    schedule_kwargs=schedule_kwargs,
    batched=True,
    return_xarray=True,
)
```

```{admonition} Breaking change
:class: warning
This is a breaking change for quantify-core users. In order to use `MeasurementControl` with quantify-scheduler v0.22.0 or later, you will have to update quantify-core as well to v0.7.9 or later.
```

### Fine delay support in QTM instructions

We have added support for fine delay in QTM instructions. This allows for picosecond control of the `MarkerPulse` operation. e.g.

```{code-block} python
schedule.add(
    MarkerPulse(
        duration=40e-9,
        fine_start_delay=391e-12,
        fine_end_delay=781e-12,
        port="qe1:switch",
    )
)
```

the fine delay parameters should be multiples of 1/128 ns. 

### New bin mode for trigger count

We have renamed a bin mode for trigger count. The `AVERAGE` mode has been replaced with `DISTRIBUTION`. The `DISTRIBUTION` mode records the number of times the trigger condition was met. 

For example, if a schedule ran 8 times and we measured:

- in 1 run, the trigger was counted 5 times,
- in 2 runs, the trigger was counted 4 times,
- in 3 runs, the trigger was counted 2 times,
- in 2 runs, the trigger was counted once.

a trigger count acquisition, together with the `DISTRIBUTION` bin mode, will return a dataset of the following form:

```{glue:} trigger_dataset
```

For more details, please visit the [Trigger count](https://quantify-os.org/docs/quantify-scheduler/dev/reference/acquisition_protocols.html#trigger-count) reference guide.


### Improve plotting of final data point

Due to interpolation, the last data point in most waveform plots were slightly off. This was most visible in ramp and staircase pulses. This has now been fixed.

e.g. a before and after plot for a RampPulse and GaussPulse:

```{image} ../images/before-after-pulses.png
:alt: Ramp and Gauss pulse before and after
:width: 100%
```

### Allow user to change `rel_tolerance` in `to_grid_time` function

Sometimes floating point errors accumulate causing the `to_grid_time` to raise a `ValueError` if the time is not a multiple of the grid time within the default tolerance:

```
ValueError: Attempting to use a time value of 335872000000.0059 ns. 
Please ensure that the durations of operations and wait times between 
operations are multiples of 4 ns (tolerance: 1e-03 ns).
```

You can now lower the tolerance by setting `constants.GRID_TIME_TOLERANCE_TIME` to a larger value, e.g.

```{code-block} python
from quantify_scheduler.backends.qblox import constants
constants.GRID_TIME_TOLERANCE_TIME = 0.1
```

The default is set to 0.0011 ns. 

### Fix unused digitization_thresholds

When setting the digitization thresholds in the hardware options, e.g.

```{code-block} python
hardware_options = {
    "digitization_thresholds": {
        "qe0:optical_readout-qe0.ge0": {"in_threshold_primary": 0.5}
    }
}
```

These were not properly propagated to the hardware. This is now fixed.

### Fix thresholded NaN values in Qblox backend

In some cases NaN values are returned by acquisitions. For example when using a dummy hardware configuration, or when scheduling acquisitions inside a conditional operation. This would lead to a `ValueError` when trying to convert NaN to an integer. This is now fixed, by mapping them to `-1`. 

### More serialization features

We have added more (de)serialization features to allow for more flexible and reusable schedules. We can now use `to_json`, `from_json`, `to_json_file` and `from_json_file` on `QuantumDevice`, `DeviceElement` (e.g. `BasicTransmonElement`, `BasicNVCenterElement`, & `BasicSpinElement`) and `Schedule` objects.

For more examples, please refer to the [Serialization](https://quantify-os.org/docs/quantify-scheduler/dev/examples/serialization.html) page.

### Reloading hardware config

In additional to serializing the quantum device, we can now also (re)load hardware configurations from JSON files:

```{code-block} python
from quantify_scheduler import QuantumDevice

quantum_device = QuantumDevice("")
quantum_device.hardware_config.load_from_json_file("path/to/hardware_config.json")
```

Similarly, we can serialize the hardware configuration to a JSON file:

```{code-block} python
quantum_device.hardware_config.write_to_json_file("path/to/hardware_config.json")
```
