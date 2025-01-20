# Release Notes

## Release v0.22.3 (2025-03-19)

Intermediate release to add the `WeightedThresholdedAcquisition` operation. This allows the user to pass integration weights to thresholded acquisitions:

**use as an Acquisition Operation**
```python
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

**or, use qubit parameters together with the Measure gate**
```python
q0.measure.acq_weights_a(np.zeros(3, dtype=complex))
q0.measure.acq_weights_b(np.ones(3, dtype=complex))
q0.measure.acq_rotation(90)
q0.measure.acq_threshold(0.5)
schedule.add(Measure("q0", acq_protocol="WeightedThresholdedAcquisition"))
```

**or, pass weights and thresholds as override parameters to the Measure gate**
```python
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
more info: [WeightedThresholdedAcquisition](https://quantify-os.org/docs/quantify-scheduler/dev/reference/operations/WeightedThresholdedAcquisition.html)

---

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
