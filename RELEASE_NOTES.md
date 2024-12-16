# Release Notes

## Release v0.22.1

A small bug fix to make the package compatible with the latest version of pydantic. 

## Release v0.22.0

Many updates and improvements in this version! Some of the highlights include:

- [Friendlier import paths for common quantify operations and other classes](#friendly-imports)
- [More spin operations](#more-spin-operations)
- [Virtual gates](#virtual-gates)
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

- [`quantify_scheduler.operations.SpinEdge`](https://quantify-os.org/docs/quantify-scheduler/dev/autoapi/quantify_scheduler/operations/spin_operations/index.html)
- [`quantify_scheduler.operations.SpinInit`](https://quantify-os.org/docs/quantify-scheduler/dev/autoapi/quantify_scheduler/operations/spin_operations/index.html)
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

### Virtual gates

For Qblox users we have added support for virtual gates (cross talk compensation) to compensate unwanted interference and interactions between qubits. For example, if we have a 2 qubit device with qubits `q0` and `q1`, we can define a virtual gate in the hardware configuration:

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
\end{pmatrix} 
\begin{pmatrix} 
f(t)\\
g(t)
\end{pmatrix} 
\Rightarrow 
\begin{pmatrix} 
f(t) + 0.5\,g(t)\\
0.5\,f(t) + g(t)
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
