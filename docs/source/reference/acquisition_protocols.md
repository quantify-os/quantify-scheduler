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
myst:
  substitutions:
    InstrumentCoordinator: "{class}`~quantify_scheduler.instrument_coordinator.instrument_coordinator.InstrumentCoordinator`"
---

(sec-acquisition-protocols)=
# Acquisition protocols

The dataset returned by
{meth}`InstrumentCoordinator.retrieve_acquisition() <quantify_scheduler.instrument_coordinator.instrument_coordinator.InstrumentCoordinator.retrieve_acquisition>`
consists of a number of {class}`~xarray.DataArray`s containing data for every
acquisition channel.
This document specifies the format of these data arrays.

```{code-cell} ipython3
---
tags: [hide-cell]
mystnb:
    code_prompt_show: "Imports and auxiliary definitions"
    code_prompt_hide: "Hide imports and auxiliary definitions"
---
import numpy as np
import xarray as xr
import hvplot.xarray

intermodulation_freq = 1e8  # 100 MHz
voltage_iq = 0.32 + 0.25j
sampling_rate = 1.8e9  # 1.8 GSa/s
readout_duration = 1e-7  # 100 ns
time_grid = xr.DataArray(
    np.arange(0, readout_duration, sampling_rate**-1), dims="trace_index"
)
```

(sec-acquisition-protocols-trace)=
## (Demodulated) Trace Acquisition Protocol

- Referred to as `"Trace"`.
- Supported by {mod}`Qblox <quantify_scheduler.backends.qblox>` and
  {mod}`Zurich Instruments <quantify_scheduler.backends.zhinst>` backends.

Readout equipment digitizes a {math}`V(t)`, where {math}`V = V_I + i V_Q` is a complex
voltage on inputs of a readout module (up/down conversion of a signal with an IQ mixer
is assumed).
The signal is demodulated with an intermodulation frequency configured for a readout
port.

For example, if we have a readout module (like Qblox QRM or Zurich Instruments
UHFQA) that is perfect, and connect its outputs to its inputs directly, raw input on a readout port
will look like this:

```{code-cell} ipython3
---
tags: [hide-input]
---
raw_trace = (
    voltage_iq * np.exp(2j * np.pi * intermodulation_freq * time_grid)
).assign_coords({"trace_time": time_grid})
xr.Dataset({"I": raw_trace.real, "Q": raw_trace.imag}).hvplot(
    x="trace_time", xlabel="t [s]", ylabel="V", group_label="Channel"
)
```

Demodulated trace will unroll this data with respect to the intermodulation frequency,
so the resulting I and Q readouts will look like this:

```{code-cell} ipython3
---
tags: [hide-input]
---
demodulated_trace = raw_trace * np.exp(-2j * np.pi * intermodulation_freq * time_grid)

xr.Dataset({"I": demodulated_trace.real, "Q": demodulated_trace.imag}).hvplot(
    x="trace_time", xlabel="t [s]", ylabel="V", group_label="Channel"
)
```

This acquisition protocol is currently supported only in `BinMode.AVERAGE` binning mode.
The resulting dataset must contain data arrays with two dimensions for each acquisition
channel: acquisition index (number of an acquisition in a schedule) and trace index
(that corresponds to time from the start of the acquisition).
All the dimension names should be suffixed with the acquisition channel to avoid
conflicts while merging the datasets.
It is recommended to annotate the trace index dimension with a coordinate that describes the
time since the start of the acquisition.
For example, if two acquisition channels read out once, the resulting dataset should have
the following structure:

```{code-cell} ipython3
---
tags: [hide-input]
---
xr.Dataset(
    {
        0: demodulated_trace.expand_dims("acq_index_0", 0).rename(
            {"trace_index": "trace_index_0", "trace_time": "trace_time_0"}
        ),
        1: demodulated_trace.expand_dims("acq_index_1", 0).rename(
            {"trace_index": "trace_index_1", "trace_time": "trace_time_1"}
        ),
    }
)
```

(sec-acquisition-protocols-ssb-integration-complex)=
## Single-sideband Complex Integration

- Referred to as `"SSBIntegrationComplex"`.
- Supported by {mod}`Qblox <quantify_scheduler.backends.qblox>` and
  {mod}`Zurich Instruments <quantify_scheduler.backends.zhinst>` backends.

In this acquisition protocol acquired voltage trace gets demodulated and averaged.
For each acquisition, a single complex voltage value is returned
({math}`V_I + i V_Q`).

This acquisition protocol supports `BinMode.APPEND` binning mode for single-shot readout
and `BinMode.AVERAGE` binning mode for returning data averaged for several
executions of a schedule.
In the first case data arrays for each acquisition channel will have two dimensions:
repetition and acquisition index.
All the dimension names except repetition should be suffixed with the acquisition
channel to avoid conflicts while merging the datasets, the repetition dimension must be
named `"repetition"`.
For example, two acquisition channels of which acquisition channel 0 read out three
times and acquisition channel two read out two times, the resulting dataset should have
the following structure in `BinMode.APPEND`:

```{code-cell} ipython3
---
tags: [hide-input]
---
xr.Dataset(
    {
        0: demodulated_trace.reduce(np.average, "trace_index").expand_dims(
            {"repetition": 5, "acq_index_0": 3}
        ),
        2: demodulated_trace.reduce(np.average, "trace_index").expand_dims(
            {"repetition": 5, "acq_index_2": 2}
        ),
    }
)
```

In `BinMode.AVERAGE` repetition dimension gets reduced and only the acquisition index
dimension is left for each channel:

```{code-cell} ipython3
---
tags: [hide-input]
---
xr.Dataset(
    {
        0: demodulated_trace.reduce(np.average, "trace_index").expand_dims(
            {"acq_index_0": 3}
        ),
        2: demodulated_trace.reduce(np.average, "trace_index").expand_dims(
            {"acq_index_2": 2}
        ),
    }
)
```

(sec-acquisition-protocols-thresholded-acquisition)=
## Thresholded Acquisition

- Referred to as `"ThresholdedAcquisition"`.
- Supported by the {mod}`Qblox <quantify_scheduler.backends.qblox>` backend.

This acquisition protocol is similar to the {ref}`SSB complex integration <sec-acquisition-protocols-ssb-integration-complex>`, but in this case, the obtained results are compared against a threshold value to obtain 0 or 1. The purpose of this protocol is to discriminate between qubit states. 

For example, when acquiring on a single acquisition channel with `BinMode.APPEND` and `repetitions=12`, the corresponding dataset could look like:
```{code-cell} ipython3
---
tags: [hide-input]
---
thresholded_data = np.array([0,0,1,0,1,0,0,1,1,0,0,1])
xr.Dataset(
    {0: xr.DataArray(thresholded_data.reshape(1,12), dims = ['acq_index_0', 'repetitions'])}
)
```

In using `BinMode.AVERAGE`, the corresponding dataset could like:

```{code-cell} ipython3
---
tags: [hide-input]
---
xr.Dataset(
    {0: xr.DataArray(np.mean(thresholded_data, keepdims=1), dims = ['acq_index_0'])}

)
```

(sec-acquisition-protocols-numerical-weighted-integration-separated)=
## Numerical Separated Weighted Integration

- Referred to as `"NumericalSeparatedWeightedIntegration"`.
- Supported by the {mod}`Qblox <quantify_scheduler.backends.qblox>` backend.

Equivalent to
{ref}`SSB complex integration <sec-acquisition-protocols-ssb-integration-complex>`,
but instead of a simple average of a demodulated signal, the signal is weighted
with two waveforms and then integrated. One waveform for the real part of the
signal, and one for the imaginary part. The dataset format is also the same.

Integration weights should normally be calibrated in a separate experiment
(see, for example, {cite:t}`magesan2015machine`).

(sec-acquisition-protocols-numerical-weighted-integration)=
## Numerical Weighted Integration

- Referred to as `"NumericalWeightedIntegration"`.
- Supported by the {mod}`Qblox <quantify_scheduler.backends.qblox>` backend.

Equivalent to
{ref}`Numerical Separated Weighted Integration <sec-acquisition-protocols-numerical-weighted-integration-separated>`,
but the real part of the output is the sum of the real and imaginary part
of the output from the {ref}`Numerical Separated Weighted Integration <sec-acquisition-protocols-numerical-weighted-integration-separated>` protocol. The dataset format is also the same.
This is equivalent to multiplying the complex signal with complex waveform
weights, and only returning the real part of the result. If the integration
weights are calibrated as in {cite:t}`magesan2015machine`, i.e. the complex
weights are the difference between the two signals we wish to distinguish,
then the real part of the complex weighted multiplication contains all the
relevant information required to distinguish between the states, and the
imaginary part contains only noise.

Integration weights should normally be calibrated in a separate experiment
(see, for example, {cite:t}`magesan2015machine`).

(sec-acquisition-protocols-weighted-thresholded-acquisition)=
## Numerical Weighted Thresholded Acquisition

- Referred to as `"WeightedThresholdedAcquisition"`.
- Supported by the {mod}`Qblox <quantify_scheduler.backends.qblox>` backend.

This protocol will perform a numerical separated weighted integration before thresholding the results.

### Examples

```{code-block} python
# use as an Acquisition Operation
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

# or, use qubit parameters together with the Measure gate
q0.measure.acq_weights_a(np.zeros(3, dtype=complex))
q0.measure.acq_weights_b(np.ones(3, dtype=complex))
q0.measure.acq_rotation(90)
q0.measure.acq_threshold(0.5)
schedule.add(Measure("q0", acq_protocol="WeightedThresholdedAcquisition"))

# or, pass weights and thresholds as override parameters to the Measure gate
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

```{note}
The resulting datasets are similar to those in {ref}`Thresholded Acquisition <sec-acquisition-protocols-thresholded-acquisition>`. While the dataset returned by {{ InstrumentCoordinator }} when using a regular thresholded acquisition returns a single row of data for each acquisition channel, the weighted thresholded acquisition currently returns two rows per channel: one with actual thresholded values, and the second with all zeros.
```


(sec-acquisition-protocols-trigger-count)=
## Trigger Count

- Referred to as `"TriggerCount"`.
- Supported by the {mod}`Qblox <quantify_scheduler.backends.qblox>` backend.

```{admonition} Note
Please also see {ref}`sec-qblox-acquisition-details` for more information on Qblox module-specific behavior of this operation.
```

This acquisition protocol measures how many times a predefined voltage threshold has been
passed. For the QRM, the threshold is set via {class}`~quantify_scheduler.backends.types.qblox.SequencerOptions.ttl_acq_threshold` (see also {ref}`sec-qblox-sequencer-options`), while for the QTM this threshold setting is a dedicated hardware option called `analog_threshold`, see {ref}`sec-qblox-digitization-thresholds`.

First, let's see an example when the bin mode is `BinMode.APPEND`.
The returned data for the acquisition channel contains the number of triggers counted for each acquisition index. In the following example, suppose a schedule with one trigger count acquisition was executed 5 times (`repetitions=5`). In order, the number of triggers counted is `[6, 3, 8, 1, 3]`. The resulting dataset would then look like:

```{code-cell} ipython3
---
tags: [hide-input]
---
trigger_data = np.array([6, 3, 8, 1, 3])
xr.Dataset(
    {0: xr.DataArray(trigger_data.reshape(1, 5), dims = ['acq_index_0', 'repetitions'])}
)
```

In `BinMode.DISTRIBUTION` mode, the result is a _distribution_ that maps the trigger count numbers to the number of occurrences of each trigger count number.
This provides insights into the overall occurrence of triggers when running the acquisition multiple times.

For example, in the example below, the schedule ran 8 times. From these 8 runs,
- in 1 run, the trigger was counted 5 times,
- in 2 runs, the trigger was counted 4 times,
- in 3 runs, the trigger was counted 2 times,
- in 2 runs, the trigger was counted once.

Note: 0 counts are removed from the returned data, so there will be no entry for "3 times".

You can think of the append mode values as the cumulative distribution of the distribution mode values.
See an example below.
```{code-cell} ipython3
---
tags: [hide-input]
---
trigger_data = [1, 2, 3, 2]
counts = [5, 4, 2, 1]
xr.Dataset(
    {0: xr.DataArray([trigger_data],
            dims=["repetition", "counts"],
            coords={"repetition": [0], "counts": counts},
        )
    }
)
```

(sec-acquisition-protocols-thresholded-trigger-count)=
## Thresholded Trigger Count

- Referred to as `"ThresholdedTriggerCount"`.
- Supported by the {mod}`Qblox <quantify_scheduler.backends.qblox>` backend.

```{admonition} Note
Please also see {ref}`sec-qblox-acquisition-details` for more information on Qblox module-specific behavior of this operation.
```

This acquisition protocol returns the result of the threshold comparison. I.e., it returns a `1` if the comparison is `True` and a `0` if it is `False`. It can also be used together with the {class}`~quantify_scheduler.operations.control_flow_library.ConditionalOperation` to execute instructions based on the threshold comparison result.

Note that the _analog_ thresholded for registering a single trigger is set via the hardware configuration. For the QRM, the analog threshold is set via {class}`~quantify_scheduler.backends.types.qblox.SequencerOptions.ttl_acq_threshold` (see also {ref}`sec-qblox-sequencer-options`), while for the QTM this threshold setting is a dedicated hardware option called `analog_threshold`, see {ref}`sec-qblox-digitization-thresholds`.

The only available bin mode is `BinMode.APPEND`.

As mentioned above, the returned data for the acquisition channel contains the result of the threshold comparison for each acquisition index. For example, suppose a schedule with one trigger count acquisition was executed 5 times (`repetitions=5`) with a threshold set to 4 counts and `trigger_condition=TriggerCondition.LESS_THAN`. In order, the number of triggers counted is `[6, 3, 8, 1, 3]`. The resulting dataset would then look like:

```{code-cell} ipython3
---
tags: [hide-input]
---
trigger_data = np.array([0, 1, 0, 1, 1])
xr.Dataset(
    {0: xr.DataArray(trigger_data.reshape(1, 5), dims = ['acq_index_0', 'repetitions'])}
)
```

(sec-acquisition-protocols-dual-thresholded-trigger-count)=
## Dual Thresholded Trigger Count

- Referred to as `"DualThresholdedTriggerCount"`.
- Supported by the {mod}`Qblox <quantify_scheduler.backends.qblox>` backend.

```{admonition} Note
Please also see {ref}`sec-qblox-acquisition-details` for more information on Qblox module-specific behavior of this operation.
```

Just like the `ThresholdedTriggerCount`, this acquisition protocol can be used together with the {class}`~quantify_scheduler.operations.control_flow_library.ConditionalOperation` to execute instructions based on the threshold comparison result. With `DualThresholdedTriggerCount`, there are 2 thresholds you can pass and 4 possible results:

- "low" when `counts < low_threshold`,
- "mid" when `low_threshold <= counts < high_threshold`,
- "high" when `counts >= high_threshold`,
- "invalid" when the counts are invalid (can occur in extremely rare cases, for example when the counter overflows).

In the returned acquisition data, only the raw counts are present.

You can use any number of the possible threshold comparison results by providing an appropriate label (`"label_<type>"` where type is one of "low", "mid", "high", "invalid") as an argument, and using that same label for a `ConditionalOperation`.

The only available bin mode is `BinMode.APPEND`.


(sec-acquisition-protocols-timetag)=
## Timetag acquisition

- Supported by the {mod}`Qblox <quantify_scheduler.backends.qblox>` backend, only on QTM modules.

```{admonition} Note
Please also see {ref}`sec-qblox-acquisition-details` for more information on Qblox module-specific behavior of this operation.
```

The {class}`~quantify_scheduler.operations.acquisition_library.Timetag` acquisition protocol (referred to as `"Timetag"`) measures the point in time at which a voltage threshold was passed with a rising edge (for Qblox QTM modules, this voltage threshold is set with the {ref}`sec-qblox-digitization-thresholds` hardware option). The timetag is the difference between a time source and a time reference.

The source of the timetag itself can be one of:

- The first recorded rising edge (`TimeSource.FIRST`),
- The second recorded rising edge (`TimeSource.SECOND`),
- The last recorded rising edge (`TimeSource.LAST`).

The time reference can be one of:

- The start of the acquisition window (`TimeRef.START`),
- The end of the acquisition window (`TimeRef.END`),
- The first measured rising edge (`TimeRef.FIRST`),
- A scheduled {class}`~quantify_scheduler.operations.pulse_library.Timestamp` operation (`TimeRef.TIMESTAMP`).
- A timetag acquisition on another port (`TimeRef.PORT` in combination with an argument for `time_ref_port`)

The protocol always returns one timetag per acquisition bin. If `BinMode.APPEND` is used, the acquisition bin index is incremented automatically and each timetag measurement is put in a separate bin. For example, let's look at the schedule below, which is repeated three times.

```{code-cell} ipython3
---
mystnb:
  remove_code_outputs: true
---
from quantify_scheduler import Schedule
from quantify_scheduler.enums import BinMode, TimeSource, TimeRef
from quantify_scheduler.operations.pulse_library import Timestamp
from quantify_scheduler.operations.acquisition_library import Timetag

sched = Schedule("Timetag", repetitions=3)

sched.add(Timestamp(port="qe0:optical_readout", clock="qe0.ge0"))
sched.add(
    Timetag(
        duration=10e-6,
        port="qe0:optical_readout",
        clock="qe0.ge0",
        time_source=TimeSource.FIRST,
        time_ref=TimeRef.TIMESTAMP,
        bin_mode=BinMode.APPEND,
    ),
    rel_time=500e-9,
)
```

An experiment with this schedule will return a dataset that may look like this:

```{code-cell} ipython3
---
tags: [hide-input]
---
data_array = xr.DataArray(
        np.array([5438.2, 756.16, 1059.2]).reshape((3, 1)),
        dims=["repetition", "acq_index_0"],
        coords={"acq_index_0": [0]},
        attrs={"acq_protocol": "Timetag"},
    )
xr.Dataset({0: data_array})
```

If `BinMode.AVERAGE` is used, the acquisition data will contain the average of the timetags recorded in each bin. If only bin index 0 was used for the three acquisitions in the above example, the data set may look like this:

```{code-cell} ipython3
---
tags: [hide-input]
---
data_array = xr.DataArray(
        [2417.853333333333],
        dims=["acq_index_0"],
        coords={"acq_index_0": [0]},
        attrs={"acq_protocol": "Timetag"},
    )
xr.Dataset({0: data_array})
```

### Measuring a time difference between pulses on two ports

If you have pulses coming from two different ports and you want to measure the time difference between them, you can use `TimeRef.PORT` on one `Timetag` acquisition, in combination with providing the port of another `Timetag` for the `time_ref_port` parameter.

Note that the timetag acquisition that you are recording relative to (i.e., the one that is _not_ using `TimeRef.PORT`) _must_ end either before or at the end of the other timetag acquisition.

For both timetag acquisitions, all values of `time_source` are allowed, and for the acquisition that is not using `TimeRef.PORT`, all other `TimeRef` values are allowed as well.

For example, the code below shows an experiment where the acquisition with `acq_channel=0` records the timetag of the first measured pulse, relative to the start of the acquisition, and `acq_channel=1` records the timetag of the first measured pulse, relative to the timetag in `acq_channel=0`.

```{code-cell} ipython3
---
mystnb:
  remove_code_outputs: true
---
from quantify_scheduler import Schedule
from quantify_scheduler.enums import BinMode, TimeSource, TimeRef
from quantify_scheduler.operations.pulse_library import Timestamp
from quantify_scheduler.operations.acquisition_library import Timetag

schedule = Schedule("Relative timetag", repetitions=1)
schedule.add(
    Timetag(
        duration=10e-6,
        port="qe0:optical_readout",
        clock="qe0.ge0",
        acq_channel=0,
        time_source=TimeSource.FIRST,
        time_ref=TimeRef.START,
    )
)
schedule.add(
    Timetag(
        duration=10e-6,
        port="qe1:optical_readout",
        clock="qe1.ge0",
        acq_channel=1,
        time_source=TimeSource.FIRST,
        time_ref=TimeRef.PORT,
        time_ref_port="qe0:optical_readout",
    ),
    # Start the acquisition simultaneously with the previous one.
    ref_pt="start",
)
```

(sec-acquisition-protocols-timetag-trace)=
## TimetagTrace acquisition

- Supported by the {mod}`Qblox <quantify_scheduler.backends.qblox>` backend, only on QTM modules.

```{admonition} Note
Please also see {ref}`sec-qblox-acquisition-details` for more information on Qblox module-specific behavior of this operation.
```

The {class}`~quantify_scheduler.operations.acquisition_library.TimetagTrace` acquisition protocol (referred to as `"TimetagTrace"`) measures all points in time at which a voltage threshold is passed (with a rising edge), while the acquisition window is active. For Qblox QTM modules, this voltage threshold is set with the {ref}`sec-qblox-digitization-thresholds` hardware option. Each timetag value is the difference between the time of the rising edge and a time reference.

The time reference can be one of:

- The start of the acquisition window,
- The end of the acquisition window,
- The first measured rising edge,
- A scheduled {class}`~quantify_scheduler.operations.pulse_library.Timestamp` operation.

The only usable bin mode at this moment is `BinMode.APPEND`. If the schedule is repeated multiple times, timetags of each repetition will be appended to the acquisition data. Please note that the returned xarray Dataset is always **rectangular**. This means that in the case that different amount of pulses are timetagged in each repetition, the sub-arrays are padded with `np.NaN` to ensure uniformity.

For example, let's take the following schedule:

```{code-cell} ipython3
---
mystnb:
  remove_code_outputs: true
---
from quantify_scheduler import Schedule
from quantify_scheduler.enums import BinMode, TimeSource, TimeRef
from quantify_scheduler.operations.pulse_library import Timestamp
from quantify_scheduler.operations.acquisition_library import TimetagTrace

sched = Schedule("Timetag", repetitions=3)

sched.add(Timestamp(port="qe0:optical_readout", clock="qe0.ge0"))
sched.add(
    TimetagTrace(
        duration=10e-6,
        port="qe0:optical_readout",
        clock="qe0.ge0",
        time_ref=TimeRef.TIMESTAMP,
    ),
    rel_time=500e-9,
)
```

This schedule could produce data that looks like this:

```{code-cell} ipython3
---
tags: [hide-input]
---
data_array = xr.DataArray(
        np.array([1227.94775391, 605.43261719, 3720.31591797, np.nan, np.nan, 4307.07177734, 6605.31689453, np.nan, np.nan, np.nan, 2063.68652344, 3743.87255859, 3121.44726562, 1534.71484375, 3273.87792969]).reshape((3, 1, 5)),
        dims=["repetition", "acq_index_0", "trace_index_0"],
        coords={"acq_index_0": [0], "trace_index_0": list(range(5))},
        attrs={"acq_protocol": "TimetagTrace"},
    )
xr.Dataset({0: data_array})
```
