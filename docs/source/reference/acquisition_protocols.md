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
## Numerical Separated Weighted

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

(sec-acquisition-protocols-trigger-count)=
## Trigger Count

- Referred to as `"TriggerCount"`.
- Supported by the {mod}`Qblox <quantify_scheduler.backends.qblox>` backend.

This acquisition protocol measures how many times a predefined voltage threshold has been
passed. The threshold is set via {class}`~quantify_scheduler.backends.types.qblox.SequencerOptions.ttl_acq_threshold` (see also {ref}`sec-qblox-sequencer-options-new`).

First, let's see an example when the bin mode is `BinMode.APPEND` and the schedule repeats once (`repetitions=1`).
The returned data for the acquisition channel is a list, with as many elements as the number of times the trigger was activated. For each element, the value is `1`. In the following example, the threshold was passed 5 times for acquisition channel 0, and therefore `acq_index_0` goes from `0` to `4`, with values `1`.
```{code-cell} ipython3
---
tags: [hide-input]
---
trigger_data = [1, 1, 1, 1, 1]
xr.Dataset(
    {0: xr.DataArray([trigger_data],
            dims=["repetition", "acq_index_0"],
            coords={"repetition": [0], "acq_index_0": range(len(trigger_data))},
        )
    }
)
```

If there are multiple repetitions of the schedule, the acquisition data is still a list for each channel.
The list's first element is a number that counts how many times the threshold was passed **at least** once.
The second element counts how many times the threshold was passed **at least** twice, and so on for the other elements.
Don't be confused by the `repetition` dimension: for trigger count, this dimension has one coordinate, namely `0`.
See an example below.
```{code-cell} ipython3
---
tags: [hide-input]
---
trigger_data = [8, 6, 3, 3, 1]
xr.Dataset(
    {0: xr.DataArray([trigger_data],
            dims=["repetition", "acq_index_0"],
            coords={"repetition": [0], "acq_index_0": range(len(trigger_data))},
        )
    }
)
```

In `BinMode.AVERAGE` mode, the data is very similar. Each element in the list shows how many times the threshold was passed in each repetition **exactly** as many times as it's shown in the `"count"` dimension.
For example, in the example below, the schedule ran 8 times. From these 8 runs,
- in 1 run, the trigger was counted 5 times,
- in 2 runs, the trigger was counted 4 times,
- in 3 runs, the trigger was counted 2 times,
- in 2 runs, the trigger was counted once.

Note: 0 counts are removed from the returned data, so there will be no entry for "3 times".

You can think of the append mode values as the cumulative distribution of the average mode values.
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
