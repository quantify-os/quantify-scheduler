---
file_format: mystnb
kernelspec:
    name: python3
---

(sec-acquisition-protocols)=
# Acquisition protocols

The dataset returned by
{meth}`InstrumentCoordinator.retrieve_acquisition() <quantify_scheduler.instrument_coordinator.instrument_coordinator.InstrumentCoordinator.retrieve_acquisition>`
consists of a number of {class}`~xarray.DataArray`s containing data for every
acquisition channel.
This document specifies the format of these data arrays.

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
%matplotlib inline

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

intermodulation_freq = 1e8  # 100 MHz
voltage_iq = 0.32 + 0.25j
sampling_rate = 1.8e9  # 1.8 GSa/s
readout_duration = 1e-7  # 100 ns
time_grid = xr.DataArray(
    np.arange(0, readout_duration, sampling_rate ** -1), dims="trace_index"
)

raw_trace = (
    voltage_iq * np.exp(2j * np.pi * intermodulation_freq * time_grid)
).assign_coords({"trace_time": time_grid})

fig, ax = plt.subplots()
xr.plot.line(raw_trace.real, ax=ax, label="I")
xr.plot.line(raw_trace.imag, ax=ax, label="Q")
ax.legend(loc="upper right")
ax.set_xlabel(r"$t[s]$")
ax.set_ylabel(r"$V$");
```

Demodulated trace will unroll this data with respect to the intermodulation frequency,
so the resulting I and Q readouts will look like this:

```{code-cell} ipython3
---
tags: [hide-input]
---
demodulated_trace = raw_trace * np.exp(-2j * np.pi * intermodulation_freq * time_grid)

fig, ax = plt.subplots()
xr.plot.line(demodulated_trace.real, ax=ax, label="I")
xr.plot.line(demodulated_trace.imag, ax=ax, label="Q")
ax.legend(loc="upper right")
ax.set_xlabel(r"$t[s]$")
ax.set_ylabel(r"$V$");
```

This acquisition protocol is currently supported only in `BinMode.AVERAGE` binning mode.
The resulting dataset must contain data arrays with two dimensions for each acquisition
channel: acquisition index (number of an acquisition in a schedule) and trace index
(that corresponds to time from the start of the acquisition).
All the dimension names should be suffixed with the acquisition channel to avoid
conflicts while merging the datasets.
It is recommended to annotate the trace index dimension with a coordinate that describes
time since the start of the acquisition.
For example, if two acquisition channels read out once, the resulting dataset should have
the following structure:

```{code-cell} ipython3
---
tags: [hide-input]
---
xr.Dataset({
    0: demodulated_trace.expand_dims("acq_index_0", 0).rename(
        {"trace_index": "trace_index_0", "trace_time": "trace_time_0"}
    ),
    1: demodulated_trace.expand_dims("acq_index_1", 0).rename(
        {"trace_index": "trace_index_1", "trace_time": "trace_time_1"}
    ),
})
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
For example two acquisition channels of which acquisition channel 0 read out three
times and acquisition channel two read out two times, the resulting dataset should have
the following structure in `BinMode.APPEND`:

```{code-cell} ipython3
---
tags: [hide-input]
---
xr.Dataset({
    0: demodulated_trace.reduce(np.average, "trace_index").expand_dims(
        {"repetition": 5, "acq_index_0": 3}
    ),
    2: demodulated_trace.reduce(np.average, "trace_index").expand_dims(
        {"repetition": 5, "acq_index_2": 2}
    ),
})
```

In `BinMode.AVERAGE` repetition dimension gets reduced and only the acquisition index
dimension is left for each channel:

```{code-cell} ipython3
---
tags: [hide-input]
---
xr.Dataset({
    0: demodulated_trace.reduce(np.average, "trace_index").expand_dims(
        {"acq_index_0": 3}
    ),
    2: demodulated_trace.reduce(np.average, "trace_index").expand_dims(
        {"acq_index_2": 2}
    ),
})
```

(sec-acquisition-protocols-numerical-weighted-integration-complex)=
## Numerical Weighted Complex Integration

- Referred to as `"NumericalWeightedIntegrationComplex"`.
- Supported by {mod}`Qblox <quantify_scheduler.backends.qblox>`.

Equivalent to
{ref}`SSB complex integration <sec-acquisition-protocols-ssb-integration-complex>`,
but instead of a simple average of a demodulated signal weighted average is taken.
The dataset format is also the same.

Integration weights should normally be calibrated in a separate experiment
(see, for example, {cite:t}`magesan2015machine`).
