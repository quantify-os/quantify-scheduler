---
file_format: mystnb
kernelspec:
    name: python3

---

```{code-cell} ipython3
---
mystnb:
  remove_code_source: true
---

# Make output easier to read
from rich import pretty
pretty.install()

```

(sec-acquisition-framework)=
# Acquisition framework

```{warning}
 :class: dropdown
This reference guide is under construction.

This means that until the acquisition framework detailed in [&5](https://gitlab.com/groups/quantify-os/-/epics/5) is completed this document will be subject to change.
This document reflects our most up-to-date understanding of the concepts relevant to the acquisition framework as it is being implemented, but it will be updated and modified as our understanding progresses.
```

## Introduction 

In this reference guide, we provide a detailed description of the acquisition framework. 
Specifically, the problem the acquisition framework intends to solve is that of specifying how acquired signals should be processed, and where the resulting data should be stored. 
A user should be able to relate individual entries in a dataset to the different acquisitions specified in a schedule. 


```{figure} /images/compilation_overview.svg
:align: center
:name: compilation_overview_acq
:width: 900px

A schematic overview of the different abstraction layers and the compilation process.
```

To explain how these concepts work together, we start by defining several concepts. 


## Definitions and specifications

```{note}
This section defines the concepts that are relevant to understand  the acquisition framework. 
Take note that not all concepts have an implementation as a class in the code. 
*If* an implementation as a python class exists within `quantify`, a reference will be provided here. 

Although, in the ideal case the (definition of) the concept, and the implementation are identical, there might be differences between the definitions provided here and the implementation in the code. 
E.g., the implementation might be incomplete, or more limited in scope. 
Should you find a case where the two directly contradict each other, this is considered a defect and we kindly ask you to fill an issue to report the defect.
```


### Measure

In quantify, a {class}`~quantify_scheduler.operations.gate_library.Measure`ment at the quantum-circuit layer can be expressed as an {class}`~quantify_scheduler.operations.acquisition_library.Acquisition` at the quantum-device layer. 

When representing a {class}`~quantify_scheduler.operations.gate_library.Measure` at the quantum-circuit layer, the default behavior is that the `AcquisitionChannel`, and `AcquisitionProtocol` are taken from the {class}`~quantify_scheduler.backends.graph_compilation.DeviceCompilationConfig`, and the `AcquisitionIndex` is determined automatically. 
However, a user may want to specify these parameters manually and thereby overwrite the defaults that are specified in the {class}`~quantify_scheduler.backends.graph_compilation.DeviceCompilationConfig`. 


```{code-cell} ipython3
:tags: [hide-input]

from quantify_scheduler import Schedule
from quantify_scheduler.operations.gate_library import Measure

schedule = Schedule("Measurement")
schedule.add(Measure("q0"))

_ = schedule.plot_circuit_diagram()

```

### Acquisition
An {class}`~quantify_scheduler.operations.acquisition_library.Acquisition` is an {class}`~quantify_scheduler.operations.operation.Operation` that can be added to a {class}`~quantify_scheduler.schedules.schedule.Schedule` that must consist of (at least) an `AcquisitionProtocol` specifying how the acquired signal is to be processed, and an `AcquisitionChannel` and `AcquisitionIndex` specifying where the acquired data is to be stored in the `RawDataset`.


### Experiment 

An `Experiment` is a procedure carried out under controlled conditions in order to make a discovery, test a hypothesis, or demonstrate a known fact.


        
### ExperimentDescription
An `ExperimentDescription` is a description of the procedure that is carried out in an `Experiment`. 
A valid `ExperimentDescription` can consist of `Settable`(s), `Gettable`(s), instructions to determine the `Setpoints`, and predefined `DataProcessing` step(s).


### Dataset
A `Dataset` is structured data with metadata (e.g., an {doc}`Xarray <xarray:index>`  dataset). 
Within the quantify framework we like to associate specific metadata to a dataset. 
This is specified in the {doc}`dataset design <quantify-core:dev/design/dataset/index>`.

### RawDataset
A `RawDataset` is a valid `Dataset`. 
The structure of a `RawDataset` is defined by what is returned by the Hardware Abstraction Layer upon execution of a {class}`~quantify_scheduler.schedules.schedule.Schedule`. 
Note that this implies that this format is backend independent. 
Data entries in the `RawDataset` are labeled by an `AcquisitionChannel`, and an `AcquisitionIndex`. 

The structure of a `RawDataset` (shape, type and units of the data)  should be predictable before executing a {class}`~quantify_scheduler.schedules.schedule.Schedule`. 

### ProcessedDataset
A `ProcessedDataset` is a valid `Dataset`. 
The structure is defined by the `Experiment` that is performed and is described by the `DataProcessing` step of the `ExperimentDescription`.

### DataProcessing

`DataProcessing`: A predefined procedure of operations that can be performed on a `Dataset` to return another `Dataset` (which may also include figures or quantities of interest).

### AcquisitionProtocol
An `AcquisitionProtocol` describes how to process an acquired signal. 
Each acquisition protocol should have a corresponding data schema defined and documented that specifies the type, shape, and `units`, of the data that performing the protocol will return. 
The reference guide on {ref}`acquisition protocols <sec-acquisition-protocols>` provides an overview of different acquisition protocols included in quantify. 

### AcquisitionChannel
An `AcquisitionChannel` is a stream of data that corresponds to a device element that is measured sequentially in a specified regime (i.e., using the specified `AcquisitionProtocol`). 
Each acquisition channel must have a `name`; a `str` or `int` that is used to refer to the acquisition channel (within e.g., operations, schedules and datasets), and an optional `long_name` that serves as a human-readable variant of the name and can be associated to `RawDataset`.
As a consequence of these definitions, all {class}`~quantify_scheduler.operations.acquisition_library.Acquisition`s associated to an `AcquisitionChannel` must have the same `AcquisitionProtocol` and `BinMode`. 

An `AcquisitionChannel` commonly corresponds to a qubit but also makes sense in isolation (e.g., when performing spectroscopy). 
A qubit can in principle have multiple acquisition channels associated with it. 

In the resulting `RawDataset` data from each acquisition channel will be formatted as a separate data array. The exact shape and structure of the data is determined by the `AcquisitionProtocol` and `BinMode`


### AcquisitionIndex

An `AcquisitionIndex` is an identifier of an acquisition within a single repetition of a schedule, unique per acquisition channel (i.e., an index value occurs only once per `AcquisitionChannel`).

In the resulting `RawDataset`, the acquisition index corresponds to a data array [dimension](https://docs.xarray.dev/en/stable/user-guide/terminology.html#term-Dimension) and determines the order in which data appears in the data array for each acquisition channel.


### AcquisitionCoordinates
`AcquisitionCoordinates` are an additional piece of information associated with an acquisition, provided by a user during the schedule construction.

In a `RawDataset` coordinates correspond to `xarray` coordinates along the `AcquistionIndex` dimension of the acquisition channel data arrays.
Coordinates provided by a user are formatted as a data array using `numpy` conventions, thus, for performance reasons they should have uniform data type that can be handled with `numpy`.
`AcquisitionCoordinates` can optionally have `units` and `long_name` attributes associated with it.


### Bin mode
A {class}`~quantify_scheduler.enums.BinMode` is a property of an acquisition channel that describes how to handle multiple schedule repetitions of the same {class}`~quantify_scheduler.operations.acquisition_library.Acquisition` operation.
The most common use-case for this is when iterating over multiple repetitions of a {class}`~quantify_scheduler.schedules.schedule.Schedule` When the `bin_mode` is set to `APPEND` new entries will be added as a list along the `repetitions` dimension.

Common {class}`~quantify_scheduler.enums.BinMode`s are `APPEND` and `AVERAGE`, which will append entries along the "repetition" dimension or average all repetitions for the schedule.

