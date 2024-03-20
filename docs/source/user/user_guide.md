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

# Overview

`quantify-scheduler` is a python module for writing quantum programs featuring a hybrid gate-pulse control model with explicit timing control.
It extends the circuit model from quantum information processing by adding a pulse-level representation to operations defined at the gate-level, and the ability to specify timing constraints between operations.
Thus, a user is able to mix gate- and pulse-level operations in a quantum circuit.

In `quantify-scheduler`, both a quantum circuit consisting of gates and measurements and a timed sequence of control pulses are described as a {class}`.Schedule` .
The {class}`.Schedule` contains information on *when* operations should be performed.
When adding operations to a schedule, one does not need to specify how to represent this {class}`.Operation` on all (both gate and pulse) abstraction levels.
Instead, this information can be added later during {ref}`Compilation`.
This allows the user to effortlessly mix the gate- and pulse-level descriptions as is required for many experiments.
We support similar flexibility in the timing constraints, one can either explicitly specify the timing using {attr}`.ScheduleBase.schedulables`, or rely on the compilation which will use the duration of operations to schedule them back-to-back.


(sec-user-guide-creating-a-schedule)=

# Creating a schedule

The most convenient way to interact with a {class}`.Schedule` is through the {mod}`quantify_scheduler` API.
In the following example, we will create a function to generate a {class}`.Schedule` for a [Bell experiment](https://en.wikipedia.org/wiki/Bell%27s_theorem) and visualize one instance of such a circuit.

```{code-cell} ipython3
---
mystnb:
  remove_code_outputs: true
---

# import the Schedule class and some basic operations.
from quantify_scheduler import Schedule
from quantify_scheduler.operations.gate_library import Reset, Measure, CZ, Rxy, X90

def bell_schedule(angles, q0:str, q1:str, repetitions: int):

    for acq_index, angle in enumerate(angles):

        sched = Schedule(f"Bell experiment on {q0}-{q1}")

        sched.add(Reset(q0, q1))  # initialize the qubits
        sched.add(X90(qubit=q0))
        # Here we use a timing constraint to explicitly schedule the second gate to start
        # simultaneously with the first gate.
        sched.add(X90(qubit=q1), ref_pt="start", rel_time=0)
        sched.add(CZ(qC=q0, qT=q1))
        sched.add(Rxy(theta=angle, phi=0, qubit=q0) )
        sched.add(Measure(q0, acq_index=acq_index))  # denote where to store the data
        sched.add(Measure(q1, acq_index=acq_index), ref_pt="start")

    return sched


sched = bell_schedule(
    angles=[45.0],
    q0="q0",
    q1="q1",
    repetitions=1024)


```

```{code-cell} ipython3

# visualize the circuit
f, ax = sched.plot_circuit_diagram()

```

```{tip}
Creating schedule generating functions is a convenient design pattern when creating measurement code. See {ref}`the section on execution <sec-user-guide-execution>` for an example of how this is used in practice.
```

# Concepts and terminology

`quantify-scheduler` can be understood by understanding the following concepts.

- {class}`.Schedule`s describe when an operation needs to be applied.
- {class}`.Operation`s describe what needs to be done.
- {class}`~quantify_scheduler.resources.Resource`s describe where an operation should be applied.
- {ref}`Compilation <sec-compilation>`: between different abstraction layers for execution on physical hardware.

The following table shows an overview of the different concepts and how these are represented at the quantum-circuit layer and quantum-device layer.

```{list-table} Overview of concepts and their representation at different levels of abstraction.
:widths: 25 25 25 25
:header-rows: 0

- *
  * Concept
  * Quantum-circuit layer
  * Quantum-device layer
- * When
  * {class}`.Schedule`
  * --
  * --
- * What
  * {class}`.Operation`
  * {ref}`Gates and Measurements <sec-user-guide-gates-measurement>`
  * {ref}`Pulses and acquisition protocols <sec-user-guide-pulses-acq-operations>`
- * Where
  * {class}`~quantify_scheduler.resources.Resource`
  * {ref}`Qubits <sec-user-guide-qubits>`
  * {ref}`Ports and clocks <sec-user-guide-ports-clocks>`

```

(sec-user-guide-quantum-circuit)=

## Quantum-circuit layer

The Quantum-circuit description is an idealized mathematical description of a schedule.

(sec-user-guide-gates-measurement)=

### Gates and measurements

In this description operations are
[quantum gates](https://en.wikipedia.org/wiki/Quantum_logic_gate) that act on idealized qubits as part of a [quantum circuit](https://en.wikipedia.org/wiki/Quantum_circuit).
Operations can be represented by (idealized) unitaries acting on qubits.
The {mod}`~quantify_scheduler.operations.gate_library` contains common operations (including the measurement operation) described at the quantum-circuit level.

The {class}`~quantify_scheduler.operations.gate_library.Measure` is a special operation that represents a measurement on a qubit.
In addition to the qubit it acts on, one also needs to specify where to store the data.

(sec-user-guide-qubits)=

### Qubits

At the gate-level description, operations are applied to qubits.
Qubits are represented by strings corresponding to the name of a qubit (e.g., {code}`q0`, {code}`q1`, {code}`A1`, {code}`QL`, {code}`qubit_1`, etc.).
Valid qubits are strings that appear in the {ref}`device configuration file<sec-device-config>` used when compiling the schedule.

### Visualization

A {class}`.Schedule` containing operations can be visualized as a circuit diagram by calling its method {meth}`.plot_circuit_diagram`.
Alternatively, one can plot the waveforms in schedules using {meth}`.plot_pulse_diagram`, which requires that the absolute timing of the schedule has been determined (see {ref}`sec-tutorial-sched-pulse` for an example).

```{code-cell} ipython3

from quantify_scheduler.operations.gate_library import X90, Measure

schedule = Schedule("X90 schedule")
schedule.add(X90("q0"))
schedule.add(Measure("q0"))

_ = schedule.plot_circuit_diagram()

```

### Summary

- Gates are described by unitaries.
- Gates are applied to qubits.
- Measurements are applied to qubits.
- Qubits are represented by strings.

(sec-user-guide-quantum-device)=
## Quantum-device layer

The quantum-device layer describes waveforms and acquisition protocols applied
to a device. These waveforms can be used to implement the idealized operations
expressed on the quantum-circuit layer, or can be used without specifying
a corresponding representation at the quantum-circuit layer:

```{code-cell} ipython3

from quantify_scheduler.operations.pulse_library import SquarePulse, RampPulse

schedule = Schedule("waveforms")
schedule.add(SquarePulse(amp=0.2, duration=4e-6, port="P"))
schedule.add(RampPulse(amp=-0.1, offset=.2, duration=6e-6, port="P"))
schedule.add(SquarePulse(amp=0.1, duration=4e-6, port="Q"), ref_pt='start')
```


(sec-user-guide-pulses-acq-operations)=
### Pulses and acquisition operations


The pulse-level description typically contains parameterization information,
such as amplitudes, durations and so forth required to synthesize the waveform
on control hardware.
{mod}`~quantify_scheduler.operations.pulse_library` module contains
a collection of commonly used pulses.

Measurements are decomposed into pulses and acquisition operations.
Similarly to pulse operations, acquisition operations contain their timing information
and correspondent {ref}`acquisition protocols <sec-user-guide-acquisition-protocols>`.
{mod}`~quantify_scheduler.operations.acquisition_library` module contains
a collection of commonly used acquisition operations.

(sec-user-guide-ports-clocks)=
### Ports and clocks

To specify *where* an operation is applied, the quantum-device layer description needs to specify both the location in physical space as well as in frequency space.

For many systems, it is possible to associate a qubit with an element or location on a device that a signal can be applied to.
We call such a location on a device a port.
Like qubits, ports are represented as strings (e.g., {code}`P0`, {code}`feedline_in`, {code}`q0:mw_drive`, etc.).
In the last example, a port is associated with a qubit by including the qubit name at the beginning of the port name (separated by a colon {code}`:`).

Associating a qubit can be useful when visualizing a schedule and or keeping configuration files readable.
It is, however, not required to associate a port with a single qubit.
This keeps matters simple when ports are associated with multiple qubits or with non-qubit elements such as tunable couplers.

Besides the physical location on a device, a pulse is typically applied at a certain frequency and with a phase.
These two parameters are stored in a {class}`~quantify_scheduler.resources.ClockResource`.
Each {class}`~quantify_scheduler.resources.ClockResource` also has a {code}`name` to be easily identified.
The {code}`name` should identify the purpose of the clock resource, not the value of the frequency.
By storing the frequency and phase in a clock, we can adjust the frequency of a transition, but refer to it with the same name.

Similar to ports, clocks can be associated with qubits by including the qubit name in the clock name (again, this is not required).
If the frequency of a clock is set to 0 (zero), the pulse is applied at baseband and is assumed to be real-valued.

{numref}`resources_fig` shows how the resources (qubit, port and clock) map to a physical device.

```{figure} /images/Device_ports_clocks.svg
:name: resources_fig
:width: 800

Resources are used to indicate *where* operations are applied.
(a) Ports (purple) indicate a location on a device.
By prefixing the name of a qubit in a port name (separated by a colon {code}`:`) a port can be associated with a qubit (red), but this is not required.
(b) Clocks (blue) denote the frequency and phase of a signal.
They can be set to track the phase of a known transition.
By prefixing the name of a qubit in a clock name (separated by a colon {code}`:`) a clock can be associated with a qubit (red), but this is not required.
Device image from {cite:t}`Dickel2018`).
```

(sec-user-guide-acquisition-protocols)=
### Acquisition protocols

When we define an acquisition, we must specify how to acquire data and how to process it
to provide a meaningful result.
For example, typical readout of a superconducting qubit state will consist of the following steps:

1. Sending a spectroscopy pulse to a port of a device, that is connected to a readout
   resonator of the qubit.
2. Acquiring a raw voltage trace of the reflected or transmitted signal using
   microwave digitizer equipment.
3. Taking a weighted sum of the returned signal to obtain a single complex number.
4. Assign most likely readout outcome (0 or 1) based on a pre-calibrated threshold.

Description of these processing steps is called an *acquisition protocol*.
To define an acquisition protocol, the developer must define two things:
an *algorithm* to process the data and the *schema* of a data array returned by
an {ref}`instrument coordinator component <sec-user-guide-hal>` (ICC).
ICC's job is to retrieve data from the instrument, process it
and return it in a required format.
If the readout equipment supports hardware acceleration for part of the processing
steps, its ICC can and should utilize it.

The {ref}`sec-user-guide-acq-data-format` chapter briefly describes the implication of
acquisition protocols and binning mode on a format of the data returned by ICC.
For a detailed description of all acquisition protocols defined in `quantify-scheduler`
and supported by at least some of the backends consult
the {ref}`acquisition protocols reference section <sec-acquisition-protocols>`.

(sec-user-guide-acquisition-channel-index)=
### Acquisition channel and acquisition index

`quantify-scheduler` identifies each acquisition in a resulting dataset with
two integer numbers: *acquisition channel* and *acquisition index*.

*Acquisition channel* is a stream of acquisition data that corresponds to a single
device element measured with the same acquisition protocol.
Each acquisition within the same acquisition channel must return data of a uniform size
and structure, that is described by the definition of an acquisition protocol.
The order number of an acquisition within the acquisition channel during a schedule run
is called an *acquisition index*.

On a quantum circuit layer they roughly correspond to a qubit being measured and
a number of the measurement of a given qubit in a schedule.
However, if a qubit (or, more precisely, a device element) can be measured using more
than one acquisition protocol, that will require defining several
acquisition channels for it.
When you are specifying a schedule on circuit level, acquisition channel is supposed
to be configured within a submodule of a device element, that corresponds to
a given measurement (i.e.
{meth}`~quantify_scheduler.device_under_test.transmon_element.BasicTransmonElement.measure`
submodule of a
{class}`~quantify_scheduler.device_under_test.transmon_element.BasicTransmonElement`}).
Acquisition index should be specified at the instantiation of a measurement operation.

### Summary

- Pulses are described as parameterized waveforms.
- Pulses are applied to *ports* at a frequency specified by a *clock*.
- Ports and clocks are represented by strings.
- Acquisition protocols describe the processing steps to perform on an acquired signal
  in order to interpret it.
- Acquisition channel and acquisition index describe how to find an acquisition result
  in a dataset.

(sec-compilation)=
# Compilation

Different compilation steps are required to go from a high-level description of a schedule to something that can be executed on hardware.
The scheduler supports multiple compilation steps, the most important ones are the step from the quantum-circuit layer (gates) to the quantum-device layer (pulses), and the one from the quantum-device layer into instructions suitable for execution on physical hardware.
The compilation is performed by a {class}`~.QuantifyCompiler`, which is configured through the {class}`~.CompilationConfig`:

```{eval-rst}
.. autoapiclass:: quantify_scheduler.backends.graph_compilation.CompilationConfig
    :noindex:
    :members: backend, device_compilation_config, hardware_compilation_config

```

The compilers are described in detail in {ref}`Compilers`.
This is schematically shown in {numref}`compilation_overview`.

```{figure} /images/compilation_overview.svg
:align: center
:name: compilation_overview
:width: 900px

A schematic overview of the different abstraction layers and the compilation process.
Both a quantum circuit, consisting of gates and measurements of qubits, and timed sequences of control pulses are represented as a {class}`.Schedule` .
The information specified in the {ref}`device configuration<sec-device-config>` is used during compilation to add information on how to represent {class}`.Operation` s specified at the quantum-circuit layer as pulses and acquisitions at the quantum-device layer.
The information in the {ref}`hardware description <sec-hardware-description>`, {ref}`hardware options <sec-hardware-options>`, and {ref}`connectivity <sec-connectivity>` is then used to compile the control pulses into instructions suitable for hardware execution.
Once executed on the hardware, a dataset is returned to the user. 
```

In the first compilation step, pulse information is added to all operations that are not valid pulses (see {attr}`.Operation.valid_pulse`) based on the information specified in the {ref}`sec-device-config`.

A second compilation step takes the schedule at the pulse level and translates this for use on a hardware backend.
This compilation step is performed using a hardware dependent compiler and uses the information specified in the {ref}`sec-hardware-compilation-config`.

```{note}
We use the term "**device**" to refer to the physical object(s) on the receiving end of the control pulses, e.g. a thin-film chip inside a dilution refrigerator.

And we employ the term "**hardware**" to refer to the instruments (electronics) that are involved in the pulse generations / signal digitization.
```

(sec-device-config)=
## Device compilation configuration

The device compilation configuration is used to compile from the quantum-circuit layer to the quantum-device layer.
This datastructure is auto-generated by the {class}`~.device_under_test.quantum_device.QuantumDevice` using the parameters stored in the {class}`~.device_under_test.device_element.DeviceElement`s.

```{eval-rst}
.. autoapiclass:: quantify_scheduler.backends.graph_compilation.DeviceCompilationConfig
    :noindex:

```

(sec-hardware-compilation-config)=
## Hardware compilation configuration
The hardware compilation configuration is used to compile from the quantum-device layer to the control-hardware layer.
Currently, this datastructure is parsed from a user-defined dict that should be passed to the `quantum_device.hardware_config` parameter.

```{eval-rst}
.. autoapiclass:: quantify_scheduler.backends.types.common.HardwareCompilationConfig
    :noindex:
    :members:

```

(user-guide-qblox-hardware-compilation-config)=
````{admonition} Example Qblox hardware compilation configuration file
:class: dropdown
```{literalinclude} ../../../quantify_scheduler/schemas/examples/qblox_hardware_compilation_config.json
:language: JSON
```
````

(user-guide-zhinst-hardware-compilation-config)=
````{admonition} Example Zurich Instruments hardware compilation configuration file
:class: dropdown
```{literalinclude} ../../../quantify_scheduler/schemas/examples/zhinst_hardware_compilation_config.json
:language: JSON
```
````

(sec-hardware-description)=
### Hardware Description
The {obj}`~.backends.types.common.HardwareDescription` datastructure specifies a control hardware instrument in the setup, along with its instrument-specific settings.
There is a specific {obj}`~.backends.types.common.HardwareDescription` datastructure for each of the currently supported instrument types, which are discriminated through the `instrument_type` field.

```{eval-rst}
.. autoapiclass:: quantify_scheduler.backends.types.common.HardwareDescription
  :noindex:
  :members:

```

(sec-hardware-options)=

### Hardware Options
The {class}`~.backends.types.common.HardwareOptions` datastructure contains the settings used in compiling from the quantum-device layer to a set of instructions for the control hardware. Most hardware options are structured as `Dict[str, HardwareOption]`, where the keys are the port-clock combinations on which these settings should be applied.

```{eval-rst}
.. autoapiclass:: quantify_scheduler.backends.types.common.HardwareOptions
  :noindex:
  :members:

```

(sec-connectivity)=
### Connectivity
The {class}`~.backends.types.common.Connectivity` datastructure describes how ports on the quantum device are connected to the control hardware. The connectivity is represented as an undirected graph, where the nodes are the ports on the quantum device and the control hardware inputs/outputs, and the edges are the connections between them.

```{eval-rst}
.. autoapiclass:: quantify_scheduler.backends.types.common.Connectivity
  :noindex:
  :members: graph

```

```{admonition} One-to-many, many-to-one, many-to-many
See {ref}`sec-connectivity-examples` for different ways of specifying connectivity graphs.
```

This information is used in the compilation backends to assign the pulses and acquisitions in the schedule to ports on the control hardware. To achieve this, the nodes in the connectivity graph should be consistent with the inputs/outputs in the {ref}`sec-hardware-description` and the schedule (see the note below). Each backend has its own requirements on the connectivity graph, which are described in the documentation of the backends themselves (see the sections on Connectivity for {ref}`Qblox  <sec-qblox-connectivity>` and {ref}`Zurich Instruments <sec-zhinst-connectivity>`). For example, a backend can support adding one or more components (such as attenuators or IQ mixers) between a control-hardware output and a quantum-device port.

```{important}
  Nodes that correspond to an input/output channel of an instrument should be named
  ``"instrument_name.channel_name"``, where the ``instrument_name`` should correspond
  to a {class}`~quantify_scheduler.backends.types.common.HardwareDescription` in 
  the {class}`~quantify_scheduler.backends.types.common.HardwareCompilationConfig`.

  Nodes that correspond to a port on the quantum device should be identical
  to a port that is used in the {class}`~quantify_scheduler.schedules.schedule.Schedule`. 
  If you use gate-level operations, you should use:

  - `"device_element_name:mw"` for `Rxy` operation (and its derived operations),
  - `"device_element_name:res"` for any measure operation,
  - `"device_element_name:fl"` for the flux port.
  ```

(sec-user-guide-execution)=
# Execution

## Different kinds of instruments

In order to execute a schedule, one needs both physical instruments to execute the compiled instructions as well as a way to manage the calibration parameters used to compile the schedule.
Although one could use manually written configuration files and send the compiled files directly to the hardware, the Quantify framework provides different kinds of {class}`~qcodes.instrument.base.Instrument`s to control the experiments and the management of the configuration files ({numref}`instruments_overview`).

```{figure} /images/instruments_overview.svg
:align: center
:name: instruments_overview
:width: 600px

A schematic overview of the different kinds of instruments present in an experiment.
Physical instruments are QCoDeS drivers that are directly responsible for executing commands on the control hardware.
On top of the physical instruments is a hardware abstraction layer, that provides a hardware agnostic interface to execute compiled schedules.
The instruments responsible for experiment control are treated to be as stateless as possible [^id3] .
The knowledge about the system that is required to generate the configuration files is described by the {class}`~quantify_scheduler.device_under_test.quantum_device.QuantumDevice` and {code}`DeviceElement`s.
Several utility instruments are used to control the flow of the experiments.
```

### Physical instruments

[QCoDeS instrument drivers](https://microsoft.github.io/Qcodes/drivers_api/index.html) are used to represent the physical hardware.
For the purpose of quantify-scheduler, these instruments are treated as stateless, the desired configurations for an experiment being described by the compiled instructions.
Because the instruments correspond to physical hardware, there is a significant overhead in querying and configuring these parameters.
As such, the state of the instruments in the software is intended to track the state of the physical hardware to facilitate lazy configuration and logging purposes.

(sec-user-guide-hal)=
### Hardware abstraction layer

Because different physical instruments have different interfaces, a hardware abstraction layer serves to provide a uniform interface.
This hardware abstraction layer is implemented as the {class}`~.InstrumentCoordinator` to which individual {class}`InstrumentCoordinatorComponent <.InstrumentCoordinatorComponentBase>`s are added that provide the uniform interface to the individual instruments.

(sec-user-guide-quantum-device-elements)=

### The quantum device and the device elements

The knowledge of the system is described by the {class}`~quantify_scheduler.device_under_test.quantum_device.QuantumDevice` and {code}`DeviceElement`s.
The {class}`~quantify_scheduler.device_under_test.quantum_device.QuantumDevice` directly represents the device under test (DUT) and contains a description of the connectivity to the control hardware as well as parameters specifying quantities like cross talk, attenuation and calibrated cable-delays.
The {class}`~quantify_scheduler.device_under_test.quantum_device.QuantumDevice` also contains references to individual {code}`DeviceElement`s, representations of elements on a device (e.g, a transmon qubit) containing the (calibrated) control-pulse parameters.

Because the {class}`~quantify_scheduler.device_under_test.quantum_device.QuantumDevice` and the {code}`DeviceElement`s are an {class}`~qcodes.instrument.base.Instrument`, the parameters used to generate the configuration files can be easily managed and are stored in the snapshot containing the experiment's metadata.

(sec-user-guide-experiment-flow)=
## Experiment flow

To use schedules in an experimental setting, in which the parameters used for compilation as well as the schedules themselves routinely change, we provide a framework for performing experiments making use of the concepts of `quantify-core`.
Central in this framework are the schedule {mod}`quantify_scheduler.gettables` that can be used by the `quantify_core.measurement.control.MeasurementControl` and are responsible for the experiment flow.

This flow is schematically shown in {numref}`experiments_control_flow`.

```{figure} /images/experiments_control_flow.svg
:align: center
:name: experiments_control_flow
:width: 800px

A schematic overview of the experiments control flow.
```

Let us consider the example of an experiment used to measure the coherence time {math}`T_1`.
In this experiment, a {math}`\pi` pulse is used to excite the qubit, which is left to idle for a time {math}`\tau` before it is measured.
This experiment is then repeated for different {math}`\tau` and averaged.

In terms of settables and gettables to use with the `quantify_core.measurement.control.MeasurementControl`, the settable in this experiment is the delay time {math}`\tau`, and the gettable is the execution of the schedule.

We represent the settable as a {class}`qcodes.instrument.parameter.ManualParameter`:

```{code-cell} ipython3

from qcodes.instrument.parameter import ManualParameter

tau = ManualParameter("tau", label=r"Delay time", initial_value=0, unit="s")

```

To execute the schedule with the right parameters, the {code}`ScheduleGettable` needs to have a reference to a template function that generates the schedule, the appropriate keyword arguments for that function, and a reference to the {class}`~quantify_scheduler.device_under_test.quantum_device.QuantumDevice` to generate the required configuration files.

For the {math}`T_1` experiment, quantify-scheduler provides a schedule generating function as part of the {mod}`quantify_scheduler.schedules.timedomain_schedules`: the {func}`quantify_scheduler.schedules.timedomain_schedules.t1_sched`.

```{code-cell} ipython3

from quantify_scheduler.schedules.timedomain_schedules import t1_sched
schedule_function = t1_sched

```

Inspecting the {func}`quantify_scheduler.schedules.timedomain_schedules.t1_sched`, we find that we need to provide the times {math}`\tau`, the name of the qubit, and the number of times we want to repeat the schedule.
Rather than specifying the values of the delay times, we pass the parameter {code}`tau`.

```{code-cell} ipython3

qubit_name = "q0"
sched_kwargs = {
    "times": tau,
    "qubit": qubit_name,
    "repetitions": 1024 # could also be a parameter
}
```

The {code}`ScheduleGettable` is set up to evaluate the value of these parameter on every call of {code}`ScheduleGettable.get`.
This flexibility allows the user to create template schedules that can then be measured by varying any of it's input parameters using the `quantify_core.measurement.control.MeasurementControl`.

Similar to how the schedule keyword arguments are evaluated for every call to {code}`ScheduleGettable.get`, the device config and hardware config files are re-generated from the {class}`~quantify_scheduler.device_under_test.quantum_device.QuantumDevice` for every iteration.
This ensures that if a calibration parameter is changed on the {class}`~quantify_scheduler.device_under_test.quantum_device.QuantumDevice`, the compilation will be affected as expected.

```{code-cell} ipython3

from quantify_scheduler.device_under_test.quantum_device import QuantumDevice
device = QuantumDevice(name="quantum_sample")
```

These ingredients can then be combined to perform the experiment:

```{code-cell} ipython3


from quantify_core.measurement import MeasurementControl
meas_ctrl = MeasurementControl("meas_ctrl")
```

```{code-block} python
t1_gettable = ScheduleGettable(
    device=device,
    schedule_function=schedule_function,
    schedule_kwargs=sched_kwargs
)

meas_ctrl.settables(tau)
meas_ctrl.setpoints(times)
meas_ctrl.gettables(t1_gettable)
label = f"T1 experiment {qubit_name}"
dataset = meas_ctrl.run(label)
```

and the resulting dataset can be analyzed using

```{code-cell} ipython3

# from quantify_core.analysis.t1_analysis import T1Analysis
# analysis = T1Analysis(label=label).run()
```

(sec-user-guide-acq-data-format)=
# Acquisition data format

`quantify-scheduler` has two primary interfaces for retrieving acquisition results: using
{class}`~quantify_scheduler.instrument_coordinator.instrument_coordinator.InstrumentCoordinator`
and using {class}`quantify_scheduler.gettables.ScheduleGettable`.

Each acquisition and measurement operation in the schedule has associated
{ref}`acquisition channel and acquisition index <sec-user-guide-acquisition-channel-index>`.
If you specifiy a schedule using raw acquisition operations (for example, using
{class}`~quantify_scheduler.operations.acquisition_library.SSBIntegrationComplex`),
use `acq_channel` and `acq_index` arguments of the operation to specify them.
Optionally you may also specify the requested binning mode in `bin_mode`
(it is `AVERAGE` by default):

```{code-block} python
schedule.add(
    SSBIntegrationComplex(
        t0=0,
        duration=100e-9,
        port="q0:res",
        clock="q0.ro",
        acq_channel=3,
        acq_index=1,
        bin_mode=BinMode.AVERAGE
    )
)
```

For a selected acquisition channel
{ref}`acquisition protocol <sec-user-guide-acquisition-protocols>` and binning mode
must be the same, otherwise compilation will fail.

When circuit-to-device compilation machinery is used, `acq_channel` should be specified
in the
{class}`DeviceElement <quantify_scheduler.device_under_test.device_element.DeviceElement>`
being measured, for example in the
{class}`~quantify_scheduler.device_under_test.transmon_element.DispersiveMeasurement`
submodule of
{class}`~quantify_scheduler.device_under_test.transmon_element.BasicTransmonElement`.
`acq_index` and `bin_mode` are still specified as input parameters to
{class}`~quantify_scheduler.operations.gate_library.Measure` (or another specialised
measurement operation supported by the device element).

## Retrieve acquisitions through `InstrumentCoordinator`

{meth}`InstrumentCoordinator.retrieve_acquisition() <quantify_scheduler.instrument_coordinator.instrument_coordinator.InstrumentCoordinator.retrieve_acquisition>`
method returns an `xarray.Dataset`:

- Each `xarray.DataArray` in the dataset corresponds to one `acq_channel`.
- Exact structure of a data array is defined by an acquisition protocol and `bin_mode`,
  that are associated with this acquisition channel.

For example, if a schedule contains two qubits (each one has its own acquisition channel,
say, `0` and `1`), the first of which has been measured three times and second twice using
`SSBIntegrationComplex`
{ref}`acquisition protocol <sec-user-guide-acquisition-protocols>` in `BinMode.APPEND`,
the resulting dataset will have the form:

```{code-cell} ipython3
---
mystnb:
  remove_code_source: true
---

import numpy as np
import xarray as xr

num_repetitions = 5
num_meas_0 = 3
num_meas_1 = 2
shape_0 = (num_repetitions, num_meas_0)
shape_1 = (num_repetitions, num_meas_1)

ds = xr.merge([
    xr.DataArray(
        np.random.rand(*shape_0).round(3) + 1j * np.random.rand(*shape_0).round(3),
        dims=("repetition", "acq_index_0"),
        name=0,
    ),
    xr.DataArray(
        np.random.rand(*shape_1).round(3) + 1j * np.random.rand(*shape_1).round(3),
        dims=("repetition", "acq_index_1"),
        name=1,
    ),
])
ds
```

Definitions of acquisition protocols and correspondent data format can be found in the
{ref}`acquisition protocols reference section <sec-acquisition-protocols>`.
Note that acquisition protocols define the meaning of each dimension of a data array,
but do not guarantee a consistent naming of the dimensions in a dataset.
Instead, the exact names of dimensions should be retrieved dynamically during
the processing of the dataset.

(sec-user-guide-acquisition-data-schedulegettable)=

## Retrieve acquisition through `ScheduleGettable`

{class}`~quantify_scheduler.gettables.ScheduleGettable` proxies the instrument-coordinator format to a format that can be used with
{class}`~quantify_core.measurement.control.MeasurementControl`.
Effectively it flattens the data arrays retrieved from the instrument coordinator,
splits complex numbers into either real and imaginary parts (if `real_imag` is set to
`True`) or absolute value and phase (if `real_imag` is `False`) and pads the data
with `nan`s to fit it into a single array.
Data that corresponds to acquisition channel number {math}`n` will end up in
items number {math}`2n` and {math}`2n+1` of that array.
For example, if `real_imag` is set to `True` in the `ScheduleGettable`,
the dataset above will be converted to:

```{code-cell} ipython3
---
mystnb:
  remove_code_source: true
---

(
    ds[0].real.as_numpy().data.reshape(-1),
    ds[0].imag.as_numpy().data.reshape(-1),
    ds[1].real.as_numpy().data.reshape(-1),
    ds[1].imag.as_numpy().data.reshape(-1),
)
```

```{rubric} Footnotes
```

[^id3]: `quantify-scheduler` treats physical instruments as stateless in the sense that the compiled instructions contain all information that specifies the execution of a schedule. However, for performance reasons, it is important to not reconfigure all parameters of all instruments whenever a new schedule is executed. The parameters (state) of the instruments are used to track the state of physical instruments to allow lazy configuration as well as ensure metadata containing the current settings is stored correctly.
