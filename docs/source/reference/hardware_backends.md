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

(sec-hardware-backends)=
# Hardware backends

The compiler for a hardware backend for Quantify takes a {class}`~.Schedule` defined on the {ref}`sec-user-guide-quantum-device` and a {class}`~.HardwareCompilationConfig`. The `InstrumentCoordinatorComponent` for the hardware runs the operations within the schedule and can return the acquired data via the {class}`~.InstrumentCoordinator` method {meth}`~.InstrumentCoordinator.retrieve_acquisition`. To that end, it generally consists of the following components:
1. {class}`~.CompilationNode`s that generate the compiled {class}`Schedule` from the device-level {class}`~.Schedule` and a {class}`~.HardwareCompilationConfig`. This compiled {class}`Schedule` generally consists of (1) the hardware level instructions that should be executed and (2) instrument settings that should be applied on the hardware.

2. A backend-specific `HardwareCompilationConfig` that contains the information that is used to compile from the quantum-device layer to the control-hardware layer (see {ref}`sec-compilation`). This generally consists of:
    - A {class}`~.backends.types.common.HardwareDescription` (e.g., the available channels, IP addresses, etc.).
    - The {class}`~.backends.types.common.Connectivity` between the quantum-device ports and the control-hardware ports.
    - The {class}`~.backends.types.common.HardwareOptions` that includes the specific settings that are supported by the backend (e.g., the gain on an output port).
    - A list of {class}`~.backends.graph_compilation.SimpleNodeConfig`s, which specifies the {class}`~.backends.graph_compilation.CompilationNode`s that should be executed to compile the {class}`~.Schedule` to the hardware-specific instructions.
3. `InstrumentCoordinatorComponent`s that send compiled instructions to the instruments, retrieve data, and convert the acquired data into a standardized, backend independent dataset (see {ref}`sec-acquisition-protocols`).

## Architecture overview

The interfaces between Quantify and a hardware backend are illustrated in the following diagram:

```{mermaid}
graph TD;
    user[User]

    subgraph Quantify
        ScheduleGettable
        QuantumDevice
        QuantifyCompiler
        InstrumentCoordinator
    
        QuantumDevice -->|CompilationConfig| ScheduleGettable
        ScheduleGettable -->|Schedule\n CompilationConfig| QuantifyCompiler
        InstrumentCoordinator -->|Raw Dataset| ScheduleGettable
        QuantifyCompiler -->|CompiledSchedule| ScheduleGettable
        ScheduleGettable -->|CompiledSchedule| InstrumentCoordinator
    end

    subgraph QuantifyBackend
        InstrumentCoordinatorComponent
        CompilationModule[Compilation Module] -->|CompilationNodes| HardwareCompilationConfig
        HardwareCompilationConfig
    end

    subgraph Hardware
        drivers[Hardware-specific drivers]
        instruments[Physical instruments]
    end

    user -->|Schedule| ScheduleGettable
    user -->|Device Description, Hardware Description,\n Connectivity, Hardware Options| QuantumDevice
    ScheduleGettable -->|Processed Dataset| user

    InstrumentCoordinatorComponent -->|Partial Raw Dataset| InstrumentCoordinator
    QuantumDevice -->|Hardware Description\n Connectivity\n Hardware Options| HardwareCompilationConfig
    HardwareCompilationConfig -->|Validated HardwareCompilationConfig| QuantumDevice
    
    InstrumentCoordinator -->|Compiled Instructions| InstrumentCoordinatorComponent

    InstrumentCoordinatorComponent -->|Compiled Instructions| drivers
    drivers -->|Data points| InstrumentCoordinatorComponent

    drivers <--> instruments
```

## Experiment flow

```{seealso}
This diagram is similar to the the one in the {ref}`sec-user-guide-experiment-flow` section in the User Guide, but provides more details on the interfaces between the objects.
```

```{mermaid}
:caption: Diagram of the experiment flow in Quantify. Dotted lines represent the output and the non-dotted lines represent the input.

sequenceDiagram
    participant SG as ScheduleGettable
    participant QuantumDevice
    participant QuantifyCompiler
    participant IC as InstrumentCoordinator
    SG->>+QuantumDevice: generate_compilation_config()
    QuantumDevice-->>-SG: CompilationConfig
    SG->>+QuantifyCompiler: compile(Schedule, CompilationConfig)
    QuantifyCompiler-->>-SG: CompiledSchedule
    SG->>+IC: prepare(CompiledSchedule)
    SG->>IC: start()
    SG->>IC: retrieve_acquisition()
    IC-->>-SG: RawDataset	
```

In the above diagram several methods are called. The `get()` method of the `ScheduleGettable` executes the sequence and returns the data from the acquisitions in the `Schedule`. The `generate_compilation_config()` method of the `QuantumDevice` generates a `CompilationConfig` that is used to compile the `Schedule` within the `ScheduleGettable`. The `compile()` method of the `QuantifyCompiler` compiles the `Schedule` to the hardware-specific instructions. The `prepare()`, `start()`, and `retrieve_acquisition()` methods of the `InstrumentCoordinator` are used to prepare the hardware, start the acquisition, and retrieve the acquired data, respectively. 

## Developing a new backend

To develop a new backend, the following approach is advised:
1. Implement a custom `Gettable` that takes a set of instructions in the form that the hardware can directly execute. The compiled schedule that the backend should return once developed should therefore be the same as the instructions that this `Gettable` accepts. This enables testing of the `InstrumentCoordinatorComponent`s without having to worry about the compilation of the `Schedule` to the hardware.
2. Implement `InstrumentCoordinatorComponent`s for each of the instruments (starting with the instrument with the acquisition channel to enable testing).
3. Implement `CompilationNode`s to generate hardware instructions from a `Schedule` and a `CompilationConfig`.
4. Integrate the (previously developed + tested) `QuantifyCompiler` and `InstrumentCoordinatorComponent`s with the `ScheduleGettable`.
5. Optionally test the `ScheduleGettable` with the `MeasurementControl`.

### Mock device 

To illustrate how to do this, let's consider a basic example of an interface with a mock hardware device for which we would like to create a corresponding Quantify hardware backend. Our mock device resembles a readout module that can play and simultaneously acquire waveforms through the "TRACE" instruction. It is described by the following class:

```{eval-rst}
.. autoapiclass:: quantify_scheduler.backends.mock.mock_rom.MockReadoutModule
    :noindex:

```
The device has three methods, `execute`, `upload_waveforms`, and `upload_instructions`, and a property `sampling_rate`.

In our endeavor to create a backend between Quantify and the mock device we start by creating a mock readout module so we can show its functionality:

```{code-cell} ipython3
import numpy as np
from quantify_scheduler.backends.mock.mock_rom import MockReadoutModule

rom = MockReadoutModule(name="mock_rom")
```

We first upload two waveforms (defined on a 1 ns grid) to the readout module:

```{code-cell} ipython3
intermodulation_freq = 1e8  # 100 MHz
amplitude = 0.5 # 0.5 V
duration = 1e-7  # 100 ns

time_grid = np.arange(0, duration, 1e-9)
complex_trace = np.exp(2j * np.pi * intermodulation_freq * time_grid)

wfs = {
    "I": complex_trace.real,
    "Q": complex_trace.imag
}
rom.upload_waveforms(wfs)
```

The mock readout module samples the uploaded waveforms and applies a certain gain to the acquired data. The sampling rate and gain can be set as follows:

```{code-cell} ipython3
rom.sampling_rate = 1.5e9  # 1.5 GSa/s
rom.gain = 2.0
```

The mock readout module takes a list of strings as instructions input:

```{code-cell} ipython3
rom.upload_instructions(["TRACE_input0_0"])
```

We can now execute the instructions on the readout module:

```{code-cell} ipython3
rom.execute()
```

The data that is returned by our mock readout module is a dictionary containing the acquired I and Q traces:

```{code-cell} ipython3
import matplotlib.pyplot as plt

data = rom.get_results()
plt.plot(data[0][0])
plt.plot(data[0][1])
plt.show()
```

The goal is now to implement a backend that can compile and execute a `Schedule` that consists of a single trace acquisition and returns a Quantify dataset. 

### 1. Implement a custom `Gettable`

A good first step to integrating this mock readout module with Quantify is to implement a custom `Gettable` that takes a set of instructions and waveforms that can be readily executed on the hardware. This `Gettable` can then be used to retrieve the executed waveforms from the hardware. This enables testing of the `InstrumentCoordinatorComponent`s without having to worry about the compilation of the `Schedule` to the hardware.

```{eval-rst}
.. autoapiclass:: quantify_scheduler.backends.mock.mock_rom.MockROMGettable
    :noindex:

```

We check that this works as expected:

```{code-cell} ipython3
from quantify_scheduler.backends.mock.mock_rom import MockROMGettable

mock_rom_gettable = MockROMGettable(mock_rom=rom, waveforms=wfs, instructions=["TRACE_input0_0"], sampling_rate=1.5e9, gain=2.0)
data = mock_rom_gettable.get()
plt.plot(data[0][0])
plt.plot(data[0][1])
plt.show()
```

From the plot, we observe that the waveforms are the same as what was sent into the `MockReadoutModule`.

### 2. Acquisition mappings

With regards to acquisition data, the ultimate role of the backend is to map the acquisition data on the hardware to the returned structured data (`xarray.Dataset`). This mapping must be generated at compilation time, and then after the experiment is run on the hardware, this mapping is needed to generate the returned structured `xarray.Dataset` to the user.

The structured acquisition data consists of an `xarray.Dataset`, which is practically a dictionary where the keys are user given acquisition channel names and the values are the corresponding data to that acquisition channel. The acquisition channel data `xarray.DataArray` is one dimensional data (assuming no append mode repetitions): each acquired data point has an index (acquisition index), and other user-given coordinates.

As an example, in case of several binned acquisition (multiple acquisition indices), but on the same acquisition channel, a typical data looks like the following for a schedule.

```{code-block} ipython3
schedule = Schedule("example")
schedule.add(Measure(qubit="q0", acq_channel="ch_0", coords={"freq": 100}))
schedule.add(Measure(qubit="q0", acq_channel="ch_0", coords={"freq": 200}))
schedule.add(Measure(qubit="q0", acq_channel="ch_0", coords={"freq": 300}))
```

```{code-cell} ipython3
---
tags: [hide-input]
---
import xarray

xarray.Dataset(
    data_vars=dict(
        ch_0=(["acq_index_ch_0"], [0.0, 0.2, 0.4]),
    ),
    coords=dict(
        freq=(["acq_index_ch_0"], [100, 200, 300]),
        acq_index_ch_0=[0, 1, 2],
    ),
)
```

In case of trace acquisition, there is one acquisition index, and the other dimension is the time
for our mock backend.

```{code-block} ipython3
schedule = Schedule("example")
schedule.add(Measure(qubit="q0", acq_channel="ch_0", acq_protocol="Trace"))
```

```{code-cell} ipython3
---
tags: [hide-input]
---
import xarray

xarray.Dataset(
    data_vars=dict(
        ch_0=(["acq_index_ch_0", "time_ch0"], [[2.5, 2.6, 2.7, 2.8, 2.9, 3.0]]),
    ),
    coords=dict(
        time_ch0=[0.0, 0.001, 0.002, 0.003, 0.004, 0.005],
        acq_index_ch_0=[0],
    ),
)
```

The compiler needs to generate a mapping of each acquisition channel, acquisition index to data on the hardware and other user-given additional coords. This mapping is divided into two parts: hardware independent mapping and hardware dependent mapping. The functions defined in the hardware independent layer (frontend) can generate the hardware independent mapping and the functions defined in the backend can generate the hardware dependent mapping. This helps standardizing how acquisitions work. Development of a quantify backend would then only require implementing hardware dependent mapping.

The hardware independent mapping ({class}`~.schedules.schedule.AcquisitionChannelData`) stores the following information for each acquisition channel:

```{eval-rst}
.. autoapiclass:: quantify_scheduler.schedules.schedule.AcquisitionChannelData
    :noindex:
    :members:
```

The hardware dependent mapping stores how each acquisition channel and acquisition index is related to hardware. To help backend developers, a function defined in the frontend generates a mapping which could be converted by backend developers to the required acquisition hardware mapping. This is the {data}`~.helpers.generate_acq_channels_data.SchedulableLabelToAcquisitionIndex`. This maps every schedulable in the schedule to an acquisition index. (Note: {data}`~.helpers.generate_acq_channels_data.SchedulableLabelToAcquisitionIndex` is only generated for binned acquisitions!)

```{code-block} ipython3
SchedulableLabel = str
FullSchedulableLabel = Tuple[SchedulableLabel]
SchedulableLabelToAcquisitionIndex = Dict[
    Tuple[FullSchedulableLabel, int], Union[int, List[int]]
]
```

The high-level overview of acquisition mapping creation is the following. The function defined in hardware independent frontend ({func}`~.helpers.generate_acq_channels_data.generate_acq_channels_data`) generates the {data}`~.schedules.schedule.AcquisitionChannelsData` and {data}`~.helpers.generate_acq_channels_data.SchedulableLabelToAcquisitionIndex`, and hardware specific backend converts the {data}`~.helpers.generate_acq_channels_data.SchedulableLabelToAcquisitionIndex` to its own backend dependent hardware mapping.

```{mermaid}
:caption: Diagram of the acquisition compilation logic. (For clarity, some details are omitted.)
sequenceDiagram
    participant User
    participant C as Compiler UI
    participant CF as Compiler functions frontend
    participant CB as Compiler functions backend

    User->>+C: compile(Schedule)
    C->>+CF: generate_acq_channels_data(schedule)
    CF-->>-C: AcquisitionChannelsData, SchedulableLabelToAcquisitionIndex
    C->>+CB: Compile backend using SchedulableLabelToAcquisitionIndex
    CB-->>-C: Hardware mapping
    C-->>-User: CompiledSchedule
```

```{mermaid}
:caption: Diagram of retrieving data using the acquisition mappings. (For clarity, some details are omitted.)
sequenceDiagram
    participant User
    participant IC as InstrumentCoordinatorComponent
    participant Hardware

    User->>+IC: retrieve_acquisition using AcquisitionChannelsData and hardware mapping
    IC->>+Hardware: retrieve hardware acquisition data
    Hardware-->>-IC: hardware acquisition data
    IC-->>-User: Acquisition dataset
```

(Note: currently the backend calls {func}`~.helpers.generate_acq_channels_data.generate_acq_channels_data`, but this function is defined still in the frontend, because it's convenient for all backends, and standardizes acquisitions, but currently it's not feasible to call this directly from the frontend.)

The generated {data}`~.schedules.schedule.AcquisitionChannelsData` and the backend generated hardware mapping need to be in `CompiledSchedule`. After running the experiment, these mappings will be used to create the `xarray.Dataset`, which will be returned to the user.

### 3. Implement `InstrumentCoordinatorComponent`(s)

Within Quantify, the `InstrumentCoordinatorComponent`s are responsible for sending compiled instructions to the instruments, retrieving data, and converting the acquired data into a quantify-compatible Dataset (see {ref}`sec-acquisition-protocols`). The `InstrumentCoordinatorComponent`s are instrument-specific and should be based on the {class}`~.quantify_scheduler.instrument_coordinator.components.base.InstrumentCoordinatorComponentBase` class.

There are two mappings which are needed in order to convert the hardware acquired data to the structured `xarray.Dataset`: {data}`~.schedules.schedule.AcquisitionChannelsData` (for backend independent mappings) and `MockROMSettings.hardware_acq_mapping` (for backend dependent mappings). The settings for the mock readout module can be set via the {class}`~.quantify_scheduler.backends.mock.mock_rom.MockROMSettings` class using the `prepare` method of the {class}`~.quantify_scheduler.backends.mock.mock_rom.MockROMInstrumentCoordinatorComponent`. The `start` method is used to start the acquisition and the `retrieve_acquisition` method is used to retrieve the acquired data:

```{eval-rst}
.. autoapiclass:: quantify_scheduler.backends.mock.mock_rom.MockROMSettings
    :noindex:
    :members:
```

We can now implement the `InstrumentCoordinatorComponent` for the mock readout module:

```{eval-rst}
.. autoapiclass:: quantify_scheduler.backends.mock.mock_rom.MockROMInstrumentCoordinatorComponent
    :noindex:
    :members:
```

Now we can control the mock readout module through the `InstrumentCoordinatorComponent`:

```{code-cell} ipython3
from quantify_scheduler.backends.mock.mock_rom import MockROMInstrumentCoordinatorComponent, MockROMSettings
from quantify_scheduler.helpers.generate_acq_channels_data import AcquisitionChannelData
from quantify_scheduler.enums import BinMode

rom_icc = MockROMInstrumentCoordinatorComponent(mock_rom=rom)
acq_channels_data = {
    0: AcquisitionChannelData(
        acq_index_dim_name="acq_index_0",
        protocol="Trace",
        bin_mode=BinMode.AVERAGE,
        coords={}
    )
}
settings = MockROMSettings(
    waveforms=wfs,
    instructions=["TRACE_input0_0"],
    sampling_rate=1.5e9,
    gain=2.0,
    hardware_acq_mapping_trace={0: 0},
    hardware_acq_mapping_binned={},
    acq_channels_data=acq_channels_data,
)


rom_icc.prepare(settings)
rom_icc.start()
dataset = rom_icc.retrieve_acquisition()
```

The acquired data is:

```{code-cell} ipython3
dataset
```

This is the expected returned dataset, with 2 dimensions: time and acquisition index.

### 4. Implement `CompilationNode`s

The next step is to implement a `QuantifyCompiler` that generates the hardware instructions from a `Schedule` and a `CompilationConfig`. The `QuantumDevice` class already includes the `generate_compilation_config()` method that generates a `CompilationConfig` that can be used to perform the compilation from the quantum-circuit layer to the quantum-device layer. For the backend-specific compiler, we need to add a `CompilationNode` and an associated `HardwareCompilationConfig` that contains the information that is used to compile from the quantum-device layer to the control-hardware layer (see {ref}`sec-compilation`).

First, we define a `DataStructure` that contains the information that is required to compile to the mock readout module:

```{eval-rst}
.. autoapiclass:: quantify_scheduler.backends.mock.mock_rom.MockROMHardwareCompilationConfig
    :noindex:
    :members:

```

We then implement a `CompilationNode` that uses a `Schedule` and a `MockROMHardwareCompilationConfig` and generates the hardware-specific instructions:

```{eval-rst}
.. autoapifunction:: quantify_scheduler.backends.mock.mock_rom.hardware_compile
    :noindex:

```

To test the implemented `CompilationNode`, we first create a raw trace `Schedule`:

```{code-cell} ipython3
from quantify_scheduler.schedules.trace_schedules import trace_schedule

sched = trace_schedule(
    pulse_amp=0.1,
    pulse_duration=1e-7,
    pulse_delay=0,
    frequency=3e9,
    acquisition_delay=0,
    integration_time=2e-7,
    port="q0:res",
    clock="q0.ro"
)
sched.plot_circuit_diagram()
```

Currently, the `QuantumDevice` is responsible for generating the full `CompilationConfig`. We therefore create an empty `QuantumDevice`:

```{code-cell} ipython3
from quantify_scheduler.device_under_test.quantum_device import QuantumDevice

quantum_device = QuantumDevice("quantum_device")
```

We supply the hardware compilation config input to the `QuantumDevice`, which will be used to generate the full `CompilationConfig`:

```{code-cell} ipython3
from quantify_scheduler.backends.mock.mock_rom import hardware_compilation_config as hw_cfg
quantum_device.hardware_config(hw_cfg)
```

The `QuantumDevice` can now generate the full `CompilationConfig`:

```{code-cell} ipython3
from rich import print

print(quantum_device.generate_compilation_config())
```

We can now compile the `Schedule` to settings for the mock readout module:

```{code-cell} ipython3
from quantify_scheduler.backends.graph_compilation import SerialCompiler

compiler = SerialCompiler(name="compiler", quantum_device=quantum_device)
compiled_schedule = compiler.compile(schedule=sched)
```

We can now check that the compiled settings are correct:

```{code-cell} ipython3
print(compiled_schedule.compiled_instructions)
```

### 5. Integration with the `ScheduleGettable`

The `ScheduleGettable` integrates the `InstrumentCoordinatorComponent`s with the `QuantifyCompiler` to provide a straightforward interface for the user to execute a `Schedule` on the hardware and retrieve the acquired data. The `ScheduleGettable` takes a `QuantumDevice` and a `Schedule` as input and returns the data from the acquisitions in the `Schedule`.

```{note}
This is also mainly a validation step. If we did everything correctly, no new development should be needed in this step.
```

We first instantiate the `InstrumentCoordinator`, add the `InstrumentCoordinatorComponent` for the mock readout module, and add a reference to the `InstrumentCoordinator` to the `QuantumDevice`:

```{code-cell} ipython3
from quantify_scheduler.instrument_coordinator.instrument_coordinator import InstrumentCoordinator

ic = InstrumentCoordinator("IC")
ic.add_component(rom_icc)
quantum_device.instr_instrument_coordinator(ic.name)
```

We then create a `ScheduleGettable`:

```{code-cell} ipython3
from quantify_scheduler.gettables import ScheduleGettable
from quantify_scheduler.schedules.trace_schedules import trace_schedule

schedule_gettable = ScheduleGettable(
    quantum_device=quantum_device,
    schedule_function=trace_schedule,
    schedule_kwargs=dict(
        pulse_amp=0.1,
        pulse_duration=1e-7,
        pulse_delay=0,
        frequency=3e9,
        acquisition_delay=0,
        integration_time=2e-7,
        port="q0:res",
        clock="q0.ro"
    ),
    batched=True
)
I, Q = schedule_gettable.get()
```
We can now plot the acquired data:

```{code-cell} ipython3
plt.plot(I)
plt.plot(Q)
plt.show()
```

### 6. Integration with the `MeasurementControl`

```{note}
This is mainly a validation step. If we did everything correctly, no new development should be needed in this step.
```

Example of a `MeasurementControl` that sweeps the pulse amplitude in the trace `ScheduleGettable`:

```{code-cell} ipython3
from quantify_core.measurement.control import MeasurementControl
from quantify_core.data.handling import set_datadir, to_gridded_dataset

set_datadir()

meas_ctrl = MeasurementControl(name="meas_ctrl")

from qcodes.parameters import ManualParameter

amp_par = ManualParameter(name="trace_amplitude")
amp_par.batched = False
sample_par = ManualParameter("sample", label="Sample time", unit="s")
sample_par.batched = True

amps = [0.1,0.2,0.3]
integration_time = 2e-7
sampling_rate = quantum_device.generate_hardware_compilation_config().hardware_description["mock_rom"].sampling_rate

schedule_gettable = ScheduleGettable(
    quantum_device=quantum_device,
    schedule_function=trace_schedule,
    schedule_kwargs=dict(
        pulse_amp=amp_par,
        pulse_duration=1e-7,
        pulse_delay=0,
        frequency=3e9,
        acquisition_delay=0,
        integration_time=integration_time,
        port="q0:res",
        clock="q0.ro"
    ),
    batched=True
)
meas_ctrl.settables([amp_par, sample_par])
# problem: we don't necessarily know the size of the returned traces (solved by using xarray?)
# workaround: use sample_par ManualParameter that predicts this using the sampling rate
meas_ctrl.setpoints_grid([np.array(amps), np.arange(0, integration_time, 1 / sampling_rate)])
meas_ctrl.gettables(schedule_gettable)

data = meas_ctrl.run()
gridded_data = to_gridded_dataset(data)
```

Data processing and plotting:

```{code-cell} ipython3
import xarray as xr

magnitude_data = abs(gridded_data.y0 + 1j * gridded_data.y1)
phase_data = xr.DataArray(np.angle(gridded_data.y0 + 1j * gridded_data.y1))
```

```{code-cell} ipython3
magnitude_data.plot()
```

```{code-cell} ipython3
phase_data.plot()
```

### 7. Binned acquisition

Above, we implemented the trace acquisition protocol in our mock backend.
In this section we implement a binned acquisition.
Binned acquisitions are more complicated, because
1. multiple operations can use the same acquisition channel, for example
```{code-block} ipython3
schedule = Schedule("example")
schedule.add(SSBIntegrationComplex(acq_channel="ch_0", coords={"freq": 100}))
schedule.add(SSBIntegrationComplex(acq_channel="ch_0", coords={"freq": 200}))
```
2. the same operation can be used for multiple acquisitions, for example when the exact same operation is added to the schedule as different schedulables,
```{code-block} ipython3
schedule = Schedule("example")

acquisition = SSBIntegrationComplex(acq_channel="ch_0")

schedule.add(acquisition)
schedule.add(acquisition)
```

The data the user receives is a 1 dimensional `xarray.DataArray` for each acquisition channel (assuming no append mode repetitions), indexed by **acquisition index**. If unspecified by the user, the acquisition index is autogenerated by {func}`~.helpers.generate_acq_channels_data.generate_acq_channels_data`. The ultimate goal of the backend is to map each acquisition index to an acquired data from the hardware.

First let's understand what {func}`~.helpers.generate_acq_channels_data.generate_acq_channels_data` returns.

```{code-cell} ipython3
from quantify_scheduler.schedules.schedule import Schedule
from quantify_scheduler.operations.acquisition_library import SSBIntegrationComplex
from quantify_scheduler.helpers.generate_acq_channels_data import generate_acq_channels_data
from quantify_scheduler.compilation import _determine_absolute_timing

schedule = Schedule("example")

subschedule = Schedule("subschedule")
subschedule.add(
    SSBIntegrationComplex(
        acq_channel="ch_0",
        coords={"amp": 0.1},
        acq_index=None,
        bin_mode=BinMode.AVERAGE,
        port="q0:res",
        clock="q0.01",
        duration=100e-9
    )
)

schedule.add(subschedule)
schedule.add(subschedule)

schedule = _determine_absolute_timing(schedule)

acq_channels_data, schedulable_to_acq_index = generate_acq_channels_data(schedule)
acq_channels_data, schedulable_to_acq_index
```

In the example schedule, there are two acquisitions, so for acquisition channel `"ch_0"` two acquisition indices are generated: the `coords` has 2 elements. The generated acquisition indices are not explicitly written out in `acq_channels_data`, they're just the indices of the `coords`.

The backends job is to map each acquisition to a hardware acquisition data, but unfortunately, if the compiler added the acquisition index to the operation, that would be incorrect, because there is only one operation, but two acquisition indices. Therefore, {func}`~.helpers.generate_acq_channels_data.generate_acq_channels_data` maps each schedulable label to an acquisition index. To be more precise, it maps the tuple of the schedulable of the subschedule and schedulable of the acquisition operation to the acquisition index. (Also, there's an acquisition info at the end (`0`), because multiple acquisition info might be present, with possibly different acquisition indices.) Note: using schedulables instead of operations to reference acquisitions can also be important for loops, and other control flow structures.

Our mock hardware does binned acquisition with the `"ACQ_input0_0"` instruction where `input0` is the hardware port and the `0` at the end is the "bin" for that acquisition. The backend compiler maps each schedulable to a bin. {func}`~.backends.mock.mock_rom.hardware_compile` generates this mapping, using {data}`~.helpers.generate_acq_channels_data.SchedulableLabelToAcquisitionIndex`.

```{code-cell} ipython3
---
tags: [hide-output]
---
from quantify_scheduler.schedules.schedule import Schedule
from quantify_scheduler.operations.acquisition_library import SSBIntegrationComplex, Trace
from quantify_scheduler.compilation import _determine_absolute_timing
from quantify_scheduler.backends.mock.mock_rom import hardware_compile
from quantify_scheduler.resources import ClockResource
from quantify_scheduler.operations.pulse_library import SquarePulse

schedule = Schedule("example")

schedule.add_resource(ClockResource("q0.ro", 3e9))

schedule.add(SquarePulse(duration=100e-9, amp=0.1, port="q0:res", clock="q0.ro"))
schedule.add(Trace(acq_channel="ch_trace", duration=100e-9, port="q0:res", clock="q0.ro"))

schedule.add(
    SSBIntegrationComplex(
        acq_channel="ch_0",
        coords={"freq": 100},
        acq_index=None,
        bin_mode=BinMode.AVERAGE,
        port="q0:res",
        clock="q0.01",
        duration=100e-9
    )
)
schedule.add(
    SSBIntegrationComplex(
        acq_channel="ch_0",
        coords={"freq": 200},
        acq_index=None,
        bin_mode=BinMode.AVERAGE,
        port="q0:res",
        clock="q0.01",
        duration=100e-9
    )
)
```

This example schedule has two binned acquisitions, both on the same acquisition channel, so we expect that the {func}`~.helpers.generate_acq_channels_data.generate_acq_channels_data` function generates two acquisition index and related coordinates.

```{code-cell} ipython3
schedule = _determine_absolute_timing(schedule)

acq_channels_data, schedulable_to_acq_index = generate_acq_channels_data(schedule)
acq_channels_data, schedulable_to_acq_index
```

Based on this, the hardware backend creates a hardware acquisition mapping.

```{code-cell} ipython3
schedule = _determine_absolute_timing(schedule)

compiled_schedule = hardware_compile(schedule, quantum_device.generate_compilation_config())

compiled_schedule["compiled_instructions"]["mock_rom"].hardware_acq_mapping_binned
```

This hardware mapping is in the compiled schedule, and later is used to map the hardware data to the `xarray.Dataset`. In practice, each acquisition channel and acquisition index is mapped to a hardware bin.

There are a set of utility functions that help backend developers to format the retrieved data: {func}`~.instrument_coordinator.utility.add_acquisition_coords_binned` and {func}`~.instrument_coordinator.utility.add_acquisition_coords_nonbinned`. The mock backend calls these functions to add the “coords” to the retrieved data before returning it to users.

The retrieved data includes both acquisition indices and coords on the relevant acquisition channel, and does not contain the hardware bins.

```{code-cell} ipython3
settings = compiled_schedule["compiled_instructions"]["mock_rom"]
rom_icc.prepare(settings)
rom_icc.start()
rom_icc.retrieve_acquisition()
```
