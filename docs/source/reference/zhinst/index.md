(sec-backend-zhinst)=
# Zurich Instruments

## Introduction

`quantify-scheduler` provides a stateless module: {mod}`~quantify_scheduler.backends.zhinst_backend`,
that abstracts the complexity of setting up experiments using [Zurich Instruments](https://www.zhinst.com) hardware.
`quantify-scheduler` uses information on the quantum device
and instrument properties to compile a {class}`quantify_scheduler.schedules.schedule.Schedule` into waveforms and sequencing instructions suitable for execution on Zurich Instruments hardware.
More information about `compilation` can be found in the {ref}`User Guide <sec-user-guide>`.

Using existing programming interfaces provided via {doc}`zhinst-qcodes <zhinst-qcodes:index>` and {doc}`zhinst-toolkit <zhinst-toolkit:index>`, `quantify-scheduler` prepares the instruments that are present in the {ref}`sec-hardware-description <Hardware Description>`.

Finally, after configuring and running {func}`~quantify_scheduler.backends.zhinst_backend.compile_backend`
successfully the instruments are prepared for execution.

The Zurich Instruments backend provides:

- Synchronized execution of a program in a UHFQA and an HDAWG through the use of Triggers and Markers.
- Automatic generation of the SeqC Sequencer instructions.
- Waveform generation and modulation.
- Memory-efficient Sequencing with the CommandTable.
- Configuration of the relevant settings on all instruments.
- Configuration for Triggers and Markers.

## Supported Instruments

The Zurich Instruments backend currently supports the **HDAWG** and the **UHFQA** instruments.
In addition to the Zurich Instruments devices, the hardware backend supports using microwave sources such as the R&S SGS100A.

## Basic paradigm and inner workings

The Zurich Instruments back end threats a {class}`~.Schedule` as a linear timeline of operations executed on the quantum device and translates this into pulses and acquisitions to be executed on the different input and output channels of an HDAWG and a UHFQA.
In this context a single HDAWG is treated as the master device that sends out a single marker pulse for synchronization at the start of each iteration of the schedule that is used to trigger all other devices.
After the synchronization trigger is given, all devices execute a compiled program for which the timings of all instructions have been calculated.

The compilation from operations at the quantum-device layer to instructions that the hardware can execute is done in several steps.
The zhinst {func}`~quantify_scheduler.backends.zhinst_backend.compile_backend` starts from the {attr}`.ScheduleBase.timing_table` and maps the operations to channels on the hardware using the information specified in the
{ref}`sec-connectivity <Connectivity>`.
Corrections for channel latency, as well as moving operations around to ensure all measurements start at the first sample (0) of a clock cycle are also done at this stage.

Once the starting time and sample of each operation are known, the numerical waveforms that have to be uploaded to the hardware can be generated.
The numerical waveforms differ from the idealized waveforms of the device description in that they include corrections for effects such as mixer-skewness and linear-dynamical distortions (not implemented yet), and intermediate frequency modulation if required.

Both the {attr}`.CompiledSchedule.hardware_timing_table` and {attr}`.CompiledSchedule.hardware_waveform_dict` are available as properties of the {class}`.CompiledSchedule`.

In the next step, the clock-accurate Seqc instructions for each awg of each device are determined, as well as the settings (nodes) to be configured including the linking of the numerical waveforms to these nodes.
All of this information is combined in {class}`~.backends.zhinst.settings.ZISettingsBuilder`s and added to the compiled instructions of the {class}`.CompiledSchedule`.

## Limitations

There are several limitations to the paradigm and to the current implementation.
Some of these are relatively easy to address while others are more fundamental to the paradigm.
Here we give an overview of the known limitations.
Note that some of these can be quite specific.

### Inherent limitations

There are some inherent limitations to the paradigm of describing the program as a single linear timeline that is started using a single synchronization trigger.
These limitations cannot easily be addressed so should be taken into account when thinking about experiments.

- Because the **synchronization of the HDAWG and the UFHQA relies on a trigger on two devices operating at different clock frequencies** one cannot guarantee at what sample within the clock domain the slave device gets triggered. The consequence is that although the triggering is stable within an experiment, the exact time difference (in the number of samples) between the different devices varies between different experiments. This problem is inherent to the triggering scheme and cannot be easily resolved.
- The paradigm of a single fixed timeline with a single synchronizing trigger is **not compatible with a control loop affecting feedback**.

### Limitations with the current implementation

There are also some practical limitations to the implementation.
Keep these in mind when operating the hardware.

- **Real-time modulation is currently not supported**, relying on pre-modulated waveforms, it is important to start waveforms at a multiple of the modulation frequency. Sticking to a 10 ns grid, it is recommended to use a modulation frequency of 100 MHz.
- **All operations need to start at an integer number of samples**. Because of the choice of sampling rates of 2.4 GSps (~0.416 ns) and 1.8 GSps (~0.555 ns) it is useful to stick to a 10 ns grid for HDAWG (microwave and flux) pulses and a 40 ns grid for UHFQA (readout) pulses.
- **Different instructions on the same "awg" cannot start in the same clock cycle.** This implies that the readout acquisition delay cannot be 0 (but it can be 40 ns or - 40ns).
- **All measurements are triggered simultaneously** using the `StartQA(QA_INT_ALL, true)` instruction. This implies it is not possible to read out only a specific qubit/channel.
- **All measurements have to start at the same sample within a clock cycle** because one can only define a single integration weight per channel. To guarantee this, all operations are shifted around a bit (the measurement fixpoint correction). As a consequence, the reset/initialization operation can sometimes be a bit longer than specified in the schedule.
- Because the **timing between two devices needs to align over longer schedules**, it is important that the clock-rates are accurate. To ensure phase stability, use a 10 MHz shared reference and operate the hardware in external reference mode.
- Only a single HDAWG supported as the primary device, other HDAWGs need to be configured as secondary devices.
- **Multiplexed readout is currently not supported**. One can only read out a single channel. (#191)

(zhinst-hardware)=
## Hardware compilation configuration

The {mod}`~quantify_scheduler.backends.zhinst_backend` allows Zurich Instruments to be
configured individually or collectively by enabling master/slave configurations via
Triggers and Markers.

The compilation onto Zurich Instruments hardware is configured by the {ref}`Hardware Compilation Config <sec-hardware-compilation-config>`.
The configuration file contains parameters about the Instruments, their connectivity to the quantum device, and options used in mapping {class}`quantify_scheduler.operations.operation.Operation`s, which act on qubits, onto physical properties of the instruments.

To use the Zurich Instruments backend in compilation, one should pass a valid hardware compilation configuration to the `quantum_device.hardware_config` parameter, such that it can be used to generate a full `CompilationConfig` using `quantum_device.generate_compilation_config()`, which can finally be used to compile a {class}`~.Schedule` using {meth}`~quantify_scheduler.backends.graph_compilation.QuantifyCompiler.compile`. The entry {code}`"backend": "quantify_scheduler.backends.zhinst_backend.compile_backend"` specifies to the scheduler that we are using the Zurich Instruments backend (specifically the {func}`~quantify_scheduler.backends.zhinst_backend.compile_backend` function).
See {ref}`the hardware verification tutorial <hardware-verfication-tutorial>` for an example.


````{admonition} Example Zurich Instruments hardware compilation configuration file
:class: dropdown
```{literalinclude} ../../../../quantify_scheduler/schemas/examples/zhinst_hardware_compilation_config.json
:language: JSON
```
````

### Hardware Description

The {ref}`Hardware Description <sec-hardware-description>` describes the instruments that are used in the setup, along with some instrument-specific settings. The currently supported instruments are:

```{eval-rst}
.. autoapiclass:: quantify_scheduler.backends.types.zhinst.ZIHDAWG4Description
    :noindex:
    :members: ref, instrument_type, channelgrouping, clock_select, channel_0, channel_1

```

```{eval-rst}
.. autoapiclass:: quantify_scheduler.backends.types.zhinst.ZIHDAWG8Description
    :noindex:
    :members: ref, instrument_type, channelgrouping, clock_select, channel_0, channel_1, channel_2, channel_3

```

```{eval-rst}
.. autoapiclass:: quantify_scheduler.backends.types.zhinst.ZIUHFQADescription
    :noindex:
    :members: ref, instrument_type, channel_0

```

```{warning}
In order for the backend to find the QCodes Instrument it is required that the keys of the
HardwareDescription map 1-to-1 to the names given to the QCodes Instrument during instantiation, with an `ic` prepend.

> - Example: If the hdawg QCodes Instrument name is "hdawg_dev8831" then the {class}`~quantify_scheduler.backends.types.zhinst.Device`'s `name` is "ic_hdawg_dev8831"
```

The channels of these instruments are described by

```{eval-rst}
.. autoapiclass:: quantify_scheduler.backends.types.zhinst.ZIChannelDescription
    :noindex:
    :members:

```

Local oscillators can also be included by using the following generic datastructure.

```{eval-rst}
.. autoapiclass:: quantify_scheduler.backends.types.common.LocalOscillatorDescription
    :noindex:
    :members:

```

### Connectivity
The {class}`~.backends.types.common.Connectivity` describes how the inputs/outputs of the Zurich Instruments devices are connected to ports on the {class}`~.device_under_test.quantum_device.QuantumDevice`.

```{note}
The {class}`~.backends.types.common.Connectivity` datastructure is currently under development. Information on the connectivity between port-clock combinations on the quantum device and ports on the control hardware is currently included in the old-style hardware configuration file, which should be included in the `"connectivity"` field of the {class}`~.backends.types.common.HardwareCompilationConfig`.
```

### Hardware Options
The {class}`~.backends.types.zhinst.ZIHardwareOptions` datastructure contains the settings used in compiling from the quantum-device layer to a set of instructions for the control hardware.

```{eval-rst}
.. autoapiclass:: quantify_scheduler.backends.types.zhinst.ZIHardwareOptions
    :noindex:
    :members: latency_corrections, distortion_corrections, modulation_frequencies, mixer_corrections, output_gain

```

```{note}
In the Zurich Instruments backend, a `LatencyCorrection` is implemented by incrementing the `abs_time` of all operations applied to the port-clock combination.
```

## Tutorials

```{toctree}
:maxdepth: 3

T_verification_programs
```
