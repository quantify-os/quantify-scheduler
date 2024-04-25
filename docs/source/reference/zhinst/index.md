---
file_format: mystnb
kernelspec:
    name: python3

---
(sec-backend-zhinst)=
# Zurich Instruments
```{admonition} Info
:class: note
The zhinst backend is currently only compatible with python versions 3.8 and 3.9. Please see the [installation page](zhinst-backend-install) for more information.
```
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

````{admonition} Old-style hardware config dictionary
:class: dropdown
The {class}`~quantify_scheduler.backends.types.common.HardwareCompilationConfig` is a {class}`~quantify_scheduler.structure.model.DataStructure` that adds validation and structure to the information that was previously stored in the hardware configuration dictionary. The old-style hardware configuration dictionary is still supported, but will be deprecated in the future.
````

```{code-cell} ipython3
---
tags: [hide-cell]
mystnb:
  code_prompt_show: "Example old-style hardware configuration"
  remove_code_source: true  
---

import json

from quantify_scheduler.backends.zhinst.zhinst_hardware_config_old_style import hardware_config as old_style_hardware_config

print(json.dumps(old_style_hardware_config, indent=4, sort_keys=False))
```

The {mod}`~quantify_scheduler.backends.zhinst_backend` allows Zurich Instruments to be
configured individually or collectively by enabling master/slave configurations via
Triggers and Markers.

The compilation onto Zurich Instruments hardware is configured by the {ref}`Hardware Compilation Config <sec-hardware-compilation-config>`.
The configuration file contains parameters about the Instruments, their connectivity to the quantum device, and options used in mapping {class}`quantify_scheduler.operations.operation.Operation`s, which act on qubits, onto physical properties of the instruments.

To use the Zurich Instruments backend in compilation, one should pass a valid hardware compilation configuration to the `quantum_device.hardware_config` parameter, such that it can be used to generate a full `CompilationConfig` using `quantum_device.generate_compilation_config()`, which can finally be used to compile a {class}`~.Schedule` using {meth}`~quantify_scheduler.backends.graph_compilation.QuantifyCompiler.compile`. The `"config_type"` entry specifies to the scheduler that we are using the Zurich Instruments backend (specifically, the {class}`quantify_scheduler.backends.zhinst_backend.ZIHardwareCompilationConfig` DataStructure will be parsed).
See {ref}`the hardware verification tutorial <hardware-verfication-tutorial>` for an example.

```{code-cell} ipython3
---
tags: [hide-cell]
mystnb:
  code_prompt_show: "Example hardware compilation configuration"  
---
import json
from quantify_scheduler.schemas.examples.utils import load_json_example_scheme
from quantify_scheduler.backends.zhinst_backend import ZIHardwareCompilationConfig

hardware_compilation_config = load_json_example_scheme("zhinst_hardware_compilation_config.json")
print(json.dumps(hardware_compilation_config, indent=4, sort_keys=False))

ZIHardwareCompilationConfig.model_validate(hardware_compilation_config)
```

(sec-zhinst-hardware-description)=
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
(sec-zhinst-connectivity)=
### Connectivity
The {class}`~.backends.types.common.Connectivity` describes how the inputs/outputs of the Zurich Instruments devices are connected to ports on the {class}`~.device_under_test.quantum_device.QuantumDevice`. As described in {ref}`sec-connectivity`, the connectivity datastructure can be parsed from a list of edges, which are described by a pair of strings that each specify a port on the quantum device, on a HDAWG/UHFQA, or on other auxiliary instruments (like external IQ mixers).

Each input/output node of the HDAWG/UHFQA should be specified in the connectivity as `"{instrument_name}.{channel_name}"`. The possible channel names are the same as the allowed fields in the corresponding {obj}`~.backends.types.zhinst.ZIHardwareDescription` datastructure:
- for `"HDAWG4"`: `"channel_{0,1}"`,
- for `"HDAWG8"`: `"channel_{0,1,2,3}"`, 
- for `"UHFQA"`: `"channel_0"`.

```{note}
The UHFQA has both an output and an input channel. However, this is currently represented in the connectivity as a single channel that is connected (through IQ mixers) to a port on the quantum device. Both pulses and acquisitions will be assigned to this channel.
```

The connectivity can be visualized using:

```{code-cell} ipython3
from quantify_scheduler.backends.types.common import Connectivity
connectivity = Connectivity.model_validate(hardware_compilation_config["connectivity"])
connectivity.draw()
```

#### External IQ mixers and local oscillators
HDAWG/UHFQA channels can be connected to external IQ mixers and local oscillators. To achieve this, you should add a {class}`~.quantify_scheduler.backends.types.common.IQMixerDescription` and {class}`~.quantify_scheduler.backends.types.common.LocalOscillatorDescription` to the `"hardware_description"` part of the hardware compilation config, and specify the connections of the `"if"`, `"lo"` and `"rf"` ports on the IQ mixer in the `"connectivity"` part of the hardware compilation config. The compiler will then use this information to assign the pulses and acquisitions to the port on the HDAWG/UHFQA that is connected to the `"if"` port on the IQ mixer, and set the local oscillator and intermodulation frequencies accordingly.

```{code-block} python
---
  emphasize-lines: 5,6,7,8,14,20,21,22
  linenos: true
---
hardware_compilation_cfg = {
    "config_type": "quantify_scheduler.backends.qblox_backend.ZIHardwareCompilationConfig",
    "hardware_description": {
        "ic_hdawg0": {...},
        "lo1": {
            "instrument_type": "LocalOscillator",
            "power": 20
        },
        "iq_mixer1": {"instrument_type": "IQMixer"},
    },
    "hardware_options": {
        "modulation_frequencies": {
            "q1:mw-q1.01": {
                "lo_freq": 5e9
            }
        }
    },
    "connectivity": {
        "graph": [
            ("ic_hdawg0.channel_0", "iq_mixer1.if"),
            ("lo1.output", "iq_mixer1.lo"),
            ("iq_mixer1.rf", "q1:mw"),
        ]
    }
}
```

(sec-zhinst-hardware-options)=
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
