---
file_format: mystnb
kernelspec:
    name: python3

---
(sec-qblox-cluster-new)=

# Cluster (HardwareCompilationConfig)

```{admonition} Under construction
The {class}`~.backends.types.common.HardwareCompilationConfig` replaces the the old-style unvalidated json/dict hardware configuration, adding validation of the contents
and restructuring into `"hardware_description"`, `"hardware_options"` and `"connectivity"`. It is still under construction,
once finished the old-style hardware config will be deprecated but still supported until further notice.
```

```{code-cell} ipython3
---
mystnb:
  remove_code_source: true
  remove_code_outputs: true  
---

# in the hidden cells we include some code that checks for correctness of the examples
from tempfile import TemporaryDirectory

from quantify_scheduler import Schedule
from quantify_scheduler.device_under_test.quantum_device import QuantumDevice
from quantify_scheduler.operations.pulse_library import SquarePulse
from quantify_scheduler.compilation import _determine_absolute_timing
from quantify_scheduler.backends.qblox_backend import hardware_compile
from quantify_scheduler.resources import ClockResource

from quantify_core.data.handling import set_datadir

temp_dir = TemporaryDirectory()
set_datadir(temp_dir.name)
```

In this section we introduce how to configure [Qblox Clusters](https://www.qblox.com/products#cluster) and the options available for them via Quantify.
For information about their lower-level functionality, you can consult the [Qblox Instruments documentation](https://qblox-qblox-instruments.readthedocs-hosted.com/en/main/).
For information on the process of compilation to hardware, see {ref}`sec-tutorial-compiling`.

(example-cluster-hardware-compilation-config)=
## General hardware compilation config structure, example

We start by looking at an example config for a single Cluster. The hardware compilation configuration specifies which modules are used (`"hardware_descriptions"`) and how they are connected to the quantum device (`"connectivity"`), along with some (optional) `"hardware_options"`, like modulation frequencies, gains and attenuations, or mixer corrections. The general structure of this configuration file is described in the {ref}`sec-hardware-compilation-config` section of the User guide.

```{code-cell} ipython3
---
mystnb:
  number_source_lines: true
  remove_code_outputs: true
---
hardware_compilation_cfg = {
    "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
    "hardware_description": {
        "cluster0": {
            "instrument_type": "Cluster",
            "ref": "internal",
            "modules": {
                "1": {
                    "instrument_type": "QCM"
                },
                "2": {
                    "instrument_type": "QCM_RF"
                },
            }
        },
        "lo0": {
            "instrument_type": "LocalOscillator",
            "power": 20
        },
        "iq_mixer_lo0": {
            "instrument_type": "IQMixer"
        }
    },
    "hardware_options": {
        "modulation_frequencies": {
            "q4:mw-q4.01": {
                "interm_freq": 200e6
            },
            "q5:mw-q5.01": {
                "interm_freq": 50e6
            },
        },
        "mixer_corrections": {
            "q4:mw-q4.01": {
                "amp_ratio": 0.9999,
                "phase_error": -4.2
            }
        },
    },
    "connectivity": {
        "graph": [
            ("cluster0.module1.complex_output_0", "iq_mixer_lo0.if"),
            ("lo0.output", "iq_mixer_lo0.lo"),
            ("iq_mixer_lo0.rf", "q4:mw"),
            ("cluster0.module2.complex_output_0", "q5:mw"),
        ]
    },
}
```

```{code-cell} ipython3
---
mystnb:
  remove_code_source: true
  remove_code_outputs: true
---

# Validate hardware_compilation_cfg

test_sched = Schedule("test_sched")
test_sched.add(
    SquarePulse(amp=1, duration=1e-6, port="q4:mw", clock="q4.01")
)
test_sched.add(
    SquarePulse(amp=0.25, duration=1e-6, port="q5:mw", clock="q5.01")
)
test_sched.add_resource(ClockResource(name="q4.01", freq=7e9))
test_sched.add_resource(ClockResource(name="q5.01", freq=8e9))
test_sched = _determine_absolute_timing(test_sched)

quantum_device = QuantumDevice("DUT")
quantum_device.hardware_config(hardware_compilation_cfg)

hardware_compile(schedule=test_sched, config=quantum_device.generate_compilation_config())
```

Notice the {class}`quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig` config type is used.
In the example, the Cluster is specified using an instrument with {code}`"instrument_type": "Cluster"`. In the backend, the Cluster instrument functions as a collection of modules.

The only instrument types that can be at the top level are:
- {code}`"Cluster"`,
- {code}`"LocalOscillator"`.

(sec-cluster-hardware-description)=
## Hardware description

To compile to a Cluster, one should include a valid {class}`~.backends.types.qblox.ClusterDescription` in the `"hardware_description"` part of the hardware compilation config.
The name of the Cluster (the key of the structure, `"cluster0"` in the example) can be chosen freely.

```{eval-rst}
.. autoapiclass:: quantify_scheduler.backends.types.qblox.ClusterDescription
    :noindex:
    :members: ref, sequence_to_file, instrument_type, modules

```

Here the modules are described by their respective {obj}`~.backends.types.qblox.ClusterModuleDescription`. For example, a QRM-RF module is described by

```{eval-rst}
.. autoapiclass:: quantify_scheduler.backends.types.qblox.QRMRFDescription
    :noindex:
    :members:

```

Channel-specific settings can be set in the `{Complex,Real,Digital}ChannelDescription` datastructures. 
For example, for a QRM-RF module, the {class}`~.backends.types.qblox.ComplexChannelDescription` is used to describe the settings for the complex output.
To use the default settings,  one can omit the channel description from the {obj}`~.backends.types.qblox.ClusterModuleDescription`, as is done in the {ref}`example-cluster-hardware-compilation-config` above.

For a complex input/output, this datastructure is:

```{eval-rst}
.. autoapiclass:: quantify_scheduler.backends.types.qblox.ComplexChannelDescription
    :noindex:
    :members: 

```

### Marker configuration

The markers can be configured by adding a `"marker_debug_mode_enable"` key to the {class}`~.backends.types.qblox.ComplexChannelDescription` 
(or {class}`~.backends.types.qblox.RealChannelDescription`). 
If the value is set to `True`, the operations defined for this channel will be accompanied by a 4 ns trigger pulse on the marker located next to the I/O port.
The marker will be pulled high at the same time as the module starts playing or acquiring.
```{code-block} python
---
emphasize-lines: 11
linenos: true
---

hardware_compilation_cfg = {
    "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
    "hardware_description": {
        "cluster0": {
            "instrument_type": "Cluster",
            "ref": "internal",
            "modules": {
                "1": {
                    "instrument_type": "QCM",
                    "complex_output_0": {
                        "marker_debug_mode_enable": True,
                    }
                }
            }
        }
    },
    "hardware_options": {...},
    "connectivity": {...},
}
```
### Write sequencer program to files

It is possible to optionally include the `"sequence_to_file"` key. If set to `True`, a file will be created for each sequencer with the program that's uploaded to the sequencer with the filename `<data_dir>/schedules/<year><month><day>-<hour><minute><seconds>-<milliseconds>-<random>_<port>_<clock>.json` in a JSON format, where `<random>` is 6 random characters in the range `0-9`, `a-f`. The value defaults to `False` in case `"sequence_to_file"` is not included.

It is also possible to set this parameter per module via its module configuration.

```{code-block} python
---
emphasize-lines: 7
linenos: true
---
hardware_compilation_cfg = {
    "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
    "hardware_description": {
        "cluster0": {
            "instrument_type": "Cluster",
            "ref": "internal",
            "sequence_to_file": True,
            "modules": {...}
        }
    },
    "hardware_options": {...},
    "connectivity": {...}
}
```

### Downconverter

```{note}
This section is only relevant for users with custom Qblox downconverter hardware.
```

Some users employ a custom Qblox downconverter module. In order to use it with this backend, we specify a {code}`"downconverter_freq"` entry in the outputs that are connected to this module, as exemplified below.

The result is that the clock frequency is downconverted such that the signal reaching the target port is at the desired clock frequency, i.e. {math}`f_\mathrm{out} = f_\mathrm{downconverter} - f_\mathrm{in}`.

For baseband modules, downconversion will not happen if `"mix_lo"` is not `True` and there is no external LO specified (`"mix_lo"` is `True` by default). For RF modules, the `"mix_lo"` setting is not used (effectively, always `True`). Also see helper function {func}`~quantify_scheduler.backends.qblox.helpers.determine_clock_lo_interm_freqs`.

```{code-block} python
---
emphasize-lines: 11,12,18
linenos: true
---

hardware_compilation_cfg = {
    "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
    "hardware_description": {
        "cluster0": {
            "instrument_type": "Cluster",
            "ref": "internal",
            "modules": {
                "1": {
                    "instrument_type": "QCM",
                    "complex_output_0": {
                        "downconverter_freq": 9e9,
                        "mix_lo": True,
                    }
                },
                "2": {
                    "instrument_type": "QCM_RF",
                    "complex_output_0": {
                        "downconverter_freq": 9e9,
                    }
                },
            }
        },
        "lo0": {"instrument_type": "LocalOscillator", "power": 20},
        "iq_mixer_lo0": {"instrument_type": "IQMixer"},
    },
    "hardware_options": {
        "modulation_frequencies": {
            "q0:mw-q0.01": {
                "interm_freq": 50e6
            },
            "q1:mw-q1.01": {
                "interm_freq": 50e6
            },
        },
    },
    "connectivity": {
        "graph": [
            ("cluster0.module1.complex_output_0", "iq_mixer_lo0.if"),
            ("lo0.output", "iq_mixer_lo0.lo"),
            ("iq_mixer_lo0.rf", "q0:mw"),
            ("cluster0.module2.complex_output_0", "q1:mw"),
        ]
    },
}
```

(sec-qblox-connectivity)=

## Connectivity
The {class}`~.backends.types.common.Connectivity` describes how the inputs/outputs of the Cluster modules are connected to ports on the {class}`~.device_under_test.quantum_device.QuantumDevice`. As described in {ref}`sec-connectivity` in the User guide, the connectivity datastructure can be parsed from a list of edges, which are described by a pair of strings that each specify a port on the quantum device, on the cluster modules, or on other auxiliary instruments (like external IQ mixers).

Each input/output node of the Cluster should be specified in the connectivity as `"{cluster_name}.module{module_slot_index}.{channel_name}"`. For each module, the possible input/output names are the same as the allowed fields in the corresponding {obj}`~.backends.types.qblox.ClusterModuleDescription` datastructure:
- for `"QCM"`: `"complex_output_{0,1}"`, `"real_output_{0,1,2,3}"`,
- for `"QRM"`: `"complex_{output,input}_0"`, `"real_{output,input}_{0,1}"`,
- for `"QCM_RF"`: `"complex_output_{0,1}"`,
- for `"QRM_RF"`: `"complex_{output,input}_0"`.

```{note}
For RF hardware, if an output is unused, it will be turned off. This is to ensure that unused local oscillators do not interfere with used outputs.
```

The connectivity can be visualized using:

```{code-cell} ipython3
from quantify_scheduler.backends.types.common import Connectivity

connectivity = Connectivity.model_validate(hardware_compilation_cfg["connectivity"])
connectivity.draw()
```

### Ports and clocks
Each module can target at most 6 port-clock combinations within a schedule. Each of these port-clock combinations is associated with one sequencer in the Qblox hardware. See the {ref}`sec-user-guide-ports-clocks` section in the User guide for more information on the role of ports and clocks within `quantify-scheduler`.

```{code-cell} ipython3
---
mystnb:
  number_source_lines: true
  remove_code_outputs: true
---
hardware_compilation_cfg = {
    "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
    "hardware_description": {
        "cluster0": {
            "instrument_type": "Cluster",
            "ref": "internal",
            "modules": {
                "1": {
                    "instrument_type": "QCM"
                },
            }
        },
    },
    "hardware_options": {},
    "connectivity": {
        "graph": [
            ("cluster0.module1.complex_output_0", "q0:mw"),
        ]
    }
}

sched = Schedule("Single pulse schedule")
sched.add(
    SquarePulse(amp=1, duration=1e-6, port="q0:mw", clock="q0.01")
)
sched.add_resource(ClockResource(name="q0.01", freq=200e6))
```

```{code-cell} ipython3
---
mystnb:
  remove_code_source: true
  remove_code_outputs: true
---
sched = _determine_absolute_timing(sched)

quantum_device.hardware_config(hardware_compilation_cfg)
hardware_compile(schedule=sched, config=quantum_device.generate_compilation_config())
```

```{note}
With gate-level operations, you have to follow strict port naming:
- `"<device element name>:mw"` for `Rxy` operation (and its derived operations),
- `"<device element name>:res"` for any measure operation,
- `"<device element name>:fl"` for the flux port.
```

### Complex channel

A complex channel is defined by including a `"complex_{output, input}_<n>"` in the connectivity.
Complex outputs (e.g. `complex_output_0`) are used for playbacks, while complex inputs (e.g. `complex_input_0`) are used for acquisitions.
However, for readout modules it is possible to use the `complex_output_<n>` key for both playbacks and acquisitions.

```{note}
When using a port and clock combination for both playback and acquisition, only set up the `complex_output_<n>`.
```

```{code-block} python
---
emphasize-lines: 17,18
linenos: true
---
hardware_compilation_cfg = {
    "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
    "hardware_description": {
        "cluster0": {
            "instrument_type": "Cluster",
            "ref": "internal",
            "modules": {
                "1": {
                    "instrument_type": "QRM"
                },
            }
        }
    }
    "hardware_options": {...},
    "connectivity": {
        "graph": [
            ("cluster0.module1.complex_output_0", "q0:res"),
            ("cluster0.module1.complex_input_0", "q1:res")
        ]
    }
}
```

### Real channel

A real channel is defined by including a `real_{output, input}_<n>` in the connectivity.
Real outputs (e.g. `real_output_0`) are used for playbacks, while real inputs (e.g. `real_input_0`) are used for acquisitions.
However, for readout modules it is possible to use the `real_output_<n>` key for both playbacks and acquisitions.
When using a real channel, the backend automatically maps the signals to the correct output paths.

Note that the backend throws an error when using a real channel for pulses with an imaginary component. For example, square and ramp pulses are allowed, but DRAG pulses are not.

```{note}
When using a port and clock combination for both playback and acquisition, only set up the `real_output_<n>`.
```


```{code-block} python
---
emphasize-lines: 17,18,19
linenos: true
---
hardware_compilation_cfg = {
    "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
    "hardware_description": {
        "cluster0": {
            "instrument_type": "Cluster",
            "ref": "internal",
            "modules": {
                "1": {
                    "instrument_type": "QRM"
                },
            }
        }
    }
    "hardware_options": {...},
    "connectivity": {
        "graph": [
            ("cluster0.module1.real_output_0", "q0:mw"),
            ("cluster0.module1.real_output_1", "q0:res"),
            ("cluster0.module1.real_input_0", "q1:res")
        ]
    }
}
```

### Digital channel

The markers can be controlled by defining a digital channel, and adding a `MarkerPulse` on this channel.
A digital channel is defined by adding a `"digital_output_n"` to the connectivity. `n` is the number of the digital output port.
For a digital channel only a port is required, no clocks or other parameters are needed.

```{code-block} python
---
emphasize-lines: 7
linenos: true
---
hardware_compilation_cfg = {
    "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
    "hardware_description": {...},
    "hardware_options": {...},
    "connectivity": {
        "graph": [
            ("cluster0.module1.digital_output_0", "q0:switch")
        ]
    }
}
```

The `MarkerPulse` is defined by adding a `MarkerPulse` to the sequence in question. It takes the same parameters as any other pulse.
```{code-block} python
schedule.add(MarkerPulse(duration=52e-9, port="q0:switch"))
```

#### Clock names

Clocks in digital channels serve simply as a label and are automatically set to {attr}`"digital" <quantify_scheduler.resources.DigitalClockResource.IDENTITY>` at `MarkerPulse` initialization, but it is also possible to specify a custom clock name (for example, a clock name from the device configuration, like `qe0.ge0`). This makes it possible to connect a digital channel to a given port-clock combination in a device element, for example. Similar to clocks for non-digital channels, the clock must be either

- specified in the device configuration,
- added to the {class}`~quantify_scheduler.schedules.schedule.Schedule` as a {class}`~quantify_scheduler.resources.ClockResource`, or
- a clock that is present by default in the schedule resources, i.e. {attr}`"digital" <quantify_scheduler.resources.DigitalClockResource.IDENTITY>` or {attr}`"cl0.baseband" <quantify_scheduler.resources.BasebandClockResource.IDENTITY>`.


### External IQ mixers and local oscillators

Baseband modules can be connected to external IQ mixers and local oscillators. To achieve this, you should add a {class}`~.quantify_scheduler.backends.types.common.IQMixerDescription` and {class}`~.quantify_scheduler.backends.types.common.LocalOscillatorDescription` to the `"hardware_description"` part of the hardware compilation config, and specify the connections of the `"if"`, `"lo"` and `"rf"` ports on the IQ mixer in the `"connectivity"` part of the hardware compilation config. The compiler will then use this information to assign the pulses and acquisitions to the port on the baseband module that is connected to the `"if"` port on the IQ mixer, and set the local oscillator and intermodulation frequencies accordingly.

```{admonition} Local Oscillator Description
It is possible to add `"generic_icc_name"` as an optional parameter to the local oscillator hardware description, but only the default name `"generic"` is supported currently with the Qblox backend.
```

```{code-block} python
---
  emphasize-lines: 5,6,7,8,9,14,20,21,22
  linenos: true
---
hardware_compilation_cfg = {
    "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
    "hardware_description": {
        "cluster0": {...},
        "lo0": {
            "instrument_type": "LocalOscillator",
            "power": 20
        },
        "iq_mixer_lo0": {"instrument_type": "IQMixer"},
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
            ("cluster0.module1.complex_output_1", "iq_mixer_lo0.if"),
            ("lo0.output", "iq_mixer_lo0.lo"),
            ("iq_mixer_lo0.rf", "q1:mw"),
        ]
    }
}
```

### Frequency multiplexing

It is possible to do frequency multiplexing of the signals without changing the connectivity: by adding operations on the same port, but with different clocks.

```{code-cell} ipython3
---
mystnb:
  number_source_lines: true
  remove_code_outputs: true
---
hardware_compilation_cfg = {
    "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
    "hardware_description": {
        "cluster0": {
            "instrument_type": "Cluster",
            "ref": "internal",
            "modules": {
                "1": {
                    "instrument_type": "QCM"
                },
            }
        },
    },
    "hardware_options": {},
    "connectivity": {
        "graph": [
            ("cluster0.module1.complex_output_0", "q0:mw"),
        ]
    }
}

sched = Schedule("Multiplexed schedule")
sched.add(
    SquarePulse(amp=1, duration=1e-6, port="q0:mw", clock="q0.01")
)
sched.add(
    SquarePulse(amp=0.5, duration=1e-6, port="q0:mw", clock="some_other_clock")
)
sched.add_resource(ClockResource(name="q0.01", freq=200e6))
sched.add_resource(ClockResource(name="some_other_clock", freq=100e6))
```

```{code-cell} ipython3
---
mystnb:
  remove_code_source: true
  remove_code_outputs: true
---
sched = _determine_absolute_timing(sched)

quantum_device.hardware_config(hardware_compilation_cfg)
hardware_compile(schedule=sched, config=quantum_device.generate_compilation_config())
```

In the given example, we add two pulses on the same port but with different clocks. Now any signal on port {code}`"q0:mw"` with clock {code}`"some_other_clock"` will be added digitally to the signal with the same port but clock {code}`"q0.01"`. 
The Qblox modules have six sequencers available, which sets the upper limit to our multiplexing capabilities.


(sec-cluster-hardware-options)=
## Hardware options
The {class}`~.backends.types.qblox.QbloxHardwareOptions` datastructure contains the settings used in compiling from the quantum-device layer to a set of instructions for the control hardware.

```{eval-rst}
.. autoapiclass:: quantify_scheduler.backends.types.qblox.QbloxHardwareOptions
    :noindex:
    :members: latency_corrections, distortion_corrections, modulation_frequencies, mixer_corrections, input_gain, output_att, input_att, sequencer_options

```

(sec-qblox-modulation-frequencies)=
### Modulation frequencies

The aim of `quantify-scheduler` is to only specify the final RF frequency when the signal arrives at the chip, rather than any parameters related to I/Q modulation. However, you still need to provide some parameters for the up/downconversion.

The backend assumes that upconversion happens according to the relation

```{math} f_{RF} = f_{IF} + f_{LO}
```

These frequencies are specified for each port-clock combination in the `"modulation_frequencies"` in the `"hardware_options"`.

You can specify {math}`f_{RF}` in multiple ways. You can specify it when you add a `ClockResource` with `freq` argument to your `Schedule`, or when you specify the `BasicTransmonElement.clock_freqs`.

```{note}
If you use gate-level operations, you have to follow strict rules for the naming of the clock resource, for each kind of operation:
- `"<device element name>.01"` for `Rxy` operation (and its derived operations),
- `"<device element name>.ro"` for any measure operation,
- `"<device element name>.12"` for the {math}`|1\rangle \rightarrow |2\rangle` transition.
```

Then:
- For baseband modules, you can optionally specify a local oscillator by its name using the `"lo_name"` key in the {ref}`sec-qblox-connectivity`. If you specify it, the `"lo_freq"` key in the `"modulation_frequencies"` (see the example below) specifies {math}`f_{LO}` of this local oscillator. Otherwise, {math}`f_{LO} = 0` and {math}`f_{RF} = f_{IF}`. {math}`f_{RF} = f_{IF}` can also be set in the hardware options explicitly with the `"interm_freq"` key in the `"modulation_frequencies"`.
- For RF modules, you can specify {math}`f_{IF}` through the `"interm_freq"` key, and/or you can specify the local oscillator frequency for the output used for the port-clock combination with the `"lo_freq"`, because they have internal local oscillators. Note, if you specify both, the relationship between these frequencies should hold, otherwise you get an error message. It's important to note, that fast frequency sweeps only work when {math}`f_{LO}` is fixed, and {math}`f_{IF}` is unspecified. Because of this, it is generally advised to specify {math}`f_{LO}` only.

In the following example:
- For the baseband modules, `"complex_output_0"`'s {math}`f_{IF}` is the same as the `"q0.01"` clock resource's frequency, and `"complex_output_1"`'s {math}`f_{IF}` is calculated using the frequency of `"lo1"` (specified in `"modulation_frequencies"` under `"q1:mw-q1.01"` ) and `"q1.01"`.
- For the RF modules, `"complex_output_0"`'s {math}`f_{IF}` is calculated using the provided `"lo_freq"` for `"q2:mw-q2.01"` and the frequency of `"q2.01"`, and for `"complex_output_1"`, the {math}`f_{LO}` is calculated using the provided `"interm_freq"` for `"q3:mw-q3.01"` and the frequency of `"q3.01"`.

```{code-cell} ipython3
---
mystnb:
  remove_code_source: true
  remove_code_outputs: true
---
QuantumDevice.find_instrument("DUT").close()
```

```{code-cell} ipython3
---
mystnb:
  number_source_lines: true
  remove_code_outputs: true
---
from quantify_scheduler import Schedule
from quantify_scheduler.backends.graph_compilation import SerialCompiler
from quantify_scheduler.device_under_test.quantum_device import QuantumDevice
from quantify_scheduler.operations.pulse_library import SquarePulse
from quantify_scheduler.resources import ClockResource

hardware_compilation_cfg = {
    "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
    "hardware_description": {
        "cluster0": {
            "instrument_type": "Cluster",
            "ref": "internal",
            "modules": {
                "1": {
                    "instrument_type": "QCM"
                },
                "2": {
                    "instrument_type": "QCM_RF"
                },
            }
        },
        "lo0": {"instrument_type": "LocalOscillator", "power": 20},
        "iq_mixer_lo0": {"instrument_type": "IQMixer"},
    },
    "hardware_options": {
        "modulation_frequencies": {
            "q1:mw-q1.01": {
                "lo_freq": 5e9
            },
            "q2:mw-q2.01": {
                "lo_freq": 7e9
            },
            "q3:mw-q3.01": {
                "interm_freq": 50e6
            },
        },
    },
    "connectivity": {
        "graph": [
            ("cluster0.module1.complex_output_0", "q0:mw"),
            ("cluster0.module1.complex_output_1", "iq_mixer_lo0.if"),
            ("lo0.output", "iq_mixer_lo0.lo"),
            ("iq_mixer_lo0.rf", "q1:mw"),
            ("cluster0.module2.complex_output_0", "q2:mw"),
            ("cluster0.module2.complex_output_1", "q3:mw"),
        ]
    }
}

test_sched = Schedule("test_sched")
test_sched.add_resource(ClockResource(name="q0.01", freq=8e9))
test_sched.add_resource(ClockResource(name="q1.01", freq=9e9))
test_sched.add_resource(ClockResource(name="q2.01", freq=8e9))
test_sched.add_resource(ClockResource(name="q3.01", freq=9e9))

test_sched.add(SquarePulse(amp=1, duration=1e-6, port="q0:mw", clock="q0.01"))
test_sched.add(SquarePulse(amp=0.25, duration=1e-6, port="q1:mw", clock="q1.01"))
test_sched.add(SquarePulse(amp=0.25, duration=1e-6, port="q2:mw", clock="q2.01"))
test_sched.add(SquarePulse(amp=0.25, duration=1e-6, port="q3:mw", clock="q3.01"))

quantum_device = QuantumDevice("DUT")
quantum_device.hardware_config(hardware_compilation_cfg)
compiler = SerialCompiler(name="compiler")
_ = compiler.compile(
    schedule=test_sched, config=quantum_device.generate_compilation_config()
)
```


(sec-qblox-mixer-corrections-new)=
### Mixer corrections

The backend also supports setting the parameters that are used by the hardware to correct for mixer imperfections in real-time.

We configure this by adding the `"mixer_corrections"` to the hardware options for a specific port-clock combination. See the following example.

```{code-block} python
---
emphasize-lines: 8,9,10,11
linenos: true
---
hardware_compilation_cfg = {
    "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
    "hardware_description": {...},
    "connectivity": {...},
    "hardware_options": {
        "mixer_corrections": {
            "q4:mw-q4.01": {
                "dc_offset_i": -0.054,
                "dc_offset_q": -0.034,
                "amp_ratio": 0.9997,
                "phase_error": -4.0,
            }
        }
    }
}

```

### Gain and attenuation

For QRM, QRM-RF and QCM-RF modules you can set the gain and attenuation parameters in dB in the `"hardware_options"`.

#### Gain configuration

* The `"input_gain"` parameter for QRM corresponds to the qcodes parameters [in0_gain](https://qblox-qblox-instruments.readthedocs-hosted.com/en/main/api_reference/module.html#QRM.in0_gain) and [in1_gain](https://qblox-qblox-instruments.readthedocs-hosted.com/en/main/api_reference/module.html#QRM.in1_gain).

Note, these parameters only affect the QRM modules. For complex inputs you have to specify a tuple (for the I and Q inputs), and for real inputs a scalar value.

```{code-block} python
---
emphasize-lines: 18-24
linenos: true
---
hardware_compilation_cfg = {
    "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
    "hardware_description": {
        "cluster0": {
            "instrument_type": "Cluster",
            "ref": "internal",
            "modules": {
                "1": {
                    "instrument_type": "QRM"
                },
                "2": {
                    "instrument_type": "QRM"
                },
            }
        },
    },
    "hardware_options": {
        "input_gain": {
            "q0:res-q0.ro": {
                "gain_I": 2,
                "gain_Q": 3
            }
            "q0:fl-cl0.baseband": 2
        },
    },
    "connectivity": {
        "graph": [
            ("cluster0.module1.complex_input_0", "q0:res"),
            ("cluster0.module2.real_input_0", "q0:fl"),
        ]
    }
}
```

#### Attenuation configuration

* The parameters `"output_att"` and `"input_att"` for QRM-RF correspond to the qcodes parameters [out0_att](https://qblox-qblox-instruments.readthedocs-hosted.com/en/main/api_reference/module.html#QRM_RF.out0_att) and [in0_att](https://qblox-qblox-instruments.readthedocs-hosted.com/en/main/api_reference/module.html#QRM_RF.in0_att) respectively.
* The parameter `"output_att"` for QCM-RF correspond to the qcodes parameters [out0_att](https://qblox-qblox-instruments.readthedocs-hosted.com/en/main/api_reference/module.html#QCM_RF.out0_att) and [out1_att](https://qblox-qblox-instruments.readthedocs-hosted.com/en/main/api_reference/module.html#QCM_RF.out1_att).

Note, that these parameters only affect RF modules.
See [Qblox Instruments: QCM-QRM](https://qblox-qblox-instruments.readthedocs-hosted.com/en/main/api_reference/module.html) documentation for allowed values.

```{code-block} python
---
emphasize-lines: 18-24
linenos: true
---
hardware_compilation_cfg = {
    "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
    "hardware_description": {
        "cluster0": {
            "instrument_type": "Cluster",
            "ref": "internal",
            "modules": {
                "1": {
                    "instrument_type": "QRM_RF"
                },
                "2": {
                    "instrument_type": "QCM_RF"
                },
            }
        },
    },
    "hardware_options": {
        "output_att": {
            "q0:res-q0.ro": 12,
            "q0:mw-q0.01": 4
        },
        "input_att": {
            "q0:res-q0.ro": 10
        }
    },
    "connectivity": {
        "graph": [
            ("cluster0.module1.complex_output_0", "q0:res"),
            ("cluster0.module2.complex_output_0", "q0:mw"),
        ]
    }
}
```

### Maximum AWG output voltage

```{note}
This subsection on `max_awg_output_voltage` is still under construction.
```

### Latency corrections

Latency corrections is a `dict` containing the delays for each port-clock combination. It is possible to specify them under the key `"latency_corrections"` in the hardware options. See the following example.

```{code-block} python
---
emphasize-lines: 6
linenos: true
---
hardware_compilation_cfg = {
    "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
    "hardware_description": {...},
    "connectivity": {...},
    "hardware_options": {
        "latency_corrections": {
            "q4:mw-q4.01": 8e-9,
            "q5:mw-q5.01": 4e-9
        }
    }
}
```

Each correction is in nanoseconds. For each specified port-clock, the program start will be delayed by this amount of time. Note, the delay still has to be a multiple of the grid time.

### Distortion corrections

Distortion corrections apply a function on the pulses which are in the schedule. Note, that this will not be applied to outputs generated by modifying the offset and gain/attenuation. The `"distortion_corrections"` is an optional key in the hardware options. See the following example.

```{code-block} python
---
emphasize-lines: 6
linenos: true
---
hardware_compilation_cfg = {
    "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
    "hardware_description": {...},
    "connectivity": {...},
    "hardware_options": {
        "distortion_corrections": {
            "q0:fl-cl0.baseband": {
                "filter_func": "scipy.signal.lfilter",
                "input_var_name": "x",
                "kwargs": {
                    "b": [0.0, 0.5, 1.0],
                    "a": [1]
                },
                "clipping_values": [-2.5, 2.5]
            }
        }
    }
}
```

If `"distortion_corrections"` are set, then `"filter_func"`, `"input_var_name"` and `"kwargs"` are required. If `"clipping_values"` are set, its value must be a list with exactly 2 floats.

Clipping values are the boundaries to which the corrected pulses will be clipped,
upon exceeding, these are optional to supply.

The `"filter_func"` is a python function that we apply with `"kwargs"` arguments. The waveform to be modified will be passed to this function in the argument name specified by `"input_var_name"`. The waveform will be passed as a `np.ndarray`.


(sec-qblox-sequencer-options-new)=
### Sequencer options

Several options are available that are set on the sequencer that is assigned to a certain port-clock combination.
These can be set by adding `"sequencer_options"` to the hardware options.

```{eval-rst}
.. autoapiclass:: quantify_scheduler.backends.types.qblox.SequencerOptions
    :noindex:
    :members:

```

(sec-qblox-qasm-hook-new)=
#### QASM hook

It is possible to inject custom qasm instructions for each port-clock combination (sequencer), see the following example to insert a NOP (_no operation_) at the beginning of the program at line 0.

```{code-block} python
---
emphasize-lines: 12
linenos: true
---
def _func_for_hook_test(qasm: QASMProgram):
    qasm.instructions.insert(
        0, QASMProgram.get_instruction_as_list(q1asm_instructions.NOP)
    )

hardware_compilation_cfg = {
    "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
    "hardware_description": {...},
    "hardware_options": {
        "sequencer_options": {
            "q0:mw-q0.01": {
                "qasm_hook_func": _func_for_hook_test,
            }
        }
    },
    "connectivity": {...}
}
```

## Debug mode compilation

Debug mode can help with debugging by modifying the compilation process slightly.

If `"debug_mode"` key in the compilation configuration is set to `True` (`False` by default), the formatting of the compiled QASM program is made more human-readable by aligning all labels, instructions, argument lists and comments in the program in columns (same indentation level).

Note that adding indentation worsens performance and has no functional value besides aiding the debugging process.

```{code-cell} ipython3
---
mystnb:
    remove_code_outputs: true
---
from quantify_scheduler.backends import SerialCompiler

compiler = SerialCompiler(name="compiler")
compilation_config = quantum_device.generate_compilation_config()
compilation_config.debug_mode = True
_ = compiler.compile(
    schedule=test_sched, config=compilation_config
)
```
