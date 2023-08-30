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
from quantify_scheduler.compilation import determine_absolute_timing
from quantify_scheduler.backends.qblox_backend import hardware_compile
from quantify_scheduler.resources import ClockResource

from quantify_core.data.handling import set_datadir

temp_dir = TemporaryDirectory()
set_datadir(temp_dir.name)
```

In this section we introduce how to configure [Qblox Clusters](https://www.qblox.com/cluster) and the options available for them via Quantify.
For information about their lower-level functionality, you can consult the [Qblox Instruments documentation](https://qblox-qblox-instruments.readthedocs-hosted.com/en/master/).
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
    "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
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
        "cluster0": {
            "cluster0_module1": {
                "complex_output_0": {
                    "lo_name": "lo0",
                    "portclock_configs": [
                        {
                            "clock": "q4.01",
                            "port": "q4:mw",
                        },
                    ]
                },
            },
            "cluster0_module2": {
                "complex_output_0": {
                    "portclock_configs": [
                        {
                            "clock": "q5.01",
                            "port": "q5:mw"
                        }
                    ]
                },
            },
        },
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
test_sched = determine_absolute_timing(test_sched)

quantum_device = QuantumDevice("DUT")
quantum_device.hardware_config(hardware_compilation_cfg)

hardware_compile(schedule=test_sched, config=quantum_device.generate_compilation_config())
```

Notice the {code}`"quantify_scheduler.backends.qblox_backend.hardware_compile"` backend is used. 
In the example, the Cluster is specified using an instrument with {code}`"instrument_type": "Cluster"`. In the backend, the Cluster instrument functions as a collection of modules.

The only instrument types that can be at the top level are:
- {code}`"Cluster"`,
- {code}`"LocalOscillator"`.

## Hardware description

To compile to a Cluster, one should include a valid {class}`~.backends.types.qblox.ClusterDescription` in the `"hardware_description"` part of the hardware compilation config.
The name of the Cluster (the key of the structure, `"cluster0"` in the example) can be chosen freely.

```{eval-rst}
.. autoclass:: quantify_scheduler.backends.types.qblox.ClusterDescription
    :noindex:
    :members:
    :inherited-members: BaseModel

```

Here the modules are described by their respective {obj}`~.backends.types.qblox.ClusterModuleDescription`. For example, a QRM-RF module is described by

```{eval-rst}
.. autoclass:: quantify_scheduler.backends.types.qblox.QRMRFDescription
    :noindex:
    :members:
    :inherited-members: BaseModel

```

Channel-specific settings can be set in the `{Complex,Real,Digital}ChannelDescription` datastructures. 
For example, for a QRM-RF module, the {class}`~.backends.types.qblox.ComplexChannelDescription` is used to describe the settings for the complex output.
To use the default settings,  one can omit the channel description from the {obj}`~.backends.types.qblox.ClusterModuleDescription`, as is done in the {ref}`example-cluster-hardware-compilation-config` above.

For a complex input/output, this datastructure is:

```{eval-rst}
.. autoclass:: quantify_scheduler.backends.types.qblox.ComplexChannelDescription
    :noindex:
    :members:
    :inherited-members: BaseModel

```

### Marker configuration

The markers can be configured by adding a `"marker_debug_mode_enable"` key to the {class}`~.backends.types.qblox.ComplexChannelDescription` 
(or {class}`~.backends.types.qblox.RealChannelDescription`). 
If the value is set to `True`, the operations defined for this I/O will be accompanied by a 4 ns trigger pulse on the marker located next to the I/O port.
The marker will be pulled high at the same time as the module starts playing or acquiring.
```{code-block} python
---
emphasize-lines: 11
linenos: true
---

hardware_compilation_cfg = {
    "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
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
    "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
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
    "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
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
        "lo1": {"instrument_type": "LocalOscillator", "power": 20},
    },
    "hardware_options": {
        "modulation_frequencies": {
            "q0:mw-q0.01": {
                "interm_freq": 50e6
            },
        },
    },
    "connectivity": {
        "cluster0": {
            "cluster0_module1": {
                "complex_output_0": {
                    "portclock_configs": [
                        {
                            "clock": "q0.01",
                            "port": "q0:mw"
                        }
                    ]
                },
            },
            "cluster0_module2": {
                "complex_output_0": {
                    "portclock_configs": [
                        {
                            "clock": "q0.01",
                            "port": "q0:mw"
                        }
                    ]
                },
            },
        },
    },
}
```

### QASM program indentation

By default, the fields of the compiled QASM program are separated by a space, but this formatting can be made more human-readable by adding indentation. If the `"align_qasm_fields"` key in the hardware configuration is set to `True` (`False` by default), then for the whole program, labels, instructions, argument lists and comments are aligned in columns (same indentation level). Note that adding indentation worsens performance and has no functional value besides making the program more human-readable.

```{code-block} python
---
emphasize-lines: 8
linenos: true
---
hardware_compilation_cfg = {
    "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
    "hardware_description":
    {
        "cluster0": {
            "instrument_type": "Cluster",
            "ref": "internal",
            "align_qasm_fields": True,
            "modules": {...}
        }
    },
    "hardware_options": {...},
    "connectivity": {...}
}
```

### Local Oscillator description

A local oscillator instrument can be used with baseband modules. After adding the instrument, we can reference it in a baseband-module output via the `"lo_name"` key.

The two mandatory parameters are the `"instrument_type"` (which should be `"LocalOscillator"`), and `"power"`. The local oscillator frequency is then controlled through the `"modulation_frequencies"` hardware option (see {ref}`sec-qblox-modulation-frequencies`), under the key of the port-clock combination associated with the baseband-module output.

Note that it is possible to add `"generic_icc_name"` as an optional parameter to the local oscillator hardware description, but only the default name `"generic"` is supported currently with the Qblox backend.

```{code-block} python
---
  emphasize-lines: 5,6,7,8,13,21
  linenos: true
---
hardware_compilation_cfg = {
    "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
    "hardware_description": {
        "cluster0": {...},
        "lo1": {
            "instrument_type": "LocalOscillator",
            "power": 20
        },
    },
    "hardware_options": {
        "modulation_frequencies": {
            "q1:mw-q1.01": {
                "lo_freq": 5e9
            }
        }
    },
    "connectivity": {
        "cluster0": {
            "cluster0_module1": {
                "complex_output_1": {
                    "lo_name": "lo1",
                    "portclock_configs": [
                        {
                            "clock": "q1.01",
                            "port": "q1:mw"
                        }
                    ]
                },
            },
        },
    }
}
```

(sec-qblox-connectivity)=

## Connectivity
The {class}`~.backends.types.common.Connectivity` describes how the inputs/outputs of the Cluster modules are connected to ports on the {class}`~.device_under_test.quantum_device.QuantumDevice`.

```{note}
The {class}`~.backends.types.common.Connectivity` datastructure is currently under development. Information on the connectivity between port-clock combinations on the quantum device and ports on the control hardware is currently included in the old-style hardware configuration file, which should be included in the `"connectivity"` field of the {class}`~.backends.types.common.HardwareCompilationConfig`.
```

The possible inputs/outputs are
- for `"QCM"`: `"complex_output_{0,1}"`, `"real_output_{0,1,2,3}"`,
- for `"QRM"`: `"complex_{output,input}_0"`, `"real_{output,input}_{0,1}"`,
- for `"QCM_RF"`: `"complex_output_{0,1}"`,
- for `"QRM_RF"`: `"complex_{output,input}_0"`.

```{note}
For RF hardware, if an output is unused, it will be turned off. This is to ensure that unused local oscillators do not interfere with used outputs.
```


### Port-clock configuration

Under these `"{complex,real}_{x}"` keys, we specify the port-clock combinations an output may target
(see the {ref}`sec-user-guide-ports-clocks` for more information on the role of ports and clocks within `quantify-scheduler`).

Each module can have at most 6 port-clock combinations defined, and the name for each `"port"` and `"clock"` combination must be unique. Each of these port-clock combinations is associated with one sequencer in the Qblox hardware.

```{note}
If you use gate-level operations, you have to follow strict rules for each kind of operation on which port name you can use (what's the naming convention for each port resource).
- `"<device element name>:mw"` for `Rxy` operation (and its derived operations),
- `"<device element name>:res"` for any measure operation,
- `"<device element name>:fl"` for the flux port.
```

### Frequency multiplexing

It is possible to do frequency multiplexing of the signals by adding multiple port-clock configurations to the same output.

```{code-cell} ipython3
---
mystnb:
  number_source_lines: true
  remove_code_outputs: true
---
hardware_compilation_cfg = {
    "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
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
    "connectivity": {
        "cluster0": {
            "cluster0_module1": {
                "complex_output_0": {
                    "portclock_configs": [
                        {
                            "port": "q0:mw",
                            "clock": "q0.01",
                        },
                        {
                            "port": "q0:mw",
                            "clock": "some_other_clock",
                        }
                    ]
                },
                "complex_output_1": {
                    "portclock_configs": [
                        {
                            "port": "q1:mw",
                            "clock": "q1.01",
                        }
                    ]
                }
            },
        }
    }
}
```

```{code-cell} ipython3
---
mystnb:
  remove_code_source: true
  remove_code_outputs: true
---

test_sched = Schedule("test_sched")
test_sched.add(
    SquarePulse(amp=1, duration=1e-6, port="q0:mw", clock="q0.01")
)
test_sched.add_resource(ClockResource(name="q0.01", freq=200e6))
test_sched.add_resource(ClockResource(name="some_other_clock", freq=100e6))

test_sched = determine_absolute_timing(test_sched)

quantum_device.hardware_config(hardware_compilation_cfg)
hardware_compile(schedule=test_sched, config=quantum_device.generate_compilation_config())
```

In the given example, we added a second port-clock configuration to output 0. Now any signal on port {code}`"q0:mw"` with clock {code}`"some_other_clock"` will be added digitally to the signal with the same port but clock {code}`"q0.01"`. 
The Qblox modules have six sequencers available, which sets the upper limit to our multiplexing capabilities.

```{note}
The backend requires that each _combination_ of a port and a clock is unique, that is, it is possible to use a certain port or clock multiple times but the combination of a port with a certain clock can only be used once in the hardware compilation config.
```

### Complex I/O

A complex I/O is defined by adding a `"complex_{output, input}_<n>"` to the module configuration.
Complex outputs (e.g. `complex_output_0`) are used for playbacks, while complex inputs (e.g. `complex_input_0`) are used for acquisitions.
However, for readout modules it is possible to use the `complex_output_<n>` key for both playbacks and acquisitions.

```{note}
It is not possible to use the same port-clock combination multiple times in the hardware compilation config. In that case, it is required to use only the `complex_output_<n>` key.
```

```{code-block} python
---
emphasize-lines: 9,18,26,34
linenos: true
---
hardware_compilation_cfg = {
    "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
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
        "cluster0": {
            "cluster0_module1": {
                "complex_output_0": {
                    "portclock_configs": [
                        {
                            "port": "q0:mw",
                            "clock": "q0.01",
                        }
                    ]
                },
                "complex_output_1": {
                    "portclock_configs": [
                        {
                            "port": "q0:res",
                            "clock": "q0.ro",
                        }
                    ]
                },
                "complex_input_0": {
                    "portclock_configs": [
                        {
                            "port": "q1:res",
                            "clock": "q1.ro",
                        }
                    ]
                }
            }
        },
    }
}
```

### Real I/O

A real I/O is defined by adding a `real_{output, input}_<n>` to the module configuration.
Real outputs (e.g. `real_output_0`) are used for playbacks, while real inputs (e.g. `real_input_0`) are used for acquisitions.
However, for readout modules it is possible to use the `real_output_<n>` key for both playbacks and acquisitions.
When using a real I/O, the backend automatically maps the signals to the correct output paths.

```{note}
It is not possible to use the same port-clock combination multiple times in the hardware compilation config. In that case, it is required to use only the `real_output_<n>` key.
```

For a real I/O, it is not allowed to use any pulses that have an imaginary component, i.e., only real valued pulses are allowed.
If you were to use a complex pulse, the backend will produce an error, e.g., square and ramp pulses are allowed but DRAG pulses are not.

```{code-block} python
---
emphasize-lines: 9,18,26,34
linenos: true
---
hardware_compilation_cfg = {
    "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
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
        "cluster0": {
            "cluster0_module1": {
                "real_output_0": {
                    "portclock_configs": [
                        {
                            "port": "q0:mw",
                            "clock": "q0.01",
                        }
                    ]
                },
                "real_output_1": {
                    "portclock_configs": [
                        {
                            "port": "q0:res",
                            "clock": "q0.ro",
                        }
                    ]
                },
                "real_input_0": {
                    "portclock_configs": [
                        {
                            "port": "q1:res",
                            "clock": "q1.ro",
                        }
                    ]
                }
            }
        },
    }
}
```

### Digital I/O

The markers can be controlled by defining a digital I/O, and adding a `MarkerPulse` on this I/O.
A digital I/O is defined by adding a `"digital_output_n"` to the module configuration. `n` is the number of the digital output port.
For a digital I/O only a port is required, no clocks or other parameters are needed.

```{code-block} python
---
emphasize-lines: 8
linenos: true
---
hardware_compilation_cfg = {
    "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
    "hardware_description": {...},
    "hardware_options": {...},
    "connectivity": {
        "cluster0": {
            "cluster0_module1": {
                "digital_output_0": {
                    "portclock_configs": [
                        {
                            "port": "q0:switch",
                        },
                    ],   
                },
            },
        },
    }
}
```

The `MarkerPulse` is defined by adding a `MarkerPulse` to the sequence in question. It takes the same parameters as any other pulse.
```{code-block} python
schedule.add(MarkerPulse(duration=52e-9, port="q0:switch"))
```



## Hardware options
The {class}`~.backends.types.qblox.QbloxHardwareOptions` datastructure contains the settings used in compiling from the quantum-device layer to a set of instructions for the control hardware.

```{eval-rst}
.. autoclass:: quantify_scheduler.backends.types.qblox.QbloxHardwareOptions
    :noindex:
    :members: 
    :inherited-members: BaseModel

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
- `"<transmon name>.01"` for `Rxy` operation (and its derived operations),
- `"<transmon name>.ro"` for any measure operation,
- `"<transmon name>.12"` for the {math}`|1\rangle \rightarrow |2\rangle` transition.
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
    "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
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
        "lo1": {"instrument_type": "LocalOscillator", "power": 20},
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
        "cluster0": {
            "cluster0_module1": {
                "complex_output_0": {
                    "portclock_configs": [
                        {
                            "clock": "q0.01",
                            "port": "q0:mw"
                        }
                    ]
                },
                "complex_output_1": {
                    "lo_name": "lo1",
                    "portclock_configs": [
                        {
                            "clock": "q1.01",
                            "port": "q1:mw"
                        }
                    ]
                },
            },
            "cluster0_module2": {
                "complex_output_0": {
                    "portclock_configs": [
                        {
                            "clock": "q2.01",
                            "port": "q2:mw"
                        }
                    ]
                },
                "complex_output_1": {
                    "portclock_configs": [
                        {
                            "clock": "q3.01",
                            "port": "q3:mw"
                        }
                    ]
                },
            },
        },
    },
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
    "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
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

* The `"input_gain"` parameter for QRM corresponds to the qcodes parameters [in0_gain](https://qblox-qblox-instruments.readthedocs-hosted.com/en/master/api_reference/qcm_qrm.html#QRM.in0_gain) and [in1_gain](https://qblox-qblox-instruments.readthedocs-hosted.com/en/master/api_reference/qcm_qrm.html#QRM.in1_gain).

Note, these parameters only affect the QRM modules. For complex inputs you have to specify a tuple (for the I and Q inputs), and for real inputs a scalar value.

```{code-block} python
---
emphasize-lines: 18-24
linenos: true
---
hardware_compilation_cfg = {
    "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
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
        "cluster0": {
            "cluster0_module1": {
                "complex_input_0": {
                    "portclock_configs": [
                        {
                            "clock": "q0.ro",
                            "port": "q0:res",
                        },
                    ]
                },
            },
            "cluster0_module2": {
                "real_input_0": {
                    "portclock_configs": [
                        {
                            "clock": "cl0.baseband",
                            "port": "q0:fl"
                        }
                    ]
                },
            },
        }
    }
}
```

#### Attenuation configuration

* The parameters `"output_att"` and `"input_att"` for QRM-RF correspond to the qcodes parameters [out0_att](https://qblox-qblox-instruments.readthedocs-hosted.com/en/master/api_reference/qcm_qrm.html#QRM_RF.out0_att) and [in0_att](https://qblox-qblox-instruments.readthedocs-hosted.com/en/master/api_reference/qcm_qrm.html#QRM_RF.in0_att) respectively.
* The parameter `"output_att"` for QCM-RF correspond to the qcodes parameters [out0_att](https://qblox-qblox-instruments.readthedocs-hosted.com/en/master/api_reference/qcm_qrm.html#QCM_RF.out0_att) and [out1_att](https://qblox-qblox-instruments.readthedocs-hosted.com/en/master/api_reference/qcm_qrm.html#QCM_RF.out1_att).

Note, that these parameters only affect RF modules.
See [Qblox Instruments: QCM-QRM](https://qblox-qblox-instruments.readthedocs-hosted.com/en/master/api_reference/qcm_qrm.html) documentation for allowed values.

```{code-block} python
---
emphasize-lines: 18-24
linenos: true
---
hardware_compilation_cfg = {
    "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
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
        "cluster0": {
            "cluster0_module1": {
                "complex_output_0": {
                    "portclock_configs": [
                        {
                            "clock": "q0.res",
                            "port": "q0:ro",
                        },
                    ]
                },
            },
            "cluster0_module2": {
                "complex_output_0": {
                    "portclock_configs": [
                        {
                            "clock": "q0.01",
                            "port": "q0:mw"
                        }
                    ]
                },
            },
        }
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
    "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
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
    "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
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
```

If `"distortion_corrections"` are set, then `"filter_func"`, `"input_var_name"` and `"kwargs"` are required. If `"clipping_values"` are set, its value must be a list with exactly 2 floats.

Clipping values are the boundaries to which the corrected pulses will be clipped,
upon exceeding, these are optional to supply.

The `"filter_func"` is a python function that we apply with `"kwargs"` arguments. The waveform to be modified will be passed to this function in the argument name specified by `"input_var_name"`. The waveform will be passed as a `np.ndarray`.

### Sequencer options

Several options are available that are set on the sequencer that is assigned to a certain port-clock combination.
These can be set by adding `"sequencer_options"` to the hardware options.

```{eval-rst}
.. autoclass:: quantify_scheduler.backends.types.qblox.SequencerOptions
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
    "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
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

(sec-qblox-cluster-long-waveform-support-new)=
## Long waveform support

It is possible to play waveforms that are too long to fit in the waveform memory of Qblox modules. For a few standard waveforms, the square pulse, ramp pulse and staircase pulse, the following helper functions create operations that can readily be added to schedules:

```{code-cell} ipython3
---
mystnb:
  number_source_lines: true
  remove_code_outputs: true
---

from quantify_scheduler.operations.pulse_factories import (
    long_ramp_pulse,
    long_square_pulse,
    staircase_pulse,
)

ramp_pulse = long_ramp_pulse(amp=0.5, duration=1e-3, port="q0:mw")
square_pulse = long_square_pulse(amp=0.5, duration=1e-3, port="q0:mw")
staircase_pulse = staircase_pulse(
    start_amp=0.0, final_amp=1.0, num_steps=20, duration=1e-4, port="q0:mw"
)
```

More complex waveforms can be created from the {class}`~quantify_scheduler.operations.stitched_pulse.StitchedPulseBuilder`. This class allows you to construct complex waveforms by stitching together available pulses, and adding voltage offsets in between. Voltage offsets can be specified with or without a duration. In the latter case, the offset will hold until the last operation in the {class}`~quantify_scheduler.operations.stitched_pulse.StitchedPulse` ends. For example,

```{code-cell} ipython3
---
mystnb:
  number_source_lines: true
  remove_code_outputs: true
---

from quantify_scheduler.operations.pulse_library import RampPulse
from quantify_scheduler.operations.stitched_pulse import StitchedPulseBuilder

trapezoid_pulse = (
    StitchedPulseBuilder(port="q0:mw", clock="q0.01")
    .add_pulse(RampPulse(amp=0.5, duration=1e-8, port="q0:mw"))
    .add_voltage_offset(path_0=0.5, path_1=0.0, duration=1e-7)
    .add_pulse(RampPulse(amp=-0.5, offset=0.5, duration=1e-8, port="q0:mw"))
    .build()
)

repeat_pulse_with_offset = (
    StitchedPulseBuilder(port="q0:mw", clock="q0.01")
    .add_pulse(RampPulse(amp=0.2, duration=8e-6, port="q0:mw"))
    .add_voltage_offset(path_0=0.4, path_1=0.0)
    .add_pulse(RampPulse(amp=0.2, duration=8e-6, port="q0:mw"))
    .build()
)
```

Pulses and offsets are appended to the end of the last added operation by default. By specifying the `append=False` keyword argument in the `add_pulse` and `add_voltage_offset` methods, in combination with the `rel_time` argument, you can insert an operation at the specified time relative to the start of the {class}`~quantify_scheduler.operations.stitched_pulse.StitchedPulse`. The example below uses this to generate a series of square pulses of various durations and amplitudes.

```{code-cell} ipython3
---
mystnb:
  number_source_lines: true
  remove_code_outputs: true
---

from quantify_scheduler.operations.stitched_pulse import StitchedPulseBuilder

offsets = [0.3, 0.4, 0.5]
durations = [1e-6, 2e-6, 1e-6]
start_times = [0.0, 2e-6, 6e-6]

builder = StitchedPulseBuilder(port="q0:mw", clock="q0.01")

for offset, duration, t_start in zip(offsets, durations, start_times):
    builder.add_voltage_offset(
        path_0=offset, path_1=0.0, duration=duration, append=False, rel_time=t_start
    )

pulse = builder.build()
```
