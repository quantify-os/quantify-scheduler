---
file_format: mystnb
kernelspec:
    name: python3

---
(sec-qblox-pulsar)=

# Pulsar QCM/QRM

```{warning}
Pulsar hardware is deprecated. Use cluster modules instead if possible!
```

```{code-cell} ipython3
---
mystnb:
  remove_code_source: true
  remove_code_outputs: true
---

# in the hidden cells we include some code that checks for correctness of the examples
from tempfile import TemporaryDirectory

from quantify_scheduler.device_under_test.quantum_device import QuantumDevice
from quantify_scheduler.operations import pulse_library
from quantify_scheduler.compilation import determine_absolute_timing
from quantify_scheduler.backends.qblox_backend import hardware_compile
from quantify_scheduler import Schedule
from quantify_scheduler.resources import ClockResource

from quantify_core.data.handling import set_datadir

temp_dir = TemporaryDirectory()
set_datadir(temp_dir.name)
```

Each device in the setup can be individually configured using the entry in the config. For instance:

```{code-cell} ipython3
---
mystnb:
  remove_code_outputs: true
  number_source_lines: true
---

hardware_compilation_cfg = {
    "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
    "hardware_description": {
        "qcm0": {
            "hardware_type": "Qblox",
            "instrument_type": "Pulsar_QCM",
            "ref": "internal",
        },
       "lo0": {"hardware_type": "LocalOscillator", "power": 20},
       "lo1": {"hardware_type": "LocalOscillator", "power": 20} 
    },
    "hardware_options": {
        "modulation_frequencies": {
            "q0:mw-q0.01": {"interm_freq": 50e6},
            "q1:mw-q1.01": {"lo_freq": 7.2e9}
        }
    },
    "connectivity": {
        "qcm0": {
            "complex_output_0": {
                "lo_name": "lo0",
                "portclock_configs": [
                    {
                        "port": "q0:mw",
                        "clock": "q0.01",
                    }
                ]
            },
            "complex_output_1": {
                "lo_name": "lo1",
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
```

```{code-cell} ipython3
---
mystnb:
  remove_code_source: true
  remove_code_outputs: true
---

test_sched = Schedule("test_sched")
test_sched.add(
    pulse_library.SquarePulse(amp=0.2, duration=1e-6, port="q0:mw", clock="q0.01")
)
test_sched.add_resource(ClockResource(name="q0.01", freq=7e9))
test_sched = determine_absolute_timing(test_sched)

quantum_device = QuantumDevice("DUT")
quantum_device.hardware_config(hardware_compilation_cfg)

hardware_compile(schedule=test_sched, config=quantum_device.generate_compilation_config())
```

Here we specify a setup containing only a Pulsar QCM, with both outputs connected to local oscillator sources.

The first entry specifies the backend, the function that will compile a schedule using the information specified in this hardware compilation config.
The other entries specify the instruments that are used (`"hardware_description"`), how they are connected to ports on the quantum device (`"connectivity"`), and options used in the compilation (`"hardware_options"`).
The instrument names need to match the names of the corresponding QCoDeS instruments.

## Hardware description

The hardware description of {code}`"qcm0"` contain settings and information for the entire device:

```{eval-rst}
.. autoclass:: quantify_scheduler.backends.types.qblox.PulsarQCMDescription
    :noindex:
    :members:
    :inherited-members: BaseModel

```

The examples given below will be for a single Pulsar QCM, but the other devices can be configured similarly. In order to use a Pulsar QRM, QCM-RF or QRM-RF, change the {code}`"instrument_type"` entry to {code}`"Pulsar_QRM"`, {code}`"Pulsar_QCM_RF"` or {code}`"Pulsar_QRM_RF"`
respectively. Multiple devices can be added to the config, similar to how we added the local oscillators in the example given above.
The name of the Pulsar (the key of the structure, `"qcm0"` in the example) can be chosen freely.

## Connectivity
The {class}`~.backends.graph_compilation.Connectivity` describes how the inputs/outputs of the Pulsar are connected to ports on the {class}`~.device_under_test.quantum_device.QuantumDevice`.

```{note}
The {class}`~.backends.graph_compilation.Connectivity` datastructure is currently under development. Information on the connectivity between port-clock combinations on the quantum device and ports on the control hardware is currently included in the old-style hardware configuration file, which should be included in the `"connectivity"` field of the {class}`~.backends.graph_compilation.HardwareCompilationConfig`.
```

Most notably under the {code}`complex_output_0`, we specify the port-clock combinations the output may target (see the {ref}`User guide <sec-user-guide>`
for more information on the role of ports and clocks within `quantify-scheduler`).

```{code-block} python
:linenos: true

"portclock_configs": [
    {
        "port": "q0:mw",
        "clock": "q0.01",
    }
]
```

### Usage without an LO

In order to use the backend without an LO, we simply remove the {code}`"lo_name"` and all other related parameters. This includes the
mixer correction parameters as well as the frequencies.

```{code-cell} ipython3
---
mystnb:
  number_source_lines: true
  remove_code_outputs: true
---

hardware_compilation_cfg = {
    "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
    "hardware_description": {
        "qcm0": {
            "hardware_type": "Qblox",
            "instrument_type": "Pulsar_QCM",
            "ref": "internal",
        },
    },
    "connectivity": {
        "qcm0": {
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
                        "port": "q1:mw",
                        "clock": "q1.01",
                    }
                ]
            }
        },
    }
}
```

```{code-cell} ipython3
---
mystnb:
  remove_code_source: true
  remove_code_outputs: true
---

quantum_device.hardware_config(hardware_compilation_cfg)
hardware_compile(schedule=test_sched, config=quantum_device.generate_compilation_config())
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
        "qcm0": {
            "hardware_type": "Qblox",
            "instrument_type": "Pulsar_QCM",
            "ref": "internal",
        },
    },
    "connectivity": {
        "qcm0": {
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
```

```{code-cell} ipython3
---
mystnb:
  remove_code_source: true
  remove_code_outputs: true
---

test_sched = Schedule("test_sched")
test_sched.add(
    pulse_library.SquarePulse(amp=1, duration=1e-6, port="q0:mw", clock="q0.01")
)
test_sched.add_resource(ClockResource(name="q0.01", freq=200e6))
test_sched.add_resource(ClockResource(name="some_other_clock", freq=100e6))

test_sched = determine_absolute_timing(test_sched)

quantum_device.hardware_config(hardware_compilation_cfg)
hardware_compile(schedule=test_sched, config=quantum_device.generate_compilation_config())
```

In the given example, we added a second port-clock configuration to output 0. Now any signal on port {code}`"q0:mw"` with clock {code}`"some_other_clock"` will be added digitally to the signal with the same port but clock {code}`"q0.01"`. The Qblox modules currently have six sequencers available, which sets the upper limit to our multiplexing capabilities.

```{note}
We note that it is a requirement of the backend that each combination of a port and a clock is unique, i.e. it is possible to use the same port or clock multiple times in the hardware config but the combination of a port with a certain clock can only occur once.
```

### Real mode

```{note}
This setting will soon move to a different place in the {class}`~.backends.graph_compilation.HardwareCompilationConfig`.
```

For the baseband modules, it is also possible to use the backend to generate signals for the outputs individually rather than using IQ pairs.

In order to do this, instead of {code}`"complex_output_X"`, we use {code}`"real_output_X"`. In the case of a QCM, we have four of those outputs. The QRM has two available.

The resulting config looks like this:

```{code-cell} ipython3
---
mystnb:
  number_source_lines: true
  remove_code_outputs: true
---

hardare_compilation_cfg = {
    "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
    "hardware_description": {
        "qcm0": {
            "hardware_type": "Qblox",
            "instrument_type": "Pulsar_QCM",
            "ref": "internal"
        }
    },
    "connectivity": {
        "qcm0": {
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
                        "port": "q1:mw",
                        "clock": "q1.01",
                    }
                ]
            },
            "real_output_2": {
                "portclock_configs": [
                    {
                        "port": "q2:mw",
                        "clock": "q2.01",
                    }
                ]
            }
        },
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
    pulse_library.SquarePulse(amp=1, duration=1e-6, port="q0:mw", clock="q0.01")
)
test_sched.add(
    pulse_library.SquarePulse(amp=1, duration=1e-6, port="q1:mw", clock="q1.01")
)
test_sched.add_resource(ClockResource(name="q0.01", freq=200e6))
test_sched.add_resource(ClockResource(name="q1.01", freq=100e6))

test_sched = determine_absolute_timing(test_sched)
quantum_device.hardware_config(hardware_compilation_cfg)

hardware_compile(schedule=test_sched, config=quantum_device.generate_compilation_config())
```

When using real outputs, the backend automatically maps the signals to the correct output paths. We note that for real outputs, it is not allowed to use any pulses that have an imaginary component i.e. only real-valued pulses are allowed. If you were to use a complex pulse, the backend will produce an error, e.g. square and ramp pulses are allowed but DRAG pulses are not.

```{warning}
When using real mode, we highly recommend using it in combination with the instrument coordinator as the outputs need to be configured correctly in order for this to function.
```

```{code-cell} ipython3
---
tags: [raises-exception]
mystnb:
  remove_code_source: true
  remove_code_outputs: true
---

test_sched.add(
    pulse_library.DRAGPulse(
        G_amp=1, D_amp=1, duration=1e-6, port="q1:mw", clock="q1.01", phase=0
    )
)

test_sched = determine_absolute_timing(test_sched)

hardware_compile(schedule=test_sched, config=quantum_device.generate_compilation_config())

```

(sec-qblox-pulsar-instruction-generated)=
### Instruction-generated pulses

```{note}
This setting will soon move to a different place in the {class}`~.backends.graph_compilation.HardwareCompilationConfig`.
```

```{warning}
The {code}`instruction_generated_pulses_enabled` option is deprecated and will be removed in a future version. Long square pulses and staircase pulses can be generated with the newly introduced {class}`~quantify_scheduler.operations.stitched_pulse.StitchedPulseBuilder`. More information can be found in the {ref}`relevant section of the Cluster user guide <sec-qblox-cluster-long-waveform-support>`.
```

The Qblox backend contains some intelligence that allows it to generate certain specific waveforms from the pulse library using a more complicated series of sequencer instructions, which helps conserve waveform memory. Though in order to keep the backend fully transparent, all such advanced capabilities are disabled by default.

In order to enable the advanced capabilities we need to add line {code}`"instruction_generated_pulses_enabled": True` to the port-clock configuration.

```{code-cell} ipython3
---
mystnb:
  number_source_lines: true
  remove_code_outputs: true
---

hardware_compilation_cfg = {
    "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
    "hardware_description": {
        "qcm0": {
            "hardware_type": "Qblox",
            "instrument_type": "Pulsar_QCM",
            "ref": "internal",
        },
    },
    "connectivity": {
        "qcm0": {
            "complex_output_0": {
                "portclock_configs": [
                    {
                        "port": "q0:mw",
                        "clock": "q0.01",
                        "instruction_generated_pulses_enabled": True,
                    }
                ]
            }
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
    pulse_library.SquarePulse(amp=1, duration=1e-3, port="q0:mw", clock="q0.01")
)

test_sched.add_resource(ClockResource(name="q0.01", freq=200e6))

test_sched = determine_absolute_timing(test_sched)

quantum_device.hardware_config(hardware_compilation_cfg)
hardware_compile(schedule=test_sched, config=quantum_device.generate_compilation_config())
```

Currently, this has the following effects:

- Long square pulses get broken up into separate pulses with durations \<= 1 us, which allows the modules to play square pulses longer than the waveform memory normally allows.
- Staircase pulses are generated using offset instructions instead of using waveform memory

## Hardware options
The {ref}`Hardware Options <sec-hardware-options>` provide a way of specifying some specific settings on the Pulsar.

### I/Q modulation

To perform upconversion using an I/Q mixer and an external local oscillator, simply specify a local oscillator in the `"connectivity"` using the {code}`lo_name` entry.
{code}`complex_output_0` is connected to a local oscillator instrument named
{code}`lo0` and {code}`complex_output_1` to {code}`lo1`.
Since the aim of `quantify-scheduler` is to only specify the final RF frequency when the signal arrives at the chip, rather than any parameters related to I/Q modulation, we specify this information here.

The backend assumes that upconversion happens according to the relation

```{math} f_{RF} = f_{IF} + f_{LO}
```

This means that in order to generate a certain {math}`f_{RF}`, we need to specify either an IF or an LO frequency. This is done in the `"modulation_frequencies"` within the `"hardware_options"`, where we either set the {code}`lo_freq` or the {code}`interm_freq` and leave the other to be calculated by the backend. Specifying both will raise an error if it violates {math}`f_{RF} = f_{IF} + f_{LO}`.

### Mixer corrections

The backend also supports setting the parameters that are used by the hardware to correct for mixer imperfections in real-time.

We configure this by adding these settings to the `"hardware_options"`:

```{code-block} python
:linenos: true

"hardware_options": {
    "mixer_corrections": {
        "dc_offset_i": -0.054,
        "dc_offset_q": -0.034,
        "amp_ratio": 0.9997,
        "phase_error": -4.0,
    }
}
```

Here, the `"dc_offset_i"` and `"dc_offset_q"` parameters add a DC offset to the outputs to correct for feed-through of the local oscillator signal.
The `"amp_ratio"` and `"phase_error"` are used to correct for imperfect rejection of the unwanted sideband.

### Gain and attenuation

For QRM, you can set the gain and attenuation parameters in dB. See the example below.
* The `"input_gain"` parameter for QRM corresponds to the qcodes parameters [in0_gain](https://qblox-qblox-instruments.readthedocs-hosted.com/en/master/api_reference/qcm_qrm.html#pulsar-qrm-pulsar-in0-gain) and [in1_gain](https://qblox-qblox-instruments.readthedocs-hosted.com/en/master/api_reference/qcm_qrm.html#pulsar-qrm-pulsar-in1-gain) respectively.

```{code-block} python
:linenos: true

hardware_compilation_cfg = {
    "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
    "hardware_description": {
        "qrm0": {
            "hardware_type": "Qblox",
            "instrument_type": "Pulsar_QRM",
            "ref": "internal",
        }
    },
    "hardware_options": {
        "power_scaling": {
            "q0:res-q0.ro": {"input_gain": (2,3)}
        }
    },
    "connectivity": {
        "qrm0": {
            "complex_output_0": {
                "portclock_configs": [
                    {
                        "port": "q0:res",
                        "clock": "q0.ro",
                    }
                ]
            }
        }
    }
}
```

See [Qblox Instruments: QCM-QRM](https://qblox-qblox-instruments.readthedocs-hosted.com/en/master/api_reference/qcm_qrm.html) documentation for allowed values.
