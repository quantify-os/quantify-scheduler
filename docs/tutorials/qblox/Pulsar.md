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

mapping_config = {
    "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
    "qcm0": {
        "instrument_type": "Pulsar_QCM",
        "ref": "internal",
        "complex_output_0": {
            "lo_name": "lo0",
            "portclock_configs": [
                {
                    "port": "q0:mw",
                    "clock": "q0.01",
                    "interm_freq": 50e6
                }
            ]
        },
        "complex_output_1": {
            "lo_name": "lo1",
            "portclock_configs": [
                {
                    "port": "q1:mw",
                    "clock": "q1.01",
                    "interm_freq": None
                }
            ]
        }
    },
    "lo0": {"instrument_type": "LocalOscillator", "frequency": None, "power": 20},
    "lo1": {"instrument_type": "LocalOscillator", "frequency": 7.2e9, "power": 20}
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

hardware_compile(test_sched, mapping_config)
```

Here we specify a setup containing only a Pulsar QCM, with both outputs connected to a local oscillator sources.

The first entry specifies the backend, the function that will compile a schedule using the information specified in this hardware config.
All other entries at the highest level are instruments ({code}`"qcm0"`, {code}`"lo0"`, {code}`"lo1"`).
These names need to match the names of the corresponding QCoDeS instruments.

The first few entries of {code}`"qcm0"` contain settings and information for the entire device:
{code}`"type": "Pulsar_QCM"` specifies that this device is a Pulsar QCM,
and {code}`"ref": "internal"` sets the reference source to internal (as opposed to {code}`"external"`). Under the entries
{code}`complex_output_0` (corresponding to O{sup}`1/2`) and {code}`complex_output_1` (corresponding to O{sup}`3/4`),
we set all the parameters that are configurable per output.

The examples given below will be for a single Pulsar QCM, but the other devices can be configured similarly. In order to use a Pulsar QRM, QCM-RF or QRM-RF, change the {code}`"instrument_type"` entry to {code}`"Pulsar_QRM"`, {code}`"Pulsar_QCM_RF"` or {code}`"Pulsar_QRM_RF"`
respectively. Multiple devices can be added to the config, similar to how we added the local oscillators in the example given above.

## Output settings

Most notably under the {code}`complex_output_0`, we specify the port-clock combinations the output may target (see the {ref}`User guide <sec-user-guide>`
for more information on the role of ports and clocks within the Quantify-Scheduler).

```{code-block} python
:linenos: true

"portclock_configs": [
    {
        "port": "q0:mw",
        "clock": "q0.01",
        "interm_freq": 50e6
    }
]
```

Additionally, the entry {code}`interm_freq` specifies the intermediate frequency to use for I/Q modulation (in Hz) when targeting this port and clock.

## I/Q modulation

To perform upconversion using an I/Q mixer and an external local oscillator, simply specify a local oscillator in the config using the {code}`lo_name` entry.
{code}`complex_output_0` is connected to a local oscillator instrument named
{code}`lo0` and {code}`complex_output_1` to {code}`lo1`.
Since the Quantify-Scheduler aim is to only specify the final RF frequency when the signal arrives at the chip, rather than any parameters related to I/Q modulation, we specify this information here.

The backend assumes that upconversion happens according to the relation

```{math} f_{RF} = f_{IF} + f_{LO}
```

This means that in order to generate a certain {math}`f_{RF}`, we need to specify either an IF or an LO frequency. In the
dictionary, we therefore either set the {code}`lo_freq` or the {code}`interm_freq` and leave the other to be calculated by
the backend by specifying it as {code}`None`. Specifying both will raise an error if it violates {math}`f_{RF} = f_{IF} + f_{LO}`.

### Downconverter

Some users may have a custom Qblox downconverter module operating at 4.4 GHz.
In order to use it with this backend, we should specify a {code}`"downconverter": True` entry in the outputs that are connected to this module, as exemplified below.
The result is that the downconversion stage will be taken into account when calculating the IF or LO frequency (whichever was undefined) during compilation, such that the signal reaching the target port is at the desired clock frequency.

```{code-block} python
---
emphasize-lines: 7
linenos: true
---

mapping_config_rf = {
    "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
    "qcm0": {
        "instrument_type": "Pulsar_QCM_RF",
        "ref": "internal",
        "complex_output_0": {
            "downconverter": True,
            "portclock_configs": [
                {
                    "port": "q0:mw",
                    "clock": "q0.01",
                    "interm_freq": 50000000.0
                }
            ]
        }
    }
}
hardware_compile(test_sched, mapping_config_rf)

```

## Mixer corrections

The backend also supports setting the parameters that are used by the hardware to correct for mixer imperfections in real-time.

We configure this by adding the lines

```{code-block} python
:linenos: true

"dc_mixer_offset_I": -0.054,
"dc_mixer_offset_Q": -0.034,
```

to {code}`complex_output_0` (or {code}`complex_output_1`) in order to add a DC offset to the outputs to correct for feed-through of the local oscillator signal. And we add

```{code-block} python
:linenos: true

"mixer_amp_ratio": 0.9997,
"mixer_phase_error_deg": -4.0,
```

to the port-clock configuration in order to set the amplitude and phase correction to correct for imperfect rejection of the unwanted sideband.

## Usage without an LO

In order to use the backend without an LO, we simply remove the {code}`"lo_name"` and all other related parameters. This includes the
mixer correction parameters as well as the frequencies.

```{code-cell} ipython3
---
mystnb:
  number_source_lines: true
  remove_code_outputs: true
---

mapping_config = {
    "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
    "qcm0": {
        "instrument_type": "Pulsar_QCM",
        "ref": "internal",
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
```

```{code-cell} ipython3
---
mystnb:
  remove_code_source: true
  remove_code_outputs: true
---

hardware_compile(test_sched, mapping_config)
```

## Frequency multiplexing

It is possible to do frequency multiplexing of the signals by adding multiple port-clock configurations to the same output.

```{code-cell} ipython3
---
mystnb:
  number_source_lines: true
  remove_code_outputs: true
---

mapping_config = {
    "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
    "qcm0": {
        "instrument_type": "Pulsar_QCM",
        "ref": "internal",
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

hardware_compile(test_sched, mapping_config)
```

In the given example, we added a second port-clock configuration to output 0. Now any signal on port {code}`"q0:mw"` with clock {code}`"some_other_clock"` will be added digitally to the signal with the same port but clock {code}`"q0.01"`. The Qblox modules currently have six sequencers available, which sets the upper limit to our multiplexing capabilities.

```{note}
We note that it is a requirement of the backend that each combination of a port and a clock is unique, i.e. it is possible to use the same port or clock multiple times in the hardware config but the combination of a port with a certain clock can only occur once.
```

## Gain and attenuation

For QRM, QRM-RF and QCM-RF modules you can set the gain and attenuation parameters in dB. See the example below for the possible gain and attenuation parameters for each module type.
* The parameters `input_gain_I` and `input_gain_Q` for QRM correspond to the qcodes parameters [in0_gain](https://qblox-qblox-instruments.readthedocs-hosted.com/en/master/api_reference/qcm_qrm.html#pulsar-qrm-pulsar-in0-gain) and [in1_gain](https://qblox-qblox-instruments.readthedocs-hosted.com/en/master/api_reference/qcm_qrm.html#pulsar-qrm-pulsar-in1-gain) respectively.
* The parameters `output_att` and `input_att` for QRM-RF correspond to the qcodes parameters [out0_att](https://qblox-qblox-instruments.readthedocs-hosted.com/en/master/api_reference/qcm_qrm.html#cluster-qrm-rf-module-out0-att) and [in0_att](https://qblox-qblox-instruments.readthedocs-hosted.com/en/master/api_reference/qcm_qrm.html#cluster-qrm-rf-module-in0-att) respectively.
* The parameters `output_att`s for QCM-RF correspond to the qcodes parameters [out0_att](https://qblox-qblox-instruments.readthedocs-hosted.com/en/master/api_reference/qcm_qrm.html#cluster-qcm-rf-module-out0-att) and [out1_att](https://qblox-qblox-instruments.readthedocs-hosted.com/en/master/api_reference/qcm_qrm.html#cluster-qcm-rf-module-out1-att).

```{code-block} python
:linenos: true

mapping_config = {
    ...
    "qrm0": {
        "instrument_type": "Pulsar_QRM",
        "complex_output_0": {
            "input_gain_I": 2,
            "input_gain_Q": 3,
            ...
        },
    },
    "qrm_rf0": {
        "instrument_type": "QRM_RF",
        "complex_output_0": {
            "output_att": 12,
            "input_att": 10,
            ...
        },
    },
    "qcm_rf0": {
        "instrument_type": "QCM_RF",
        "complex_output_0": {
            "output_att": 4,
            ...
        },
        "complex_output_1": {
            "output_att": 6,
            ...
        },
    },
}
```

See [Qblox Instruments: QCM-QRM](https://qblox-qblox-instruments.readthedocs-hosted.com/en/master/api_reference/qcm_qrm.html) documentation for allowed values.

## Real mode

For the baseband modules, it is also possible to use the backend to generate signals for the outputs individually rather than using IQ pairs.

In order to do this, instead of {code}`"complex_output_X"`, we use {code}`"real_output_X"`. In case of a QCM, we have four of those outputs. The QRM has two available.

The resulting config looks like:

```{code-cell} ipython3
---
mystnb:
  number_source_lines: true
  remove_code_outputs: true
---

mapping_config = {
    "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
    "qcm0": {
        "instrument_type": "Pulsar_QCM",
        "ref": "internal",
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

hardware_compile(test_sched, mapping_config)
```

When using real outputs, the backend automatically maps the signals to the correct output paths. We note that for real outputs, it is not allowed to use any pulses that have an imaginary component i.e. only real valued pulses are allowed. If you were to use a complex pulse, the backend will produce an error, e.g. square and ramp pulses are allowed but DRAG pulses not.

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

hardware_compile(test_sched, mapping_config)

```

## Experimental features

The Qblox backend contains some intelligence that allows it to generate certain specific waveforms from the pulse library using a more complicated series of sequencer instructions, which helps conserve waveform memory. Though in order to keep the backend fully transparent, all such advanced capabilities are disabled by default.

In order to enable the advanced capabilities we need to add line {code}`"instruction_generated_pulses_enabled": True` to the port-clock configuration.

```{code-cell} ipython3
---
mystnb:
  number_source_lines: true
  remove_code_outputs: true
---

mapping_config = {
    "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
    "qcm0": {
        "instrument_type": "Pulsar_QCM",
        "ref": "internal",
        "complex_output_0": {
            "portclock_configs": [
                {
                    "port": "q0:mw",
                    "clock": "q0.01",
                    "instruction_generated_pulses_enabled": True
                }
            ]
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

test_sched = Schedule("test_sched")
test_sched.add(
    pulse_library.SquarePulse(amp=1, duration=1e-3, port="q0:mw", clock="q0.01")
)

test_sched.add_resource(ClockResource(name="q0.01", freq=200e6))

test_sched = determine_absolute_timing(test_sched)

hardware_compile(test_sched, mapping_config)
```

Currently, this has the following effects:

- Long square pulses get broken up into separate pulses with durations \<= 1 us, which allows the modules to play square pulses longer than the waveform memory normally allows.
- Staircase pulses are generated using offset instructions instead of using waveform memory
