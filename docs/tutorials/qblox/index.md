(sec-backend-qblox)=

# Backend: Qblox

```{warning}
The {mod}`quantify_scheduler.backends.qblox_backend` is still under development.
Breaking changes at this stage are not excluded.
```

## Introduction

{mod}`quantify_scheduler` provides a modular system: {mod}`~quantify_scheduler.backends.qblox_backend`,
that abstracts the full experimental setup using [Qblox](https://www.qblox.com) hardware for
experiments in a modern and automated fashion.

Functionality included in this backend:

- Full compilation of the schedule to a program for the sequencer.
- Waveform generation and modulation, in a parameterized fashion as supported by Qblox hardware.
- Built-in version handling to ensure the backend works correctly with the installed driver version.
- Automatic handling of the hardware constraints such as output voltage ranges and sampling rates.
- Calculation of the optimal hardware settings for execution of the provided schedule using the Instrument Coordinator.
- Real mode, which allows to address the individual outputs separately rather than using IQ signals.
- Full support of frequency multiplexing.
- Automatic calculation of the required parameters for external local oscillators.
- Correction of the mixer errors using specified correction parameters.
- Flexible configuration via JSON data structures.

No special configuration is required to use this backend. Simply specify {obj}`quantify_scheduler.backends.qblox_backend.hardware_compile`
in the hardware configuration to use this backend or call the function directly. Please see {ref}`Usage of the backend <sec-qblox-how-to-configure>`
for information on how to set this up.
After a schedule is compiled into a program, uploading to the hardware can be done using the usual
[qblox-instruments](https://pypi.org/project/qblox-instruments/) drivers. Installation of these drivers
is done through

```console
$ pip install qblox-instruments
```

Please visit the [Qblox instruments documentation](https://qblox-qblox-instruments.readthedocs-hosted.com)
for more information.

## Supported Instruments

- ✅ QCM
- ✅ QRM
- ✅ QCM-RF
- ✅ QRM-RF
- ✅ Cluster
- ✅ Local Oscillator
- ⬜️ SPI

## Recent changes

For recent changes regarding the Qblox backend see the following.

```{toctree}
recent
```

## How to use

```{toctree}
:maxdepth: 2

How to use
Pulsar
Cluster
```
