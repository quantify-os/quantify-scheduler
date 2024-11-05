(sec-backend-qblox)=
# Qblox

`quantify-scheduler` provides the {mod}`~quantify_scheduler.backends.qblox_backend`
that facilitates setting up experiments using [Qblox](https://www.qblox.com) control hardware.

Functionality included in this backend:

- Full compilation of the schedule to a program for the sequencer.
- Waveform generation and modulation, in a parameterized fashion as supported by Qblox hardware.
- Built-in version handling to ensure the backend works correctly with the installed driver version.
- Automatic handling of the hardware constraints such as output voltage ranges and sampling rates.
- Calculation of the optimal hardware settings for the execution of the provided schedule using the Instrument Coordinator.
- Real mode, which allows addressing the individual outputs separately rather than using IQ signals.
- Full support of frequency multiplexing.
- Automatic calculation of the required parameters for external local oscillators.
- Correction of the mixer errors using specified correction parameters.
- Flexible configuration via JSON data structures.

Simply specify {code}`"quantify_scheduler.backends.qblox_backend.hardware_compile"`
in the hardware configuration to use this backend.
After a schedule is compiled into a program, uploading to the control hardware can be done using the usual
[qblox-instruments](https://pypi.org/project/qblox-instruments/) driver under the hood. The driver is automatically included in installing `quantify-scheduler`.
Please visit the [Qblox Instruments setup documentation](https://docs.qblox.com/en/main/getting_started/setup.html)
for detailed setup instructions.

Supported instruments:

- ✅ QCM
- ✅ QRM
- ✅ QCM-RF
- ✅ QRM-RF
- ✅ Local Oscillator
- ⬜️ SPI

```{toctree}
:hidden: true

recent
```

```{toctree}
:hidden: true
:maxdepth: 2

Cluster
Voltage offsets and long waveforms
Acquisition details
Hardware config versioning
```
