---
file_format: mystnb
kernelspec:
    name: python3

---
(hardware-verfication-tutorial)=

# Tutorial: Zhinst hardware verification

``````{seealso}
The complete source code of this tutorial can be found in

{nb-download}`T_verification_programs.ipynb`

``````

```{code-cell} ipython3
---
mystnb:
  remove_code_source: true
---

# Make output easier to read
from rich import pretty

pretty.install()
```

## Introduction

This tutorial gives an overview of how to use the {mod}`~quantify_scheduler.backends.zhinst_backend` for quantify-scheduler.
The example we follow in this notebook corresponds to a standard circuit QED measurement setup in which readout is performed by generating a pulse on a UFHQA, send to the device and the returning signal is measured in transmission using the same UFHQA, and an HDAWG is then used to generate the control pulses.
We set up this test setup so that it is easy to verify the hardware is behaving correctly and that it can be used for real experiments with minimal modifications.

In {ref}`how to connect <sec-zhinst-how-to-connect>` we discuss how to set up the test setup, connect all the cables, and use quantify to connect to the hardware.
In {ref}`how to configure <sec-zhinst-how-to-configure>` we discuss how to write a hardware configuration file that can be used to compile instructions for the Zurich Instruments hardware.
In {ref}`verification programs <sec-zhinst-verification-programs>` we go over how to write and compile several verification programs, how to verify that these are correct, and how to execute these on the physical hardware.

(sec-zhinst-how-to-connect)=

## How to connect

```{note}
This documentation is a work in progress. See issue #237.
Improvements to this tutorial will include adding instructions on how to connect the hardware and how to set up the initialization script as well as more example programs.
```

(sec-zhinst-how-to-configure)=

## How to describe the hardware configuration

``````{admonition} Example Zurich Instruments hardware configuration file
:class: dropdown

In this tutorial we make use of the example configuration file that contains an HDAWG, a UHFQA and a few local oscillators. This same file is also used for testing purposes in the CI.

```{literalinclude} ../../../quantify_scheduler/schemas/examples/zhinst_test_mapping.json
:language: JSON
```
``````

(sec-zhinst-verification-programs)=

## Verifying the hardware compilation

Quantify-scheduler comes with several build in verification programs in the {mod}`~.schedules.verification` module.
Here we will start by building and compiling the first schedule by hand to show how one can construct such a schedule and debug it, before giving an overview of test programs.

### Bining and averaging - AWG staircase

#### Description

In this schedule we will play pulses of increasing amplitude on the HDAWG.
These pulses are modulated with a local oscillator so that they appear on the same frequency as the readout pulse.
This ensures that they are visible in the acquired signal.

#### Expected outcome

One would expect to see a monotonic increase in the measured amplitude.
The actual amplitudes would probably not match the input amplitudes 1-to-1 because there is likely some loss on the signal path from the up- and down-conversion.
Additionally, depending on the overlap between the pulse and the integration window, the average measured voltage will be slightly lower, and the phase can be slightly different resulting in not all signal being in the I-quadrature.

#### Creating the staircase program

We start by manually recreating the {func}`~quantify_scheduler.schedules.verification.awg_staircase_sched`, a schedule in which (modulated) square pulses are played on an HDAWG and the UHFQA is triggered subsequently to observe the result of that schedule.

```{code-cell} ipython3

# import statements required to make a schedule

import numpy as np

from quantify_scheduler import Schedule
from quantify_scheduler.operations.acquisition_library import SSBIntegrationComplex
from quantify_scheduler.operations.pulse_library import IdlePulse, SquarePulse
from quantify_scheduler.resources import ClockResource


```

```{code-cell} ipython3

pulse_amps = np.linspace(0.05, 0.9, 3)
repetitions = 1024
init_duration = 4000e-6  # 4us should allow for plenty of wait time
mw_port = "q0:mw"
ro_port = "q0:res"
mw_clock = "q0.01"  # chosen to correspond to values in the hardware cfg
ro_clock = "q0.ro"
readout_frequency = (
    6.5e9  # this frequency will be used for both the AWG pulse as well as
)
# for the readout.

pulse_duration = 1e-6
acq_channel = 0
integration_time = 2e-6
acquisition_delay = 0


sched = Schedule(name="AWG staircase", repetitions=repetitions)

sched.add_resource(ClockResource(name=mw_clock, freq=readout_frequency))
sched.add_resource(ClockResource(name=ro_clock, freq=readout_frequency))
pulse_amps = np.asarray(pulse_amps)


for acq_index, pulse_amp in enumerate(pulse_amps):

    sched.add(IdlePulse(duration=init_duration))

    pulse = sched.add(
        SquarePulse(
            duration=pulse_duration,
            amp=pulse_amp,
            port=mw_port,
            clock=mw_clock,
        ),
        label=f"SquarePulse_{acq_index}",
    )

    sched.add(
        SSBIntegrationComplex(
            duration=integration_time,
            port=ro_port,
            clock=ro_clock,
            acq_index=acq_index,
            acq_channel=acq_channel,
        ),
        ref_op=pulse,
        ref_pt="start",
        rel_time=acquisition_delay,
        label=f"Acquisition_{acq_index}",
    )

sched


```

Now that we have generated the schedule we can compile it and verify if the hardware output is correct.

```{code-cell} ipython3

from quantify_scheduler.backends.circuit_to_device import DeviceCompilationConfig
from quantify_scheduler.compilation import qcompile
from quantify_scheduler.schemas.examples import utils
from quantify_scheduler.schemas.examples.circuit_to_device_example_cfgs import (
    example_transmon_cfg,
)

transmon_device_cfg = DeviceCompilationConfig.parse_obj(example_transmon_cfg)
zhinst_hardware_cfg = utils.load_json_example_scheme("zhinst_test_mapping.json")

comp_sched = qcompile(
    schedule=sched, device_cfg=transmon_device_cfg, hardware_cfg=zhinst_hardware_cfg
)


```

##### The timing table

The {attr}`.ScheduleBase.timing_table` can be used after the absolute timing has been determined. It gives an overview of all operations in the schedule at the quantum-device level.

```{code-cell} ipython3

# Pandas dataframes do not render correctly in the sphinx documentation environment. See issue #238.
comp_sched.timing_table


```

##### The hardware timing table

The {attr}`.CompiledSchedule.hardware_timing_table` is populated during the hardware compilation. It gives an overview of all operations in the schedule at the control-electronics layer. This means that this the signals are corrected for effects such as gain and latency, and that modulations have been applied.

The "waveform_id" key can be used to find the numerical waveforms in {attr}`.CompiledSchedule.hardware_waveform_dict`.

```{code-cell} ipython3

comp_sched.hardware_timing_table


```

##### The hardware waveform dict

```{code-cell} ipython3

comp_sched.hardware_waveform_dict


```

##### The compiled instructions

The compiled instructions can be found in the `compiled_instructions` of the compiled schedule.

```{code-cell} ipython3

comp_sched.compiled_instructions


```

The setting for the Zurich Instruments instruments are stored as a {class}`~.ZIDeviceConfig`, of which the settings_builder contains the {class}`~.backends.zhinst.settings.ZISettingsBuilder` containing both the settings to set on all the nodes in the Zurich Instruments drivers as well as the compiled `seqc` instructions.

```{code-cell} ipython3

# the .as_dict method can be used to generate a "readable" overview of the settings.
hdawg_settings_dict = (
    comp_sched.compiled_instructions["ic_hdawg0"].settings_builder.build().as_dict()
)
# hdawg_settings_dict
hdawg_settings_dict


```

The compiler source string for each awg channel can be printed to see the instructions the ZI hardware will execute.
The clock-cycles are tracked by the assembler backend and can be compared to the hardware_timing_table.

```{code-cell} ipython3

awg_index = 0
print(hdawg_settings_dict["compiler/sourcestring"][awg_index])


```

```{code-cell} ipython3

# the .as_dict method can be used to generate a "readable" overview of the settings.
uhfqa_settings_dict = (
    comp_sched.compiled_instructions["ic_uhfqa0"].settings_builder.build().as_dict()
)
# uhfqa_settings_dict
uhfqa_settings_dict


```

```{code-cell} ipython3

awg_index = 0
print(uhfqa_settings_dict["compiler/sourcestring"][awg_index])


```

## Verification programs

Quantify-scheduler comes with several test programs that can be used to verify that the software and the hardware is configured and functioning correctly.
You should be able to run this notebook on your setup directly if you replace the mock_setup initialization with your own initialization script.

```{note}
This documentation is a work in progress. See issue #237.
Here we provide an overview of schedules that are used to verify different kinds of functionality.
This section will be expanded to include working examples.
```

### Time trace acquisition - readout pulse

#### Description

In this experiment, a square readout pulse is applied. This pulse should be visible in the acquisition window and can be used to calibrate the timing delay of the integration window.

This experiment can be used to verify the time-trace acquisition functionality of the readout module (e.g., Qblox QRM or ZI UHFQA) is working.

#### Expected outcome

A square pulse with some modulation is visible in the integration window.

{func}`~quantify_scheduler.schedules.trace_schedules.trace_schedule`

### Time trace acquisition - two pulses

#### Description

In this experiment, a square pulse is applied on the microwave drive line. This pulse should be visible in the acquisition window and can be used to calibrate the timing delay between the readout and control pulses.

This experiment can be used to verify the time-trace acquisition functionality of the readout module (e.g., Qblox QRM or ZI UHFQA) is working in combination with the synchronization between the readout module (e.g., Qblox QRM or ZI UHFQA) and the pulse generating module (e.g., Qblox QCM or ZI HDAWG).

#### Expected outcome

A square pulse with some modulation is visible on top of a second pulse with a different modulation frequency in the integration window.

{func}`~quantify_scheduler.schedules.trace_schedules.two_tone_trace_schedule`

### Weighted integration and averaging - Heterodyne spectroscopy

#### Description

#### Expected outcome

{func}`~quantify_scheduler.schedules.spectroscopy_schedules.heterodyne_spec_sched`

### Bining and averaging - acquisition staircase

#### Description

#### Expected outcome

One would expect to see a monotonic increase in the measured amplitude.
The actual amplitudes would probably not match the input amplitudes 1-to-1 because there is likely some loss on the signal path from the up- and down-conversion.
Additionally, depending on the overlap between the pulse and the integration window, the average measured voltage will be slightly lower, and the phase can be slightly different resulting in not all signal being in the I-quadrature.

{func}`~quantify_scheduler.schedules.verification.acquisition_staircase_sched`

```{code-cell} ipython3

from quantify_scheduler.schedules.verification import acquisition_staircase_sched

acq_channel = 0
schedule = acquisition_staircase_sched(
    readout_pulse_amps=np.linspace(0, 1, 4),
    readout_pulse_duration=1e-6,
    readout_frequency=6e9,
    acquisition_delay=100e-9,
    integration_time=2e-6,
    port="q0:res",
    clock="q0.ro",
    repetitions=1024,
    acq_channel=acq_channel,
)


comp_sched = qcompile(
    schedule, device_cfg=transmon_device_cfg, hardware_cfg=zhinst_hardware_cfg
)
```
