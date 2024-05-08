# Recent interface changes

## 0.20.1: DistortionCorrection

- `DistortionCorrection` is now deprecated in favor of `SoftwareDistortionCorrection`. 

## 0.18.0: HardwareCompilationConfig and DeviceCompilationConfig

- `DeviceCompilationConfig`: The `backend` field was replaced by the `config_type` field.
  - If you are loading a stored `DeviceCompilationConfig` (instead of relying on the device config generation of the `QuantumDevice`), remove the `"backend"` key.
- `HardwareCompilationConfig`: The `backend` field was removed.
  - Only if you are already using the `HardwareCompilationConfig`, remove the `"backend"` key and add `"config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig"`.


## 0.15.0: HardwareCompilationConfig structure

The {class}`~.backends.types.common.HardwareCompilationConfig` datastructure was introduced to structure the information that was previously stored in the hardware configuration dict.
The information is now divided into sub-datastructures for the `"hardware_description"`, `"hardware_options"` and `"connectivity"`.
For more information, see {ref}`sec-hardware-compilation-config` and {ref}`Cluster (HardwareCompilationConfig) <sec-qblox-cluster-new>`.

## 0.13.0: Long waveform support

The {code}`instruction_generated_pulses_enabled` option is deprecated and will be removed in a future version. Long square pulses, staircase pulses and long ramps can be generated with the newly introduced helper functions {class}`~quantify_scheduler.backends.qblox.operations.pulse_factories.long_square_pulse`, {class}`~quantify_scheduler.backends.qblox.operations.pulse_factories.staircase_pulse` and {class}`~quantify_scheduler.backends.qblox.operations.pulse_factories.long_ramp_pulse`. More complex long waveforms can now also be created from the {class}`~quantify_scheduler.backends.qblox.operations.stitched_pulse.StitchedPulseBuilder`, which generates a {class}`~quantify_scheduler.backends.qblox.operations.stitched_pulse.StitchedPulse`. For more information, see {ref}`Long waveform support <sec-long-waveforms-via-stitchedpulse>`.

## 0.12.0: Marker behavior for RF modules; Custom Qblox downconverter

1. For the Qblox QCM-RF and QRM-RF modules, the marker override QCoDeS parameter `marker_ovr_en`
   is now always set to `False` before the execution of a schedule. This way, the markers
   behave as expected, even if they were previously overridden.
   Please refer to the
   [qblox-instruments documentation](https://qblox-qblox-instruments.readthedocs-hosted.com/en/main/cluster/qrm_rf.html#marker-output-channels) for more information about the `marker_ovr_en` and `marker_ovr_value` parameters.

2. For deactivating the custom Qblox downconverter, set `downconverter_freq` to `null` (json) or `None` (instead of `0` before).
    ```{note}
    Using `downconverter_freq` requires custom Qblox hardware, do not use otherwise.
    ``` 

## 0.11.1: Input attenuation parameter both for input and output

Ability to set `input_att` parameter for both `complex_input_0` and `complex_output_0` on QRM-RF.
To make sure that you can use the same sequencer for both input and output
on QRM-RF with input modulation and input attenuation, you can set `input_att` on either input or output,
but not on both in the same module at the same time.

## 0.11.0: Gain and attenuation parameters

Ability to set input/output gain/attenuation parameters via the Qblox hardware configuration file.

```{note}
`input_gain` introduced in version 0.9.0 is now renamed to `input_gain_<n>`.
```

The following parameters are available per device type:

- QRM (baseband)

  - `input_gain_I` (for `complex_output_0`) or `input_gain_0` (for `real_output_0`)
  - `input_gain_Q` (for `complex_output_0`) or `input_gain_1` (for `real_output_1`)

- QRM-RF

  - `output_att` (for `complex_output_0`)
  - `input_att` (for `complex_input_0`)

- QCM-RF

  - `output_att` (for `complex_output_0`)
  - `output_att` (for `complex_output_1`)

See `quantify_scheduler/schemas/examples/qblox_test_mapping.json` for concrete examples (see {ref}`sec-qblox-how-to-configure`).

## 0.8.0: Dynamic Sequencer Allocation

The Qblox backend now dynamically allocates sequencers to port-clock combinations as required.
This enables the user to target an arbitrary number of port-clock combinations using the same module, as long as they are not being *simultaneously* targeted.
Given that Qblox instruments have 6 sequencers, up to 6 port-clock combinations may be simultaneously targeted.

This change has introduced a new syntax for the hardware configuration file:

1. We now specify a list of `portclock_configs` instead of sequencers.

2. Furthermore, we moved latency correction to top-level key `latency_corrections`

3. We provide a helper function that may be used to convert from old to new syntax:

    ```{warning}
    This helper function has been removed in quantify-scheduler 0.18.0.
    ```

   - `quantify_scheduler.backends.qblox.helpers.convert_hw_config_to_portclock_configs_spec`
   - Temporarily, this method is called by the Qblox backend before compilation.

The old syntax:

```python
hardware_cfg = {
    "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
    "cluster0": {
        "instrument_type": "Cluster",
        "ref": "internal",
        "cluster0_module1: {
            "instrument_type": "QCM",
            "complex_output_0": {
                "lo_name": "lo0",
                "seq0": {
                    "port": "q0:mw",
                    "clock": "q0.01",
                    "interm_freq": 50e6,
                    "latency_correction" : 4e-9
                },
                "seq1": {
                    "port": "q1:mw",
                    "clock": "q1.01",
                    "interm_freq": 100e6
                }
            },
            "complex_output_1": {
                "lo_name": "lo1",
                "seq2": {
                    "port": "q2:mw",
                    "clock": "q2.01",
                    "interm_freq": None
                }
            },
        },
    "lo0": {"instrument_type": "LocalOscillator", "frequency": None, "power": 20},
    "lo1": {"instrument_type": "LocalOscillator", "frequency": 7.2e9, "power": 20}
}
```

The new syntax:

```python
hardware_cfg = {
    "backend": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
    "latency_corrections": {
        "q0:mw-q0.01": 4e-9
    },
    "cluster0": {
        "instrument_type": "Cluster",
        "ref": "internal",
        "cluster0_module1: {
            "complex_output_0": {
                "instrument_type": "QCM",
                "lo_name": "lo0",
                "portclock_configs": [
                    {
                        "port": "q0:mw",
                        "clock": "q0.01",
                        "interm_freq": 50e6
                    },
                    {
                        "port": "q1:mw",
                        "clock": "q1.01",
                        "interm_freq": 100e6
                    }
                ]
            },
            "complex_output_1": {
                "lo_name": "lo1",
                "portclock_configs": [
                    {
                        "port": "q2:mw",
                        "clock": "q2.01",
                        "interm_freq": None
                    }
                ]
            }
        },
    "lo0": {"instrument_type": "LocalOscillator", "frequency": None, "power": 20},
    "lo1": {"instrument_type": "LocalOscillator", "frequency": 7.2e9, "power": 20}
}
```
