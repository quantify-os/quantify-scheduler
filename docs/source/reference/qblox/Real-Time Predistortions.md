---
file_format: mystnb
kernelspec:
    name: python3

---
(sec-qblox-rtp-details)=

# Hardware Distortion Corrections

If you're using Qblox hardware, you can take advantage of the real-time hardware predistortion features that apply predistortion to your waveform. This functionality is available in the QCM module of the Qblox Cluster.

For more information, please visit the [Qblox documentation on real-time predistortion](https://docs.qblox.com/en/main/cluster/real_time_predistortions.html).

## Interface

The distortion correction coefficients can be specified using the `distortion_corrections` key inside the `hardware_options` section of the `hardware_config`.

For example:

```python
from quantify_scheduler.backends.types.qblox import QbloxHardwareDistortionCorrection

hardware_config = {
    "hardware_options": {
        "distortion_corrections": {
            "q0:fl-cl0.baseband": QbloxHardwareDistortionCorrection(
                exp1_coeffs=[2000, -0.1],
                fir_coeffs=[1.025] + [0.03, 0.02] * 15 + [0],
            )
        }
    }
}

```
## Delay Compensation

In order to compensate the delay caused by the filters, on other output channels that do not have filters enabled, you can set the `distortion_correction_latency_compensation` flag inside the hardware configuration.

Example configuration snippet:

```python
from quantify_scheduler.backends.qblox.enums import DistortionCorrectionLatencyEnum

hardware_compilation_cfg ={
    "hardware_description": {
    "cluster0": {
        "instrument_type": "Cluster",
        "ref": "internal",
        "modules": {
            "1": {
                "instrument_type": "QCM",
                "complex_output_0": {
                    "distortion_correction_latency_compensation": (
                        DistortionCorrectionLatencyEnum.EXP0
                        | DistortionCorrectionLatencyEnum.EXP1
                        | DistortionCorrectionLatencyEnum.EXP3
                    )
                }
            }
        }
    }
}
}
``` 

Each `DistortionCorrectionLatencyEnum` value specifies which type of correction delay to compensate for; see possible values: {class}`~quantify_scheduler.backends.qblox.enums.DistortionCorrectionLatencyEnum`

You can combine multiple flags using the bitwise OR (|) operator to enable compensation for multiple filters