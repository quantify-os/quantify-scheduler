---
file_format: mystnb
kernelspec:
    name: python3

---
(sec-qblox-hw-cfg-versioning)=

# Hardware config versioning

Qblox hardware compilation configs have a "version" field that indicates the specific structure and fields the configs have. The versions are classified using "x.y" tags, where "x" and "y" indicate major and minor version changes.

## Version 0.2

In v0.2 we changed slightly the definitions for complex and real channels. Before an output channels would also activate input ports. In v0.2 output channels are used for output only, and input channels for input only. If you want to play and acquire the same qubit, you will have to explicitly add both channels to the same port in the connectivity (see example below). Allowed combinations are:

- QRM
  - `complex_output_0` + `complex_input_0`
  - `real_output_{i}` + `real_input_{j}`

- QRM_RF:
  - `complex_output_0` + `complex_input_0`

```python
{
    "version": "0.2",
    "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
    "hardware_description": {
        "cluster0": {
            "instrument_type": "Cluster",
            "ref": "internal",
            "ip": None,
            "modules": {
                "1": {"instrument_type": "QRM"},
                "2": {"instrument_type": "QRM_RF"},
                },
            }
        },
    },
    "hardware_options": {},
    "connectivity": {
        "graph": [
            ["cluster0.module1.real_output_0", "q0:res"],
            ["cluster0.module1.real_input_1", "q0:res"],
            ["cluster0.module2.complex_output_0", "q1:res"],
            ["cluster0.module2.complex_input_0", "q1:res"],
        ]
    }
}
```


## Version 0.1

First version of the hardware compilation config, built on a pydantic basemodel.

```python
{
    "version": "0.1",
    "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
    "hardware_description": {
        "cluster0": {
            "instrument_type": "Cluster",
            "ref": "internal",
            "ip": None,
            "modules": {"1": {"instrument_type": "QCM"}},
            }
        },
    },
    "hardware_options": {},
    "connectivity": {
        "graph": [["cluster0.module1.complex_output_0", "q0:mw"]]
    }
}
```