---
file_format: mystnb
kernelspec:
    name: python3

---
(sec-qblox-hw-cfg-versioning)=

# Hardware config versioning

Qblox hardware compilation configs have a "version" field that indicates the specific structure and fields the configs have. The versions are classified using "x.y" tags, where "x" and "y" indicate major and minor version changes.

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