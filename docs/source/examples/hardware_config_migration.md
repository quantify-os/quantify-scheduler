---
file_format: mystnb
kernelspec:
    name: python3

---
(sec-hardware-config-migration)=
# `HardwareCompilationConfig` migration guide

The {class}`~.backends.types.common.HardwareCompilationConfig` replaces the old-style unvalidated json/dict hardware configuration, adding the validation of the contents and restructuring into `"hardware_description"`, `"hardware_options"` and `"connectivity"`. These parts are described in detail in the {ref}`sec-hardware-compilation-config` section in the User Guide. Here, we describe how to migrate from the old-style configuration to the new one for the current hardware backends, which each define their own `HardwareCompilationConfig` datastructure with backend-specific fields  (e.g., {class}`~.backends.qblox_backend.QbloxHardwareCompilationConfig` and {class}`~.backends.zhinst_backend.ZIHardwareCompilationConfig`).

```{admonition} Custom backends
If you have implemented a custom backend, or have added custom instruments and/or options to the Qblox or Zurich Instruments backends, you will need to define your own {class}`~.backends.types.common.HardwareCompilationConfig` datastructure, as is already done for the Qblox and Zurich Instruments backends.
```

## Qblox
The {class}`~.backends.qblox_backend.QbloxHardwareCompilationConfig` inherits from the {class}`~.backends.types.common.HardwareCompilationConfig` and contains the following backend-specific fields:

1. The {obj}`~.backends.types.qblox.QbloxHardwareDescription`, which specifies the instruments that are used in the setup, along with their instrument-specific settings. For the Qblox Cluster, we refer to the {ref}`sec-cluster-hardware-description` section for the allowed instrument types and settings.
2. The {class}`~.backends.types.qblox.QbloxHardwareOptions`, which adds some backend-specific options to the generic {class}`~.backends.types.common.HardwareOptions` (see {ref}`sec-cluster-hardware-options` for more details).
3. The `compilation_passes` field, which specifies the compilation passes that are used to compile the schedule. In the example below, we do not specify the compilation passes, which means that the default passes are used. 

```{admonition} Under construction
The new-style {class}`~.backends.qblox_backend.QbloxHardwareCompilationConfig` is currently still under construction and subject to change.
```

To show how to migrate, we first load an example old-style hardware configuration for the Qblox backend:

```{code-cell} ipython3
---
tags: [hide-output]
---
import rich
from quantify_scheduler.backends.qblox.qblox_hardware_config_old_style import hardware_config as qblox_hardware_config_old_style

rich.print(qblox_hardware_config_old_style)
```

This config can be migrated to the new {class}`~.backends.qblox_backend.QbloxHardwareCompilationConfig` datastructure using the built-in validation, which will recognize the old-style config and convert it to the new-style config:

```{code-cell} ipython3
---
tags: [hide-output]
---
from quantify_scheduler.backends.qblox_backend import QbloxHardwareCompilationConfig

qblox_hardware_config_new_style = QbloxHardwareCompilationConfig.model_validate(
    qblox_hardware_config_old_style
)
rich.print(qblox_hardware_config_new_style)
```

This new-style config can then be passed to the {class}`~.device_under_test.quantum_device.QuantumDevice` in order to compile a schedule for the Qblox backend, as is shown in the {ref}`sec-tutorial-compiling-to-hardware-compilation` section of the {ref}`sec-tutorial-compiling`.

## Zurich Instruments
The {class}`~.backends.zhinst_backend.ZIHardwareCompilationConfig` inherits from the {class}`~.backends.types.common.HardwareCompilationConfig` and contains the following backend-specific fields:

1. The {obj}`~.backends.types.qblox.QbloxHardwareDescription`, which specifies the instruments that are used in the setup, along with their instrument-specific settings. We refer to the Zurich Instruments {ref}`sec-zhinst-hardware-description` section for the allowed instrument types and settings.
2. The {class}`~.backends.types.zhinst.ZIHardwareOptions`, which adds backend-specific options to the generic {class}`~.backends.types.common.HardwareOptions`, see {ref}`sec-zhinst-hardware-options` for more details.
3. The `compilation_passes` field, which specifies the compilation passes that are used to compile the schedule. In the example below, we do not specify the compilation passes, which means that the default passes are used.

To show how to migrate, we first load an example old-style hardware configuration for the Zurich Instruments backend:

```{code-cell} ipython3
---
tags: [hide-output]
---
import rich
from quantify_scheduler.backends.zhinst.zhinst_hardware_config_old_style import hardware_config as zhinst_hardware_config_old_style

rich.print(zhinst_hardware_config_old_style)
```

This config can be migrated to the new {class}`~.backends.zhinst_backend.ZIHardwareCompilationConfig` datastructure using the built-in validation, which will recognize the old-style config and convert it to the new-style config:

```{code-cell} ipython3
---
tags: [hide-output]
---
from quantify_scheduler.backends.zhinst_backend import ZIHardwareCompilationConfig

zhinst_hardware_config_new_style = ZIHardwareCompilationConfig.model_validate(
    zhinst_hardware_config_old_style
)
rich.print(zhinst_hardware_config_new_style)
```

This new-style config can then be passed to the {class}`~.device_under_test.quantum_device.QuantumDevice` in order to compile a schedule for the Zurich Instruments backend, as is shown in the {ref}`sec-tutorial-compiling-to-hardware-compilation` section of the {ref}`sec-tutorial-compiling`.