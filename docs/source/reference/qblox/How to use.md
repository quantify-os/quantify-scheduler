---
file_format: mystnb
kernelspec:
    name: python3

---
(sec-qblox-how-to-configure)=

# Usage of the backend

Configuring the backend is done by specifying a Python dictionary (or loading it from a JSON file)
that describes your experimental setup. An example of such a hardware configuration is shown below.

Two variants of the same hardware configuration are shown: 
- The current old-style unvalidated json/dict hardware configuration (see {ref}`sec-qblox-cluster`).
- The new {class}`~.backends.types.common.HardwareCompilationConfig` variant that adds validation of the contents and divides into `"hardware_description"`, `"hardware_options"` and `"connectivity"` (see {ref}`Cluster (HardwareCompilationConfig) <sec-qblox-cluster-new>`,  and the {ref}`sec-hardware-compilation-config` section of the User guide). 

```{code-cell} ipython3
---
tags: [hide-cell]
mystnb:
  code_prompt_show: "Example hardware configuration"
  remove_code_source: true  
---

import json

from quantify_scheduler.backends.qblox.hardware_config_transmon_old_style import hardware_config as hardware_config_transmon_old_style

print(json.dumps(hardware_config_transmon_old_style, indent=4, sort_keys=False))
```


```{code-cell} ipython3
---
tags: [hide-cell]
mystnb:
  code_prompt_show: "Example hardware configuration (HardwareCompilationConfig)"
  remove_code_source: true  
---

import json
import inspect
from pathlib import Path
import quantify_scheduler.schemas.examples as es

esp = inspect.getfile(es)
cfg_f = Path(esp).parent / "qblox_hardware_config_transmon.json"

with open(cfg_f, "r") as f:
  qblox_hardware_config_transmon = json.load(f)

print(json.dumps(qblox_hardware_config_transmon, indent=4, sort_keys=False))  # Do not sort to retain the order as in the file
```