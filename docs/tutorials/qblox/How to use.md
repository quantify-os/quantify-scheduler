---
file_format: mystnb
kernelspec:
    name: python3

---
(sec-qblox-how-to-configure)=

# Usage of the backend

Configuring the backend is done by specifying a python dictionary (or loading it from a JSON file)
that describes your experimental setup. An example of such a config is shown below.

Here the entry {code}`"backend": "quantify_scheduler.backends.qblox_backend.hardware_compile"` specifies to the scheduler
that we are using the Qblox backend (specifically the {func}`~quantify_scheduler.backends.qblox_backend.hardware_compile` function).

The general structure of this hardware compilation configuration file is explained in the {ref}`sec-hardware-compilation-config` section of the User guide.
In the other sections, we will look at the specific instrument configurations in more detail.

```{code-cell} ipython3
---
mystnb:
  remove_code_source: true
---

import json
import os, inspect
from pathlib import Path
import quantify_scheduler.schemas.examples as es

esp = inspect.getfile(es)

cfg_f = Path(esp).parent / 'qblox_hardware_compilation_config.json'

with open(cfg_f, 'r') as f:
  qblox_hardware_compilation_config = json.load(f)

print(json.dumps(qblox_hardware_compilation_config, indent=4, sort_keys=False))  # Do not sort to retain the order as in the file
```
