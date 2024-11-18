---
file_format: mystnb
kernelspec:
    name: python3

---
```{seealso}
This notebook can be downloaded {nb-download}`here <serialization.ipynb>`
```

# Serialization

Quantify allows for serialization of :class:`~quantify_scheduler.QuantumDevice`, :class:`~quantify_scheduler.DeviceElement` (e.g. :class:`~quantify_scheduler.BasicTransmonElement`) and :class:`~quantify_scheduler.Schedule` objects to json strings and json files. Each class has the following methods:

- `to_json` : Converts the object into a json string.
- `from_json` : Converts a json string back into the object.
- `to_json_file` : Stores the json string to a file.
- `from_json_file` : Reads the json string from a file and converts it back into the object.

## Examples

### (De)Serializing a `QuantumDevice` object to json string

```{code-cell} ipython3
from quantify_scheduler import BasicTransmonElement, QuantumDevice
import json

QuantumDevice.close_all()
device = QuantumDevice("single_qubit_device")
q0 = BasicTransmonElement("q0")
device.cfg_sched_repetitions(512)
device.add_element(q0)
...

device_json = device.to_json()
print(json.dumps(json.loads(device_json), indent=4))

```

Loading the object from the json string is done using the `from_json` method:

```{code-cell} ipython3
:tags: [raises-exception, remove-output]
deserialized_device = QuantumDevice.from_json(device_json)
```

### (De)Serializing a `QuantumDevice` object to json file

You can optionally specify the path as an argument to the `to_json_file` method.

```{code-cell} ipython3
device.to_json_file("/tmp")
```

or save it automatically to the current data directory using the objects name:

```{code-cell} ipython3
from quantify_core.data.handling import set_datadir
set_datadir("/tmp")
device.to_json_file() # Saves to "/tmp/single_qubit_device_2024-11-14_13-36-59_UTC.json"
```

and the timestamp can be omitted by setting `add_timestamp=False`:

```{code-cell} ipython3
device.to_json_file(add_timestamp=False) # Saves to "/tmp/single_qubit_device.json"
```

loading the object from the json file is done using the `from_json_file` method:

```{code-cell} ipython3
:tags: [raises-exception, remove-output]
deserialized_device = QuantumDevice.from_json_file("/tmp/single_qubit_device.json")
```
