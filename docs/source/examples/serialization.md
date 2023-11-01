---
file_format: mystnb
kernelspec:
    name: python3

---
```{seealso}
This notebook can be downloaded {nb-download}`here <serialization.ipynb>`
```

# Serialization

Quantify allows for serialization of `QuantumDevice` and `Schedule` objects to json strings. Serializing to json string is done using the method `to_json`, and deserializing using `from_json`. It is also possible to directly store in / read from a file using the methods `to_json_file` and `from_json_file`.

The code block below shows the serialization of a `QuantumDevice` object into a json string:

```{code-cell} ipython3
from quantify_scheduler.device_under_test.composite_square_edge import CompositeSquareEdge
from quantify_scheduler.device_under_test.quantum_device import QuantumDevice
from quantify_scheduler.device_under_test.transmon_element import BasicTransmonElement

q0 = BasicTransmonElement("q0")
q1 = BasicTransmonElement("q1")
edge_q0_q1 = CompositeSquareEdge(parent_element_name=q0.name, child_element_name=q1.name)

quantum_device = QuantumDevice("quantum_device")
quantum_device.add_element(q0)
quantum_device.add_element(q1)
quantum_device.add_edge(edge_q0_q1)

quantum_device.cfg_sched_repetitions(512)

# Serialize QuantumDevice into a json string
serialized_quantum_device = quantum_device.to_json()
# Close before deserialization to avoid duplicated instrument error: 
# this closes *any* open instrument
QuantumDevice.close_all()

print(f"Class: {serialized_quantum_device.__class__}")
```

The previous `QuantumDevice` object can now be instantiated again from the json string:

```{code-cell} ipython3
deserialized_quantum_device = QuantumDevice.from_json(serialized_quantum_device)

print(f"Class: {deserialized_quantum_device.__class__}")
print(f"Elements: {deserialized_quantum_device.elements()}")
print(f"Edges: {deserialized_quantum_device.edges()}")
print(f"Repetitions: {deserialized_quantum_device.cfg_sched_repetitions()}")
```

```{admonition} Note
Inherited attributes from the QCoDeS `Instrument` class are not included when serializing `QuantumDevice`.
```
