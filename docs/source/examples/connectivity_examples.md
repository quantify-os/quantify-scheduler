---
file_format: mystnb
kernelspec:
    name: python3

---
```{seealso}
This notebook can be downloaded {nb-download}`here <connectivity_examples.ipynb>`
```
(sec-connectivity-examples)=
# Connectivity examples
As described in the {ref}`sec-connectivity` in the User guide, the {class}`~.backends.types.common.Connectivity` datastructure indicates how ports on the quantum device are connected to the control hardware. Here we show examples of the different ways of specifying connections in the connectivity graph.

## One-to-many, many-to-one, and many-to-many connections

```{code-cell} ipython3
from quantify_scheduler.backends.types.common import Connectivity

connectivity = Connectivity.model_validate(
    {"graph":
        [
            # One-to-one
            ("instrument_0.port0", "q0:a"),
            # One-to-many
            ("instrument_0.port1", ["q0:b", "q0:c", "q0:d", "q0:e", "q0:f"]),
            # Many-to-one
            (["instrument_1.port0", "instrument_1.port1", "instrument_1.port2"], "q1:a"),
            # Many-to-many
            (["instrument_2.port0", "instrument_2.port1"], ["q2:a", "q2:b"]),
        ]
    }
)
connectivity.draw()
```
