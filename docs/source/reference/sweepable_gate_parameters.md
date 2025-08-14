---
file_format: mystnb
kernelspec:
    name: python3

---
(sec-qblox-gate-sweep-param-details)=

# Sweepable Gate Parameters

Quantify Scheduler offers the option to sweep gate parameters (amplitude, frequency, etc) within the schedule. This is an alternative to changing/sweeping such parameters at the pulse level.

For example, please have a look at the Measure gate parameters in an experiment where we sweep both frequency and the amplitude in the readout line:

```python
import numpy as np
from quantify_scheduler.operations import Measure, IdlePulse
from quantify_scheduler import Schedule


def resonator_punchout_schedule(
    qubit,  # noqa: ANN001
    freqs: np.array,
    ro_pulse_amps: np.array,
    repetitions: int = 1,
) -> Schedule:
    """Schedule to sweep the resonator frequency."""
    sched = Schedule("schedule", repetitions=repetitions)
    index = 0
    freqs, ro_pulse_amps = np.unique(freqs), np.unique(ro_pulse_amps)
    for freq in freqs:
        for amp in ro_pulse_amps:
            sched.add(Measure(qubit.name, acq_index=index, freq=freq, pulse_amp=amp))
            sched.add(IdlePulse(8e-9))
            index += 1
    return sched
```
This functionality is available for all gates outlined in `quantify_scheduler.operations`.

To know the list of parameters that can be swept for each gate, you can run the following 

```python

from quantify_scheduler import BasicTransmonElement

def sweepable_params(element):
    dev_cfg = element.generate_device_config()          # public method
    elem_cfg = dev_cfg.elements[element.name]           # dict[str, OperationCompilationConfig]
    out = {}
    for gate, op in elem_cfg.items():
        factory_keys = list((op.factory_kwargs or {}).keys())
        gate_info_keys = list(op.gate_info_factory_kwargs or [])
        out[gate] = {
            "factory_kwargs": factory_keys,
            "gate_info_kwargs": gate_info_keys,
        }
    return out

# Example
q0 = BasicTransmonElement("q0")
params = sweepable_params(q0)
for gate, d in params.items():
    print(f"{gate}:")
    print("  Sweepable Parameters :", d["factory_kwargs"])
```
This gives you the sweepable parameters for each gate for a transmon element for each gate. 

:::{warning}
Passing parameters to a gate that are not obtained from the list above will not cause a compilation error, but they will have no effect on the experiment. Please ensure you only use the parameters listed above when using this functionality.
:::