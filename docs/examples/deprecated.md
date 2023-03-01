---
file_format: mystnb

---

# Deprecated Code Suggestions

```{seealso}
Download the notebook: {nb-download}`deprecated.ipynb`
```

- {ref}`1. Qcompile => SerialCompiler`
- {ref}`2. Qblox Hardware Configuration`
- {ref}`3. TransmonElement => BasicTransmonElement`

As of `quantify-scheduler==0.10.0`, deprecation warnings are shown by default (as `FutureWarning`).

## Compilation Setup

```{code-cell} ipython3
from quantify_core.data import handling as dh
from quantify_core.measurement.control import MeasurementControl
from quantify_scheduler.instrument_coordinator import InstrumentCoordinator
from quantify_scheduler.instrument_coordinator.components.qblox import ClusterComponent

from qblox_instruments import Cluster, ClusterType
from qcodes import Instrument

dh.set_datadir(dh.default_datadir())

Instrument.close_all()
meas_ctrl = MeasurementControl("meas_ctrl")
ic = InstrumentCoordinator("ic")

cluster = Cluster(
    "cluster",
    dummy_cfg={
        1: ClusterType.CLUSTER_QRM_RF,
    },
)

ic_cluster = ClusterComponent(cluster)
ic.add_component(ic_cluster)

# Always picks the first module of a certain type, and ignores the others of same type!
qcm_rf, qrm_rf, qcm, qrm = [None] * 4
for module in cluster.modules:
    try:
        if module.is_rf_type:
            if module.is_qcm_type:
                if qcm_rf is None:
                    qcm_rf = module
            else:
                if qrm_rf is None:
                    qrm_rf = module
        else:
            if module.is_qcm_type:
                if qcm is None:
                    qcm = module
            else:
                if qrm is None:
                    qrm = module
    except KeyError:
        continue

print(f"qcm    => {qcm}\nqrm    => {qrm}\nqcm_rf => {qcm_rf}\nqrm_rf => {qrm_rf}")
```

```{code-cell} ipython3
from quantify_scheduler.device_under_test.quantum_device import QuantumDevice
from quantify_scheduler.device_under_test.transmon_element import BasicTransmonElement

q0 = BasicTransmonElement("q0")

quantum_device = QuantumDevice("quantum_device")
quantum_device.add_element(q0)
quantum_device.instr_measurement_control(meas_ctrl.name)
quantum_device.instr_instrument_coordinator(ic.name)

q0.clock_freqs.f01(7.3e9)
q0.clock_freqs.f12(7.0e9)
q0.clock_freqs.readout(8.2e9)
q0.measure.acq_delay(100e-9)
q0.measure.acq_channel(0)
q0.measure.pulse_amp(0.2)

device_cfg = quantum_device.generate_device_config()
```

```{code-cell} ipython3
hardware_cfg = {
    "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
    "cluster": {
        "ref": "internal",
        "instrument_type": "Cluster",
        f"cluster_module{qrm_rf.slot_idx}": {
            "instrument_type": "QRM_RF",
            "complex_output_0": {
                "lo_freq": 2e9,
                "portclock_configs": [
                    {
                        "port": "q0:res",
                        "clock": "q0.ro",
                    },
                ],
            },
        },
    },
}
```

```{code-cell} ipython3
from quantify_scheduler import Schedule
from quantify_scheduler.operations.gate_library import Measure, Reset
from quantify_scheduler.operations.pulse_library import DRAGPulse
from quantify_scheduler.resources import ClockResource


def simple_trace_sched(
    repetitions: int,
    pulse_amp: float = 0.2, 
) -> Schedule:
    sched = Schedule("Simple trace schedule", repetitions)

    port = "q0:res"
    clock = "q0.ro"

    sched.add(Reset("q0"))
    sched.add(Measure("q0", acq_index=0, acq_protocol="Trace"))
    sched.add(
        DRAGPulse(
            G_amp=pulse_amp,
            D_amp=0,
            phase=0,
            duration=160e-9,
            port=port,
            clock=clock,
        )
    )

    return sched


sched = simple_trace_sched(repetitions=1)
```

## 1. Qcompile => SerialCompiler

First, run {ref}`Compilation Setup`.

```{code-cell} ipython3
from quantify_scheduler.compilation import qcompile

compiled_schedule = qcompile(sched, device_cfg, hardware_cfg)
```

```{code-cell} ipython3
from quantify_scheduler.backends.graph_compilation import SerialCompiler

quantum_device.hardware_config(hardware_cfg)

compiler = SerialCompiler(name="compiler")
compiled_schedule = compiler.compile(
    schedule=sched, config=quantum_device.generate_compilation_config()
)
```

```{code-cell} ipython3
compiled_schedule.timing_table
```

## 2. Qblox Hardware Configuration
1. `seqx` => `portclock_configs`  
1. `latency_correction` => standalone/top-level `latency_corrections`
1. `line_gain_db` removed


```{code-cell} ipython3
depr_hardware_cfg = {
    "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
    "cluster": {
        "ref": "internal",
        "instrument_type": "Cluster",
        "cluster_module1": {
            "instrument_type": "QRM_RF",
            "complex_output_0": {
                "line_gain_db": 0,
                "seq0": {
                    "port": "q6:res",
                    "clock": "q6.ro",
                    "latency_correction": 4e-9,
                },
                "seq1": {
                    "port": "q1:res",
                    "clock": "q1.ro",
                },
            },
        },
    },
}
```

```{code-cell} ipython3
from quantify_scheduler.backends.qblox.helpers import (
    convert_hw_config_to_portclock_configs_spec,
)

new_hardware_cfg = convert_hw_config_to_portclock_configs_spec(depr_hardware_cfg)


fnc = lambda sub: {
    key1: fnc(val1) if isinstance(val1, dict) else val1
    for key1, val1 in sub.items()
    if key1 != "line_gain_db"
}

new_hardware_cfg = fnc(new_hardware_cfg)
```

```{code-cell} ipython3
import json

print(json.dumps(new_hardware_cfg, indent=4))
```

## 3. TransmonElement => BasicTransmonElement

```{code-cell} ipython3
from qcodes import Instrument

Instrument.close_all()
```

```{code-cell} ipython3
from quantify_scheduler.device_under_test.transmon_element import (
    BasicTransmonElement,
    TransmonElement,
)

transmon = TransmonElement("transmon")
print(f"{transmon.name}: {list(transmon.parameters.keys())}")
print()

basic = BasicTransmonElement("basic")
print(f"{basic.name}: {list(basic.parameters.keys()) + list(basic.submodules.keys())}")
for submodule_name, submodule in basic.submodules.items():
    print(f"{basic.name}.{submodule_name}: {list(submodule.parameters.keys())}")
```

```{code-cell} ipython3
spec_str = f'via:\tschedule.add(SpectroscopyOperation("{basic.name}")), not implemented for BasicTransmonElement, see BasicElectronicNVElement.spectroscopy_operation'

convert = {
    ".IDN": ".IDN",
    ".init_duration": ".reset.duration",
    ".mw_amp180": ".rxy.amp180",
    ".mw_motzoi": ".rxy.motzoi",
    ".mw_pulse_duration": ".rxy.duration",
    ".mw_ef_amp180": None,
    ".mw_port": ".ports.microwave",
    ".fl_port": ".ports.flux",
    ".ro_port": ".ports.readout",
    ".mw_01_clock": f'no longer settable, always "{basic.name}.01"',
    ".mw_12_clock": f'no longer settable, always "{basic.name}.12"',
    ".ro_clock": f'no longer settable, always "{basic.name}.ro"',
    ".freq_01": ".clock_freqs.f01",
    ".freq_12": ".clock_freqs.f12",
    ".ro_freq": ".clock_freqs.readout",
    ".ro_pulse_amp": ".measure.pulse_amp",
    ".ro_pulse_duration": ".measure.pulse_duration",
    ".ro_pulse_type": ".measure.pulse_type",
    ".ro_pulse_delay": "via:\tschedule.add(..., rel_time=...)",
    ".ro_acq_channel": ".measure.acq_channel",
    f'schedule.add(Measure("{transmon.name}", acq_channel=...))': ".measure.acq_channel",
    ".ro_acq_delay": ".measure.acq_delay",
    ".ro_acq_integration_time": ".measure.integration_time",
    ".spec_pulse_duration": spec_str,
    ".spec_pulse_frequency": spec_str,
    ".spec_pulse_amp": spec_str,
    ".spec_pulse_clock": spec_str,
    ".acquisition": f'via:\tschedule.add(Measure("{basic.name}", acq_protocol=...))',
    ".ro_acq_weight_type": ".measure.acq_weight_type",
}
```

```{code-cell} ipython3
transmon_params = [f".{param}" for param in transmon.parameters]

for transmon_param in transmon_params + list(convert.keys() - transmon_params):
    basic_param = str(convert.get(transmon_param, None))
    print(
        f"{transmon.name if transmon_param.startswith('.') else ''}{transmon_param:<42}   =>    {basic.name if basic_param.startswith('.') else ''}{basic_param}"
    )
```

```{code-cell} ipython3
import pprint

device_config_transmon = transmon.generate_device_config().dict()
pprint.pprint(device_config_transmon)
```

```{code-cell} ipython3
import pprint

device_config_basic_transmon = basic.generate_device_config().dict()
pprint.pprint(device_config_basic_transmon)
```

