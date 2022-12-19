---
file_format: mystnb
kernelspec:
    name: python3
---

# Quantify Deprecated Code Suggestions

```{seealso}
Download the notebook: {nb-download}`deprecated.ipynb`
```

- {ref}`1. Qcompile => SerialCompiler`
- {ref}`2. Qblox Hardware Configuration`
- {ref}`3. TransmonElement => BasicTransmonElement`

As of `quantify-scheduler==0.10.0`, deprecation warnings are shown by default (as `FutureWarning`).

### Compilation Setup

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
    clock_freq: float = 1.8e9,  # Below 2e9 to be able to visualize on oscilliscope
) -> Schedule:
    sched = Schedule("Simple trace schedule", repetitions)

    port = "q0:res"
    clock = "q0.ro"

    sched.add_resources([ClockResource(clock, clock_freq)])

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
depr_hardware_cfg = {
    "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
    "cluster1": {
        "ref": "internal",
        "instrument_type": "Cluster",
        "cluster1_module16": {
            "instrument_type": "QRM_RF",
            "complex_output_0": {
                "line_gain_db": 0,
                "dc_mixer_offset_I": -8.5195e-3,
                "dc_mixer_offset_Q": 1.5106e-3,
                "lo_freq": 6_640_000_000.0,
                "seq0": {
                    "mixer_amp_ratio": 1.005,
                    "mixer_phase_error_deg": -6.3357,
                    "port": "q11:res",
                    "clock": "q11.ro",
                },
                "seq1": {
                    "mixer_amp_ratio": 1.005,
                    "mixer_phase_error_deg": -6.3357,
                    "port": "q12:res",
                    "clock": "q12.ro",
                },
                "seq2": {
                    "mixer_amp_ratio": 1.005,
                    "mixer_phase_error_deg": -6.3357,
                    "port": "q13:res",
                    "clock": "q13.ro",
                },
                "seq3": {
                    "mixer_amp_ratio": 1.005,
                    "mixer_phase_error_deg": -6.3357,
                    "port": "q14:res",
                    "clock": "q14.ro",
                },
                "seq4": {
                    "mixer_amp_ratio": 1.005,
                    "mixer_phase_error_deg": -6.3357,
                    "port": "q15:res",
                    "clock": "q15.ro",
                },
            },
        },
        "cluster1_module17": {
            "instrument_type": "QRM_RF",
            "complex_output_0": {
                "line_gain_db": 0,
                "dc_mixer_offset_I": -4.9804e-3,
                "dc_mixer_offset_Q": 1.4753e-3,
                "lo_freq": 6_770_000_000.0,
                "seq5": {
                    "mixer_amp_ratio": 1.1059,
                    "mixer_phase_error_deg": -5.5432,
                    "port": "q16:res",
                    "clock": "q16.ro",
                },
                "seq6": {
                    "mixer_amp_ratio": 1.1059,
                    "mixer_phase_error_deg": -5.5432,
                    "port": "q17:res",
                    "clock": "q17.ro",
                },
                "seq7": {
                    "mixer_amp_ratio": 1.1059,
                    "mixer_phase_error_deg": -5.5432,
                    "port": "q18:res",
                    "clock": "q18.ro",
                },
                "seq8": {
                    "mixer_amp_ratio": 1.1059,
                    "mixer_phase_error_deg": -5.5432,
                    "port": "q19:res",
                    "clock": "q19.ro",
                },
                "seq9": {
                    "mixer_amp_ratio": 1.1059,
                    "mixer_phase_error_deg": -5.5432,
                    "port": "q20:res",
                    "clock": "q20.ro",
                },
            },
        },
        "cluster1_module1": {
            "instrument_type": "QCM_RF",
            "complex_output_0": {
                "line_gain_db": 0,
                "dc_mixer_offset_I": -13.819e-3,
                "dc_mixer_offset_Q": -0.60749e-3,
                "lo_freq": 3_790_000_000.0,
                "seq0": {
                    "mixer_amp_ratio": 0.98902,
                    "mixer_phase_error_deg": -29.644,
                    "port": "q11:mw",
                    "clock": "q11.01",
                },
                "seq0_12": {
                    "mixer_amp_ratio": 0.98902,
                    "mixer_phase_error_deg": -29.644,
                    "port": "q11:mw",
                    "clock": "q11.12",
                },
            },
            "complex_output_1": {
                "line_gain_db": 0,
                "dc_mixer_offset_I": -3.6511e-3,
                "dc_mixer_offset_Q": -8.9219e-3,
                "lo_freq": 3_440_000_000.0,
                "seq1": {
                    "mixer_amp_ratio": 1.003,
                    "mixer_phase_error_deg": -25.307,
                    "port": "q12:mw",
                    "clock": "q12.01",
                },
                "seq1_12": {
                    "mixer_amp_ratio": 1.003,
                    "mixer_phase_error_deg": -25.307,
                    "port": "q12:mw",
                    "clock": "q12.12",
                },
            },
        },
        "cluster1_module2": {
            "instrument_type": "QCM_RF",
            "complex_output_0": {
                "line_gain_db": 0,
                "dc_mixer_offset_I": -13.804e-3,
                "dc_mixer_offset_Q": -3.026e-3,
                "lo_freq": 3_660_000_000.0,
                "seq0": {
                    "mixer_amp_ratio": 0.96898,
                    "mixer_phase_error_deg": -18.599,
                    "port": "q13:mw",
                    "clock": "q13.01",
                },
                "seq0_12": {
                    "mixer_amp_ratio": 0.96898,
                    "mixer_phase_error_deg": -18.599,
                    "port": "q13:mw",
                    "clock": "q13.12",
                },
            },
            "complex_output_1": {
                "line_gain_db": 0,
                "dc_mixer_offset_I": -5.388e-3,
                "dc_mixer_offset_Q": -0.1745e-3,
                "lo_freq": 3_400_000_000.0,
                "seq1": {
                    "mixer_amp_ratio": 1.001,
                    "mixer_phase_error_deg": 83.864,
                    "port": "q14:mw",
                    "clock": "q14.01",
                },
                "seq1_12": {
                    "mixer_amp_ratio": 1.001,
                    "mixer_phase_error_deg": 83.864,
                    "port": "q14:mw",
                    "clock": "q14.12",
                },
            },
        },
        "cluster1_module3": {
            "instrument_type": "QCM_RF",
            "complex_output_0": {
                "line_gain_db": 0,
                "dc_mixer_offset_I": -9.6247e-3,
                "dc_mixer_offset_Q": -9.2564e-3,
                "lo_freq": 3_950_000_000.0,
                "seq0": {
                    "mixer_amp_ratio": 0.96999,
                    "mixer_phase_error_deg": -25.042,
                    "port": "q15:mw",
                    "clock": "q15.01",
                },
                "seq0_12": {
                    "mixer_amp_ratio": 0.96999,
                    "mixer_phase_error_deg": -25.042,
                    "port": "q15:mw",
                    "clock": "q15.12",
                },
            },
            "complex_output_1": {
                "line_gain_db": 0,
                "dc_mixer_offset_I": -1.5163e-3,
                "dc_mixer_offset_Q": -7.2648e-3,
                "lo_freq": 3_260_000_000.0,
                "seq1": {
                    "mixer_amp_ratio": 1.016,
                    "mixer_phase_error_deg": -24.979,
                    "port": "q16:mw",
                    "clock": "q16.01",
                },
                "seq1_12": {
                    "mixer_amp_ratio": 1.016,
                    "mixer_phase_error_deg": -24.979,
                    "port": "q16:mw",
                    "clock": "q16.12",
                },
            },
        },
        "cluster1_module4": {
            "instrument_type": "QCM_RF",
            "complex_output_0": {
                "line_gain_db": 0,
                "dc_mixer_offset_I": -14.715e-3,
                "dc_mixer_offset_Q": -6.0985e-3,
                "lo_freq": 3_950_000_000.0,
                "seq0": {
                    "mixer_amp_ratio": 0.99201,
                    "mixer_phase_error_deg": -19.942,
                    "port": "q17:mw",
                    "clock": "q17.01",
                },
                "seq0_12": {
                    "mixer_amp_ratio": 0.99201,
                    "mixer_phase_error_deg": -19.942,
                    "port": "q17:mw",
                    "clock": "q17.12",
                },
            },
            "complex_output_1": {
                "line_gain_db": 0,
                "dc_mixer_offset_I": 1.8529e-3,
                "dc_mixer_offset_Q": -6.2015e-3,
                "lo_freq": 3_260_000_000.0,
                "seq1": {
                    "mixer_amp_ratio": 1.035,
                    "mixer_phase_error_deg": -8.964,
                    "port": "q18:mw",
                    "clock": "q18.01",
                },
                "seq1_12": {
                    "mixer_amp_ratio": 1.035,
                    "mixer_phase_error_deg": -8.964,
                    "port": "q18:mw",
                    "clock": "q18.12",
                },
            },
        },
        "cluster1_module6": {
            "instrument_type": "QCM_RF",
            "complex_output_0": {
                "line_gain_db": 0,
                "dc_mixer_offset_I": -14.766e-3,
                "dc_mixer_offset_Q": -2.3082e-3,
                "lo_freq": 3_990_000_000.0,
                "seq0": {
                    "mixer_amp_ratio": 0.969,
                    "mixer_phase_error_deg": -4.7546,
                    "port": "q19:mw",
                    "clock": "q19.01",
                },
                "seq0_12": {
                    "mixer_amp_ratio": 0.969,
                    "mixer_phase_error_deg": -4.7546,
                    "port": "q19:mw",
                    "clock": "q19.12",
                },
            },
            "complex_output_1": {
                "line_gain_db": 0,
                "dc_mixer_offset_I": -8.2093e-3,
                "dc_mixer_offset_Q": -8.0902e-3,
                "lo_freq": 3_410_000_000.0,
                "seq1": {
                    "mixer_amp_ratio": 1.019,
                    "mixer_phase_error_deg": -34.116,
                    "port": "q20:mw",
                    "clock": "q20.01",
                },
                "seq1_12": {
                    "mixer_amp_ratio": 1.019,
                    "mixer_phase_error_deg": -34.116,
                    "port": "q20:mw",
                    "clock": "q20.12",
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

```{code-cell} ipython3

```
