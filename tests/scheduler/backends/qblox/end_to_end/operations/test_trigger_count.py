# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
import numpy as np
import xarray as xr

from quantify_scheduler import Schedule, SerialCompiler
from quantify_scheduler.enums import BinMode
from quantify_scheduler.operations import TriggerCount
from quantify_scheduler.schemas.examples import utils
from tests.scheduler.instrument_coordinator.components.test_qblox import (
    make_cluster_component,
)

EXAMPLE_QBLOX_HARDWARE_CONFIG_NV_CENTER = utils.load_json_example_scheme(
    "qblox_hardware_config_nv_center.json"
)


def test_trigger_count_bin_mode_sum_qrm(
    mock_setup_basic_nv,
    make_cluster_component,
    mocker,
):
    cluster_name = "cluster0"

    qrm_name = f"{cluster_name}_module4"

    cluster = make_cluster_component(
        name=cluster_name,
        modules={"4": "QRM"},
    )
    qrm = cluster._cluster_modules[qrm_name]

    dummy_data = {
        "0": {
            "index": 0,
            "acquisition": {
                "scope": {},
                "bins": {
                    "integration": {},
                    "threshold": [1.0] * 2,
                    "avg_cnt": [24.0, 20.0],
                },
            },
        }
    }

    count = np.array(dummy_data["0"]["acquisition"]["bins"]["avg_cnt"]).astype(int)
    dataarray = xr.DataArray(
        count,
        dims=["acq_index_0"],
        coords={"acq_index_0": list(range(2))},
        attrs={"acq_protocol": "TriggerCount"},
    )
    expected_dataset = xr.Dataset({0: dataarray})

    quantum_device = mock_setup_basic_nv["quantum_device"]
    quantum_device.hardware_config(EXAMPLE_QBLOX_HARDWARE_CONFIG_NV_CENTER)

    sched = Schedule("digital_pulse_and_acq", repetitions=3)
    sched.add(
        TriggerCount(
            duration=1e-6,
            port="qe0:optical_readout",
            clock="qe0.ge0",
            acq_index=0,
            bin_mode=BinMode.SUM,
        )
    )
    sched.add(
        TriggerCount(
            duration=1e-6,
            port="qe0:optical_readout",
            clock="qe0.ge0",
            acq_index=1,
            bin_mode=BinMode.SUM,
        ),
        rel_time=100e-9,
    )

    mocker.patch.object(
        cluster.instrument.module4,
        "get_acquisitions",
        return_value=dummy_data,
    )

    compiler = SerialCompiler(name="compiler")
    compiled_schedule = compiler.compile(
        schedule=sched,
        config=quantum_device.generate_compilation_config(),
    )
    prog = compiled_schedule["compiled_instructions"][cluster_name]

    cluster.prepare(prog)
    cluster.start()

    xr.testing.assert_identical(qrm.retrieve_acquisition(), expected_dataset)


def test_trigger_count_bin_mode_sum_qtm(
    mock_setup_basic_nv,
    make_cluster_component,
    mocker,
):
    cluster_name = "cluster0"

    qrm_name = f"{cluster_name}_module5"

    cluster = make_cluster_component(
        name=cluster_name,
        modules={"5": "QTM"},
    )
    qrm = cluster._cluster_modules[qrm_name]

    dummy_data = {
        "0": {
            "index": 0,
            "acquisition": {
                "bins": {
                    "count": [
                        24.0,
                        20.0,
                    ],
                    "timedelta": [
                        1898975.0,
                        326098.0,
                    ],
                    "threshold": [1.0, 1.0],
                    "avg_cnt": [3, 3],
                }
            },
        }
    }

    count = np.array(dummy_data["0"]["acquisition"]["bins"]["count"]).astype(int)
    dataarray = xr.DataArray(
        count,
        dims=["acq_index_0"],
        coords={"acq_index_0": list(range(2))},
        attrs={"acq_protocol": "TriggerCount"},
    )
    expected_dataset = xr.Dataset({0: dataarray})

    quantum_device = mock_setup_basic_nv["quantum_device"]
    quantum_device.hardware_config(EXAMPLE_QBLOX_HARDWARE_CONFIG_NV_CENTER)

    sched = Schedule("digital_pulse_and_acq", repetitions=3)
    sched.add(
        TriggerCount(
            duration=1e-6,
            port="qe1:optical_readout",
            clock="qe1.ge0",
            acq_index=0,
            bin_mode=BinMode.SUM,
        )
    )
    sched.add(
        TriggerCount(
            duration=1e-6,
            port="qe1:optical_readout",
            clock="qe1.ge0",
            acq_index=1,
            bin_mode=BinMode.SUM,
        ),
        rel_time=100e-9,
    )

    mocker.patch.object(
        cluster.instrument.module5,
        "get_acquisitions",
        return_value=dummy_data,
    )

    compiler = SerialCompiler(name="compiler")
    compiled_schedule = compiler.compile(
        schedule=sched,
        config=quantum_device.generate_compilation_config(),
    )
    prog = compiled_schedule["compiled_instructions"][cluster_name]

    cluster.prepare(prog)
    cluster.start()

    xr.testing.assert_identical(qrm.retrieve_acquisition(), expected_dataset)
