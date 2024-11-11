# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
import json
import os
import zipfile
from datetime import datetime
from unittest.mock import MagicMock

import numpy as np
import pytest
from packaging.requirements import Requirement
from qblox_instruments import ClusterType
from qcodes.instrument.parameter import ManualParameter
from xarray import Dataset

from quantify_scheduler.device_under_test.quantum_device import QuantumDevice
from quantify_scheduler.gettables import ScheduleGettable
from quantify_scheduler.instrument_coordinator import InstrumentCoordinator
from quantify_scheduler.instrument_coordinator.components.qblox import (
    ClusterComponent,
)
from quantify_scheduler.operations.gate_library import Measure, Reset
from quantify_scheduler.schedules.schedule import CompiledSchedule, Schedule
from quantify_scheduler.schedules.spectroscopy_schedules import (
    heterodyne_spec_sched,
)
from quantify_scheduler.schedules.timedomain_schedules import (
    t1_sched,
)
from tests.scheduler.backends.test_qblox_backend import (
    dummy_cluster,
)
from tests.scheduler.instrument_coordinator.components.test_qblox import (
    make_cluster_component,
)


def test_schedule_gettable_always_initialize_false(
    mock_setup_basic_transmon_with_standard_params, make_cluster_component
):
    cluster_name = "cluster0"
    hardware_cfg = {
        "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
        "hardware_description": {
            f"{cluster_name}": {
                "instrument_type": "Cluster",
                "ref": "internal",
                "modules": {
                    "4": {"instrument_type": "QRM_RF"},
                },
            }
        },
        "hardware_options": {
            "modulation_frequencies": {
                "q0:res-q0.ro": {
                    "lo_freq": 5e9,
                }
            }
        },
        "connectivity": {
            "graph": [
                ["cluster0.module4.complex_input_0", "q0:res"],
            ]
        },
    }

    quantum_device = mock_setup_basic_transmon_with_standard_params["quantum_device"]
    quantum_device.hardware_config(hardware_cfg)

    ic_cluster0 = make_cluster_component(cluster_name)
    qrm_rf = ic_cluster0._cluster_modules[f"{cluster_name}_module4"]

    ic = mock_setup_basic_transmon_with_standard_params["instrument_coordinator"]
    ic.add_component(ic_cluster0)

    qubit = quantum_device.get_element("q0")
    ro_freq = ManualParameter("ro_freq", initial_value=5e9, unit="Hz")

    schedule_kwargs = {
        "pulse_amp": qubit.measure.pulse_amp(),
        "pulse_duration": qubit.measure.pulse_duration(),
        "frequency": ro_freq,
        "acquisition_delay": qubit.measure.acq_delay(),
        "integration_time": qubit.measure.integration_time(),
        "port": qubit.ports.readout(),
        "clock": qubit.name + ".ro",
        "init_duration": qubit.reset.duration(),
    }

    # Configure the gettable
    spec_gettable = ScheduleGettable(
        quantum_device=quantum_device,
        schedule_function=heterodyne_spec_sched,
        schedule_kwargs=schedule_kwargs,
        real_imag=False,
        always_initialize=False,
    )
    assert spec_gettable.is_initialized is False

    spec_gettable.get()
    assert spec_gettable.is_initialized is True
    assert qrm_rf.instrument.arm_sequencer.call_count == 1

    spec_gettable.get()
    assert qrm_rf.instrument.arm_sequencer.call_count == 2


def test_initialize_and_get_with_report_failed_initialization(  # noqa: PLR0915
    mock_setup_basic_transmon_with_standard_params,
    mocker,
    hardware_cfg_rf,
    mock_qblox_instruments_config_manager,
    dummy_cluster,
):
    cluster_name = "cluster0"

    ic = InstrumentCoordinator("ic")
    ic.add_component(
        ClusterComponent(
            dummy_cluster(
                name=cluster_name,
                dummy_cfg={
                    2: ClusterType.CLUSTER_QCM_RF,
                    4: ClusterType.CLUSTER_QRM_RF,
                },
            )
        )
    )

    quantum_device = mock_setup_basic_transmon_with_standard_params["quantum_device"]
    quantum_device.get_element("q2").clock_freqs.readout(7.5e9)

    # ConfigurationManager belongs to qblox-instruments, but was already imported
    # in quantify_scheduler
    mocker.patch(
        "quantify_scheduler.instrument_coordinator.components.qblox.ConfigurationManager",
        return_value=mock_qblox_instruments_config_manager,
    )

    # Test report with failing compilation
    gettable = ScheduleGettable(
        quantum_device=quantum_device,
        schedule_function=t1_sched,
        schedule_kwargs={"times": np.linspace(1e-6, 50e-6, 50), "qubit": "q17"},
        batched=True,
    )

    with pytest.raises(AttributeError):
        gettable.initialize_and_get_with_report()

    quantum_device.instr_instrument_coordinator(ic.name)

    with pytest.raises(AttributeError):
        gettable.initialize_and_get_with_report()

    quantum_device.hardware_config(hardware_cfg_rf)

    report_zipfile = gettable.initialize_and_get_with_report()
    assert "failed_initialization" in os.path.basename(report_zipfile)

    # Close instruments, necessary for deserializing QuantumDevice
    for element_name in list(quantum_device.elements()):
        quantum_device.get_element(element_name).close()

    for edge_name in list(quantum_device.edges()):
        quantum_device.get_edge(edge_name).close()

    with zipfile.ZipFile(report_zipfile, mode="r") as zf:
        quantum_device_report = QuantumDevice.from_json(zf.read("quantum_device.json").decode())
        dependency_versions = json.loads(zf.read("dependency_versions.json").decode())
        timestamp = zf.read("timestamp.txt").decode()
        gettable_cfg_report = json.loads(zf.read("gettable.json").decode())
        hardware_cfg_report = json.loads(zf.read("hardware_cfg.json").decode())
        schedule_report = Schedule.from_json(zf.read("schedule.json").decode())
        report_error_trace = zf.read("error_trace.txt").decode()
        with pytest.raises(KeyError):
            Schedule.from_json(zf.read("compiled_schedule.json").decode())
        with pytest.raises(KeyError):
            json.loads(zf.read("snapshot.json").decode())
        with pytest.raises(KeyError):
            zf.read("acquisition_data.txt").decode()
        with pytest.raises(KeyError):
            zf.read(f"{cluster_name}/{cluster_name}_cmm_app_log.txt").decode()
        with pytest.raises(KeyError):
            zf.read("connection_error_trace.txt").decode()

    parsed_dependencies = [Requirement(line.split(":")[0]).name for line in dependency_versions]
    for dependency in [
        "python",
        "quantify-scheduler",
        "quantify-core",
        "qblox-instruments",
        "numpy",
    ]:
        assert dependency in parsed_dependencies

    assert quantum_device_report.get_element("q2").rxy.parameters["amp180"]() == 0.213
    assert datetime.strptime(timestamp, "%Y-%m-%d_%H-%M-%S_%Z")

    assert gettable.quantum_device.cfg_sched_repetitions() == gettable_cfg_report["repetitions"]

    assert (
        hardware_cfg_report["hardware_description"][cluster_name]["modules"]["2"]["instrument_type"]
        == "QCM_RF"
    )

    assert schedule_report == gettable.schedule_function(
        **gettable._evaluated_sched_kwargs,
        repetitions=gettable.quantum_device.cfg_sched_repetitions(),
    )

    assert report_error_trace.split(" ")[0] == "Traceback"
    assert (
        report_error_trace.split(": ")[-1].rstrip()
        == "[\\'q0\\', \\'q1\\', \\'q2\\', \\'q3\\', \\'q4\\']'"
    )

    # Test failing ic retrieval / preparation
    quantum_device_report.hardware_config(hardware_cfg_report)
    quantum_device_report.instr_instrument_coordinator(ic.name)

    gettable = ScheduleGettable(
        quantum_device=quantum_device_report,
        schedule_function=t1_sched,
        schedule_kwargs={"times": np.linspace(1e-6, 50e-6, 50), "qubit": "q2"},
        batched=True,
    )

    ic.prepare = MagicMock(side_effect=RuntimeError)
    report_zipfile = gettable.initialize_and_get_with_report()
    assert "failed_initialization" in os.path.basename(report_zipfile)

    with zipfile.ZipFile(report_zipfile, mode="r") as zf:
        compiled_schedule_report = CompiledSchedule.from_json(
            zf.read("compiled_schedule.json").decode()
        )
    assert gettable.compiled_schedule == compiled_schedule_report


def test_initialize_and_get_with_report_compiled_schedule_reset(
    mock_setup_basic_transmon_with_standard_params,
    mocker,
    hardware_cfg_rf,
    mock_qblox_instruments_config_manager,
    dummy_cluster,
):
    ic = InstrumentCoordinator("ic")
    ic.add_component(
        ClusterComponent(
            dummy_cluster(
                dummy_cfg={
                    2: ClusterType.CLUSTER_QCM_RF,
                    4: ClusterType.CLUSTER_QRM_RF,
                },
            )
        )
    )

    quantum_device = mock_setup_basic_transmon_with_standard_params["quantum_device"]
    quantum_device.instr_instrument_coordinator(ic.name)
    quantum_device.hardware_config(hardware_cfg_rf)
    quantum_device.get_element("q2").clock_freqs.readout(7.5e9)

    # ConfigurationManager belongs to qblox-instruments, but was already imported
    # in quantify_scheduler
    mocker.patch(
        "quantify_scheduler.instrument_coordinator.components.qblox.ConfigurationManager",
        return_value=mock_qblox_instruments_config_manager,
    )

    gettable = ScheduleGettable(
        quantum_device=quantum_device,
        schedule_function=t1_sched,
        schedule_kwargs={"times": np.linspace(1e-6, 50e-6, 50), "qubit": "q2"},
        batched=True,
    )

    _ = gettable.initialize_and_get_with_report()
    assert gettable.compiled_schedule is not None

    # Assert any old compiled schedule is reset when creating a report
    gettable.initialize = MagicMock(side_effect=RuntimeError)
    _ = gettable.initialize_and_get_with_report()
    assert gettable.compiled_schedule is None


def test_initialize_and_get_with_report_failed_exp(
    example_ip,
    mock_setup_basic_transmon_with_standard_params,
    mocker,
    hardware_cfg_rf,
    mock_qblox_instruments_config_manager,
    dummy_cluster,
):
    cluster_name = "cluster0"

    ic = InstrumentCoordinator("ic")
    ic.add_component(
        ClusterComponent(
            dummy_cluster(
                name=cluster_name,
                dummy_cfg={
                    2: ClusterType.CLUSTER_QCM_RF,
                    4: ClusterType.CLUSTER_QRM_RF,
                },
            )
        )
    )

    quantum_device = mock_setup_basic_transmon_with_standard_params["quantum_device"]
    quantum_device.instr_instrument_coordinator(ic.name)
    quantum_device.hardware_config(hardware_cfg_rf)
    quantum_device.get_element("q2").clock_freqs.readout(7.5e9)

    mocker.patch(
        "qblox_instruments.Cluster.get_ip_config",
        return_value=example_ip,
    )

    # ConfigurationManager belongs to qblox-instruments, but was already imported
    # in quantify_scheduler
    mocker.patch(
        "quantify_scheduler.instrument_coordinator.components.qblox.ConfigurationManager",
        return_value=mock_qblox_instruments_config_manager,
    )

    gettable = ScheduleGettable(
        quantum_device=quantum_device,
        schedule_function=t1_sched,
        schedule_kwargs={"times": np.linspace(1e-6, 50e-6, 50), "qubit": "q2"},
        batched=True,
    )

    failing_exp_trace = "Test failing exp error trace"
    gettable.get = MagicMock(side_effect=RuntimeError(failing_exp_trace))

    report_zipfile = gettable.initialize_and_get_with_report()

    assert "failed_exp" in os.path.basename(report_zipfile)

    # Close instruments, necessary for deserializing QuantumDevice
    for element_name in quantum_device.snapshot()["parameters"]["elements"]["value"]:
        quantum_device.get_element(element_name).close()

    for edge_name in quantum_device.snapshot()["parameters"]["edges"]["value"]:
        quantum_device.get_edge(edge_name).close()

    with zipfile.ZipFile(report_zipfile, mode="r") as zf:
        QuantumDevice.from_json(zf.read("quantum_device.json").decode())
        json.loads(zf.read("dependency_versions.json").decode())
        json.loads(zf.read("gettable.json").decode())
        json.loads(zf.read("hardware_cfg.json").decode())
        Schedule.from_json(zf.read("schedule.json").decode())
        compiled_schedule_report = Schedule.from_json(zf.read("compiled_schedule.json").decode())
        snap_report = json.loads(zf.read("snapshot.json").decode())
        with pytest.raises(KeyError):
            zf.read("acquisition_data.txt").decode()
        report_error_trace = zf.read("error_trace.txt").decode()
        zf.read(f"{cluster_name}/{cluster_name}_cmm_app_log.txt").decode()
        with pytest.raises(KeyError):
            zf.read("connection_error_trace.txt").decode()

    assert gettable.compiled_schedule == compiled_schedule_report

    assert (
        snap_report["instruments"]["q2"]["submodules"]["reset"]["parameters"]["duration"]["value"]
        == 0.0002
    )

    assert report_error_trace.split(" ")[0] == "Traceback"
    assert report_error_trace.split(": ")[-1].rstrip() == failing_exp_trace


def test_initialize_and_get_with_report_completed_exp(
    example_ip,
    mock_setup_basic_transmon_with_standard_params,
    mocker,
    hardware_cfg_rf,
    mock_qblox_instruments_config_manager,
    dummy_cluster,
):
    cluster_name = "cluster0"

    ic = InstrumentCoordinator("ic")
    ic.add_component(
        ClusterComponent(
            dummy_cluster(
                name=cluster_name,
                dummy_cfg={
                    2: ClusterType.CLUSTER_QCM_RF,
                    4: ClusterType.CLUSTER_QRM_RF,
                },
            )
        )
    )

    quantum_device = mock_setup_basic_transmon_with_standard_params["quantum_device"]
    quantum_device.instr_instrument_coordinator(ic.name)
    quantum_device.hardware_config(hardware_cfg_rf)
    quantum_device.get_element("q2").clock_freqs.readout(7.5e9)

    mocker.patch(
        "qblox_instruments.Cluster.get_ip_config",
        return_value=example_ip,
    )

    # ConfigurationManager belongs to qblox-instruments, but was already imported
    # in quantify_scheduler
    mocker.patch(
        "quantify_scheduler.instrument_coordinator.components.qblox.ConfigurationManager",
        return_value=mock_qblox_instruments_config_manager,
    )

    gettable = ScheduleGettable(
        quantum_device=quantum_device,
        schedule_function=t1_sched,
        schedule_kwargs={"times": np.linspace(1e-6, 50e-6, 50), "qubit": "q2"},
        batched=True,
    )

    # Prepare mock data
    acquisition_channel = 0
    data = (np.ones(50) * np.exp(1j * np.deg2rad(45))).astype(np.complex64)
    expected_data = Dataset(
        {
            acquisition_channel: (
                ["acq_index"],
                data,
                {"acq_protocol": "SSBIntegrationComplex"},
            )
        }
    )

    mocker.patch.object(
        ic,
        "retrieve_acquisition",
        return_value=expected_data,
    )

    report_zipfile = gettable.initialize_and_get_with_report()

    assert "completed_exp" in os.path.basename(report_zipfile)

    # Close instruments, necessary for deserializing QuantumDevice
    for element_name in quantum_device.snapshot()["parameters"]["elements"]["value"]:
        quantum_device.get_element(element_name).close()

    for edge_name in quantum_device.snapshot()["parameters"]["edges"]["value"]:
        quantum_device.get_edge(edge_name).close()

    with zipfile.ZipFile(report_zipfile, mode="r") as zf:
        QuantumDevice.from_json(zf.read("quantum_device.json").decode())
        json.loads(zf.read("dependency_versions.json").decode())
        json.loads(zf.read("gettable.json").decode())
        json.loads(zf.read("hardware_cfg.json").decode())
        Schedule.from_json(zf.read("schedule.json").decode())
        Schedule.from_json(zf.read("compiled_schedule.json").decode())
        json.loads(zf.read("snapshot.json").decode())
        acquisition_data = zf.read("acquisition_data.txt").decode()
        with pytest.raises(KeyError):
            zf.read("error_trace.txt").decode()
        hardware_log = zf.read(f"{cluster_name}/{cluster_name}_cmm_app_log.txt").decode()
        hardware_log_idn = zf.read(f"{cluster_name}/{cluster_name}_idn.txt").decode()
        hardware_log_mods_info = zf.read(f"{cluster_name}/{cluster_name}_mods_info.txt").decode()
        with pytest.raises(KeyError):
            zf.read("connection_error_trace.txt").decode()

    assert acquisition_data.split(" ")[1] == "0.70710677"

    # Test that hardware logs are correctly passed to the zipfile
    assert hardware_log == "Mock hardware log for app"
    assert "serial_number" in hardware_log_idn
    assert "IDN" in hardware_log_mods_info


def test_initialize_and_get_with_report_failed_hw_log_retrieval(
    mock_setup_basic_transmon_with_standard_params,
    mocker,
    hardware_cfg_rf,
    mock_qblox_instruments_config_manager,
    dummy_cluster,
):
    cluster_name = "cluster0"

    ic = InstrumentCoordinator("ic")
    ic.add_component(
        ClusterComponent(
            dummy_cluster(
                name=cluster_name,
                dummy_cfg={
                    2: ClusterType.CLUSTER_QCM_RF,
                    4: ClusterType.CLUSTER_QRM_RF,
                },
            )
        )
    )

    quantum_device = mock_setup_basic_transmon_with_standard_params["quantum_device"]
    quantum_device.instr_instrument_coordinator(ic.name)
    quantum_device.hardware_config(hardware_cfg_rf)
    quantum_device.get_element("q2").clock_freqs.readout(7.5e9)

    # ConfigurationManager belongs to qblox-instruments, but was already imported
    # in quantify_scheduler
    mocker.patch(
        "quantify_scheduler.instrument_coordinator.components.qblox.ConfigurationManager",
        return_value=mock_qblox_instruments_config_manager,
    )

    gettable = ScheduleGettable(
        quantum_device=quantum_device,
        schedule_function=t1_sched,
        schedule_kwargs={"times": np.linspace(1e-6, 50e-6, 50), "qubit": "q2"},
        batched=True,
    )

    failing_connection_trace = "Test failing connection error trace"
    ic.retrieve_hardware_logs = MagicMock(side_effect=RuntimeError(failing_connection_trace))

    report_zipfile = gettable.initialize_and_get_with_report()

    assert "failed_hw_log_retrieval" in os.path.basename(report_zipfile)

    # Close instruments, necessary for deserializing QuantumDevice
    for element_name in quantum_device.snapshot()["parameters"]["elements"]["value"]:
        quantum_device.get_element(element_name).close()

    for edge_name in quantum_device.snapshot()["parameters"]["edges"]["value"]:
        quantum_device.get_edge(edge_name).close()

    with zipfile.ZipFile(report_zipfile, mode="r") as zf:
        QuantumDevice.from_json(zf.read("quantum_device.json").decode())
        json.loads(zf.read("dependency_versions.json").decode())
        json.loads(zf.read("gettable.json").decode())
        json.loads(zf.read("hardware_cfg.json").decode())
        Schedule.from_json(zf.read("schedule.json").decode())
        Schedule.from_json(zf.read("compiled_schedule.json").decode())
        json.loads(zf.read("snapshot.json").decode())
        zf.read("acquisition_data.txt").decode()
        with pytest.raises(KeyError):
            zf.read("error_trace.txt").decode()
        with pytest.raises(KeyError):
            zf.read(f"{cluster_name}/{cluster_name}_cmm_app_log.txt").decode()
        report_connection_error_trace = zf.read("connection_error_trace.txt").decode()

    assert report_connection_error_trace.split(" ")[0] == "Traceback"
    assert report_connection_error_trace.split(": ")[-1].rstrip() == failing_connection_trace


def test_initialize_and_get_with_report_failed_connection_to_hw(
    mock_setup_basic_transmon_with_standard_params,
    mocker,
    hardware_cfg_rf,
    mock_qblox_instruments_config_manager,
    dummy_cluster,
):
    cluster_name = "cluster0"

    ic = InstrumentCoordinator("ic")
    ic.add_component(
        ClusterComponent(
            dummy_cluster(
                name=cluster_name,
                dummy_cfg={
                    2: ClusterType.CLUSTER_QCM_RF,
                    4: ClusterType.CLUSTER_QRM_RF,
                },
            )
        )
    )

    quantum_device = mock_setup_basic_transmon_with_standard_params["quantum_device"]
    quantum_device.instr_instrument_coordinator(ic.name)
    quantum_device.hardware_config(hardware_cfg_rf)
    quantum_device.get_element("q2").clock_freqs.readout(7.5e9)

    # ConfigurationManager belongs to qblox-instruments, but was already imported
    # in quantify_scheduler
    mocker.patch(
        "quantify_scheduler.instrument_coordinator.components.qblox.ConfigurationManager",
        return_value=mock_qblox_instruments_config_manager,
    )

    gettable = ScheduleGettable(
        quantum_device=quantum_device,
        schedule_function=t1_sched,
        schedule_kwargs={"times": np.linspace(1e-6, 50e-6, 50), "qubit": "q2"},
        batched=True,
    )

    gettable.get = MagicMock(side_effect=RuntimeError)
    ic.retrieve_hardware_logs = MagicMock(side_effect=RuntimeError)

    report_zipfile = gettable.initialize_and_get_with_report()

    assert "failed_connection_to_hw" in os.path.basename(report_zipfile)

    # Close instruments, necessary for deserializing QuantumDevice
    for element_name in quantum_device.snapshot()["parameters"]["elements"]["value"]:
        quantum_device.get_element(element_name).close()

    for edge_name in quantum_device.snapshot()["parameters"]["edges"]["value"]:
        quantum_device.get_edge(edge_name).close()

    with zipfile.ZipFile(report_zipfile, mode="r") as zf:
        QuantumDevice.from_json(zf.read("quantum_device.json").decode())
        json.loads(zf.read("dependency_versions.json").decode())
        json.loads(zf.read("gettable.json").decode())
        json.loads(zf.read("hardware_cfg.json").decode())
        Schedule.from_json(zf.read("schedule.json").decode())
        Schedule.from_json(zf.read("compiled_schedule.json").decode())
        json.loads(zf.read("snapshot.json").decode())
        with pytest.raises(KeyError):
            zf.read("acquisition_data.txt").decode()
        zf.read("error_trace.txt").decode()
        with pytest.raises(KeyError):
            zf.read(f"{cluster_name}/{cluster_name}_cmm_app_log.txt").decode()
        with pytest.raises(KeyError):
            zf.read("connection_error_trace.txt").decode()


def test_initialize_and_get_with_report__two_clusters(
    example_ip,
    mock_setup_basic_transmon_with_standard_params,
    mocker,
    hardware_cfg_rf_two_clusters,
    mock_qblox_instruments_config_manager,
    dummy_cluster,
):
    cluster1_name = "cluster1"
    cluster2_name = "cluster2"

    ic = InstrumentCoordinator("ic")
    ic.add_component(
        ClusterComponent(
            dummy_cluster(
                name=cluster1_name,
                dummy_cfg={
                    1: ClusterType.CLUSTER_QCM_RF,
                    2: ClusterType.CLUSTER_QRM_RF,
                },
            )
        )
    )
    ic.add_component(
        ClusterComponent(
            dummy_cluster(
                name=cluster2_name,
                dummy_cfg={
                    1: ClusterType.CLUSTER_QCM_RF,
                    2: ClusterType.CLUSTER_QRM_RF,
                },
            )
        )
    )

    quantum_device = mock_setup_basic_transmon_with_standard_params["quantum_device"]
    quantum_device.hardware_config(hardware_cfg_rf_two_clusters)
    quantum_device.instr_instrument_coordinator(ic.name)

    mocker.patch(
        "qblox_instruments.Cluster.get_ip_config",
        return_value=example_ip,
    )

    # ConfigurationManager belongs to qblox-instruments, but was already imported
    # in quantify_scheduler
    mocker.patch(
        "quantify_scheduler.instrument_coordinator.components.qblox.ConfigurationManager",
        return_value=mock_qblox_instruments_config_manager,
    )

    def schedule_function(times, repetitions=1):  # noqa: ARG001
        sched = Schedule("sched")
        sched.add(Measure("q2"))
        sched.add(Measure("q3"))
        return sched

    gettable = ScheduleGettable(
        quantum_device=quantum_device,
        schedule_function=schedule_function,
        schedule_kwargs={"times": None},
        batched=True,
    )

    report_zipfile = gettable.initialize_and_get_with_report()

    logfiles = [
        "cmm_app_log.txt",
        "cmm_system_log.txt",
        "cmm_cfg_man_log.txt",
        "module2_app_log.txt",
        "module2_system_log.txt",
    ]

    with zipfile.ZipFile(report_zipfile, mode="r") as zf:
        for cluster_name in (cluster1_name, cluster2_name):
            for logfile in logfiles:
                zf.read(f"{cluster_name}/{cluster_name}_{logfile}").decode()
