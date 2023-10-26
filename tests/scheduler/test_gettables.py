# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-module-docstring
# pylint: disable=too-many-locals
# pylint: disable=invalid-name
# pylint: disable=unused-argument

# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
import json
import os
import zipfile
from unittest import TestCase
from unittest.mock import Mock, MagicMock

import numpy as np
import pytest
from packaging.requirements import Requirement
from qblox_instruments import ClusterType
from qcodes.instrument.parameter import ManualParameter
from xarray import DataArray, Dataset

from quantify_scheduler.backends import SerialCompiler
from quantify_scheduler.enums import BinMode
from quantify_scheduler.gettables import ScheduleGettable
from quantify_scheduler.gettables_profiled import ProfiledScheduleGettable
from quantify_scheduler.helpers.schedule import (
    extract_acquisition_metadata_from_schedule,
)
from quantify_scheduler.instrument_coordinator import InstrumentCoordinator
from quantify_scheduler.instrument_coordinator.components.qblox import (
    ClusterComponent,
)
from quantify_scheduler.operations.gate_library import Reset
from quantify_scheduler.schedules.schedule import AcquisitionMetadata, Schedule
from quantify_scheduler.schedules.spectroscopy_schedules import (
    heterodyne_spec_sched,
    nv_dark_esr_sched,
)
from quantify_scheduler.schedules.timedomain_schedules import (
    allxy_sched,
    rabi_sched,
    readout_calibration_sched,
    t1_sched,
)
from quantify_scheduler.schedules.trace_schedules import trace_schedule

from tests.scheduler.backends.test_qblox_backend import (  # pylint: disable=unused-import
    dummy_cluster,
)


@pytest.mark.parametrize("num_channels, real_imag", [(1, True), (2, False), (10, True)])
def test_process_acquired_data(
    mock_setup_basic_transmon, num_channels: int, real_imag: bool
):
    # arrange
    quantum_device = mock_setup_basic_transmon["quantum_device"]
    acq_metadata = AcquisitionMetadata(
        acq_protocol="SSBIntegrationComplex",
        bin_mode=BinMode.AVERAGE,
        acq_return_type=complex,
        acq_indices={i: [0] for i in range(num_channels)},
        repetitions=1,
    )

    mock_results = np.array([4815 + 162342j], dtype=np.complex64)
    mock_dataset = Dataset(
        {i: ([f"acq_index_{i}"], mock_results) for i in range(num_channels)}
    )

    gettable = ScheduleGettable(
        quantum_device=quantum_device,
        schedule_function=lambda x: x,
        schedule_kwargs={},
        real_imag=real_imag,
    )

    # act
    with pytest.warns(FutureWarning, match=".* in quantify-scheduler-0.17."):
        processed_data = gettable.process_acquired_data(
            mock_dataset, acq_metadata, repetitions=10
        )

    # assert
    assert len(processed_data) == 2 * num_channels


def test_schedule_gettable_iterative_heterodyne_spec(mock_setup_basic_transmon, mocker):
    meas_ctrl = mock_setup_basic_transmon["meas_ctrl"]
    quantum_device = mock_setup_basic_transmon["quantum_device"]

    qubit = quantum_device.get_element("q0")

    # manual parameter for testing purposes
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

    # Prepare the mock data the spectroscopy schedule

    acq_channel = 0
    acq_indices = [0]

    data = np.exp(1j * np.deg2rad(45)).astype(np.complex64)

    # SSBIntegrationComplex with bin_mode.AVERAGE should return that data
    expected_data = Dataset(
        {
            acq_channel: (
                [
                    f"acq_index_{acq_channel}_yolo"
                ],  # the name of acquisition channel dimension should not matter
                data.reshape((len(acq_indices),)),
            )
        },
    )

    mocker.patch.object(
        mock_setup_basic_transmon["instrument_coordinator"],
        "retrieve_acquisition",
        return_value=expected_data,
    )

    # Configure the gettable
    spec_gettable = ScheduleGettable(
        quantum_device=quantum_device,
        schedule_function=heterodyne_spec_sched,
        schedule_kwargs=schedule_kwargs,
        real_imag=False,
    )
    assert spec_gettable.is_initialized is False

    freqs = np.linspace(5e9, 6e9, 11)
    meas_ctrl.settables(ro_freq)
    meas_ctrl.setpoints(freqs)
    meas_ctrl.gettables(spec_gettable)
    label = f"Heterodyne spectroscopy {qubit.name}"
    dset = meas_ctrl.run(label)
    assert spec_gettable.is_initialized is True

    exp_data = np.ones(len(freqs), dtype=np.complex64) * data
    # Assert that the data is coming out correctly.
    np.testing.assert_array_equal(dset.x0, freqs)
    np.testing.assert_array_equal(dset.y0, abs(exp_data))
    np.testing.assert_array_equal(dset.y1, np.angle(exp_data, deg=True))


# test a batched case
def test_schedule_gettable_batched_allxy(
    mock_setup_basic_transmon_with_standard_params, mocker
):
    meas_ctrl = mock_setup_basic_transmon_with_standard_params["meas_ctrl"]
    quantum_device = mock_setup_basic_transmon_with_standard_params["quantum_device"]

    qubit = quantum_device.get_element("q0")

    index_par = ManualParameter("index", initial_value=0, unit="#")
    index_par.batched = True

    sched_kwargs = {
        "element_select_idx": index_par,
        "qubit": qubit.name,
    }
    indices = np.repeat(np.arange(21), 2)
    # Prepare the mock data the ideal AllXY data
    sched = allxy_sched("q0", element_select_idx=indices, repetitions=256)

    compiler = SerialCompiler(name="compiler")
    comp_allxy_sched = compiler.compile(
        schedule=sched,
        config=quantum_device.generate_compilation_config(),
    )

    data = (
        np.concatenate(
            (
                0 * np.ones(5 * 2),
                0.5 * np.ones(12 * 2),
                np.ones(4 * 2),
            )
        )
        * np.exp(1j * np.deg2rad(45))
    ).astype(np.complex64)
    acq_metadata = extract_acquisition_metadata_from_schedule(comp_allxy_sched)
    acq_channel, acq_indices = next(iter(acq_metadata.acq_indices.items()))
    # SSBIntegrationComplex, bin_mode.AVERAGE
    expected_data = Dataset(
        {acq_channel: ([f"acq_index_{acq_channel}"], data.reshape((len(acq_indices),)))}
    )

    mocker.patch.object(
        mock_setup_basic_transmon_with_standard_params["instrument_coordinator"],
        "retrieve_acquisition",
        return_value=expected_data,
    )

    # Configure the gettable

    allxy_gettable = ScheduleGettable(
        quantum_device=quantum_device,
        schedule_function=allxy_sched,
        schedule_kwargs=sched_kwargs,
        real_imag=True,
        batched=True,
        max_batch_size=1024,
    )

    meas_ctrl.settables(index_par)
    meas_ctrl.setpoints(indices)
    meas_ctrl.gettables([allxy_gettable])
    label = f"AllXY {qubit.name}"
    dset = meas_ctrl.run(label)

    # Assert that the data is coming out correctly.
    np.testing.assert_array_equal(dset.x0, indices)
    np.testing.assert_array_equal(dset.y0 + 1j * dset.y1, data)


# test a batched case
def test_schedule_gettable_append_readout_cal(
    mock_setup_basic_transmon_with_standard_params, mocker
):
    meas_ctrl = mock_setup_basic_transmon_with_standard_params["meas_ctrl"]
    quantum_device = mock_setup_basic_transmon_with_standard_params["quantum_device"]

    repetitions = 256
    qubit = quantum_device.get_element("q0")

    prep_state = ManualParameter("prep_state", label="Prepared qubit state", unit="")
    prep_state.batched = True

    # extra repetition index will not be required after the new data format
    repetition_par = ManualParameter("repetition", label="Repetition", unit="#")
    repetition_par.batched = True

    sched_kwargs = {
        "qubit": qubit.name,
        "prepared_states": [0, 1],
    }

    quantum_device.cfg_sched_repetitions(repetitions)

    # Prepare the mock data the ideal SSRO data
    ssro_sched = readout_calibration_sched("q0", [0, 1], repetitions=repetitions)

    compiler = SerialCompiler(name="compiler")
    comp_ssro_sched = compiler.compile(
        schedule=ssro_sched,
        config=quantum_device.generate_compilation_config(),
    )

    data = (np.tile(np.arange(2, dtype=np.float64), repetitions) * np.exp(1j)).astype(
        np.complex64
    )

    acq_metadata = extract_acquisition_metadata_from_schedule(comp_ssro_sched)
    acq_channel, acq_indices = next(iter(acq_metadata.acq_indices.items()))
    # SSBIntegrationComplex, BinMode.APPEND
    expected_data = Dataset(
        {
            acq_channel: (
                ["a_repetition_index", "an_acq_index"],
                data.reshape((repetitions, len(acq_indices))),
            )
        }
    )

    mocker.patch.object(
        mock_setup_basic_transmon_with_standard_params["instrument_coordinator"],
        "retrieve_acquisition",
        return_value=expected_data,
    )

    # Configure the gettable

    ssro_gettable = ScheduleGettable(
        quantum_device=quantum_device,
        schedule_function=readout_calibration_sched,
        schedule_kwargs=sched_kwargs,
        real_imag=True,
        batched=True,
        max_batch_size=1024,
    )

    meas_ctrl.settables([prep_state, repetition_par])
    meas_ctrl.setpoints_grid([np.arange(2), np.arange(repetitions)])
    meas_ctrl.gettables(ssro_gettable)
    label = f"SSRO {qubit.name}"
    dset = meas_ctrl.run(label)

    # Assert that the data is coming out correctly.
    np.testing.assert_array_equal(dset.x0, np.tile(np.arange(2), repetitions))
    np.testing.assert_array_equal(dset.x1, np.repeat(np.arange(repetitions), 2))

    np.testing.assert_array_equal(dset.y0 + 1j * dset.y1, data)


def test_schedule_gettable_trace_acquisition(
    mock_setup_basic_transmon_with_standard_params, mocker
):
    meas_ctrl = mock_setup_basic_transmon_with_standard_params["meas_ctrl"]
    quantum_device = mock_setup_basic_transmon_with_standard_params["quantum_device"]
    # q0 is a  device element from the test setup has all the right params
    device_element = quantum_device.get_element("q0")

    sample_par = ManualParameter("sample", label="Sample time", unit="s")
    sample_par.batched = True

    schedule_kwargs = {
        "pulse_amp": device_element.measure.pulse_amp(),
        "pulse_duration": device_element.measure.pulse_duration(),
        "pulse_delay": 2e-9,
        "frequency": device_element.clock_freqs.readout(),
        "acquisition_delay": device_element.measure.acq_delay(),
        "integration_time": device_element.measure.integration_time(),
        "port": device_element.ports.readout(),
        "clock": device_element.name + ".ro",
        "init_duration": device_element.reset.duration(),
    }

    sched_gettable = ScheduleGettable(
        quantum_device=quantum_device,
        schedule_function=trace_schedule,
        schedule_kwargs=schedule_kwargs,
        batched=True,
    )

    sample_times = np.arange(0, device_element.measure.integration_time(), 1 / 1e9)
    exp_trace = (np.ones(len(sample_times)) * np.exp(1j * np.deg2rad(35))).astype(
        np.complex64
    )

    exp_data_array = DataArray(
        [exp_trace],
        coords=[[0], range(len(exp_trace))],
        dims=["repetition", "acq_index"],
    )
    exp_data = Dataset({0: exp_data_array})

    mocker.patch.object(
        mock_setup_basic_transmon_with_standard_params["instrument_coordinator"],
        "retrieve_acquisition",
        return_value=exp_data,
    )

    # Executing the experiment
    meas_ctrl.settables(sample_par)
    meas_ctrl.setpoints(sample_times)
    meas_ctrl.gettables(sched_gettable)
    label = f"Readout trace schedule of {device_element.name}"
    dset = meas_ctrl.run(label)

    # Assert that the data is coming out correctly.
    np.testing.assert_array_equal(dset.x0, sample_times)
    np.testing.assert_array_equal(dset.y0, exp_trace.real)
    np.testing.assert_array_equal(dset.y1, exp_trace.imag)


@pytest.mark.deprecated
def test_schedule_gettable_generate_diagnostic(
    mock_setup_basic_transmon_with_standard_params, mocker
):
    schedule_kwargs = {"times": np.linspace(1e-6, 50e-6, 50), "qubit": "q0"}
    quantum_device = mock_setup_basic_transmon_with_standard_params["quantum_device"]

    # Prepare the mock data the t1 schedule
    acq_channel = 0
    data = (np.ones(50) * np.exp(1j * np.deg2rad(45))).astype(np.complex64)

    # SSBIntegrationComplex, BinMode.AVERAGE
    expected_data = Dataset({acq_channel: (["acq_index"], data)})

    mocker.patch.object(
        mock_setup_basic_transmon_with_standard_params["instrument_coordinator"],
        "retrieve_acquisition",
        return_value=expected_data,
    )

    # Configure the gettable
    gettable = ScheduleGettable(
        quantum_device=quantum_device,
        schedule_function=t1_sched,
        schedule_kwargs=schedule_kwargs,
        real_imag=True,
        batched=True,
    )
    assert gettable.is_initialized is False

    with pytest.raises(RuntimeError):
        gettable.generate_diagnostics_report()

    with pytest.raises(RuntimeError):
        gettable.generate_diagnostics_report(update=True)

    filename = gettable.generate_diagnostics_report(execute_get=True)

    assert gettable.is_initialized is True

    with zipfile.ZipFile(filename, mode="r") as zf:
        _ = json.loads(zf.read("device_cfg.json").decode())
        _ = json.loads(zf.read("hardware_cfg.json").decode())
        get_cfg = json.loads(zf.read("gettable.json").decode())
        sched = Schedule.from_json(zf.read("schedule.json").decode())
        snap = json.loads(zf.read("snapshot.json").decode())

    assert (
        snap["instruments"]["q0"]["submodules"]["reset"]["parameters"]["duration"][
            "value"
        ]
        == 0.0002
    )
    assert gettable.quantum_device.cfg_sched_repetitions() == get_cfg["repetitions"]

    compiler = SerialCompiler(name="compiler")
    compiled_sched = compiler.compile(
        schedule=sched, config=quantum_device.generate_compilation_config()
    )

    assert gettable._compiled_schedule == compiled_sched


def test_profiling(mock_setup_basic_transmon_with_standard_params, tmp_test_data_dir):
    mock_setup = mock_setup_basic_transmon_with_standard_params
    quantum_device = mock_setup["quantum_device"]
    qubit = mock_setup["q0"]

    schedule_kwargs = {
        "pulse_amp": qubit.measure.pulse_amp(),
        "pulse_duration": qubit.measure.pulse_duration(),
        "frequency": qubit.clock_freqs.f01(),
        "qubit": "q0",
    }
    profiled_gettable = ProfiledScheduleGettable(
        quantum_device=quantum_device,
        schedule_function=rabi_sched,
        schedule_kwargs=schedule_kwargs,
    )

    profiled_gettable.initialize()
    profiled_ic = (
        profiled_gettable.quantum_device.instr_instrument_coordinator.get_instr()
    )
    profiled_ic.start()
    profiled_ic.wait_done()
    profiled_ic.retrieve_acquisition()
    profiled_ic.stop()
    profiled_gettable.close()

    # Test if all steps have been measured and have a value > 0
    log = profiled_gettable.log_profile()
    TestCase().assertAlmostEqual(log["schedule"][0], 0.2062336)
    verif_keys = [
        "schedule",
        "_compile",
        "prepare",
        "start",
        "wait_done",
        "retrieve_acquisition",
        "stop",
    ]
    for key in verif_keys:
        assert len(log[key]) > 0
        assert [value > 0 for value in log[key]]

    # Test logging to json
    obj = {"test": ["test"]}
    path = tmp_test_data_dir
    filename = "test"
    profiled_gettable.log_profile(
        obj=obj, path=path, filename=filename, indent=4, separators=(",", ": ")
    )
    assert os.path.getsize(os.path.join(path, filename)) > 0

    # Test plot function
    path = tmp_test_data_dir
    filename = "average_runtimes.pdf"
    profiled_gettable.plot_profile(path=path)
    assert profiled_gettable.plot is not None
    assert os.path.getsize(os.path.join(path, filename)) > 0


def test_formatting_trigger_count(mock_setup_basic_nv):
    """ScheduleGettable formats data in trigger_acquisition mode correctly"""
    # Arrange
    instrument_coordinator = mock_setup_basic_nv["instrument_coordinator"]
    nv_center = mock_setup_basic_nv["quantum_device"]
    nv_center.cfg_sched_repetitions(1)

    # data returned by the instrument coordinator
    acquired_data_array = DataArray(
        [[101, 35, 2]],
        coords=[[0], [0, 1, 2]],
        dims=["repetition", "acq_index"],
    )
    acquired_data = Dataset({0: acquired_data_array})

    # Make instrument coordinator a dummy that only returns data
    instrument_coordinator.retrieve_acquisition = Mock(return_value=acquired_data)
    instrument_coordinator.prepare = Mock()
    instrument_coordinator.stop = Mock()
    instrument_coordinator.start = Mock()
    instrument_coordinator.get = Mock()

    sched_kwargs = {
        "qubit": "qe0",
    }
    dark_esr_gettable = ScheduleGettable(
        quantum_device=nv_center,
        schedule_function=nv_dark_esr_sched,
        schedule_kwargs=sched_kwargs,
        batched=True,
        data_labels=["Trigger Count"],
    )
    dark_esr_gettable.unit = [""]

    # Act
    data = dark_esr_gettable.get()

    # Assert
    assert isinstance(data, tuple)
    assert len(data) == 1
    assert len(data[0]) == 3
    for count in data[0]:
        assert isinstance(count, np.uint64)
    np.testing.assert_array_equal(data[0], [101, 35, 2])


def test_no_hardware_cfg_raises(mock_setup_basic_transmon):
    meas_ctrl = mock_setup_basic_transmon["meas_ctrl"]
    quantum_device = mock_setup_basic_transmon["quantum_device"]

    qubit = quantum_device.get_element("q0")

    # manual parameter for testing purposes
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
    )
    assert spec_gettable.is_initialized is False

    freqs = np.linspace(5e9, 6e9, 11)
    meas_ctrl.settables(ro_freq)
    meas_ctrl.setpoints(freqs)
    meas_ctrl.gettables(spec_gettable)
    label = f"Heterodyne spectroscopy {qubit.name}"
    with pytest.raises(RuntimeError) as exc:
        _ = meas_ctrl.run(label)
    assert (
        f"InstrumentCoordinator.retrieve_acquisition() "
        f"('{mock_setup_basic_transmon['instrument_coordinator'].name}') did not "
        f"return any data, but was expected to return data based on the acquisition "
        f"metadata in the compiled schedule: acq_metadata.acq_indices="
        in str(exc.value)
    )


def test_initialize_and_get_with_report_failed_initialization__qblox(
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

    with zipfile.ZipFile(report_zipfile, mode="r") as zf:
        dependency_versions = json.loads(zf.read("dependency_versions.json").decode())
        q2_cfg_report = json.loads(zf.read("device_elements/q2.json"))
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

    parsed_dependencies = [
        Requirement(line.split(":")[0]).name for line in dependency_versions
    ]

    for dependency in [
        "python",
        "quantify-scheduler",
        "quantify-core",
        "qblox-instruments",
        "numpy",
    ]:
        assert dependency in parsed_dependencies

    assert q2_cfg_report["data"]["rxy"]["amp180"] == 0.213

    assert (
        gettable.quantum_device.cfg_sched_repetitions()
        == gettable_cfg_report["repetitions"]
    )

    assert (
        hardware_cfg_report[cluster_name][f"{cluster_name}_module2"]["instrument_type"]
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
    gettable = ScheduleGettable(
        quantum_device=quantum_device,
        schedule_function=t1_sched,
        schedule_kwargs={"times": np.linspace(1e-6, 50e-6, 50), "qubit": "q2"},
        batched=True,
    )

    ic.prepare = MagicMock(side_effect=RuntimeError)
    report_zipfile = gettable.initialize_and_get_with_report()

    assert "failed_initialization" in os.path.basename(report_zipfile)

    with zipfile.ZipFile(report_zipfile, mode="r") as zf:
        compiled_schedule_report = Schedule.from_json(
            zf.read("compiled_schedule.json").decode()
        )

    assert gettable.compiled_schedule == compiled_schedule_report


def test_initialize_and_get_with_report_compiled_schedule_reset__qblox(
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


def test_initialize_and_get_with_report_failed_exp__qblox(
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

    ic.get_component(f"ic_{cluster_name}").instrument.get_ip_config = MagicMock(
        return_value=example_ip
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

    with zipfile.ZipFile(report_zipfile, mode="r") as zf:
        json.loads(zf.read("dependency_versions.json").decode())
        json.loads(zf.read("device_elements/q2.json"))
        json.loads(zf.read("gettable.json").decode())
        json.loads(zf.read("hardware_cfg.json").decode())
        Schedule.from_json(zf.read("schedule.json").decode())
        compiled_schedule_report = Schedule.from_json(
            zf.read("compiled_schedule.json").decode()
        )
        snap_report = json.loads(zf.read("snapshot.json").decode())
        with pytest.raises(KeyError):
            zf.read("acquisition_data.txt").decode()
        report_error_trace = zf.read("error_trace.txt").decode()
        zf.read(f"{cluster_name}/{cluster_name}_cmm_app_log.txt").decode()
        with pytest.raises(KeyError):
            zf.read("connection_error_trace.txt").decode()

    assert gettable.compiled_schedule == compiled_schedule_report

    assert (
        snap_report["instruments"]["q2"]["submodules"]["reset"]["parameters"][
            "duration"
        ]["value"]
        == 0.0002
    )

    assert report_error_trace.split(" ")[0] == "Traceback"
    assert report_error_trace.split(": ")[-1].rstrip() == failing_exp_trace


def test_initialize_and_get_with_report_completed_exp__qblox(
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

    ic.get_component(f"ic_{cluster_name}").instrument.get_ip_config = MagicMock(
        return_value=example_ip
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
    expected_data = Dataset({acquisition_channel: (["acq_index"], data)})

    mocker.patch.object(
        ic,
        "retrieve_acquisition",
        return_value=expected_data,
    )

    report_zipfile = gettable.initialize_and_get_with_report()

    assert "completed_exp" in os.path.basename(report_zipfile)

    with zipfile.ZipFile(report_zipfile, mode="r") as zf:
        json.loads(zf.read("dependency_versions.json").decode())
        json.loads(zf.read("device_elements/q2.json"))
        json.loads(zf.read("gettable.json").decode())
        json.loads(zf.read("hardware_cfg.json").decode())
        Schedule.from_json(zf.read("schedule.json").decode())
        Schedule.from_json(zf.read("compiled_schedule.json").decode())
        json.loads(zf.read("snapshot.json").decode())
        acquisition_data = zf.read("acquisition_data.txt").decode()
        with pytest.raises(KeyError):
            zf.read("error_trace.txt").decode()
        hardware_log = zf.read(
            f"{cluster_name}/{cluster_name}_cmm_app_log.txt"
        ).decode()
        hardware_log_idn = zf.read(f"{cluster_name}/{cluster_name}_idn.txt").decode()
        hardware_log_mods_info = zf.read(
            f"{cluster_name}/{cluster_name}_mods_info.txt"
        ).decode()
        with pytest.raises(KeyError):
            zf.read("connection_error_trace.txt").decode()

    assert acquisition_data.split(" ")[1] == "0.70710677"

    # Test that hardware logs are correctly passed to the zipfile
    assert hardware_log == "Mock hardware log for app"
    assert "serial_number" in hardware_log_idn
    assert "IDN" in hardware_log_mods_info


def test_initialize_and_get_with_report_failed_hw_log_retrieval__qblox(
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
    ic.retrieve_hardware_logs = MagicMock(
        side_effect=RuntimeError(failing_connection_trace)
    )

    report_zipfile = gettable.initialize_and_get_with_report()

    assert "failed_hw_log_retrieval" in os.path.basename(report_zipfile)

    with zipfile.ZipFile(report_zipfile, mode="r") as zf:
        json.loads(zf.read("dependency_versions.json").decode())
        json.loads(zf.read("device_elements/q2.json"))
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
    assert (
        report_connection_error_trace.split(": ")[-1].rstrip()
        == failing_connection_trace
    )


def test_initialize_and_get_with_report_failed_connection_to_hw__qblox(
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

    with zipfile.ZipFile(report_zipfile, mode="r") as zf:
        json.loads(zf.read("dependency_versions.json").decode())
        json.loads(zf.read("device_elements/q2.json"))
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


def test_initialize_and_get_with_report__two_qblox_clusters(
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

    ic.get_component(f"ic_{cluster1_name}").instrument.get_ip_config = MagicMock(
        return_value=example_ip
    )
    ic.get_component(f"ic_{cluster2_name}").instrument.get_ip_config = MagicMock(
        return_value=example_ip
    )

    # ConfigurationManager belongs to qblox-instruments, but was already imported
    # in quantify_scheduler
    mocker.patch(
        "quantify_scheduler.instrument_coordinator.components.qblox.ConfigurationManager",
        return_value=mock_qblox_instruments_config_manager,
    )

    def schedule_function(times, repetitions=1):  # noqa: ARG001
        sched = Schedule("sched")
        sched.add(Reset("q2"))
        sched.add(Reset("q3"))
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
        "module1_app_log.txt",
        "module1_system_log.txt",
        "module2_app_log.txt",
        "module2_system_log.txt",
    ]

    with zipfile.ZipFile(report_zipfile, mode="r") as zf:
        for cluster_name in (cluster1_name, cluster2_name):
            for logfile in logfiles:
                zf.read(f"{cluster_name}/{cluster_name}_{logfile}").decode()
