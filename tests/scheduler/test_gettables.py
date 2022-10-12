# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-module-docstring
# pylint: disable=too-many-locals
# pylint: disable=invalid-name
# pylint: disable=unused-argument

# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch

from typing import Any, Dict, Tuple
from unittest import TestCase

import json
import os
import zipfile

import numpy as np
import pytest
from qcodes.instrument.parameter import ManualParameter

from quantify_scheduler.compilation import qcompile
from quantify_scheduler.device_under_test.mock_setup import set_standard_params_transmon
from quantify_scheduler.enums import BinMode
from quantify_scheduler.gettables import ScheduleGettable
from quantify_scheduler.gettables_profiled import ProfiledScheduleGettable
from quantify_scheduler.helpers.schedule import (
    extract_acquisition_metadata_from_schedule,
)

from quantify_scheduler.instrument_coordinator.components.qblox import (
    AcquisitionIndexing,
)
from quantify_scheduler.schedules.schedule import AcquisitionMetadata, Schedule
from quantify_scheduler.schedules.spectroscopy_schedules import heterodyne_spec_sched
from quantify_scheduler.schedules.timedomain_schedules import (
    allxy_sched,
    rabi_sched,
    readout_calibration_sched,
    t1_sched,
)
from quantify_scheduler.schedules.trace_schedules import trace_schedule


@pytest.mark.parametrize("num_channels, real_imag", [(1, True), (2, False), (10, True)])
def test_process_acquired_data(
    mock_setup_basic_transmon, num_channels: int, real_imag: bool
):
    # arrange
    quantum_device = mock_setup_basic_transmon["quantum_device"]
    acq_metadata = AcquisitionMetadata(
        acq_protocol="ssb_integration_complex",
        bin_mode=BinMode.AVERAGE,
        acq_return_type=complex,
        acq_indices={i: [0] for i in range(num_channels)},
    )
    mock_data = {AcquisitionIndexing(i, 0): (4815, 162342) for i in range(num_channels)}
    gettable = ScheduleGettable(
        quantum_device=quantum_device,
        schedule_function=lambda x: x,
        schedule_kwargs={},
        real_imag=real_imag,
    )

    # act
    processed_data = gettable.process_acquired_data(
        mock_data, acq_metadata, repetitions=10
    )

    # assert
    assert len(processed_data) == 2 * num_channels


def test_ScheduleGettableSingleChannel_iterative_heterodyne_spec(
    mock_setup_basic_transmon, mocker
):
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

    acq_metadata = AcquisitionMetadata(
        acq_protocol="ssb_integration_complex",
        bin_mode=BinMode.AVERAGE,
        acq_return_type=complex,
        acq_indices={0: [0]},
    )

    data = 1 * np.exp(1j * np.deg2rad(45))

    acq_indices_data = _reshape_array_into_acq_return_type(data, acq_metadata)

    mocker.patch.object(
        mock_setup_basic_transmon["instrument_coordinator"],
        "retrieve_acquisition",
        return_value=acq_indices_data,
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

    exp_data = np.ones(len(freqs)) * data
    # Assert that the data is coming out correctly.
    np.testing.assert_array_equal(dset.x0, freqs)
    np.testing.assert_array_equal(dset.y0, abs(exp_data))
    np.testing.assert_array_equal(dset.y1, np.angle(exp_data, deg=True))


# test a batched case
def test_ScheduleGettableSingleChannel_batched_allxy(mock_setup_basic_transmon, mocker):
    meas_ctrl = mock_setup_basic_transmon["meas_ctrl"]
    quantum_device = mock_setup_basic_transmon["quantum_device"]

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
    comp_allxy_sched = qcompile(sched, quantum_device.generate_device_config())
    data = np.concatenate(
        (
            0 * np.ones(5 * 2),
            0.5 * np.ones(12 * 2),
            np.ones(4 * 2),
        )
    ) * np.exp(1j * np.deg2rad(45))

    acq_indices_data = _reshape_array_into_acq_return_type(
        data, extract_acquisition_metadata_from_schedule(comp_allxy_sched)
    )

    mocker.patch.object(
        mock_setup_basic_transmon["instrument_coordinator"],
        "retrieve_acquisition",
        return_value=acq_indices_data,
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
def test_ScheduleGettableSingleChannel_append_readout_cal(
    mock_setup_basic_transmon, mocker
):
    meas_ctrl = mock_setup_basic_transmon["meas_ctrl"]
    quantum_device = mock_setup_basic_transmon["quantum_device"]

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
    comp_ssro_sched = qcompile(ssro_sched, quantum_device.generate_device_config())

    data = np.tile(np.arange(2), repetitions) * np.exp(1j)

    acq_indices_data = _reshape_array_into_acq_return_type(
        data, extract_acquisition_metadata_from_schedule(comp_ssro_sched)
    )

    mocker.patch.object(
        mock_setup_basic_transmon["instrument_coordinator"],
        "retrieve_acquisition",
        return_value=acq_indices_data,
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


def test_ScheduleGettableSingleChannel_trace_acquisition(
    mock_setup_basic_transmon, mocker
):
    meas_ctrl = mock_setup_basic_transmon["meas_ctrl"]
    quantum_device = mock_setup_basic_transmon["quantum_device"]
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
    exp_trace = np.ones(len(sample_times)) * np.exp(1j * np.deg2rad(35))

    exp_data = {
        AcquisitionIndexing(acq_channel=0, acq_index=0): (
            exp_trace.real,
            exp_trace.imag,
        )
    }

    mocker.patch.object(
        mock_setup_basic_transmon["instrument_coordinator"],
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


def test_ScheduleGettable_generate_diagnostic(mock_setup_basic_transmon, mocker):
    schedule_kwargs = {"times": np.linspace(1e-6, 50e-6, 50), "qubit": "q0"}
    quantum_device = mock_setup_basic_transmon["quantum_device"]

    # Prepare the mock data the t1 schedule
    acq_metadata = AcquisitionMetadata(
        acq_protocol="ssb_integration_complex",
        bin_mode=BinMode.AVERAGE,
        acq_return_type=complex,
        acq_indices={0: range(50)},
    )

    data = np.ones(50) * np.exp(1j * np.deg2rad(45))

    acq_indices_data = _reshape_array_into_acq_return_type(data, acq_metadata)

    mocker.patch.object(
        mock_setup_basic_transmon["instrument_coordinator"],
        "retrieve_acquisition",
        return_value=acq_indices_data,
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

    filename = gettable.generate_diagnostics_report(execute_get=True)

    assert gettable.is_initialized is True

    with zipfile.ZipFile(filename, mode="r") as zf:
        dev_cfg = json.loads(zf.read("device_cfg.json").decode())
        hw_cfg = json.loads(zf.read("hardware_cfg.json").decode())
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
    assert gettable._compiled_schedule == qcompile(
        sched, device_cfg=dev_cfg, hardware_cfg=hw_cfg
    )


# this is probably useful somewhere, it illustrates the reshaping in the
# instrument coordinator
def _reshape_array_into_acq_return_type(
    data: np.ndarray, acq_metadata: AcquisitionMetadata
) -> Dict[Tuple[int, int], Any]:
    """
    Takes one ore more complex valued arrays and reshapes the data into a dictionary
    with AcquisitionIndexing
    """

    # Temporary. Will probably be replaced by an xarray object
    # See quantify-core#187, quantify-core#233, quantify-scheduler#36
    acquisitions = dict()

    # if len is 1, we have only 1 channel in the retrieved data
    if len(np.shape(data)) == 0:
        for acq_channel, acq_indices in acq_metadata.acq_indices.items():
            for acq_index in acq_indices:
                acqs = {
                    AcquisitionIndexing(acq_channel, acq_index): (
                        data.real,
                        data.imag,
                    )
                }
                acquisitions.update(acqs)
    elif len(np.shape(data)) == 1:
        for acq_channel, acq_indices in acq_metadata.acq_indices.items():
            for acq_index in acq_indices:
                acqs = {
                    AcquisitionIndexing(acq_channel, acq_index): (
                        data[acq_index].real,
                        data[acq_index].imag,
                    )
                }
                acquisitions.update(acqs)
    else:
        for acq_channel, acq_indices in acq_metadata.acq_indices.items():
            for acq_index in acq_indices:
                acqs = {
                    AcquisitionIndexing(acq_channel, acq_index): (
                        data[acq_channel, acq_index].real,
                        data[acq_channel, acq_index].imag,
                    )
                }
                acquisitions.update(acqs)
    return acquisitions


def test_profiling(mock_setup_basic_transmon, tmp_test_data_dir):
    set_standard_params_transmon(mock_setup_basic_transmon)
    quantum_device = mock_setup_basic_transmon["quantum_device"]
    qubit = mock_setup_basic_transmon["q0"]

    schedule_kwargs = {
        "pulse_amp": qubit.measure.pulse_amp(),
        "pulse_duration": qubit.measure.pulse_duration(),
        "frequency": qubit.clock_freqs.readout(),
        "qubit": "q0",
    }
    prof_gettable = ProfiledScheduleGettable(
        quantum_device=quantum_device,
        schedule_function=rabi_sched,
        schedule_kwargs=schedule_kwargs,
    )

    prof_gettable.initialize()
    instr_coordinator = (
        prof_gettable.quantum_device.instr_instrument_coordinator.get_instr()
    )
    instr_coordinator.start()
    instr_coordinator.wait_done()
    instr_coordinator.retrieve_acquisition()
    instr_coordinator.stop()
    prof_gettable.close()

    # Test if all steps have been measured and have a value > 0
    log = prof_gettable.log_profile()
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
    prof_gettable.log_profile(
        obj=obj, path=path, filename=filename, indent=4, separators=(",", ": ")
    )
    assert os.path.getsize(os.path.join(path, filename)) > 0

    # Test plot function
    path = tmp_test_data_dir
    filename = "average_runtimes.pdf"
    prof_gettable.plot_profile(path=path)
    assert prof_gettable.plot is not None
    assert os.path.getsize(os.path.join(path, filename)) > 0
