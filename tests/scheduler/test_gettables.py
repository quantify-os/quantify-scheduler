# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
import os
from unittest import TestCase
from unittest.mock import Mock

import numpy as np
import pytest
from qcodes.instrument.parameter import ManualParameter
from xarray import DataArray, Dataset

from quantify_scheduler.backends import SerialCompiler
from quantify_scheduler.gettables import ScheduleGettable
from quantify_scheduler.gettables_profiled import ProfiledScheduleGettable
from quantify_scheduler.helpers.schedule import (
    extract_acquisition_metadata_from_schedule,
)
from quantify_scheduler.schedules.spectroscopy_schedules import (
    heterodyne_spec_sched,
    nv_dark_esr_sched,
)
from quantify_scheduler.schedules.timedomain_schedules import (
    allxy_sched,
    rabi_sched,
    readout_calibration_sched,
)
from quantify_scheduler.schedules.trace_schedules import trace_schedule


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
                {"acq_protocol": "SSBIntegrationComplex"},
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
def test_schedule_gettable_batched_allxy(mock_setup_basic_transmon_with_standard_params, mocker):
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
    first_acq_channel = next(iter(acq_metadata.acq_channels_metadata.values()))
    acq_channel, acq_indices = (
        first_acq_channel.acq_channel,
        first_acq_channel.acq_indices,
    )
    # SSBIntegrationComplex, bin_mode.AVERAGE
    expected_data = Dataset(
        {
            acq_channel: (
                [f"acq_index_{acq_channel}"],
                data.reshape((len(acq_indices),)),
                {"acq_protocol": "SSBIntegrationComplex"},
            ),
        }
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

    data = (np.tile(np.arange(2, dtype=np.float64), repetitions) * np.exp(1j)).astype(np.complex64)

    acq_metadata = extract_acquisition_metadata_from_schedule(comp_ssro_sched)
    first_acq_channel = next(iter(acq_metadata.acq_channels_metadata.values()))
    acq_channel, acq_indices = (
        first_acq_channel.acq_channel,
        first_acq_channel.acq_indices,
    )
    # SSBIntegrationComplex, BinMode.APPEND
    expected_data = Dataset(
        {
            acq_channel: (
                ["a_repetition_index", "an_acq_index"],
                data.reshape((repetitions, len(acq_indices))),
                {"acq_protocol": "SSBIntegrationComplex"},
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
    exp_trace = (np.ones(len(sample_times)) * np.exp(1j * np.deg2rad(35))).astype(np.complex64)

    exp_data_array = DataArray(
        [exp_trace],
        coords=[[0], range(len(exp_trace))],
        dims=["repetition", "acq_index"],
        attrs={"acq_protocol": "SSBIntegrationComplex"},
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
    profiled_ic = profiled_gettable.quantum_device.instr_instrument_coordinator.get_instr()
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
        attrs={"acq_protocol": "TriggerCount"},
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


def test_schedule_gettable_no_hardware_cfg_raises(mock_setup_basic_transmon):
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

    ro_freqs = np.linspace(5e9, 6e9, 11)
    meas_ctrl.settables(ro_freq)
    meas_ctrl.setpoints(ro_freqs)
    meas_ctrl.gettables(spec_gettable)
    with pytest.raises(RuntimeError) as exc:
        _ = meas_ctrl.run(f"Heterodyne spectroscopy {qubit.name}")
    assert (
        f"InstrumentCoordinator.retrieve_acquisition() "
        f"('{mock_setup_basic_transmon['instrument_coordinator'].name}') did not "
        f"return any data, but was expected to return data." in str(exc.value)
    )
