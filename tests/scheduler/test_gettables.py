# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=too-many-locals
# pylint: disable=invalid-name

from collections import namedtuple
from typing import Any, Dict, Tuple

import numpy as np
from qcodes.instrument.parameter import ManualParameter

from quantify_scheduler.compilation import qcompile
from quantify_scheduler.enums import BinMode
from quantify_scheduler.gettables import ScheduleGettableSingleChannel
from quantify_scheduler.helpers.schedule import (
    extract_acquisition_metadata_from_schedule,
)
from quantify_scheduler.schedules.schedule import AcquisitionMetadata
from quantify_scheduler.schedules.spectroscopy_schedules import heterodyne_spec_sched
from quantify_scheduler.schedules.timedomain_schedules import (
    allxy_sched,
    readout_calibration_sched,
)
from quantify_scheduler.schedules.trace_schedules import trace_schedule

# this is taken from the qblox backend and is used to make the tuple indexing of
# acquisitions more explicit. See also #179 of quantify-scheduler
AcquisitionIndexing = namedtuple("AcquisitionIndexing", "acq_channel acq_index")


def test_ScheduleGettableSingleChannel_iterative_heterodyne_spec(mock_setup, mocker):
    meas_ctrl = mock_setup["meas_ctrl"]
    quantum_device = mock_setup["quantum_device"]

    qubit = quantum_device.get_component("q0")

    # manual parameter for testing purposes
    ro_freq = ManualParameter("ro_freq", initial_value=5e9, unit="Hz")

    schedule_kwargs = {
        "pulse_amp": qubit.ro_pulse_amp,
        "pulse_duration": qubit.ro_pulse_duration,
        "frequency": ro_freq,
        "acquisition_delay": qubit.ro_acq_delay,
        "integration_time": qubit.ro_acq_integration_time,
        "port": qubit.ro_port,
        "clock": qubit.ro_clock,
        "init_duration": qubit.init_duration,
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
        mock_setup["instrument_coordinator"],
        "retrieve_acquisition",
        return_value=acq_indices_data,
    )

    # Configure the gettable
    spec_gettable = ScheduleGettableSingleChannel(
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
def test_ScheduleGettableSingleChannel_batched_allxy(mock_setup, mocker):
    meas_ctrl = mock_setup["meas_ctrl"]
    quantum_device = mock_setup["quantum_device"]

    qubit = quantum_device.get_component("q0")

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
    data = (
        np.concatenate(
            (
                0 * np.ones(5 * 2),
                0.5 * np.ones(12 * 2),
                np.ones(4 * 2),
            )
        )
        * np.exp(1j * np.deg2rad(45))
    )

    acq_indices_data = _reshape_array_into_acq_return_type(
        data, extract_acquisition_metadata_from_schedule(comp_allxy_sched)
    )

    mocker.patch.object(
        mock_setup["instrument_coordinator"],
        "retrieve_acquisition",
        return_value=acq_indices_data,
    )

    # Configure the gettable

    allxy_gettable = ScheduleGettableSingleChannel(
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
def test_ScheduleGettableSingleChannel_append_readout_cal(mock_setup, mocker):
    meas_ctrl = mock_setup["meas_ctrl"]
    quantum_device = mock_setup["quantum_device"]

    repetitions = 256
    qubit = quantum_device.get_component("q0")

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
        mock_setup["instrument_coordinator"],
        "retrieve_acquisition",
        return_value=acq_indices_data,
    )

    # Configure the gettable

    ssro_gettable = ScheduleGettableSingleChannel(
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


def test_ScheduleGettableSingleChannel_trace_acquisition(mock_setup, mocker):
    meas_ctrl = mock_setup["meas_ctrl"]
    quantum_device = mock_setup["quantum_device"]
    # q0 is a  device element from the test setup has all the right params
    device_element = quantum_device.get_component("q0")

    sample_par = ManualParameter("sample", label="Sample time", unit="s")
    sample_par.batched = True

    schedule_kwargs = {
        "pulse_amp": device_element.ro_pulse_amp,
        "pulse_duration": device_element.ro_pulse_duration,
        "pulse_delay": device_element.ro_pulse_delay,
        "frequency": device_element.ro_freq,
        "acquisition_delay": device_element.ro_acq_delay,
        "integration_time": device_element.ro_acq_integration_time,
        "port": device_element.ro_port,
        "clock": device_element.ro_clock,
        "init_duration": device_element.init_duration,
    }

    sched_gettable = ScheduleGettableSingleChannel(
        quantum_device=quantum_device,
        schedule_function=trace_schedule,
        schedule_kwargs=schedule_kwargs,
        batched=True,
    )

    sample_times = np.arange(0, device_element.ro_acq_integration_time(), 1 / 1e9)
    exp_trace = np.ones(len(sample_times)) * np.exp(1j * np.deg2rad(35))

    exp_data = {
        AcquisitionIndexing(acq_channel=0, acq_index=0): (
            exp_trace.real,
            exp_trace.imag,
        )
    }

    mocker.patch.object(
        mock_setup["instrument_coordinator"],
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
