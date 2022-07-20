# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=too-many-locals
# pylint: disable=invalid-name

from collections import namedtuple
from typing import Any, Dict, Tuple

import numpy as np
import pytest
from qcodes.instrument.parameter import ManualParameter

from quantify_scheduler.compilation import qcompile
from quantify_scheduler.enums import BinMode
from quantify_scheduler.gettables import ScheduleGettable
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


@pytest.mark.parametrize("num_channels, real_imag", [(1, True), (2, False), (10, True)])
def test_process_acquired_data(mock_setup, num_channels: int, real_imag: bool):
    # arrange
    quantum_device = mock_setup["quantum_device"]
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
        mock_setup["instrument_coordinator"],
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

    sched_gettable = ScheduleGettable(
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


from tests.scheduler.instrument_coordinator.components.test_qblox import (
    make_cluster_component,
    make_qrm_component,
)


def test_trace_acquisition_measurement_control(
    mock_setup, mocker, make_cluster_component
):
    from quantify_scheduler import Schedule

    from quantify_scheduler.operations.gate_library import (
        Measure,
        Reset,
    )
    from qcodes import ManualParameter
    from quantify_scheduler.gettables import ScheduleGettable

    from quantify_core.data.handling import set_datadir
    import tempfile

    def raw_trace(
        qubit_name: str,
        repetitions: int = 1,
    ) -> Schedule:
        """
        Generate a schedule to perform raw trace acquisition. (New-style device element.)

        Parameters
        ----------
        qubit_name
            Name of a device element
        frequency :
            The frequency of the pulse and of the data acquisition [Hz].
        repetitions
            The amount of times the Schedule will be repeated.

        Returns
        -------
        :
            The Raw Trace acquisition Schedule.
        """
        schedule = Schedule("Raw trace acquisition", repetitions)
        schedule.add(Reset(qubit_name))
        schedule.add(Measure(qubit_name, acq_protocol="Trace"))
        return schedule

    tmp_dir = tempfile.TemporaryDirectory()
    set_datadir(tmp_dir.name)

    hardware_cfg = {
        "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
        "cluster0": {
            "ref": "internal",
            "instrument_type": "Cluster",
            "cluster0_module4": {
                "instrument_type": "QRM_RF",
                "complex_output_0": {
                    "line_gain_db": 0,
                    "portclock_configs": [
                        {"port": "q2:res", "clock": "q2.ro", "interm_freq": 50e6},
                    ],
                },
            },
        },
    }

    cluster0 = make_cluster_component("cluster0")

    print("CMM system status is \n", cluster0.instrument.get_system_state())
    print("correctly connected to qblox-cluster-MM.\n")

    cluster0.instrument.reset()

    instr_coordinator = mock_setup["instrument_coordinator"]
    instr_coordinator.add_component(cluster0)

    # utility instruments
    #############################

    meas_ctrl = mock_setup["meas_ctrl"]

    # Config management instruments
    #############################
    q2 = mock_setup["q2"]

    #####################################
    # 4 Loading settings onto instruments
    #####################################

    # Output attenuation of QRM-RF
    cluster0.instrument.module4.out0_att(50)

    quantum_device = mock_setup["quantum_device"]

    quantum_device.instr_measurement_control(meas_ctrl.name)
    quantum_device.instr_instrument_coordinator(instr_coordinator.name)

    quantum_device.hardware_config(hardware_cfg)

    q2.measure.acq_delay(600e-9)
    q2.clock_freqs.readout(7404000000.0)

    meas_ctrl = quantum_device.instr_measurement_control.get_instr()
    device_element = q2

    sample_par = ManualParameter("sample", label="Sample time", unit="s")
    sample_par.batched = True

    sched_gettable = ScheduleGettable(
        quantum_device=quantum_device,
        schedule_function=raw_trace,
        schedule_kwargs=dict(
            qubit_name=q2.name,
        ),
        batched=True,
    )

    # the sampling rate of the Qblox hardware
    sampling_rate = 1e9

    # in the Qblox hardware, the trace acquisition will always return 16384 samples. But the dummy returns 16383...
    sample_size = 16384
    if cluster0.instrument.get_idn().get("serial_number") == "whatever":
        sample_size = 16383

    sample_times = np.arange(
        start=0, stop=sample_size / sampling_rate, step=1 / sampling_rate
    )

    meas_ctrl.settables(sample_par)
    meas_ctrl.setpoints(sample_times)
    meas_ctrl.gettables(sched_gettable)

    mocker.patch.object(
        meas_ctrl, "_get_fracdone", side_effect=np.linspace(start=0, stop=1, num=6)
    )

    try:
        _ = meas_ctrl.run(f"Readout trace schedule of {q2.name}")
    except Exception as ex:
        import pprint

        print()
        pprint.pprint(sched_gettable.compiled_schedule.compiled_instructions)
        raise ex

    instr_coordinator.remove_component(cluster0.name)


def test_trace_acquisition_instrument_coordinator(
    mock_setup, make_cluster_component, make_qrm_component
):
    from quantify_scheduler.compilation import qcompile

    from quantify_scheduler.instrument_coordinator.components.qblox import QRMComponent
    from quantify_scheduler.operations.gate_library import Measure, Reset
    from quantify_scheduler.schedules.schedule import Schedule

    from quantify_core.data.handling import set_datadir
    import tempfile

    tmp_dir = tempfile.TemporaryDirectory()
    set_datadir(tmp_dir.name)

    instr_coordinator = mock_setup["instrument_coordinator"]

    cluster0 = make_cluster_component("cluster0")
    instr_coordinator.add_component(cluster0)

    qrm0 = make_qrm_component("qrm0")
    instr_coordinator.add_component(qrm0)

    hardware_cfg = {
        "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
        "cluster0": {
            "ref": "internal",
            "instrument_type": "Cluster",
            "cluster0_module4": {
                "instrument_type": "QRM_RF",
                "complex_output_0": {
                    "line_gain_db": 0,
                    "portclock_configs": [
                        {"port": "q2:res", "clock": "q2.ro", "interm_freq": 50e6}
                    ],
                },
            },
        },
    }

    # hardware_cfg = {
    #     "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
    #     "cluster0": {
    #         "ref": "internal",
    #         "instrument_type": "Cluster",
    #         "cluster0_module3": {
    #             "instrument_type": "QRM",
    #             "complex_output_0": {
    #                 "line_gain_db": 0,
    #                 "portclock_configs": [{"port": "q2:res", "clock": "q2.ro"}],
    #             },
    #         },
    #     },
    # }

    # hardware_cfg = {
    #     "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
    #     "qrm0": {
    #         "instrument_type": "Pulsar_QRM",
    #         "ref": "internal",
    #         "complex_output_0": {
    #             "line_gain_db": 0,
    #             "portclock_configs": [{"port": "q2:res", "clock": "q2.ro"}],
    #         },
    #     },
    # }

    q2 = mock_setup["q2"]
    q2.measure.acq_delay(600e-9)
    q2.clock_freqs.readout(7404000000.0)

    quantum_device = mock_setup["quantum_device"]
    quantum_device.hardware_config(hardware_cfg)

    qubit_name = "q2"
    schedule = Schedule("Raw trace acquisition", repetitions=1)
    schedule.add(Reset(qubit_name))
    schedule.add(Measure(qubit_name, acq_protocol="Trace"))

    compiled_sched = qcompile(
        schedule=schedule,
        device_cfg=quantum_device.generate_device_config(),
        hardware_cfg=hardware_cfg,
    )

    print()
    print(
        "qrm0.scope_acq_sequencer_select: {}".format(
            instr_coordinator.get_component("ic_qrm0").instrument.get(
                "scope_acq_sequencer_select"
            )
        )
    )
    print(
        "cluster0.module3.scope_acq_sequencer_select: {}".format(
            instr_coordinator.get_component("ic_cluster0").instrument.module3.get(
                "scope_acq_sequencer_select"
            )
        )
    )
    print(
        "cluster0.module4.scope_acq_sequencer_select: {}".format(
            instr_coordinator.get_component("ic_cluster0").instrument.module4.get(
                "scope_acq_sequencer_select"
            )
        )
    )

    try:
        instr_coordinator.prepare(compiled_sched)
    except Exception as ex:
        import pprint

        print()
        pprint.pprint(compiled_sched.compiled_instructions)
        raise ex

    instr_coordinator.remove_component(cluster0.name)
    instr_coordinator.remove_component(qrm0.name)


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
