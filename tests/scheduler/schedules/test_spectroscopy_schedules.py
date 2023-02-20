# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

import numpy as np
import pytest
from qcodes.instrument.parameter import ManualParameter

from quantify_scheduler.backends import SerialCompiler
from quantify_scheduler.compilation import determine_absolute_timing
from quantify_scheduler.device_under_test.quantum_device import QuantumDevice
from quantify_scheduler.device_under_test.nv_element import BasicElectronicNVElement
from quantify_scheduler.enums import BinMode
from quantify_scheduler.gettables import ScheduleGettable
from quantify_scheduler.schedules import spectroscopy_schedules as sps
from quantify_scheduler.schedules.schedule import AcquisitionMetadata

from tests.scheduler.instrument_coordinator.components.test_qblox import (  # pylint: disable=unused-import
    make_cluster_component,
)
from tests.scheduler.schedules.compiles_all_backends import _CompilesAllBackends
from tests.scheduler.test_gettables import _reshape_array_into_acq_return_type


class TestHeterodyneSpecSchedule(_CompilesAllBackends):
    @classmethod
    def setup_class(cls):
        cls.sched_kwargs = {
            "pulse_amp": 0.15,
            "pulse_duration": 1e-6,
            "port": "q0:res",
            "clock": "q0.ro",
            "frequency": 7.04e9,
            "integration_time": 1e-6,
            "acquisition_delay": 220e-9,
            "init_duration": 18e-6,
            "repetitions": 10,
        }
        cls.uncomp_sched = sps.heterodyne_spec_sched(**cls.sched_kwargs)

    def test_repetitions(self):
        assert self.uncomp_sched.repetitions == self.sched_kwargs["repetitions"]

    def test_timing(self):
        """Test that the right operations are added and timing is as expected."""
        sched = determine_absolute_timing(self.uncomp_sched)

        labels = ["buffer", "spec_pulse", "acquisition"]
        abs_times = [
            0,
            self.sched_kwargs["init_duration"],
            self.sched_kwargs["init_duration"] + self.sched_kwargs["acquisition_delay"],
        ]

        for i, schedulable in enumerate(sched.schedulables.values()):
            assert schedulable["label"] == labels[i]
            assert schedulable["abs_time"] == abs_times[i]

    def test_compiles_device_cfg_only(self, device_compile_config_basic_transmon):
        # assert that files properly compile
        compiler = SerialCompiler(name="compiler")
        compiler.compile(
            schedule=self.uncomp_sched, config=device_compile_config_basic_transmon
        )


class TestHeterodyneSpecScheduleNCO(TestHeterodyneSpecSchedule):
    @classmethod
    def setup_class(cls):
        cls.sched_kwargs = {
            "pulse_amp": 0.15,
            "pulse_duration": 1e-6,
            "port": "q0:res",
            "clock": "q0.ro",
            "frequencies": np.linspace(start=6.8e9, stop=7.2e9, num=5),
            "integration_time": 1e-6,
            "acquisition_delay": 220e-9,
            "init_duration": 18e-6,
            "repetitions": 10,
        }
        cls.uncomp_sched = sps.heterodyne_spec_sched_nco(**cls.sched_kwargs)

    def test_timing(self):
        """Test that the right operations are added and timing is as expected."""
        sched = determine_absolute_timing(self.uncomp_sched)

        labels = ["buffer", "set_freq", "spec_pulse", "acquisition"]
        labels *= len(self.sched_kwargs["frequencies"])

        rel_times = [
            self.sched_kwargs["init_duration"],
            8e-9,
            self.sched_kwargs["acquisition_delay"],
            self.sched_kwargs["integration_time"],
        ]
        rel_times *= len(self.sched_kwargs["frequencies"])

        abs_time = 0.0
        for i, schedulable in enumerate(sched.schedulables.values()):
            assert labels[i] in schedulable["label"]
            assert np.isclose(
                schedulable["abs_time"], abs_time, atol=0.0, rtol=1e-15
            ), schedulable["label"]
            abs_time += rel_times[i]

    @pytest.mark.xfail(reason="SetClockFrequency not supported in Zhinst backend")
    def test_compiles_zi_backend(
        self, compile_config_basic_transmon_zhinst_hardware
    ) -> None:
        _CompilesAllBackends.test_compiles_zi_backend(
            self, compile_config_basic_transmon_zhinst_hardware
        )


def test_heterodyne_spec_sched_nco__qblox_hardware(
    mock_setup_basic_transmon_with_standard_params, make_cluster_component, mocker
):
    cluster_name = "cluster0"
    hardware_cfg = {
        "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
        f"{cluster_name}": {
            "ref": "internal",
            "instrument_type": "Cluster",
            f"{cluster_name}_module4": {
                "instrument_type": "QRM_RF",
                "complex_output_0": {
                    "lo_freq": 5e9,
                    "portclock_configs": [
                        {"port": "q0:res", "clock": "q0.ro", "interm_freq": None},
                    ],
                },
            },
        },
    }

    quantum_device = mock_setup_basic_transmon_with_standard_params["quantum_device"]
    quantum_device.hardware_config(hardware_cfg)

    ic_cluster0 = make_cluster_component(cluster_name)
    ic = mock_setup_basic_transmon_with_standard_params["instrument_coordinator"]
    ic.add_component(ic_cluster0)

    # Manual parameter for batched schedule
    ro_freq = ManualParameter("ro_freq", unit="Hz")
    ro_freq.batched = True
    ro_freqs = np.linspace(start=4.5e9, stop=5.5e9, num=11)
    quantum_device.cfg_sched_repetitions(5)

    # Configure the gettable
    qubit = quantum_device.get_element("q0")
    schedule_kwargs = {
        "pulse_amp": qubit.measure.pulse_amp(),
        "pulse_duration": qubit.measure.pulse_duration(),
        "frequencies": ro_freqs,
        "acquisition_delay": qubit.measure.acq_delay(),
        "integration_time": qubit.measure.integration_time(),
        "port": qubit.ports.readout(),
        "clock": qubit.name + ".ro",
        "init_duration": qubit.reset.duration(),
    }
    spec_gettable = ScheduleGettable(
        quantum_device=quantum_device,
        schedule_function=sps.heterodyne_spec_sched_nco,
        schedule_kwargs=schedule_kwargs,
        real_imag=False,
        batched=True,
    )
    assert spec_gettable.is_initialized is False

    # Prepare the mock data the spectroscopy schedule
    acq_metadata = AcquisitionMetadata(
        acq_protocol="ssb_integration_complex",
        bin_mode=BinMode.AVERAGE,
        acq_return_type=complex,
        acq_indices={0: [*range(len(ro_freqs))]},
        repetitions=quantum_device.cfg_sched_repetitions,
    )
    data = 1 * np.exp(1j * np.deg2rad(45))
    acq_indices_data = _reshape_array_into_acq_return_type(
        data=data, acq_metadata=acq_metadata
    )
    mocker.patch.object(
        ic,
        "retrieve_acquisition",
        return_value=acq_indices_data,
    )

    # Run the schedule
    meas_ctrl = mock_setup_basic_transmon_with_standard_params["meas_ctrl"]
    meas_ctrl.settables(ro_freq)
    meas_ctrl.setpoints(ro_freqs)
    meas_ctrl.gettables(spec_gettable)
    dataset = meas_ctrl.run(
        name=f"Fast heterodyne spectroscopy (NCO sweep) {qubit.name}"
    )
    assert spec_gettable.is_initialized is True

    # Assert that the data is coming out correctly
    exp_data = np.ones(len(ro_freqs)) * data
    np.testing.assert_array_equal(dataset.x0, ro_freqs)
    np.testing.assert_array_equal(dataset.y0, abs(exp_data))
    np.testing.assert_array_equal(dataset.y1, np.angle(exp_data, deg=True))


class TestTwoToneSpecSchedule(_CompilesAllBackends):
    @classmethod
    def setup_class(cls):
        cls.sched_kwargs = {
            "spec_pulse_amp": 0.5,
            "spec_pulse_duration": 1e-6,
            "spec_pulse_port": "q0:mw",
            "spec_pulse_clock": "q0.01",
            "spec_pulse_frequency": 6.02e9,
            "ro_pulse_amp": 0.15,
            "ro_pulse_duration": 1e-6,
            "ro_pulse_delay": 1e-6,
            "ro_pulse_port": "q0:res",
            "ro_pulse_clock": "q0.ro",
            "ro_pulse_frequency": 7.04e9,
            "ro_integration_time": 1e-6,
            "ro_acquisition_delay": 220e-9,
            "init_duration": 18e-6,
            "repetitions": 10,
        }
        cls.uncomp_sched = sps.two_tone_spec_sched(**cls.sched_kwargs)

    def test_repetitions(self):
        assert self.uncomp_sched.repetitions == self.sched_kwargs["repetitions"]

    def test_timing(self):
        """Test that the right operations are added and timing is as expected."""
        sched = determine_absolute_timing(self.uncomp_sched)

        labels = ["buffer", "spec_pulse", "readout_pulse", "acquisition"]

        t2 = (
            self.sched_kwargs["init_duration"]
            + self.sched_kwargs["spec_pulse_duration"]
            + self.sched_kwargs["ro_pulse_delay"]
        )
        t3 = t2 + self.sched_kwargs["ro_acquisition_delay"]
        abs_times = [0, self.sched_kwargs["init_duration"], t2, t3]

        for i, schedulable in enumerate(sched.schedulables.values()):
            assert schedulable["label"] == labels[i]
            assert schedulable["abs_time"] == abs_times[i]

    def test_compiles_device_cfg_only(self, device_compile_config_basic_transmon):
        # assert that files properly compile
        compiler = SerialCompiler(name="compiler")
        compiler.compile(
            schedule=self.uncomp_sched, config=device_compile_config_basic_transmon
        )


class TestTwoToneSpecScheduleNCO(TestTwoToneSpecSchedule):
    @classmethod
    def setup_class(cls):
        cls.sched_kwargs = {
            "spec_pulse_amp": 0.5,
            "spec_pulse_duration": 1e-6,
            "spec_pulse_port": "q0:mw",
            "spec_pulse_clock": "q0.01",
            "spec_pulse_frequencies": np.linspace(start=6.8e9, stop=7.2e9, num=5),
            "ro_pulse_amp": 0.15,
            "ro_pulse_duration": 1e-6,
            "ro_pulse_delay": 1e-6,
            "ro_pulse_port": "q0:res",
            "ro_pulse_clock": "q0.ro",
            "ro_pulse_frequency": 7.04e9,
            "ro_integration_time": 1e-6,
            "ro_acquisition_delay": 220e-9,
            "init_duration": 18e-6,
            "repetitions": 10,
        }
        cls.uncomp_sched = sps.two_tone_spec_sched_nco(**cls.sched_kwargs)

    def test_timing(self):
        """Test that the right operations are added and timing is as expected."""
        sched = determine_absolute_timing(self.uncomp_sched)

        labels = ["buffer", "set_freq", "spec_pulse", "readout_pulse", "acquisition"]
        labels *= len(self.sched_kwargs["spec_pulse_frequencies"])

        rel_times = [
            self.sched_kwargs["init_duration"],
            8e-9,
            self.sched_kwargs["spec_pulse_duration"]
            + self.sched_kwargs["ro_pulse_delay"],
            self.sched_kwargs["ro_acquisition_delay"],
            self.sched_kwargs["ro_integration_time"],
        ]
        rel_times *= len(self.sched_kwargs["spec_pulse_frequencies"])

        abs_time = 0.0
        for i, schedulable in enumerate(sched.schedulables.values()):
            assert labels[i] in schedulable["label"]
            assert np.isclose(
                schedulable["abs_time"], abs_time, atol=0.0, rtol=1e-15
            ), schedulable["label"]
            abs_time += rel_times[i]

    @pytest.mark.xfail(reason="SetClockFrequency not supported in Zhinst backend")
    def test_compiles_zi_backend(
        self, compile_config_basic_transmon_zhinst_hardware
    ) -> None:
        _CompilesAllBackends.test_compiles_zi_backend(
            self, compile_config_basic_transmon_zhinst_hardware
        )


class TestNVDarkESRSched:
    @classmethod
    def setup_class(cls):
        cls.sched_kwargs = {
            "qubit": "qe0",
            "repetitions": 10,
        }
        cls.uncomp_sched = sps.nv_dark_esr_sched(**cls.sched_kwargs)

    def test_repetitions(self):
        assert self.uncomp_sched.repetitions == self.sched_kwargs["repetitions"]

    def test_timing(self, mock_setup_basic_nv):
        # Arrange
        quantum_device: QuantumDevice = mock_setup_basic_nv["quantum_device"]
        qe0: BasicElectronicNVElement = mock_setup_basic_nv["qe0"]

        # For operations, whose duration is not trivial to calculate, use values that
        # allow to easily predict the duration of the operations (used below when
        # constructing abs_times).
        qe0.cr_count.acq_delay(0)
        qe0.cr_count.acq_duration(1e-6)
        qe0.cr_count.readout_pulse_duration(1e-6)
        qe0.cr_count.spinpump_pulse_duration(1e-6)
        qe0.measure.pulse_duration(2e-6)
        qe0.measure.acq_duration(2e-6)

        # Act
        compiler = SerialCompiler(name="compiler")
        sched = compiler.compile(
            schedule=self.uncomp_sched,
            config=quantum_device.generate_compilation_config(),
        )

        # Assert
        abs_times = [0]
        abs_times.append(abs_times[-1] + qe0.charge_reset.duration())
        abs_times.append(abs_times[-1] + qe0.cr_count.acq_duration())
        abs_times.append(abs_times[-1] + qe0.reset.duration())
        abs_times.append(abs_times[-1] + qe0.spectroscopy_operation.duration())
        abs_times.append(abs_times[-1] + qe0.measure.acq_duration())
        abs_times.append(abs_times[-1] + qe0.cr_count.acq_duration())

        for i, schedulable in enumerate(sched.schedulables.values()):
            assert schedulable["abs_time"] == abs_times[i]

    def test_compiles_device_cfg_only(self, mock_setup_basic_nv):
        # assert that files properly compile
        compiler = SerialCompiler(name="compiler")
        compiler.compile(
            schedule=self.uncomp_sched,
            config=mock_setup_basic_nv["quantum_device"].generate_compilation_config(),
        )

    def test_compiles_qblox_backend(self, mock_setup_basic_nv_qblox_hardware) -> None:
        # assert that files properly compile
        quantum_device: QuantumDevice = mock_setup_basic_nv_qblox_hardware[
            "quantum_device"
        ]
        compiler = SerialCompiler(name="compiler")

        schedule = compiler.compile(
            schedule=self.uncomp_sched,
            config=quantum_device.generate_compilation_config(),
        )
        assert not schedule.compiled_instructions == {}
