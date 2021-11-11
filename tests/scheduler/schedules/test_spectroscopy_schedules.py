# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

import tempfile

from quantify_core.data.handling import set_datadir

from quantify_scheduler.compilation import determine_absolute_timing, qcompile
from quantify_scheduler.schedules import spectroscopy_schedules as sps

from .compiles_all_backends import _CompilesAllBackends

# TODO to be replaced with fixture in tests/fixtures/schedule from !49 # pylint: disable=fixme
tmp_dir = tempfile.TemporaryDirectory()


class TestHeterodyneSpecSchedule(_CompilesAllBackends):
    @classmethod
    def setup_class(cls):
        set_datadir(tmp_dir.name)
        cls.sched_kwargs = {
            "pulse_amp": 0.15,
            "pulse_duration": 1e-6,
            "port": "q0:res",
            "clock": "q0.ro",
            "frequency": 4.48e9,
            "integration_time": 1e-6,
            "acquisition_delay": 220e-9,
            "init_duration": 18e-6,
            "repetitions": 10,
        }

        cls.sched = sps.heterodyne_spec_sched(**cls.sched_kwargs)

    def test_repetitions(self):
        assert self.sched.repetitions == self.sched_kwargs["repetitions"]

    def test_timing(self):
        sched = determine_absolute_timing(self.sched)
        # test that the right operations are added and timing is as expected.
        labels = ["buffer", "spec_pulse", "acquisition"]
        abs_times = [
            0,
            self.sched_kwargs["init_duration"],
            self.sched_kwargs["init_duration"] + self.sched_kwargs["acquisition_delay"],
        ]

        for i, constr in enumerate(sched.timing_constraints):
            assert constr["label"] == labels[i]
            assert constr["abs_time"] == abs_times[i]

    def test_compiles_device_cfg_only(self, load_example_transmon_config):
        # assert that files properly compile
        qcompile(self.sched, load_example_transmon_config())


class TestPulsedSpecSchedule(_CompilesAllBackends):
    @classmethod
    def setup_class(cls):
        set_datadir(tmp_dir.name)
        cls.sched_kwargs = {
            "spec_pulse_amp": 0.5,
            "spec_pulse_duration": 1e-6,
            "spec_pulse_port": "q0:mw",
            "spec_pulse_clock": "q0.01",
            "spec_pulse_frequency": 5.4e9,
            "ro_pulse_amp": 0.15,
            "ro_pulse_duration": 1e-6,
            "ro_pulse_delay": 1e-6,
            "ro_pulse_port": "q0:res",
            "ro_pulse_clock": "q0.ro",
            "ro_pulse_frequency": 4.48e9,
            "ro_integration_time": 1e-6,
            "ro_acquisition_delay": 220e-9,
            "init_duration": 18e-6,
            "repetitions": 10,
        }

        cls.sched = sps.two_tone_spec_sched(**cls.sched_kwargs)

    def test_repetitions(self):
        assert self.sched.repetitions == self.sched_kwargs["repetitions"]

    def test_timing(self):
        sched = determine_absolute_timing(self.sched)

        # test that the right operations are added and timing is as expected.
        labels = ["buffer", "spec_pulse", "readout_pulse", "acquisition"]

        t2 = (
            self.sched_kwargs["init_duration"]
            + self.sched_kwargs["spec_pulse_duration"]
            + self.sched_kwargs["ro_pulse_delay"]
        )
        t3 = t2 + self.sched_kwargs["ro_acquisition_delay"]
        abs_times = [0, self.sched_kwargs["init_duration"], t2, t3]

        for i, constr in enumerate(sched.timing_constraints):
            assert constr["label"] == labels[i]
            assert constr["abs_time"] == abs_times[i]

    def test_compiles_device_cfg_only(self, load_example_transmon_config):
        # assert that files properly compile
        qcompile(self.sched, load_example_transmon_config())
