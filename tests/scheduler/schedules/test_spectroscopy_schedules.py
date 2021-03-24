import inspect
import os
import json
import tempfile
import quantify.scheduler.schemas.examples as es
from quantify.scheduler.schedules import spectroscopy_schedules as sps
from quantify.scheduler.compilation import determine_absolute_timing, qcompile
from quantify.data.handling import set_datadir

tmp_dir = tempfile.TemporaryDirectory()
set_datadir(tmp_dir.name)

esp = inspect.getfile(es)

cfg_f = os.path.abspath(os.path.join(esp, "..", "transmon_test_config.json"))
with open(cfg_f, "r") as f:
    DEVICE_CFG = json.load(f)

map_f = os.path.abspath(os.path.join(esp, "..", "qblox_test_mapping.json"))
with open(map_f, "r") as f:
    HARDWARE_MAPPING = json.load(f)



def test_heterodynce_spec_schedule():
    pulse_amp = 0.15
    pulse_duration = 1e-6
    port = "q0:res"
    clock = "q0.ro"
    frequency = 4.48e9
    integration_time = 1e-6
    acquisition_delay = 220e-9
    buffer_time = 18e-6

    sched = sps.heterodyne_spec_sched(
        pulse_amp=pulse_amp,
        pulse_duration=pulse_duration,
        port=port,
        clock=clock,
        frequency=frequency,
        integration_time=integration_time,
        acquisition_delay=acquisition_delay,
        buffer_time=buffer_time,
    )

    sched = determine_absolute_timing(sched)
    # test that the right operations are added and timing is as expected.
    labels = ["buffer", "spec_pulse", "acquisition"]
    abs_times = [0, buffer_time, buffer_time + acquisition_delay]

    for i, constr in enumerate(sched.timing_constraints):
        assert constr["label"] == labels[i]
        assert constr["abs_time"] == abs_times[i]

    # assert that files properly compile
    qcompile(sched, DEVICE_CFG, HARDWARE_MAPPING)


def test_pulsed_spec_schedule():
    spec_pulse_amp = 0.5
    spec_pulse_duration = 1e-6
    spec_pulse_port = "q0:mw"
    spec_pulse_clock = "q0.01"
    spec_pulse_frequency = 5.4e9
    ro_pulse_amp = 0.15
    ro_pulse_duration = 1e-6
    ro_pulse_delay = 1e-6
    ro_pulse_port = "q0:res"
    ro_pulse_clock = "q0.ro"
    ro_pulse_frequency = 4.48e9
    ro_integration_time = 1e-6
    ro_acquisition_delay = 220e-9
    buffer_time = 18e-6

    sched = sps.two_tone_spec_sched(
        spec_pulse_amp=spec_pulse_amp,
        spec_pulse_duration=spec_pulse_duration,
        spec_pulse_port=spec_pulse_port,
        spec_pulse_clock=spec_pulse_clock,
        spec_pulse_frequency=spec_pulse_frequency,
        ro_pulse_amp=ro_pulse_amp,
        ro_pulse_duration=ro_pulse_duration,
        ro_pulse_delay=ro_pulse_delay,
        ro_pulse_port=ro_pulse_port,
        ro_pulse_clock=ro_pulse_clock,
        ro_pulse_frequency=ro_pulse_frequency,
        ro_acquisition_delay=ro_acquisition_delay,
        ro_integration_time=ro_integration_time,
        buffer_time=buffer_time,
    )

    sched = determine_absolute_timing(sched)
    # test that the right operations are added and timing is as expected.
    labels = ["buffer", "spec_pulse", "readout_pulse", "acquisition"]

    t2 = buffer_time + spec_pulse_duration + ro_pulse_delay
    t3 = t2 + ro_acquisition_delay
    abs_times = [0, buffer_time, t2, t3]

    for i, constr in enumerate(sched.timing_constraints):
        assert constr["label"] == labels[i]
        assert constr["abs_time"] == abs_times[i]

    # assert that files properly compile
    qcompile(sched, DEVICE_CFG, HARDWARE_MAPPING)

