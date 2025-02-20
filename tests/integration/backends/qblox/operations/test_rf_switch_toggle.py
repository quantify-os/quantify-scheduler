# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Integration test for the RF Switch toggle"""

import pytest

from quantify_scheduler import (
    BasicTransmonElement,
    ClockResource,
    QuantumDevice,
    Schedule,
    SerialCompiler,
)
from quantify_scheduler.backends.qblox.operations.rf_switch_toggle import RFSwitchToggle
from quantify_scheduler.operations import IdlePulse, MarkerPulse, X

# To be used in pytest parameterised
# qubit, module, marker index
_parameters = [
    ("q0", 6, 1),
    ("q1", 6, 2),
    ("q2", 5, 2),
]


@pytest.fixture
def rf_device():
    hw_config = {
        "config_type": "quantify_scheduler.backends.qblox_backend.QbloxHardwareCompilationConfig",
        "hardware_description": {
            "cluster0": {
                "instrument_type": "Cluster",
                "modules": {
                    "5": {"instrument_type": "QRM_RF", "rf_output_on": False},
                    "6": {"instrument_type": "QCM_RF", "rf_output_on": False},
                    "7": {"instrument_type": "QRM_RF", "rf_output_on": True},
                    "8": {"instrument_type": "QCM_RF", "rf_output_on": True},
                },
                "sequence_to_file": False,
                "ref": "internal",
            }
        },
        "hardware_options": {
            "modulation_frequencies": {
                "q0:mw-q0.01": {"lo_freq": 3.8e9},
                "q1:mw-q1.01": {"lo_freq": 3.8e9},
                "q2:mw-q2.01": {"lo_freq": 3.8e9},
                "q3:mw-q3.01": {"lo_freq": 3.8e9},
                "q4:mw-q4.01": {"lo_freq": 3.8e9},
                "q5:mw-q5.01": {"lo_freq": 3.8e9},
            },
        },
        "connectivity": {
            "graph": [
                ["cluster0.module5.complex_output_0", "q2:mw"],
                ["cluster0.module6.complex_output_0", "q0:mw"],
                ["cluster0.module6.complex_output_1", "q1:mw"],
                ["cluster0.module7.complex_output_0", "q5:mw"],
                ["cluster0.module8.complex_output_0", "q3:mw"],
                ["cluster0.module8.complex_output_1", "q4:mw"],
            ]
        },
    }
    device = QuantumDevice("single_qubit_device")
    for i in range(6):
        qubit = BasicTransmonElement(f"q{i}")
        qubit.measure.pulse_amp(0.1)
        qubit.clock_freqs.readout(4.3e9)
        qubit.clock_freqs.f01(4e9)
        qubit.measure.acq_delay(100e-9)
        qubit.rxy.amp180(0.15)
        qubit.rxy.duration(100e-9)
        device.add_element(qubit)
    device.hardware_config(hw_config)
    return device


@pytest.mark.parametrize("qubit, module, expected_index", _parameters)
def test_rf_switch_only(qubit, module, expected_index, rf_device, assert_equal_q1asm):
    schedule = Schedule("rf_test")
    schedule.add_resource(ClockResource(f"{qubit}.01", freq=4e9))
    schedule.add(RFSwitchToggle(duration=100e-9, port=f"{qubit}:mw", clock=f"{qubit}.01"))
    schedule.add(IdlePulse(duration=200e-9))

    compiled_schedule = SerialCompiler("c").compile(
        schedule=schedule, config=rf_device.generate_compilation_config()
    )
    module = compiled_schedule.compiled_instructions["cluster0"][f"cluster0_module{module}"]
    sequencers = module["sequencers"]

    assert len(sequencers) == 1
    assert_equal_q1asm(
        sequencers["seq0"].sequence["program"],
        [
            "set_mrk 0",
            " wait_sync 4",
            " upd_param 4",
            " wait 4",
            " move 1,R0",
            "start:",
            " reset_ph",
            " upd_param 4",
            f" set_mrk {expected_index}",
            " upd_param 4",
            " wait 96",
            " set_mrk 0",
            " upd_param 4",
            " wait 196",
            " loop R0,@start",
            " stop",
        ],
    )


@pytest.mark.parametrize("qubit, module, expected_index", _parameters)
def test_x_gate_after_switch(qubit, module, expected_index, rf_device, assert_equal_q1asm):
    schedule = Schedule("rf_test")
    schedule.add_resource(ClockResource(f"{qubit}.01", freq=4e9))
    switch = schedule.add(MarkerPulse(duration=100e-9, port=f"{qubit}:mw", clock=f"{qubit}.01"))
    schedule.add(X(qubit), label="pi", rel_time=40e-9, ref_op=switch)
    compiled_schedule = SerialCompiler("c").compile(
        schedule=schedule, config=rf_device.generate_compilation_config()
    )
    module = compiled_schedule.compiled_instructions["cluster0"][f"cluster0_module{module}"]
    sequencers = module["sequencers"]

    assert len(sequencers) == 1
    print(sequencers["seq0"].sequence["program"])
    assert_equal_q1asm(
        sequencers["seq0"].sequence["program"],
        [
            " set_mrk 0",
            " wait_sync 4",
            " upd_param 4",
            " wait 4",
            " move 1,R0",
            "start:",
            " reset_ph",
            " upd_param 4",
            f" set_mrk {expected_index}",
            " upd_param 4",
            " wait 96",  # auto generated wait (96 ns)
            " set_mrk 0",  # set markers to 0 (default, marker pulse)
            " upd_param 4",
            " wait 36",  # auto generated wait (36 ns)
            " set_awg_gain 4913, 0",  # setting gain for X q0
            " play 0, 0, 4",  # play X q0 (100 ns)
            " wait 96",  # auto generated wait (96 ns)
            " loop R0, @ start",
            "stop",
        ],
    )


def test_raise_error_when_rf_output_on_is_true(rf_device):
    qubit = "q3"
    schedule = Schedule("rf_test")
    schedule.add_resource(ClockResource(f"{qubit}.01", freq=4e9))
    switch = schedule.add(MarkerPulse(duration=100e-9, port=f"{qubit}:mw", clock=f"{qubit}.01"))
    schedule.add(X(qubit), label="pi", rel_time=40e-9, ref_op=switch)
    with pytest.raises(
        RuntimeError,
        match="Attempting to turn on an RF output on a module where "
        r"`rf_output_on` is set to True \(the default value\)",
    ):
        SerialCompiler("c").compile(
            schedule=schedule, config=rf_device.generate_compilation_config()
        )
