import re

import numpy as np
import pytest

from quantify_scheduler.backends.graph_compilation import SerialCompiler
from quantify_scheduler.backends.qblox.operations.control_flow_library import (
    ConditionalOperation,
)
from quantify_scheduler.backends.qblox.operations.gate_library import ConditionalReset
from quantify_scheduler.operations.gate_library import Measure, X, Y, Z
from quantify_scheduler.operations.pulse_library import DRAGPulse, SquarePulse
from quantify_scheduler.schedules.schedule import Schedule
from quantify_scheduler.schemas.examples import utils


def complicated_schedule(pulse_duration):
    body = Schedule("")
    body.add(X("q0"))
    body.add(Z("q0"))
    body.add(SquarePulse(amp=0.1, duration=pulse_duration, port="q0:mw", clock="q0.01"))
    body.add(
        DRAGPulse(
            G_amp=0.1,
            D_amp=0.1,
            phase=0,
            duration=2 * pulse_duration,
            port="q4:mw",
            clock="q4.01",
        ),
        rel_time=-52e-9,
    )

    body.add(Y("q0"))
    body.add(Measure("q0", acq_protocol="ThresholdedAcquisition", acq_index=1))

    schedule = Schedule("test")
    schedule.add(Measure("q0", acq_protocol="ThresholdedAcquisition", feedback_trigger_label="q0"))
    schedule.add(ConditionalOperation(body=body, qubit_name="q0"), label="complicated_label")
    return schedule


@pytest.mark.parametrize("pulse_duration", [100e-9, 156e-9, 200e-9, 248e-9, 300e-9])
def test_conditional_playback_compiles(
    mock_setup_basic_transmon_with_standard_params, pulse_duration
):
    schedule = complicated_schedule(pulse_duration)
    quantum_device = mock_setup_basic_transmon_with_standard_params["quantum_device"]

    hardware_config = utils.load_json_example_scheme("qblox_hardware_config_transmon.json")
    quantum_device.hardware_config(hardware_config)
    config = quantum_device.generate_compilation_config()

    compiler = SerialCompiler(name="compiler")
    compiled_schedule = compiler.compile(
        schedule,
        config=config,
    )

    key_conditional_playback = compiled_schedule.schedulables["complicated_label"]["operation_id"]
    conditional_playback = compiled_schedule.operations[key_conditional_playback]
    conditional_duration = conditional_playback.duration

    seq_settings = compiled_schedule.compiled_instructions["cluster0"]["cluster0_module4"][
        "sequencers"
    ]["seq0"]
    assert (expected_address := seq_settings["thresholded_acq_trigger_address"]) is not None
    assert seq_settings["thresholded_acq_trigger_en"] is True
    assert seq_settings["thresholded_acq_trigger_invert"] is False

    qcm_program = compiled_schedule.compiled_instructions["cluster0"]["cluster0_module2"][
        "sequencers"
    ]["seq0"]["sequence"]["program"]

    qrm_program = compiled_schedule.compiled_instructions["cluster0"]["cluster0_module4"][
        "sequencers"
    ]["seq0"]["sequence"]["program"]

    pattern = r"set_cond.+play.+set_cond.+wait.+set_cond."
    match = re.search(pattern, qrm_program, re.DOTALL)
    assert match is not None

    pattern = r"^\s*set_latch_en\s*(\d).*$"
    match = re.search(pattern, qcm_program, re.MULTILINE)
    assert match is not None

    latch_en_arg = int(match.group(1))
    assert latch_en_arg == 1

    # The (?P<enable>\d) syntax below assigns a name to the capturing group.
    pattern = r"""
        set_cond\s+1,(?P<mask>\d+),0,4.*
        set_awg_gain.*
        play\s+\d+,\d+,\d+.*
        wait.*
        set_cond\s+1,1,1,4.*
        wait\s+(?P<wait_duration>\d+).*
        set_cond\s+0,0,0,0.*
    """

    compiled_pattern = re.compile(pattern, re.MULTILINE | re.DOTALL | re.VERBOSE)
    match = compiled_pattern.search(qcm_program)
    assert match is not None

    mask = match.group("mask")

    wait_duration = match.group("wait_duration")
    expected_mask = str(2**expected_address - 1)

    assert mask == expected_mask

    assert 2**expected_address - 1 == int(mask)
    num_real_time_operations = 5

    assert np.isclose(
        int(wait_duration) + num_real_time_operations * 4 + 4,
        conditional_duration * 1e9,  # type: ignore
    )
