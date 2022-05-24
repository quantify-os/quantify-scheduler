# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Tests for pulse and acquisition corrections."""
import json
import inspect
import os
import tempfile

import numpy as np
import pytest

from quantify_core.data.handling import set_datadir

import quantify_scheduler.schemas.examples as es

from quantify_scheduler import Schedule

from quantify_scheduler.backends.corrections import distortion_correct_pulse
from quantify_scheduler.backends.qblox import constants

from quantify_scheduler.compilation import (
    determine_absolute_timing,
    qcompile,
)

from quantify_scheduler.operations.gate_library import Reset
from quantify_scheduler.operations.pulse_library import (
    DRAGPulse,
    NumericalPulse,
    RampPulse,
    SquarePulse,
)

from quantify_scheduler.resources import ClockResource

esp = inspect.getfile(es)

cfg_f = os.path.abspath(os.path.join(esp, "..", "transmon_test_config.json"))
with open(cfg_f, "r", encoding="utf-8") as f:
    DEVICE_CFG = json.load(f)


def test_apply_distortion_corrections():

    fir_filter_coeffs = np.linspace(
        0, 1, 10
    )  # TODO: Also test list(np.linspace(0, 1, 10))

    hw_config = {
        "backend": "quantify_scheduler.backends.qblox_backend.hardware_compile",
        "distortion_corrections": {
            "q0:fl-cl0.baseband": {
                "filter_func": "scipy.signal.lfilter",
                "input_var_name": "x",
                "kwargs": {"b": fir_filter_coeffs, "a": 1},
            },
        },
        "qcm0": {
            "instrument_type": "Pulsar_QCM",
            "ref": "external",
            "complex_output_0": {
                "seq0": {
                    "port": "q0:fl",
                    "clock": "cl0.baseband",
                }
            },
            "complex_output_1": {
                "seq1": {
                    "port": "q0:mw",
                    "clock": "cl0.baseband",
                }
            },
        },
    }

    composite_drag_pulse = DRAGPulse(
        G_amp=0.5,
        D_amp=-0.2,
        phase=90,
        port="q0:fl",
        duration=20e-9,
        clock="cl0.baseband",
        t0=4e-9,
    )
    composite_drag_pulse.data["pulse_info"].append(
        {**composite_drag_pulse.data["pulse_info"][0], "port": "q0:mw"}
    )

    composite_ramp_pulse = RampPulse(
        t0=2e-3, amp=0.5, duration=28e-9, port="q0:mw", clock="cl0.baseband"
    )
    composite_ramp_pulse.data["pulse_info"].append(
        {**composite_drag_pulse.data["pulse_info"][0], "port": "q0:fl"}
    )

    sched = Schedule("pulse_only_experiment")
    sched.add(Reset("q0"))
    sched.add(composite_drag_pulse)
    sched.add(composite_ramp_pulse)
    sched.add_resources(
        [ClockResource("q0:fl", freq=5e9), ClockResource("q0:mw", freq=50e6)]
    )  # Clocks need to be manually added at this stage

    determine_absolute_timing(sched)

    tmp_dir = tempfile.TemporaryDirectory()
    set_datadir(tmp_dir.name)

    full_program = qcompile(sched, DEVICE_CFG, hw_config)  # "transmon_test_config.json"

    operations_pretty_repr = "".join(
        f"\nkey:  {operation_repr}\nrepr: {repr(operation)}\n"
        for operation_repr, operation in full_program.operations.items()
    )

    assert_mssg = (
        "Only replace waveform components in need of correcting by numerical pulse;"
        f" operations: {operations_pretty_repr}"
    )
    assert [
        [None],
        [
            "quantify_scheduler.waveforms.interpolated_complex_waveform",
            "quantify_scheduler.waveforms.drag",
        ],
        [
            "quantify_scheduler.waveforms.ramp",
            "quantify_scheduler.waveforms.interpolated_complex_waveform",
        ],
    ] == [
        [pulse_info["wf_func"] for pulse_info in operation.data["pulse_info"]]
        for operation in full_program.operations.values()
    ], assert_mssg

    assert_mssg = (
        "Distortion correction converts to operation type of first entry in pulse_info;"
        f" operations: {operations_pretty_repr}"
    )
    assert [Reset, NumericalPulse, RampPulse] == [
        type(operation) for operation in full_program.operations.values()
    ], assert_mssg

    assert_mssg = (
        "Key no longer matches str(operation) if first pulse_info entry was corrected;"
        f" operations: {operations_pretty_repr}"
    )
    assert [True, False, True] == [
        operation_repr == str(operation)
        for operation_repr, operation in full_program.operations.items()
    ], assert_mssg


@pytest.mark.parametrize("duration", list(np.arange(start=1e-9, stop=16e-9, step=1e-9)))
def test_distortion_correct_pulse(
    duration,
):
    filter_coefficients = [
        1.95857073e00,
        -1.86377203e-01,
        -1.68242537e-01,
        -1.52224167e-01,
        -1.37802128e-01,
        -1.21882898e-01,
        -8.43375734e-02,
        -5.96895462e-02,
        -3.96596464e-02,
        -1.76637397e-02,
        3.30717805e-03,
        8.42734090e-03,
        6.07696990e-03,
        -5.36042501e-03,
        -1.29125589e-02,
        -4.28917964e-03,
        1.33989347e-02,
        1.62354458e-02,
        9.54868788e-03,
        1.17526984e-02,
        -1.89290954e-03,
        -9.12214872e-03,
        -1.36650277e-02,
        -1.90334368e-02,
        -1.01304462e-02,
        1.06730684e-03,
        1.09447182e-02,
        1.00001337e-02,
        3.11361952e-03,
        -1.38470050e-02,
    ]

    filter_func_name = "scipy.signal.lfilter"
    input_var_name = "x"
    kwargs_dict = {"b": filter_coefficients, "a": 1}

    pulse = SquarePulse(
        amp=220e-3, duration=duration, port="q0:fl", clock="cl0.baseband"
    )

    corrected_pulse = distortion_correct_pulse(
        pulse_data=pulse.data["pulse_info"][0],
        sampling_rate=constants.SAMPLING_RATE,
        filter_func_name=filter_func_name,
        input_var_name=input_var_name,
        kwargs_dict=kwargs_dict,
    )

    assert len(corrected_pulse.data["pulse_info"][0]["samples"]) > 1
