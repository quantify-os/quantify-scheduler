# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=redefined-outer-name

# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Tests for pulse and acquisition corrections."""
import numpy as np
import pytest

from quantify_scheduler.backends import SerialCompiler
from quantify_scheduler.backends.corrections import distortion_correct_pulse
from quantify_scheduler.backends.qblox import constants as qblox_constants
from quantify_scheduler.operations.pulse_library import NumericalPulse, SquarePulse

from tests.scheduler.backends.test_qblox_backend import (  # pylint: disable=unused-import
    hardware_cfg_two_qubit_gate as qblox_hardware_cfg_two_qubit_gate,
)
from tests.scheduler.backends.test_zhinst_backend import (  # pylint: disable=unused-import
    hardware_cfg_distortion_corrections as zhinst_hardware_cfg_distortion_corrections,
)


# --------- Test fixtures ---------
@pytest.fixture
def hardware_cfg_distortion_corrections(
    filter_coefficients,
    qblox_hardware_cfg_two_qubit_gate,
    zhinst_hardware_cfg_distortion_corrections,
    backend,
    use_numpy_array,
):
    hardware_cfg = {
        "distortion_corrections": {
            "q2:fl-cl0.baseband": {
                "filter_func": "scipy.signal.lfilter",
                "input_var_name": "x",
                "kwargs": {
                    "b": filter_coefficients,
                    "a": np.array([1]) if use_numpy_array else [1],
                },
                "clipping_values": [-2.5, 2.5],
            },
        },
    }

    if "qblox" in backend:
        hardware_cfg = {**hardware_cfg, **qblox_hardware_cfg_two_qubit_gate}
    elif "zhinst" in backend:
        hardware_cfg = {**hardware_cfg, **zhinst_hardware_cfg_distortion_corrections}

    return hardware_cfg


@pytest.fixture
def filter_coefficients(use_numpy_array):
    coeffs = [
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

    if use_numpy_array:
        return np.array(coeffs)

    return coeffs


# --------- Test correction functions ---------
@pytest.mark.parametrize(
    "clipping_values, duration, use_numpy_array",
    list(
        (clipping, duration, use_numpy)
        for clipping in [None, [-0.2, 0.4]]
        for duration in np.arange(start=1e-9, stop=16e-9, step=1e-9)
        for use_numpy in [True, False]
    ),
)
def test_distortion_correct_pulse(
    filter_coefficients, clipping_values, duration, use_numpy_array
):
    pulse = SquarePulse(amp=220e-3, duration=duration, port="", clock="")

    corrected_pulse = distortion_correct_pulse(
        pulse_data=pulse.data["pulse_info"][0],
        sampling_rate=qblox_constants.SAMPLING_RATE,
        filter_func_name="scipy.signal.lfilter",
        input_var_name="x",
        kwargs_dict={
            "b": filter_coefficients,
            "a": np.array([1]) if use_numpy_array else [1],
        },
        clipping_values=clipping_values,
    )

    corrected_pulse_samples = corrected_pulse.data["pulse_info"][0]["samples"]

    assert (
        len(corrected_pulse_samples) > 1
    ), "Correction always generates at least 2 sample points"

    if clipping_values:
        assert min(corrected_pulse_samples) >= clipping_values[0]
        assert max(corrected_pulse_samples) <= clipping_values[1]


@pytest.mark.parametrize(
    "backend, use_numpy_array",
    [
        (backend, use_numpy)
        for backend in [
            "quantify_scheduler.backends.qblox_backend.hardware_compile",
            "quantify_scheduler.backends.zhinst_backend.compile_backend",
        ]
        for use_numpy in [True, False]
    ],
)
def test_apply_distortion_corrections(  # pylint: disable=unused-argument disable=too-many-arguments
    mock_setup_basic_transmon_with_standard_params,
    hardware_cfg_distortion_corrections,
    filter_coefficients,
    two_qubit_gate_schedule,
    backend,
    use_numpy_array,
):
    quantum_device = mock_setup_basic_transmon_with_standard_params["quantum_device"]
    quantum_device.hardware_config(hardware_cfg_distortion_corrections)
    compiler = SerialCompiler(name="compiler")
    compiled_sched = compiler.compile(
        schedule=two_qubit_gate_schedule,
        config=quantum_device.generate_compilation_config(),
    )

    operations_pretty_repr = "".join(
        f"\noperations:\n  key:  {operation_repr}\n  repr: {repr(operation)}\n"
        for operation_repr, operation in compiled_sched.operations.items()
    )

    assert (
        list(compiled_sched.operations.keys())[0] == "CZ(qC='q2',qT='q3')"
    ), f'Key of CZ operation is still "CZ(...)" {operations_pretty_repr}'

    assert (  # pylint: disable=unidiomatic-typecheck
        type(list(compiled_sched.operations.values())[0]) is NumericalPulse
    ), f"Type of CZ operation is now NumericalPulse {operations_pretty_repr}"

    assert list(compiled_sched.operations.values())[0].data["pulse_info"][0][
        "samples"
    ] == pytest.approx(
        [
            0.979285365,
            0.8860967635,
            0.801975495,
            0.7258634115,
            0.6569623475,
            0.5960208985000001,
            0.5538521118,
            0.5240073387,
            0.5041775155,
            0.49534564565,
            0.496999234675,
            0.501212905125,
            0.5042513900750001,
            0.5015711775700001,
            0.49511489812000015,
            0.49297030829999994,
            0.4996697756499999,
            0.5077874985499999,
            0.5125618424899999,
            0.5184381916899999,
        ]
    )
