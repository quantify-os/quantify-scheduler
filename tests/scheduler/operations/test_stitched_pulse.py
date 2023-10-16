"""Tests for the stitched_pulse module."""
import numpy as np
import pytest

from quantify_scheduler.operations.acquisition_library import Trace
from quantify_scheduler.operations.gate_library import Rxy
from quantify_scheduler.operations.operation import Operation
from quantify_scheduler.operations.pulse_factories import long_ramp_pulse
from quantify_scheduler.operations.pulse_library import (
    RampPulse,
    ReferenceMagnitude,
    SquarePulse,
    VoltageOffset,
)
from quantify_scheduler.operations.stitched_pulse import (
    StitchedPulse,
    StitchedPulseBuilder,
    convert_to_numerical_pulse,
)


def test_constructors():
    """Test class initializers."""
    _ = StitchedPulse()
    _ = StitchedPulseBuilder(port="q0:mw", clock="q0.01")


def test_str():
    """Test the string representation."""
    pulse = (
        StitchedPulseBuilder(port="q0:mw", clock="q0.01")
        .add_pulse(SquarePulse(amp=0.2, duration=1e-6, port="q0:mw", clock="q0.01"))
        .add_voltage_offset(path_0=0.5, path_1=0.0, duration=1e-7)
        .build()
    )
    assert eval(str(pulse)) == pulse  # pylint: disable=eval-used # nosec


def test_add_operation_wrong_clock():
    """Tests that adding operations with different clocks raises an error."""
    pulse = (
        StitchedPulseBuilder(port="q0:mw", clock="q0.01")
        .add_pulse(SquarePulse(amp=0.2, duration=1e-6, port="q0:mw", clock="q0.01"))
        .add_voltage_offset(path_0=0.5, path_1=0.0, duration=1e-7)
        .build()
    )
    with pytest.raises(ValueError):
        pulse.add_pulse(
            RampPulse(amp=0.5, duration=28e-9, port="q0:res", clock="q0.01")
        )

    pulse = (
        StitchedPulseBuilder(port="q0:mw", clock="q0.01")
        .add_pulse(SquarePulse(amp=0.2, duration=1e-6, port="q0:mw", clock="q0.01"))
        .add_pulse(RampPulse(amp=0.5, duration=28e-9, port="q0:mw", clock="q0.01"))
        .build()
    )
    with pytest.raises(ValueError):
        pulse.add_pulse(
            VoltageOffset(
                offset_path_0=0.5,
                offset_path_1=0.0,
                duration=1e-7,
                t0=1e-6,
                port="q0:res",
                clock="q0.01",
            )
        )


def test_set_port_clock_t0():
    """Test that you can set the port, clock and t0."""
    builder = (
        StitchedPulseBuilder()
        .add_pulse(SquarePulse(amp=0.2, duration=1e-6, port="q0:mw"))
        .add_pulse(RampPulse(amp=0.5, duration=28e-9, port="q0:mw"))
        .add_voltage_offset(path_0=0.5, path_1=0.0, duration=1e-7)
    )
    builder.set_clock("q0.01")
    builder.set_port("q0:mw")
    builder.set_t0(1e-6)
    pulse = builder.build()
    assert pulse["pulse_info"][0]["t0"] == pytest.approx(1e-6)
    assert pulse["pulse_info"][0]["port"] == "q0:mw"
    assert pulse["pulse_info"][0]["clock"] == "q0.01"


def test_no_port_clock_fails():
    """Test that an error is raised if no port or clock is defined."""
    builder = (
        StitchedPulseBuilder()
        .add_pulse(SquarePulse(amp=0.2, duration=1e-6, port="q0:mw"))
        .add_pulse(RampPulse(amp=0.5, duration=28e-9, port="q0:mw"))
        .add_voltage_offset(path_0=0.5, path_1=0.0, duration=1e-7)
    )
    with pytest.raises(RuntimeError):
        _ = builder.build()


@pytest.mark.parametrize(
    "wrong",
    [
        Trace(1e-6, "q0:mw", "q0.ro"),
        Rxy(0.5, 0.1, "q0"),
        VoltageOffset(offset_path_0=0.5, offset_path_1=0.0, duration=1e-7),
    ],
)
def test_must_add_pulse(wrong: Operation):
    """Test that an error is raised if we add something other than a pulse."""
    builder = StitchedPulseBuilder()
    with pytest.raises(RuntimeError):
        builder.add_pulse(wrong)


def test_add_operations():
    """Test that operations are correctly added to the StitchedPulse."""
    pulse = (
        StitchedPulseBuilder(port="q0:mw", clock="q0.01")
        .add_pulse(SquarePulse(amp=0.2, duration=1e-6, port="q0:mw", clock="q0.01"))
        .add_pulse(RampPulse(amp=0.5, duration=28e-9, port="q0:mw", clock="q0.01"))
        .add_voltage_offset(
            path_0=0.5,
            path_1=0.0,
            duration=1e-7,
            reference_magnitude=ReferenceMagnitude(1, "V"),
        )
        .build()
    )
    assert len(pulse.data["pulse_info"]) == 4
    assert pulse.data["pulse_info"][0] == {
        "amp": 0.2,
        "reference_magnitude": None,
        "clock": "q0.01",
        "duration": 1e-06,
        "phase": 0,
        "port": "q0:mw",
        "t0": 0,
        "wf_func": "quantify_scheduler.waveforms.square",
    }
    assert pulse.data["pulse_info"][1] == {
        "amp": 0.5,
        "reference_magnitude": None,
        "clock": "q0.01",
        "duration": 2.8e-08,
        "offset": 0,
        "port": "q0:mw",
        "t0": 1e-06,
        "wf_func": "quantify_scheduler.waveforms.ramp",
    }
    assert pulse.data["pulse_info"][2] == {
        "clock": "q0.01",
        "duration": 1e-07,
        "offset_path_0": 0.5,
        "offset_path_1": 0.0,
        "port": "q0:mw",
        "t0": 1.028e-06,
        "wf_func": None,
        "reference_magnitude": ReferenceMagnitude(1, "V"),
    }
    assert pulse.data["pulse_info"][3] == {
        "clock": "q0.01",
        "duration": 0.0,
        "offset_path_0": 0.0,
        "offset_path_1": 0.0,
        "port": "q0:mw",
        "t0": 1.128e-06,
        "wf_func": None,
        "reference_magnitude": ReferenceMagnitude(1, "V"),
    }


def test_add_operations_insert_timing():
    """Test that operations can be inserted at a specific time."""
    pulse = (
        StitchedPulseBuilder(port="q0:mw", clock="q0.01")
        .add_pulse(SquarePulse(amp=0.2, duration=1e-6, port="q0:mw", clock="q0.01"))
        .add_pulse(RampPulse(amp=0.5, duration=28e-9, port="q0:mw", clock="q0.01"))
        .add_voltage_offset(
            path_0=0.5, path_1=0.0, duration=1e-7, rel_time=5e-7, append=False
        )
        .build()
    )
    assert len(pulse.data["pulse_info"]) == 4
    assert pulse.data["pulse_info"][0] == {
        "amp": 0.2,
        "reference_magnitude": None,
        "clock": "q0.01",
        "duration": 1e-06,
        "phase": 0,
        "port": "q0:mw",
        "t0": 0,
        "wf_func": "quantify_scheduler.waveforms.square",
    }
    assert pulse.data["pulse_info"][1] == {
        "amp": 0.5,
        "reference_magnitude": None,
        "clock": "q0.01",
        "duration": 2.8e-08,
        "offset": 0,
        "port": "q0:mw",
        "t0": 1e-06,
        "wf_func": "quantify_scheduler.waveforms.ramp",
    }
    assert pulse.data["pulse_info"][2] == {
        "clock": "q0.01",
        "duration": 1e-07,
        "offset_path_0": 0.5,
        "offset_path_1": 0.0,
        "port": "q0:mw",
        "t0": 5e-7,
        "wf_func": None,
        "reference_magnitude": None,
    }
    assert pulse.data["pulse_info"][3] == {
        "clock": "q0.01",
        "duration": 0.0,
        "offset_path_0": 0.0,
        "offset_path_1": 0.0,
        "port": "q0:mw",
        "t0": 6e-7,
        "wf_func": None,
        "reference_magnitude": None,
    }


def test_convert_to_numerical():
    pulse = long_ramp_pulse(
        amp=0.5, duration=1e-4, port="some_port", clock="some_clock", offset=-0.25
    )
    num_pulse = convert_to_numerical_pulse(pulse)

    assert (
        num_pulse.data["pulse_info"][0]["wf_func"]
        == "quantify_scheduler.waveforms.interpolated_complex_waveform"
    )
    # Last point can be off
    assert np.isclose(
        num_pulse.data["pulse_info"][0]["samples"][:-1],
        np.linspace(-0.25, 0.25, 100_001)[:-1],
    ).all()
    assert np.isclose(
        num_pulse.data["pulse_info"][0]["t_samples"], np.linspace(0, 1e-4, 100_001)
    ).all()
    assert num_pulse.data["pulse_info"][0]["duration"] == 1e-4
    assert num_pulse.data["pulse_info"][0]["interpolation"] == "linear"
    assert num_pulse.data["pulse_info"][0]["clock"] == "some_clock"
    assert num_pulse.data["pulse_info"][0]["port"] == "some_port"
    assert num_pulse.data["pulse_info"][0]["t0"] == 0.0


@pytest.mark.parametrize(
    "not_a_pulse", [Trace(1e-6, "q0:mw", "q0.ro"), Rxy(0.5, 0.1, "q0")]
)
def test_convert_to_numerical_does_nothing(not_a_pulse):
    converted_op = convert_to_numerical_pulse(not_a_pulse)
    assert converted_op == not_a_pulse


def test_convert_to_numerical_mixed_operation():
    pulse = long_ramp_pulse(
        amp=0.5, duration=1e-4, port="some_port", clock="some_clock", offset=-0.25
    )
    # Add some mock gate_info to make sure the type does not get converted
    dummy_gate_info = {
        "unitary": [[1, 0], [0, 1]],
        "operation_type": "Example",
        "qubits": ["q0"],
        "symmetric": False,
        "tex": r"example",
        "plot_func": None,
    }
    pulse.data["gate_info"] = dummy_gate_info
    num_pulse = convert_to_numerical_pulse(pulse)

    assert isinstance(num_pulse, StitchedPulse)
    assert num_pulse.data["gate_info"] == dummy_gate_info
