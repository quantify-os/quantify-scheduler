import pytest

from quantify_scheduler.backends.types.common import HardwareCompilationConfig


def test_mismatching_instrument_names_raises():
    """
    Tests if a mismatching instrument name in the Connectivity raises an error.
    """

    def foo():
        pass

    hardware_compilation_cfg = {
        "backend": foo,
        "hardware_description": {
            "instrument_0": {
                "instrument_type": "some_type",
            },
        },
        "hardware_options": {},
        "connectivity": {
            "graph": [
                ("instrument_1.output_0", "q0:mw"),
            ]
        },
    }

    with pytest.raises(
        ValueError,
        match="Invalid node. Instrument 'instrument_1' not found in "
        "hardware description.",
    ):
        HardwareCompilationConfig.model_validate(hardware_compilation_cfg)


@pytest.mark.parametrize(
    "invalid_port", ["q0microwave", "q0.mw", "quantum_device.q0:mw"]
)
def test_invalid_quantum_device_port_raises(invalid_port):
    """
    Tests if an invalid quantum device port in the Connectivity raises an error.
    """

    def foo():
        pass

    hardware_compilation_cfg = {
        "backend": foo,
        "hardware_description": {
            "instrument_0": {
                "instrument_type": "some_type",
            },
        },
        "connectivity": {
            "graph": [
                ("instrument_0.output_0", invalid_port),
            ]
        },
    }

    with pytest.raises(ValueError, match="Invalid node"):
        HardwareCompilationConfig.model_validate(hardware_compilation_cfg)
