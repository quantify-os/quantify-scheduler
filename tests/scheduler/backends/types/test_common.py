import pytest

from quantify_scheduler.backends.types.common import (
    Connectivity,
    HardwareCompilationConfig,
)


def test_mismatching_instrument_names_raises():
    """
    Tests if a mismatching instrument name in the Connectivity raises an error.
    """

    hardware_compilation_cfg = {
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
        match="Invalid node. Instrument 'instrument_1' not found in " "hardware description.",
    ):
        HardwareCompilationConfig.model_validate(hardware_compilation_cfg)


@pytest.mark.parametrize("invalid_port", ["q0microwave", "q0.mw", "quantum_device.q0:mw"])
def test_invalid_quantum_device_port_raises(invalid_port):
    """
    Tests if an invalid quantum device port in the Connectivity raises an error.
    """

    hardware_compilation_cfg = {
        "hardware_description": {
            "instrument_0": {
                "instrument_type": "some_type",
            },
        },
        "hardware_options": {},
        "connectivity": {
            "graph": [
                ("instrument_0.output_0", invalid_port),
            ]
        },
    }

    with pytest.raises(ValueError, match="Invalid node"):
        HardwareCompilationConfig.model_validate(hardware_compilation_cfg)


@pytest.mark.parametrize(
    "list_of_edges, expected_edges",
    [
        ([("instrument_0.port0", "q0:a")], {("instrument_0.port0", "q0:a")}),
        (
            [("instrument_0.port1", ["q0:b", "q0:c", "q0:d", "q0:e", "q0:f"])],
            {
                ("instrument_0.port1", "q0:b"),
                ("instrument_0.port1", "q0:c"),
                ("instrument_0.port1", "q0:d"),
                ("instrument_0.port1", "q0:e"),
                ("instrument_0.port1", "q0:f"),
            },
        ),
        (
            [
                (
                    ["instrument_1.port0", "instrument_1.port1", "instrument_1.port2"],
                    "q1:a",
                )
            ],
            {
                ("instrument_1.port0", "q1:a"),
                ("instrument_1.port1", "q1:a"),
                ("instrument_1.port2", "q1:a"),
            },
        ),
        (
            [(["instrument_2.port0", "instrument_2.port1"], ["q2:a", "q2:b"])],
            {
                ("instrument_2.port0", "q2:a"),
                ("instrument_2.port0", "q2:b"),
                ("instrument_2.port1", "q2:a"),
                ("instrument_2.port1", "q2:b"),
            },
        ),
    ],
)
def test_connectivity_edges_with_lists_of_ports(list_of_edges, expected_edges):
    connectivity = Connectivity.model_validate({"graph": list_of_edges})
    for edge in expected_edges:
        assert edge in connectivity.graph.edges
