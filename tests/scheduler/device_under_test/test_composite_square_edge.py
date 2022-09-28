# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=redefined-outer-name
import pytest

from quantify_scheduler.device_under_test.composite_square_edge import (
    CompositeSquareEdge,
)
from quantify_scheduler.device_under_test.transmon_element import BasicTransmonElement
from quantify_scheduler.backends.circuit_to_device import OperationCompilationConfig


@pytest.fixture
def edge_q2b_q3b():
    q2b = BasicTransmonElement("q2b")  # pylint: disable=invalid-name
    q3b = BasicTransmonElement("q3b")  # pylint: disable=invalid-name

    edge_q2b_q3b = CompositeSquareEdge(
        parent_element_name=q2b.name, child_element_name=q3b.name
    )

    # Transmon element is returned
    yield edge_q2b_q3b
    # after the test, teardown...
    q2b.close()
    q3b.close()
    edge_q2b_q3b.close()


def test_generate_edge_config(edge_q2b_q3b):
    # Setup
    expected_edge_cfg = {
        "q2b_q3b": {
            "CZ": OperationCompilationConfig(
                factory_func="quantify_scheduler.operations."
                + "pulse_factories.composite_square_pulse",
                factory_kwargs={
                    "square_port": "q2b:fl",
                    "square_clock": "cl0.baseband",
                    "square_amp": 0.65,
                    "square_duration": 2.5e-8,
                    "virt_z_parent_qubit_phase": 44,
                    "virt_z_parent_qubit_clock": "q2b.01",
                    "virt_z_child_qubit_phase": 63,
                    "virt_z_child_qubit_clock": "q3b.01",
                },
            ),
        }
    }

    expected_cz_dict = expected_edge_cfg["q2b_q3b"]["CZ"].dict()

    edge_q2b_q3b.cz.square_amp(expected_cz_dict["factory_kwargs"]["square_amp"])
    edge_q2b_q3b.cz.square_duration(
        expected_cz_dict["factory_kwargs"]["square_duration"]
    )
    edge_q2b_q3b.cz.q2b_phase_correction(
        expected_cz_dict["factory_kwargs"]["virt_z_parent_qubit_phase"]
    )
    edge_q2b_q3b.cz.q3b_phase_correction(
        expected_cz_dict["factory_kwargs"]["virt_z_child_qubit_phase"]
    )

    # Act
    generated_edge_cfg = edge_q2b_q3b.generate_edge_config()

    # Check
    assert generated_edge_cfg == expected_edge_cfg
