# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=redefined-outer-name
import pytest

from quantify_scheduler.device_under_test.sudden_nz_edge import SuddenNetZeroEdge
from quantify_scheduler.device_under_test.transmon_element import BasicTransmonElement
from quantify_scheduler.backends.circuit_to_device import OperationCompilationConfig


@pytest.fixture
def edge_q2b_q3b():
    q2b = BasicTransmonElement("q2b")  # pylint: disable=invalid-name
    q3b = BasicTransmonElement("q3b")  # pylint: disable=invalid-name

    edge_q2b_q3b = SuddenNetZeroEdge(
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
        "q2b-q3b": {
            "CZ": OperationCompilationConfig(
                factory_func="quantify_scheduler.operations."
                + "pulse_library.SuddenNetZeroPulse",
                factory_kwargs={
                    "port": "q2b:fl",
                    "clock": "cl0.baseband",
                    "amp_A": 0.65,
                    "amp_B": 0.55,
                    "net_zero_A_scale": 0.89,
                    "t_pulse": 2.5e-8,
                    "t_phi": 3e-9,
                    "t_integral_correction": 1.2e-8,
                },
            ),
        }
    }

    expected_q2b_phase_correction = 44
    expected_q3b_phase_correction = 63

    expected_cz_dict = expected_edge_cfg["q2b-q3b"]["CZ"].dict()

    edge_q2b_q3b.cz.amp_A(expected_cz_dict["factory_kwargs"]["amp_A"])
    edge_q2b_q3b.cz.amp_B(expected_cz_dict["factory_kwargs"]["amp_B"])
    edge_q2b_q3b.cz.net_zero_A_scale(
        expected_cz_dict["factory_kwargs"]["net_zero_A_scale"]
    )
    edge_q2b_q3b.cz.t_pulse(expected_cz_dict["factory_kwargs"]["t_pulse"])
    edge_q2b_q3b.cz.t_phi(expected_cz_dict["factory_kwargs"]["t_phi"])
    edge_q2b_q3b.cz.t_integral_correction(
        expected_cz_dict["factory_kwargs"]["t_integral_correction"]
    )
    edge_q2b_q3b.cz.q2b_phase_correction(expected_q2b_phase_correction)
    edge_q2b_q3b.cz.q3b_phase_correction(expected_q3b_phase_correction)

    # Act
    generated_edge_cfg = edge_q2b_q3b.generate_edge_config()

    # Check
    assert generated_edge_cfg == expected_edge_cfg
    assert edge_q2b_q3b.cz.q2b_phase_correction() == expected_q2b_phase_correction
    assert edge_q2b_q3b.cz.q3b_phase_correction() == expected_q3b_phase_correction
