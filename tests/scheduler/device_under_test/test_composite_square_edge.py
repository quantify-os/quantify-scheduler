import json
import pytest

from quantify_scheduler.backends.circuit_to_device import OperationCompilationConfig
from quantify_scheduler.device_under_test.composite_square_edge import (
    CompositeSquareEdge,
)
from quantify_scheduler.device_under_test.transmon_element import BasicTransmonElement
from quantify_scheduler.json_utils import SchedulerJSONEncoder, SchedulerJSONDecoder


@pytest.fixture
def edge_q2b_q3b():
    q2b = BasicTransmonElement("q2b")
    q3b = BasicTransmonElement("q3b")

    edge_q2b_q3b = CompositeSquareEdge(
        parent_element_name=q2b.name, child_element_name=q3b.name
    )

    # Transmon element is returned
    yield edge_q2b_q3b


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

    expected_cz_dict = expected_edge_cfg["q2b_q3b"]["CZ"].model_dump()

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


def test_composite_square_edge_serialization(edge_q2b_q3b):
    """
    Tests the serialization process of :class:`~CompositeSquareEdge` by comparing the
    parameter values of the submodules of the original `CompositeSquareEdge` object and
    the serialized counterpart.
    """

    edge_q2b_q3b_as_dict = json.loads(
        json.dumps(edge_q2b_q3b, cls=SchedulerJSONEncoder)
    )
    assert edge_q2b_q3b_as_dict.__class__ is dict
    assert (
        edge_q2b_q3b_as_dict["deserialization_type"]
        == "quantify_scheduler.device_under_test.composite_square_edge.CompositeSquareEdge"
    )

    # Check that all original submodule params match their serialized counterpart
    for submodule_name, submodule in edge_q2b_q3b.submodules.items():
        for parameter_name in submodule.parameters:
            assert (
                edge_q2b_q3b_as_dict["data"][submodule_name][parameter_name]
                == edge_q2b_q3b.submodules[submodule_name][parameter_name]()
            ), (
                f"Expected value {edge_q2b_q3b.submodules[submodule_name][parameter_name]()} for "
                f"{submodule_name}.{parameter_name} but got "
                f"{edge_q2b_q3b_as_dict['data'][submodule_name][parameter_name]}"
            )

    # Check that all serialized submodule params match the original
    for submodule_name, submodule_data in edge_q2b_q3b_as_dict["data"].items():
        if (
            submodule_name == "parent_element_name"
            or submodule_name == "child_element_name"
        ):
            continue
        for parameter_name, parameter_val in submodule_data.items():
            assert (
                parameter_val
                == edge_q2b_q3b.submodules[submodule_name][parameter_name]()
            ), (
                f"Expected value {edge_q2b_q3b.submodules[submodule_name][parameter_name]()} for "
                f"{submodule_name}.{parameter_name} but got {parameter_val}"
            )


def test_composite_square_edge_deserialization(edge_q2b_q3b):
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

    expected_cz_dict = expected_edge_cfg["q2b_q3b"]["CZ"].model_dump()

    for param_name, kwarg_name in [
        ("square_amp", ""),
        ("square_duration", ""),
        ("q2b_phase_correction", "virt_z_parent_qubit_phase"),
        ("q3b_phase_correction", "virt_z_child_qubit_phase"),
    ]:
        edge_q2b_q3b.cz.set(
            param_name,
            expected_cz_dict["factory_kwargs"][
                param_name if kwarg_name == "" else kwarg_name
            ],
        )

    edge_q2b_q3b_serialized = json.dumps(edge_q2b_q3b, cls=SchedulerJSONEncoder)
    assert edge_q2b_q3b_serialized.__class__ is str

    edge_q2b_q3b.close()

    edge_q2b_q3b_deserialized = json.loads(
        edge_q2b_q3b_serialized, cls=SchedulerJSONDecoder
    )
    assert edge_q2b_q3b_deserialized.__class__ is CompositeSquareEdge
    assert edge_q2b_q3b_deserialized.generate_edge_config() == expected_edge_cfg
