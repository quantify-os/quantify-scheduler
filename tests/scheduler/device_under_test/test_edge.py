import json

import pytest

from quantify_scheduler.backends.circuit_to_device import OperationCompilationConfig
from quantify_scheduler.device_under_test.composite_square_edge import (
    CompositeSquareEdge,
)
from quantify_scheduler.device_under_test.spin_edge import SpinEdge
from quantify_scheduler.device_under_test.spin_element import BasicSpinElement
from quantify_scheduler.device_under_test.transmon_element import BasicTransmonElement
from quantify_scheduler.json_utils import SchedulerJSONDecoder, SchedulerJSONEncoder
from quantify_scheduler.operations.pulse_factories import composite_square_pulse, spin_init_pulse


@pytest.fixture
def transmon_edge():
    q2b = BasicTransmonElement("q2b")
    q3b = BasicTransmonElement("q3b")

    edge_q2b_q3b = CompositeSquareEdge(parent_element_name=q2b.name, child_element_name=q3b.name)

    edge_q2b_q3b.cz.square_amp(0.65)
    edge_q2b_q3b.cz.square_duration(2.5e-8)
    edge_q2b_q3b.cz.q2b_phase_correction(44)
    edge_q2b_q3b.cz.q3b_phase_correction(63)

    yield edge_q2b_q3b


@pytest.fixture
def spin_edge():
    q2b = BasicSpinElement("q2b")
    q3b = BasicSpinElement("q3b")

    edge_q2b_q3b = SpinEdge(parent_element_name=q2b.name, child_element_name=q3b.name)

    edge_q2b_q3b.spin_init.square_duration(2e-6)
    edge_q2b_q3b.spin_init.ramp_diff(1e-6)
    edge_q2b_q3b.spin_init.q2b_square_amp(0.5)
    edge_q2b_q3b.spin_init.q2b_ramp_amp(0.25)
    edge_q2b_q3b.spin_init.q2b_ramp_rate(0.25 / 3e-6)
    edge_q2b_q3b.spin_init.q3b_square_amp(0.4)
    edge_q2b_q3b.spin_init.q3b_ramp_amp(0.2)
    edge_q2b_q3b.spin_init.q3b_ramp_rate(0.2 / 4e-6)

    yield edge_q2b_q3b


@pytest.mark.parametrize(
    ["edge", "expected_edge_cfg"],
    [
        (
            "transmon_edge",
            {
                "q2b_q3b": {
                    "CZ": OperationCompilationConfig(
                        factory_func=composite_square_pulse,
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
            },
        ),
        (
            "spin_edge",
            {
                "q2b_q3b": {
                    "SpinInit": OperationCompilationConfig(
                        factory_func=spin_init_pulse,
                        factory_kwargs={
                            "square_duration": 2e-6,
                            "ramp_diff": 1e-6,
                            "parent_port": "q2b:mw",
                            "parent_clock": "q2b.f_larmor",
                            "parent_square_amp": 0.5,
                            "parent_ramp_amp": 0.25,
                            "parent_ramp_rate": 0.25 / 3e-6,
                            "child_port": "q3b:mw",
                            "child_clock": "q3b.f_larmor",
                            "child_square_amp": 0.4,
                            "child_ramp_amp": 0.2,
                            "child_ramp_rate": 0.2 / 4e-6,
                        },
                    ),
                }
            },
        ),
    ],
)
def test_generate_edge_config(request, edge, expected_edge_cfg):
    edge = request.getfixturevalue(edge)
    # Act
    generated_edge_cfg = edge.generate_edge_config()

    # Check
    assert generated_edge_cfg == expected_edge_cfg


@pytest.mark.parametrize(
    ["edge", "expected_deserialization_type"],
    [
        (
            "transmon_edge",
            "quantify_scheduler.device_under_test.composite_square_edge.CompositeSquareEdge",
        ),
        (
            "spin_edge",
            "quantify_scheduler.device_under_test.spin_edge.SpinEdge",
        ),
    ],
)
def test_composite_square_edge_serialization(request, edge, expected_deserialization_type):
    """
    Tests the serialization process of an Edge by comparing the
    parameter values of the submodules of the original object and
    the serialized counterpart.
    """

    edge = request.getfixturevalue(edge)
    edge_as_dict = json.loads(json.dumps(edge, cls=SchedulerJSONEncoder))
    assert edge_as_dict.__class__ is dict
    assert edge_as_dict["deserialization_type"] == expected_deserialization_type

    # Check that all original submodule params match their serialized counterpart
    for submodule_name, submodule in edge.submodules.items():
        for parameter_name in submodule.parameters:
            assert (
                edge_as_dict["data"][submodule_name][parameter_name]
                == edge.submodules[submodule_name][parameter_name]()
            ), (
                f"Expected value {edge.submodules[submodule_name][parameter_name]()} for "
                f"{submodule_name}.{parameter_name} but got "
                f"{edge_as_dict['data'][submodule_name][parameter_name]}"
            )

    # Check that all serialized submodule params match the original
    for submodule_name, submodule_data in edge_as_dict["data"].items():
        if submodule_name in ("parent_element_name", "child_element_name"):
            continue
        for parameter_name, parameter_val in submodule_data.items():
            assert parameter_val == edge.submodules[submodule_name][parameter_name](), (
                f"Expected value {edge.submodules[submodule_name][parameter_name]()} for "
                f"{submodule_name}.{parameter_name} but got {parameter_val}"
            )


@pytest.mark.parametrize(
    ["edge", "expected_edge_cfg", "expected_type"],
    [
        (
            "transmon_edge",
            {
                "q2b_q3b": {
                    "CZ": OperationCompilationConfig(
                        factory_func=composite_square_pulse,
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
            },
            CompositeSquareEdge,
        ),
        (
            "spin_edge",
            {
                "q2b_q3b": {
                    "SpinInit": OperationCompilationConfig(
                        factory_func=spin_init_pulse,
                        factory_kwargs={
                            "square_duration": 2e-6,
                            "ramp_diff": 1e-6,
                            "parent_port": "q2b:mw",
                            "parent_clock": "q2b.f_larmor",
                            "parent_square_amp": 0.5,
                            "parent_ramp_amp": 0.25,
                            "parent_ramp_rate": 0.25 / 3e-6,
                            "child_port": "q3b:mw",
                            "child_clock": "q3b.f_larmor",
                            "child_square_amp": 0.4,
                            "child_ramp_amp": 0.2,
                            "child_ramp_rate": 0.2 / 4e-6,
                        },
                    ),
                }
            },
            SpinEdge,
        ),
    ],
)
def test_composite_square_edge_deserialization(request, edge, expected_edge_cfg, expected_type):
    edge = request.getfixturevalue(edge)

    edge_q2b_q3b_serialized = json.dumps(edge, cls=SchedulerJSONEncoder)
    assert edge_q2b_q3b_serialized.__class__ is str

    edge.close()

    edge_q2b_q3b_deserialized = json.loads(edge_q2b_q3b_serialized, cls=SchedulerJSONDecoder)
    assert isinstance(edge_q2b_q3b_deserialized, expected_type)
    assert edge_q2b_q3b_deserialized.generate_edge_config() == expected_edge_cfg
