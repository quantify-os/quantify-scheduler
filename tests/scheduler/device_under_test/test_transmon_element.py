# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=redefined-outer-name
import json

import pytest

from quantify_scheduler import Schedule
from quantify_scheduler.backends.circuit_to_device import DeviceCompilationConfig
from quantify_scheduler.backends.graph_compilation import SerialCompiler
from quantify_scheduler.device_under_test.transmon_element import BasicTransmonElement
from quantify_scheduler.device_under_test.quantum_device import QuantumDevice
from quantify_scheduler.json_utils import ScheduleJSONEncoder, ScheduleJSONDecoder
from quantify_scheduler.operations.gate_library import Measure


@pytest.fixture
def q0() -> BasicTransmonElement:
    q0 = BasicTransmonElement("q0")
    yield q0
    q0.close()


@pytest.fixture
def dev() -> QuantumDevice:
    dev = QuantumDevice("dev")
    yield dev
    dev.close()


def test_qubit_name(q0: BasicTransmonElement):
    assert q0.name == "q0"


def test_generate_device_config_part_of_device(
    q0: BasicTransmonElement, dev: QuantumDevice
):
    # Test that setting some values updates the correct values in the configuration
    q0.measure.pulse_type("SquarePulse")
    q0.measure.pulse_duration(400e-9)

    dev.add_element(q0)
    dev_cfg = dev.generate_device_config()

    assert isinstance(dev_cfg, DeviceCompilationConfig)

    # Assert values in right place in config
    assert (
        dev_cfg.elements["q0"]["measure"].factory_kwargs["pulse_type"] == "SquarePulse"
    )
    assert dev_cfg.elements["q0"]["measure"].factory_kwargs["pulse_duration"] == 400e-9

    assert dev_cfg.elements["q0"]["Rxy"].factory_kwargs["clock"] == "q0.01"
    assert dev_cfg.elements["q0"]["Rxy"].gate_info_factory_kwargs == ["theta", "phi"]


def test_generate_device_config(q0: BasicTransmonElement):
    _ = q0.generate_device_config()


@pytest.mark.parametrize(
    "readout_frequency, mw_frequency, acq_delay, pulse_amp",
    [
        (8.0e9, 8.2e9, 100e-9, 0.1),
    ],
)
def test_basic_transmon_serialization(
    q0: BasicTransmonElement, readout_frequency, mw_frequency, acq_delay, pulse_amp
):
    """
    Tests the serialization process of :class:`~BasicTransmonElement` by comparing the
    parameter values of the submodules of the original `BasicTransmonElement` object and
    the serialized counterpart.
    """

    q0.clock_freqs.readout(readout_frequency)
    q0.clock_freqs.f01(mw_frequency)
    q0.clock_freqs.f12(0)
    q0.measure.acq_delay(acq_delay)
    q0.rxy.amp180(pulse_amp)

    q0_as_dict = json.loads(json.dumps(q0, cls=ScheduleJSONEncoder))
    assert q0_as_dict.__class__ is dict
    assert q0_as_dict["deserialization_type"] == "BasicTransmonElement"

    # Check that all original submodule params match their serialized counterpart
    for submodule_name, submodule in q0.submodules.items():
        for parameter_name in submodule.parameters:
            assert (
                q0_as_dict["data"][submodule_name][parameter_name]
                == q0.submodules[submodule_name][parameter_name]()
            ), (
                f"Expected value {q0.submodules[submodule_name][parameter_name]()} for "
                f"{submodule_name}.{parameter_name} but got "
                f"{q0_as_dict['data'][submodule_name][parameter_name]}"
            )

    # Check that all serialized submodule params match the original
    for submodule_name, submodule_data in q0_as_dict["data"].items():
        if submodule_name == "name":
            continue
        for parameter_name, parameter_val in submodule_data.items():
            assert parameter_val == q0.submodules[submodule_name][parameter_name](), (
                f"Expected value {q0.submodules[submodule_name][parameter_name]()} for "
                f"{submodule_name}.{parameter_name} but got {parameter_val}"
            )


def test_basic_transmon_deserialization(q0: BasicTransmonElement, dev: QuantumDevice):
    """
    Tests the deserialization process of :class:`~BasicTransmonElement` by comparing the
    operations inside compiled schedules of the original and the deserialized
    `BasicTransmonElement` object.
    """

    q0.measure.acq_channel(0)
    q0.measure.pulse_amp(0.05)
    q0.clock_freqs.readout(3e9)
    dev.add_element(q0)

    sched = Schedule("test_basic_transmon_deserialization")
    sched.add(Measure(q0.name))

    compiler = SerialCompiler(name="compiler")
    compiled_sched_q0 = compiler.compile(
        schedule=sched, config=dev.generate_compilation_config()
    )

    q0_as_str = json.dumps(q0, cls=ScheduleJSONEncoder)
    assert q0_as_str.__class__ is str

    q0.close()

    deserialized_q0 = json.loads(q0_as_str, cls=ScheduleJSONDecoder)
    assert deserialized_q0.__class__ is BasicTransmonElement

    compiled_sched_deserialized_q0 = compiler.compile(
        schedule=sched, config=dev.generate_compilation_config()
    )
    assert compiled_sched_deserialized_q0.operations == compiled_sched_q0.operations, (
        f"Compiled operations of deserialized '{deserialized_q0.name}' does not match "
        f"the original's"
    )

    deserialized_q0.close()
