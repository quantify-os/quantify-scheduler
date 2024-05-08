import json

import numpy as np
import pytest
import math

from quantify_scheduler import Schedule
from quantify_scheduler.backends.circuit_to_device import DeviceCompilationConfig
from quantify_scheduler.backends.graph_compilation import SerialCompiler
from quantify_scheduler.device_under_test.transmon_element import BasicTransmonElement
from quantify_scheduler.device_under_test.quantum_device import QuantumDevice
from quantify_scheduler.json_utils import SchedulerJSONEncoder, SchedulerJSONDecoder
from quantify_scheduler.operations.gate_library import Measure


@pytest.fixture
def q0() -> BasicTransmonElement:
    q0 = BasicTransmonElement("q0")
    yield q0


@pytest.fixture
def dev() -> QuantumDevice:
    dev = QuantumDevice("dev")
    yield dev


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

    def is_serialized_ndarray(obj):
        return isinstance(obj, dict) and obj["deserialization_type"] == "ndarray"

    q0.clock_freqs.readout(readout_frequency)
    q0.clock_freqs.f01(mw_frequency)
    q0.clock_freqs.f12(0)
    q0.measure.acq_delay(acq_delay)
    q0.rxy.amp180(pulse_amp)
    q0.rxy.reference_magnitude.dBm(-10)
    q0.measure.reference_magnitude.dBm(-10)

    q0_as_dict = json.loads(json.dumps(q0, cls=SchedulerJSONEncoder))
    assert q0_as_dict.__class__ is dict
    assert (
        q0_as_dict["deserialization_type"]
        == "quantify_scheduler.device_under_test.transmon_element.BasicTransmonElement"
    )

    # Check that all original submodule params match their serialized counterpart
    for submodule_name, submodule in q0.submodules.items():
        for parameter_name in submodule.parameters:
            if is_serialized_ndarray(
                q0_as_dict["data"][submodule_name][parameter_name]
            ):
                np.testing.assert_equal(
                    q0_as_dict["data"][submodule_name][parameter_name]["data"],
                    q0.submodules[submodule_name][parameter_name](),
                )
            else:
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
            if is_serialized_ndarray(parameter_val):
                np.testing.assert_equal(
                    parameter_val["data"],
                    q0.submodules[submodule_name][parameter_name](),
                )
            else:
                assert (
                    parameter_val == q0.submodules[submodule_name][parameter_name]()
                ), (
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

    q0_as_str = json.dumps(q0, cls=SchedulerJSONEncoder)
    assert q0_as_str.__class__ is str

    q0.close()

    deserialized_q0 = json.loads(q0_as_str, cls=SchedulerJSONDecoder)
    assert deserialized_q0.__class__ is BasicTransmonElement

    compiled_sched_deserialized_q0 = compiler.compile(
        schedule=sched, config=dev.generate_compilation_config()
    )
    assert compiled_sched_deserialized_q0.operations == compiled_sched_q0.operations, (
        f"Compiled operations of deserialized '{deserialized_q0.name}' does not match "
        f"the original's"
    )


def test_reference_magnitude_overwrite_units(q0: BasicTransmonElement):
    """
    Tests that the amplitude reference parameters get correctly ovewritten when you
    call the set method of a different unit parameter
    """
    # All units should initially be nan
    assert math.isnan(q0.rxy.reference_magnitude.dBm())
    assert math.isnan(q0.rxy.reference_magnitude.V())

    # Set dBm unit
    q0.rxy.reference_magnitude.dBm(-10)
    assert q0.rxy.reference_magnitude.dBm() == -10
    assert math.isnan(q0.rxy.reference_magnitude.V())

    # Set V unit
    q0.rxy.reference_magnitude.V(10e-3)
    assert q0.rxy.reference_magnitude.V() == 10e-3
    assert math.isnan(q0.rxy.reference_magnitude.dBm())

    assert q0.rxy.reference_magnitude.get_val_unit() == (10e-3, "V")

    # Set A unit
    q0.rxy.reference_magnitude.A(1e-3)
    assert q0.rxy.reference_magnitude.A() == 1e-3
    assert math.isnan(q0.rxy.reference_magnitude.V())

    assert q0.rxy.reference_magnitude.get_val_unit() == (1e-3, "A")

    # Set nan
    q0.rxy.reference_magnitude.V(float("nan"))

    assert math.isnan(q0.rxy.reference_magnitude.V())
    assert q0.rxy.reference_magnitude.A() == 1e-3


def test_generate_config_measure(q0: BasicTransmonElement):
    """Setting values updates the correct values in the config."""
    # Set values for measure
    q0.measure.pulse_amp(0.1234)
    q0.measure.pulse_duration(300e-6)
    q0.measure.acq_channel(123)
    q0.measure.acq_delay(13e-6)
    q0.measure.integration_time(8e-7)
    q0.measure.reset_clock_phase(False)

    # Get device config
    dev_cfg = q0.generate_device_config()
    cfg_measure = dev_cfg.elements["q0"]["measure"]

    # Assert values are in right place
    assert cfg_measure.factory_kwargs["port"] == "q0:res"
    assert cfg_measure.factory_kwargs["clock"] == "q0.ro"
    assert cfg_measure.factory_kwargs["pulse_type"] == "SquarePulse"
    assert cfg_measure.factory_kwargs["pulse_amp"] == 0.1234
    assert cfg_measure.factory_kwargs["pulse_duration"] == 300e-6
    assert cfg_measure.factory_kwargs["acq_delay"] == 13e-6
    assert cfg_measure.factory_kwargs["acq_duration"] == 8e-7
    assert cfg_measure.factory_kwargs["acq_channel"] == 123
    assert not cfg_measure.factory_kwargs["reset_clock_phase"]

    # Changing values of the measure
    q0.measure.acq_channel("ch_123")

    dev_cfg = q0.generate_device_config()
    cfg_measure = dev_cfg.elements["q0"]["measure"]

    # Assert values are in right place
    assert cfg_measure.factory_kwargs["acq_channel"] == "ch_123"
