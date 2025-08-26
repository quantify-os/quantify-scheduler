import copy
import json
import sys
import types
from importlib import resources

import numpy as np
import pytest

from quantify_scheduler import QuantumDevice
from quantify_scheduler.helpers.importers import export_python_object_to_path_string
from quantify_scheduler.json_utils import (
    SchedulerJSONDecoder,
    SchedulerJSONEncoder,
    UnknownDeserializationTypeError,
)
from quantify_scheduler.operations.gate_library import Measure, Reset
from quantify_scheduler.operations.pulse_library import DRAGPulse, IdlePulse, ReferenceMagnitude
from quantify_scheduler.resources import ClockResource
from quantify_scheduler.schedules import timedomain_schedules
from quantify_scheduler.schedules.schedule import Schedule
from quantify_scheduler.schedules.spectroscopy_schedules import heterodyne_spec_sched


def test_getsetstate_json_serialization():
    # Create a test-class for testing the json serialization/deserialization
    class MockClass:
        def __init__(self):
            self.value = 0

        def __getstate__(self):
            return {"deserialization_type": "MockClass", "data": {"value": self.value}}

        def __setstate__(self, state):
            self.value = state["data"]["value"]

    # create and import a module containing this class
    test_module = types.ModuleType("test_module")
    test_module.MockClass = MockClass
    MockClass.__module__ = "test_module"
    sys.modules["test_module"] = test_module

    mockinstance1 = MockClass()
    # Check that for the test class, it works as expected
    assert mockinstance1.__dict__ == copy.copy(mockinstance1).__dict__
    serialized = json.dumps(mockinstance1, cls=SchedulerJSONEncoder)
    assert serialized == json.dumps({"deserialization_type": "MockClass", "data": {"value": 0}})

    # Try to decode an unknown class should fail with a ValueError
    with pytest.raises(UnknownDeserializationTypeError):
        assert json.loads(serialized, cls=SchedulerJSONDecoder)

    # We can tell the decoder about non-standard modules that it should be able to
    # decode
    mockinstance2 = SchedulerJSONDecoder(modules=[test_module]).decode(serialized)
    assert mockinstance2.__dict__ == mockinstance1.__dict__

    # Clear the test_module from sys.modules
    del sys.modules["test_module"]

    # Create a class which improperly implements get_state
    class MockClassFail:
        def __init__(self):
            self.value = 0

        def __getstate__(self):
            return {
                "deserialization_type": "MockClassFail",
                "data": {"value": self.value + 1},
            }

        def __setstate__(self, state):
            self.value = state["data"]["value"]

    mockinstance3 = MockClassFail()
    # Check that the copy mechanism of python indeed uses getstate and setstate
    assert copy.copy(mockinstance3).__dict__ != mockinstance3.__dict__
    assert copy.copy(mockinstance3).value == 1


class MockClass2:
    """
    Mock class to test serialization and deserialization via
    SchedulerJSONEncoder/SchedulerJSONDecoder.
    """

    def __init__(self):
        self.value = 0

    def __getstate__(self):
        return {
            "deserialization_type": export_python_object_to_path_string(self.__class__),
            "data": {"value": self.value},
        }

    def __setstate__(self, state):
        self.value = state["data"]["value"]


def test_deserialization_fully_qualified_path():
    # Test that a serialized and deserialized object is equal to the original object, if
    # the class uses a dot-separated path as "deserialization_type".
    mock_instance1 = MockClass2()
    serialized = json.dumps(mock_instance1, cls=SchedulerJSONEncoder)
    mock_instance2 = json.loads(serialized, cls=SchedulerJSONDecoder)
    assert mock_instance1.__dict__ == mock_instance2.__dict__


def test_deprecated_deserialization_fallback():
    # Test serialization/deserialization of an object whose type is known to be listed
    # in the deprecated list of types.
    pulse = IdlePulse(duration=100e-9)
    state = pulse.__getstate__()

    # Overwrite deserialization_type to use the deprecated string.
    state["deserialization_type"] = IdlePulse.__name__
    serialized = json.dumps(state)
    with pytest.warns(FutureWarning):
        deserialized = json.loads(serialized, cls=SchedulerJSONDecoder)
    assert isinstance(deserialized, IdlePulse)


def simple_schedule(ref_mag=None):
    schedule = Schedule("Test", 1)
    clock = "q0.01"
    frequency = 4509922570.610321
    schedule.add_resource(ClockResource(name=clock, freq=frequency))
    qubit = "q0"
    port = "q0:mw"
    duration = 2.0e-8
    max_duration = 2e-08

    op = schedule.add(Reset(qubit), label="Reset all qubits")

    schedule.add(
        DRAGPulse(
            duration=duration,
            G_amp=0.0,
            D_amp=0,
            port=port,
            clock=clock,
            phase=0,
            reference_magnitude=ref_mag,
        ),
        # Make sure the rabi pulses are parallel not sequential
        ref_op=op,
        label=f"Rabi_pulse on {qubit}",
    )

    schedule.add(
        Measure(qubit, coords={"index": 0}, acq_protocol="SSBIntegrationComplex"),
        label="Multiplexed measurement",
        rel_time=max_duration,
        ref_pt="start",
    )

    return schedule


@pytest.mark.parametrize(
    "schedule",
    [
        timedomain_schedules.t1_sched(np.zeros(1), "q0"),
        heterodyne_spec_sched(0.1, 0.1, 6e9, 1e-7, 1e-6, "q0:mw", "q0.01", 200e-6, 1024),
        simple_schedule(None),
        simple_schedule(ref_mag=ReferenceMagnitude(value=0, unit="dBm")),
    ],
)
def test_schedule_to_and_from_json(schedule):
    json_data = schedule.to_json()
    result = Schedule.from_json(json_data)

    assert schedule == result
    assert schedule.data == result.data
    assert schedule.__dict__ == result.__dict__


def test_quantum_device_json_compatibility():
    """Verify compatibility of old and new `QuantumDevice` JSON serialization formats."""
    with (
        resources.path("tests.data", "qdevice_with_two_qubits_new_format.json") as new_path,
        resources.path("tests.data", "qdevice_with_two_qubits_old_format.json") as old_path,
    ):
        new = QuantumDevice.from_json_file(str(new_path))
        new.close_all()
        old = QuantumDevice.from_json_file(str(old_path))
    assert new.to_json() == old.to_json()
