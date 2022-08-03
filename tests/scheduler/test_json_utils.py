# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-module-docstring

import copy
import json
import sys
import types

import pytest

from quantify_scheduler.json_utils import ScheduleJSONEncoder, ScheduleJSONDecoder


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
    serialized = json.dumps(mockinstance1, cls=ScheduleJSONEncoder)
    assert serialized == json.dumps(
        {"deserialization_type": "MockClass", "data": {"value": 0}}
    )

    # Try to decode an unknown class should fail with a keyerror
    with pytest.raises(KeyError):
        assert json.loads(serialized, cls=ScheduleJSONDecoder)

    # We can tell the decoder about non-standard modules that it should be able to
    # decode
    mockinstance2 = ScheduleJSONDecoder(modules=[test_module]).decode(serialized)
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
