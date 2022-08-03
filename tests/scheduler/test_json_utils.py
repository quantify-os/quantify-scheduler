import copy
import json
import pytest
import sys
import types

from quantify_scheduler.json_utils import ScheduleJSONEncoder, ScheduleJSONDecoder


def test_getsetstate_json_serialization():
    # Create a test-class for testing the json serialization/deserialization
    class foo:
        def __init__(self):
            self.bar = 0

        def __getstate__(self):
            return {"deserialization_type": "foo", "data": {"bar": self.bar}}

        def __setstate__(self, state):
            self.bar = state["data"]["bar"]

    # create and import a module containing this class
    test_module = types.ModuleType("test_module")
    test_module.foo = foo
    foo.__module__ = "test_module"
    sys.modules["test_module"] = test_module

    f1 = foo()
    # Check that for the test class, it works as expected
    assert f1.__dict__ == copy.copy(f1).__dict__
    serialized = json.dumps(f1, cls=ScheduleJSONEncoder)
    assert serialized == json.dumps({"deserialization_type": "foo", "data": {"bar": 0}})

    # Try to decode an unknown class should fail with a keyerror
    with pytest.raises(KeyError):
        assert json.loads(serialized, cls=ScheduleJSONDecoder)

    # We can tell the decoder about non-standard modules that it should be able to decode
    f2 = ScheduleJSONDecoder(modules=[test_module]).decode(serialized)
    assert f2.__dict__ == f1.__dict__

    # Clear the test_module from sys.modules
    del sys.modules["test_module"]

    # Create a class which improperly implements get_state
    class foo_fail:
        def __init__(self):
            self.bar = 0

        def __getstate__(self):
            return {"deserialization_type": "foo", "data": {"bar": self.bar + 1}}

        def __setstate__(self, state):
            self.bar = state["data"]["bar"]

    f1 = foo_fail()
    # Check that the copy mechanism of python indeed uses getstate and setstate
    assert copy.copy(f1).__dict__ != f1.__dict__
    assert copy.copy(f1).bar == 1
