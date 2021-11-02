# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
from quantify_scheduler.helpers import inspect


def test_get_classes():
    # Arrange
    from quantify_scheduler import (
        gate_library,
    )  # pylint: disable=import-outside-toplevel

    # Act
    classes = inspect.get_classes(gate_library)

    # Assert
    assert "X90" in classes
    assert isinstance(classes["X90"], type)
