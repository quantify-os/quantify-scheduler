from quantify_scheduler.helpers import inspect


def test_get_classes() -> None:
    # Arrange
    from quantify_scheduler.operations import gate_library

    # Act
    classes = inspect.get_classes(gate_library)

    # Assert
    assert "X90" in classes
    assert isinstance(classes["X90"], type)
