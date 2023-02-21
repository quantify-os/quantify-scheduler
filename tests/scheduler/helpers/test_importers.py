from quantify_scheduler.helpers.importers import (
    export_python_object_to_path_string,
    import_python_object_from_string,
)


def test_import_python_object_from_string_1():
    from quantify_scheduler.backends.qblox_backend import hardware_compile

    import_string = "quantify_scheduler.backends.qblox_backend.hardware_compile"
    obj = import_python_object_from_string(import_string)
    assert obj is hardware_compile


def test_import_python_object_from_string_2():
    from quantify_scheduler.operations.pulse_library import IdlePulse

    import_string = "quantify_scheduler.operations.pulse_library.IdlePulse"
    obj = import_python_object_from_string(import_string)
    assert obj is IdlePulse


def test_python_object_to_path_string_1():
    from quantify_scheduler.backends.qblox_backend import hardware_compile

    import_string = export_python_object_to_path_string(hardware_compile)
    assert import_string == "quantify_scheduler.backends.qblox_backend.hardware_compile"


def test_python_object_to_path_string_2():
    from quantify_scheduler.operations.pulse_library import IdlePulse

    import_string = export_python_object_to_path_string(IdlePulse)
    assert import_string == "quantify_scheduler.operations.pulse_library.IdlePulse"


def test_python_object_to_relative_path():
    from .test_inspect import test_get_classes

    import_string = export_python_object_to_path_string(test_get_classes)
    assert import_string == "tests.scheduler.helpers.test_inspect.test_get_classes"
