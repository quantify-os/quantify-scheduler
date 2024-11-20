from __future__ import annotations

import importlib

import pytest


def _get_public_items(module):
    """Get public items that are defined in the module, handling decorated functions."""

    def is_defined_in_module(obj, module_name):
        if hasattr(obj, "__module__"):
            return obj.__module__.startswith(module_name)
        return False

    public_items = set()
    for name in dir(module):
        if name.startswith("_"):
            continue

        obj = getattr(module, name)

        # Check if the object itself is defined in the module
        if is_defined_in_module(obj, module.__name__):
            public_items.add(name)
            continue

    return public_items


@pytest.mark.parametrize(
    "main_module,sub_modules,exceptions",
    [
        (
            "quantify_scheduler.operations",
            [
                "operation",
                "pulse_library",
                "gate_library",
                "pulse_factories",
                "nv_native_library",
                "control_flow_library",
                "acquisition_library",
            ],
            {
                "create_dc_compensation_pulse",
                "decompose_long_square_pulse",
                "Loop",
                "Conditional",
                "AcquisitionOperation",  # deprecated
            },
        ),
        ("quantify_scheduler", ["resources"], None),
        ("quantify_scheduler.instrument_coordinator", ["instrument_coordinator"], None),
    ],
)
def test_aliases(main_module: str, sub_modules: list[str], exceptions: set[str] | None):
    """
    Test to make sure all classes/methods of a given module are
    imported and put in __all__ in that module's __init__.py
    """
    exceptions = exceptions or set()
    # Import the main module
    main = importlib.import_module(main_module)

    # Get all items defined in __all__
    all_items = set(main.__all__) if hasattr(main, "__all__") else set()

    # Collect all public items from sub-modules
    expected_items = set()
    for sub_module in sub_modules:
        sub = importlib.import_module(main_module + "." + sub_module)
        expected_items.update(_get_public_items(sub))

    # Prepare detailed error message
    error = False
    error_message = []
    if exceptions - expected_items != set():
        error_message.append(
            f"Items in exceptions don't exist in the indicated submodules in {main_module}:"
        )
        for item in sorted(exceptions - expected_items):
            error_message.append(f"  - {item}")
        error = True

    # Check if all expected items are in __all__
    missing_all = expected_items - all_items - exceptions

    if missing_all:
        error_message.append(
            f"Items defined in sub-modules but missing from {main_module}.__all__:"
        )
        for item in sorted(missing_all):
            error_message.append(f"  - {item}")
        error_message.append(
            "To exclude an item from having to be imported in __init__.py, "
            "please add it as an exception in this test"
        )
        error = True

    # Assert and provide detailed error message
    assert not error, "\n".join(error_message)
