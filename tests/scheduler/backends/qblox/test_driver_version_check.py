# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=redefined-outer-name
# pylint: disable=missing-module-docstring
# pylint: disable=no-self-use

# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
"""Tests for the driver version check."""

import pytest

from qblox_instruments import build

from quantify_scheduler.backends.qblox.driver_version_check import (
    verify_qblox_instruments_version,
)
from quantify_scheduler.backends.qblox.driver_version_check import DriverVersionError


def test_verify_qblox_instruments_version():
    verify_qblox_instruments_version(build.__version__)

    nonsense_version = "nonsense.driver.version"
    with pytest.raises(DriverVersionError) as wrong_version:
        verify_qblox_instruments_version(nonsense_version)
    assert (
        wrong_version.value.args[0]
        == f"Qblox DriverVersionError: Installed driver version {nonsense_version} not "
        f"supported by backend. Please install a supported version (currently "
        f"supported: ('0.5.0', '0.5.1', '0.5.2')) to continue to use this backend."
    )

    with pytest.raises(DriverVersionError) as none_error:
        verify_qblox_instruments_version(None)

    assert (
        none_error.value.args[0]
        == "Qblox DriverVersionError: qblox-instruments version check could not be "
        "performed. Either the package is not installed correctly or a version < "
        "0.3.2 was found."
    )
