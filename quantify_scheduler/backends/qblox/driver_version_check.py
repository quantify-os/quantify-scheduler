# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
"""Helper functions to perform the version check for qblox_instruments."""

from typing import Tuple

try:
    from qblox_instruments.build import __version__ as driver_version
except ImportError:
    driver_version = None  # Prior to v0.3.2 __version__ was not there

SUPPORTED_DRIVER_VERSIONS: Tuple[str, ...] = ("0.5.0", "0.5.1", "0.5.2")
"""Tuple containing all the version supported by this version of the backend."""
raise_on_version_mismatch: bool = True
"""Can be set to false to override version check."""


class DriverVersionError(Exception):
    """
    Raise when the installed driver version is not supported
    """


def verify_qblox_instruments_version(version=driver_version):
    """
    Verifies whether the installed version is supported by the qblox_backend.

    Raises
    ------
    DriverVersionError
        When an incorrect or no installation of qblox-instruments was found.
    """
    if not raise_on_version_mismatch:
        return
    if version is None:
        raise DriverVersionError(
            "Qblox DriverVersionError: qblox-instruments version check could not be"
            " performed. Either the package is not installed "
            "correctly or a version < 0.3.2 was found."
        )
    if version not in SUPPORTED_DRIVER_VERSIONS:
        message = (
            f"Qblox DriverVersionError: Installed driver version {version}"
            f" not supported by backend."
        )
        message += (
            f" Please install version {SUPPORTED_DRIVER_VERSIONS[0]}"
            if len(SUPPORTED_DRIVER_VERSIONS) == 1
            else f" Please install a supported version (currently supported: "
            f"{SUPPORTED_DRIVER_VERSIONS})"
        )
        message += " to continue to use this backend."
        raise DriverVersionError(message)
