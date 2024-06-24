# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Exceptions used by Qblox backend."""


class NcoOperationTimingError(ValueError):
    """Exception thrown if there are timing errors for NCO operations."""
