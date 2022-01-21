# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Pytest fixtures for quantify-scheduler."""
from __future__ import annotations

import pytest
from qcodes import Instrument


@pytest.fixture(autouse=False)
def close_all_instruments():
    """Makes sure that after startup and teardown all instruments are closed"""
    Instrument.close_all()
    yield
    Instrument.close_all()
