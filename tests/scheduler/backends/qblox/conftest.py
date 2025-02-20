# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
from unittest.mock import Mock

import pytest

from quantify_scheduler.backends.qblox.instrument_compilers import (
    QCMCompiler,
    QCMRFCompiler,
    QRMCompiler,
    QRMRFCompiler,
    QTMCompiler,
)
from quantify_scheduler.backends.qblox.qasm_program import QASMProgram
from quantify_scheduler.backends.qblox.register_manager import RegisterManager
from quantify_scheduler.backends.types.qblox import (
    QCMDescription,
    QCMRFDescription,
    QRMDescription,
    QRMRFDescription,
    QTMDescription,
)


@pytest.fixture()
def empty_qasm_program_qcm():
    """Empty QASMProgram object."""
    yield QASMProgram(
        static_hw_properties=QCMCompiler.static_hw_properties,
        register_manager=RegisterManager(),
        align_fields=True,
        acq_metadata=None,
    )


@pytest.fixture()
def empty_qasm_program_qrm():
    yield QASMProgram(
        static_hw_properties=QRMCompiler.static_hw_properties,
        register_manager=RegisterManager(),
        align_fields=True,
        acq_metadata=None,
    )


@pytest.fixture()
def empty_qasm_program_qrm_rf():
    yield QASMProgram(
        static_hw_properties=QRMRFCompiler.static_hw_properties,
        register_manager=RegisterManager(),
        align_fields=True,
        acq_metadata=None,
    )


@pytest.fixture()
def empty_qasm_program_qcm_rf():
    yield QASMProgram(
        static_hw_properties=QCMRFCompiler.static_hw_properties,
        register_manager=RegisterManager(),
        align_fields=True,
        acq_metadata=None,
    )


@pytest.fixture()
def empty_qasm_program_qtm():
    yield QASMProgram(
        static_hw_properties=QTMCompiler.static_hw_properties,
        register_manager=RegisterManager(),
        align_fields=True,
        acq_metadata=None,
    )
