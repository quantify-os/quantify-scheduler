# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
"""defines interfaces for operation handling strategies."""

from __future__ import annotations

from typing import Dict, Any

from abc import ABC, abstractmethod


from quantify_scheduler.backends.types import qblox as types
from quantify_scheduler.backends.qblox.qasm_program import QASMProgram


class IOperationStrategy(ABC):
    @property
    @abstractmethod
    def operation_info(self) -> types.OpInfo:
        pass

    @abstractmethod
    def generate_data(self, wf_dict: Dict[str, Any]):
        pass

    @abstractmethod
    def insert_qasm(self, qasm_program: QASMProgram):
        pass
