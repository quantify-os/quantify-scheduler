# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Defines interfaces for operation handling strategies."""

from __future__ import annotations

from typing import Dict, Any

from abc import ABC, abstractmethod


from quantify_scheduler.backends.types import qblox as types
from quantify_scheduler.backends.qblox.qasm_program import QASMProgram


class IOperationStrategy(ABC):
    """Defines the interface operation strategies must adhere to."""

    @property
    @abstractmethod
    def operation_info(self) -> types.OpInfo:
        """Returns the pulse/acquisition information extracted from the schedule."""

    @abstractmethod
    def generate_data(self, wf_dict: Dict[str, Any]):
        """
        Generates the waveform data and adds them to the wf_dict (if not already
        present). This is either the awg data, or the acquisition weights.

        Parameters
        ----------
        wf_dict
            The dictionary to add the waveform to. N.B. the dictionary is modified in
            function.
        """

    @abstractmethod
    def insert_qasm(self, qasm_program: QASMProgram):
        """
        Add the assembly instructions for the Q1 sequence processor that corresponds to
        this pulse/acquisition.

        Parameters
        ----------
        qasm_program
            The QASMProgram to add the assembly instructions to.
        """
