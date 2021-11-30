from __future__ import annotations

from typing import Optional, Dict, Tuple, Any, Union

from abc import ABC, abstractmethod

import numpy as np

from quantify_scheduler.backends.types import qblox as types
from quantify_scheduler.backends.qblox.qasm_program import QASMProgram
from quantify_scheduler.backends.qblox import helpers, constants, q1asm_instructions


class IOperationStrategy(ABC):
    @property
    @abstractmethod
    def operation_info(self) -> types.OpInfo:
        pass

    @abstractmethod
    def generate_data(
        self, wf_dict: Dict[str, Any], output_mode: str
    ) -> Optional[np.ndarray]:
        pass

    @abstractmethod
    def insert_qasm(self, qasm_program: QASMProgram):
        pass


def get_indices_from_wf_dict(uuid: str, wf_dict: Dict[str, Any]) -> Tuple[int, int]:
    """
    Takes a waveforms_dict or weights_dict and extracts the waveform indices based
    off of the uuid of the pulse/acquisition.

    Parameters
    ----------
    uuid
        The unique identifier of the pulse/acquisition.
    wf_dict
        The awg or acq dict that holds the waveform data and indices.

    Returns
    -------
    :
        Index of the I waveform.
    :
        Index of the Q waveform.
    """
    name_real, name_imag = helpers.generate_waveform_names_from_uuid(uuid)
    idx_real = None if name_real not in wf_dict else wf_dict[name_real]["index"]
    idx_imag = None if name_imag not in wf_dict else wf_dict[name_imag]["index"]
    return idx_real, idx_imag
