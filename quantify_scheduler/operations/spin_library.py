# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch

"""Spin qubit specific operations for use with the quantify_scheduler."""
from __future__ import annotations

from .operation import Operation


class SpinInit(Operation):
    """
    Initialize a spin qubit system.

    Parameters
    ----------
    qubits
        The qubits to initialize.

    """

    def __init__(self, qC: str, qT: str, **device_overrides) -> None:
        super().__init__(name=f"SpinInit ({qC}, {qT})")
        self.data.update(
            {
                "name": self.name,
                "gate_info": {
                    "unitary": None,
                    "plot_func": "quantify_scheduler.schedules._visualization."
                    + "circuit_diagram.reset",
                    "tex": r"SpinInit",
                    "qubits": [qC, qT],
                    "operation_type": "SpinInit",
                    "device_overrides": device_overrides,
                },
            }
        )
        self.update()

    def __str__(self) -> str:
        qubits = map(lambda x: f"'{x}'", self.data["gate_info"]["qubits"])
        return f'{self.__class__.__name__}({",".join(qubits)})'
