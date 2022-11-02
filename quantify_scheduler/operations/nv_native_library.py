# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
# pylint: disable=invalid-name
"""NV center specific gateset for use with the quantify_scheduler."""
from typing import Optional
from .operation import Operation


class ChargeReset(Operation):
    r"""
    Reset a NV to its negative charge state NV$^-$.

    .. note::

        Strictly speaking this is not a gate as it can not
        be described by a unitary.
    """

    def __init__(self, *qubits: str):
        """
        Create a new instance of ChargeReset operation that is used to initialize the
        charge state of an NV center.

        Parameters
        ----------
        qubit
            The qubit to reset. NB one or more qubits can be specified, e.g.,
            :code:`Reset("q0")`, :code:`Reset("q0", "q1", "q2")`, etc..
        """

        super().__init__(name=f"ChargeReset {', '.join(qubits)}")
        self.data.update(
            {
                "name": f"ChargeReset {', '.join(qubits)}",
                "gate_info": {
                    "unitary": None,
                    "plot_func": "quantify_scheduler.visualization."
                    + "circuit_diagram.reset",
                    "tex": r"$NV^-$",
                    "qubits": list(qubits),
                    "operation_type": "charge_reset",
                },
            }
        )
        self.update()

    def __str__(self) -> str:
        qubits = map(lambda x: f"'{x}'", self.data["gate_info"]["qubits"])
        return f'{self.__class__.__name__}({",".join(qubits)})'
