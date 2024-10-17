# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Module containing shared native operations."""
from .operation import Operation


class SpectroscopyOperation(Operation):
    """
    Spectroscopy operation to find energy between computational basis states.

    Spectroscopy operations can be supported by various qubit types, but not all of
    them. They are typically translated into a spectroscopy pulse by the quantum
    device. The frequency is taken from a clock of the device element.

    Parameters
    ----------
    qubit
        The target qubit.

    """

    def __init__(
        self,
        qubit: str,
    ) -> None:
        super().__init__(name=f"Spectroscopy operation {qubit}")
        self.data.update(
            {
                "gate_info": {
                    "unitary": None,
                    "plot_func": "quantify_scheduler.schedules._visualization"
                    ".circuit_diagram.pulse_modulated",
                    "tex": r"Spectroscopy operation",
                    "qubits": [qubit],
                    "operation_type": "spectroscopy_operation",
                },
            }
        )
        self._update()

    def __str__(self) -> str:
        gate_info = self.data["gate_info"]
        qubit = gate_info["qubits"][0]
        return f'{self.__class__.__name__}(qubit="{qubit}")'
