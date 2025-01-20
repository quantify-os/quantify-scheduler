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
    qC
        The control device element.
    qT
        The target device element
    device_overrides
        Device level parameters that override device configuration values
        when compiling from circuit to device level.

    """

    def __init__(self, qC: str, qT: str, **device_overrides) -> None:
        device_element_control, device_element_target = qC, qT
        super().__init__(name=f"SpinInit ({device_element_control}, {device_element_target})")
        self.data.update(
            {
                "name": self.name,
                "gate_info": {
                    "unitary": None,
                    "plot_func": "quantify_scheduler.schedules._visualization."
                    + "circuit_diagram.reset",
                    "tex": r"SpinInit",
                    "device_elements": [device_element_control, device_element_target],
                    "operation_type": "SpinInit",
                    "device_overrides": device_overrides,
                },
            }
        )
        self.update()

    def __str__(self) -> str:
        device_elements = map(lambda x: f"'{x}'", self.data["gate_info"]["device_elements"])
        return f'{self.__class__.__name__}({",".join(device_elements)})'
