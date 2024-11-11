# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""The module provides classes related CZ operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from qcodes.instrument import InstrumentChannel
from qcodes.instrument.parameter import ManualParameter

from quantify_scheduler.backends.graph_compilation import OperationCompilationConfig
from quantify_scheduler.device_under_test.edge import Edge
from quantify_scheduler.helpers.validators import Numbers
from quantify_scheduler.operations.pulse_factories import composite_square_pulse
from quantify_scheduler.resources import BasebandClockResource

if TYPE_CHECKING:
    from qcodes.instrument.base import InstrumentBase


class CZ(InstrumentChannel):
    """Submodule containing parameters for performing a CZ operation."""

    def __init__(self, parent: InstrumentBase, name: str, **kwargs: float) -> None:
        super().__init__(parent=parent, name=name)
        self.square_amp = ManualParameter(
            "square_amp",
            docstring=r"""Amplitude of the square envelope.""",
            unit="V",
            initial_value=kwargs.get("square_amp", 0.5),
            vals=Numbers(min_value=0, max_value=1e12, allow_nan=True),
            instrument=self,
        )

        self.square_duration = ManualParameter(
            "square_duration",
            docstring=r"""The square pulse duration in seconds.""",
            unit="s",
            initial_value=kwargs.get("square_duration", 2e-8),
            vals=Numbers(min_value=0, max_value=1e12, allow_nan=True),
            instrument=self,
        )

        self.add_parameter(
            name=f"{parent._parent_element_name}_phase_correction",
            docstring=r"""The phase correction for the parent qubit after the"""
            r""" square pulse operation has been performed.""",
            unit="degrees",
            parameter_class=ManualParameter,
            initial_value=kwargs.get(f"{parent._parent_element_name}_phase_correction", 0),
            vals=Numbers(min_value=-1e12, max_value=1e12, allow_nan=True),
        )
        self.add_parameter(
            name=f"{parent._child_element_name}_phase_correction",
            docstring=r"""The phase correction for the child qubit after the"""
            r""" Square pulse operation has been performed.""",
            unit="degrees",
            parameter_class=ManualParameter,
            initial_value=kwargs.get(f"{parent._child_element_name}_phase_correction", 0),
            vals=Numbers(min_value=-1e12, max_value=1e12, allow_nan=True),
        )


class CompositeSquareEdge(Edge):
    """
    An example Edge implementation which connects two BasicTransmonElements.

    This edge implements a square flux pulse and two virtual z
    phase corrections for the CZ operation between the two BasicTransmonElements.
    """

    def __init__(
        self,
        parent_element_name: str,
        child_element_name: str,
        **kwargs,
    ) -> None:
        cz_data = kwargs.pop("cz", {})

        super().__init__(
            parent_element_name=parent_element_name,
            child_element_name=child_element_name,
            **kwargs,
        )

        self.add_submodule("cz", CZ(parent=self, name="cz", **cz_data))

    def generate_edge_config(self) -> dict[str, dict[str, OperationCompilationConfig]]:
        """
        Generate valid device config.

        Fills in the edges information to produce a valid device config for the
        quantify-scheduler making use of the
        :func:`~.circuit_to_device.compile_circuit_to_device_with_config_validation` function.
        """
        edge_op_config = {
            f"{self.name}": {
                "CZ": OperationCompilationConfig(
                    factory_func=composite_square_pulse,
                    factory_kwargs={
                        "square_port": self.parent_device_element.ports.flux(),
                        "square_clock": BasebandClockResource.IDENTITY,
                        "square_amp": self.cz.square_amp(),
                        "square_duration": self.cz.square_duration(),
                        "virt_z_parent_qubit_phase": self.cz.parameters[
                            f"{self._parent_element_name}_phase_correction"
                        ](),
                        "virt_z_parent_qubit_clock": f"{self.parent_device_element.name}.01",
                        "virt_z_child_qubit_phase": self.cz.parameters[
                            f"{self._child_element_name}_phase_correction"
                        ](),
                        "virt_z_child_qubit_clock": f"{self.child_device_element.name}.01",
                    },
                ),
            }
        }

        return edge_op_config
