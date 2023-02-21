# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch

from typing import Dict, Any

from qcodes.instrument import InstrumentChannel
from qcodes.instrument.base import InstrumentBase
from qcodes.instrument.parameter import ManualParameter

from quantify_scheduler.backends.circuit_to_device import OperationCompilationConfig
from quantify_scheduler.helpers.validators import Numbers
from quantify_scheduler.device_under_test.edge import Edge
from quantify_scheduler.operations.pulse_factories import composite_square_pulse
from quantify_scheduler.resources import BasebandClockResource


class CZ(InstrumentChannel):
    """
    Submodule containing parameters for performing a CZ operation
    """

    def __init__(self, parent: InstrumentBase, name: str, **kwargs: Any) -> None:
        super().__init__(parent=parent, name=name)
        self.add_parameter(
            name="square_amp",
            docstring=r"""Amplitude of the square envelope.""",
            unit="V",
            parameter_class=ManualParameter,
            initial_value=0.5,
            vals=Numbers(min_value=0, max_value=1e12, allow_nan=True),
        )
        self.add_parameter(
            name="square_duration",
            docstring=r"""The square pulse duration in seconds.""",
            unit="s",
            parameter_class=ManualParameter,
            initial_value=2e-8,
            vals=Numbers(min_value=0, max_value=1e12, allow_nan=True),
        )
        self.add_parameter(
            name=f"{parent._parent_element_name}_phase_correction",
            docstring=r"""The phase correction for the parent qubit after the"""
            r""" square pulse operation has been performed.""",
            unit="degrees",
            parameter_class=ManualParameter,
            initial_value=0,
            vals=Numbers(min_value=-1e12, max_value=1e12, allow_nan=True),
        )
        self.add_parameter(
            name=f"{parent._child_element_name}_phase_correction",
            docstring=r"""The phase correction for the child qubit after the"""
            r""" Square pulse operation has been performed.""",
            unit="degrees",
            parameter_class=ManualParameter,
            initial_value=0,
            vals=Numbers(min_value=-1e12, max_value=1e12, allow_nan=True),
        )


class CompositeSquareEdge(Edge):
    """
    An example Edge implementation which connects two BasicTransmonElements within a
    QuantumDevice. This edge implements a square flux pulse and two virtual z
    phase corrections for the CZ operation between the two BasicTransmonElements.
    """

    def __init__(
        self,
        parent_element_name: str,
        child_element_name: str,
        **kwargs,
    ):
        super().__init__(
            parent_element_name=parent_element_name,
            child_element_name=child_element_name,
            **kwargs,
        )

        self.add_submodule("cz", CZ(self, "cz"))

    def generate_edge_config(self) -> Dict[str, Dict[str, OperationCompilationConfig]]:
        """
        Fills in the edges information to produce a valid device config for the
        quantify-scheduler making use of the
        :func:`~.circuit_to_device.compile_circuit_to_device` function.
        """
        # pylint: disable=line-too-long
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
