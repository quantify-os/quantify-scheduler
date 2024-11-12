# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""The module provides classes related CZ operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qcodes.instrument import InstrumentChannel
from qcodes.instrument.parameter import ManualParameter

from quantify_scheduler.backends.graph_compilation import OperationCompilationConfig
from quantify_scheduler.device_under_test.edge import Edge
from quantify_scheduler.helpers.validators import (
    Numbers,
    _Amplitudes,
    _Durations,
)
from quantify_scheduler.operations.pulse_factories import spin_init_pulse

if TYPE_CHECKING:
    from qcodes.instrument.base import InstrumentBase


class SpinInit(InstrumentChannel):
    """Submodule containing parameters for performing a SpinInit operation."""

    def __init__(self, parent: InstrumentBase, name: str, **kwargs: Any) -> None:  # noqa: ANN401
        super().__init__(parent=parent, name=name)

        self.add_parameter(
            name="square_duration",
            docstring="The duration of the square pulses for parent and child elements.",
            unit="s",
            parameter_class=ManualParameter,
            vals=_Durations(),
            initial_value=kwargs.get("square_duration", 0),
        )

        self.add_parameter(
            name="ramp_diff",
            docstring="The time difference between the ramp pulses;"
            "child element ramp start minus parent element ramp start.",
            unit="s",
            parameter_class=ManualParameter,
            vals=Numbers(),
            initial_value=kwargs.get("ramp_diff", 0),
        )

        for element_name in (
            parent.parent_device_element.name,
            parent.child_device_element.name,
        ):
            self.add_parameter(
                name=f"{element_name}_square_amp",
                docstring=f"The amplitude of the square pulse for {element_name}.",
                unit="V",
                parameter_class=ManualParameter,
                vals=_Amplitudes(),
                initial_value=kwargs.get(f"{element_name}_square_amp", 0),
            )

            self.add_parameter(
                name=f"{element_name}_ramp_amp",
                docstring=f"The final amplitude of the ramp pulse for {element_name}.",
                unit="V",
                parameter_class=ManualParameter,
                vals=_Amplitudes(),
                initial_value=kwargs.get(f"{element_name}_ramp_amp", 0),
            )

            self.add_parameter(
                name=f"{element_name}_ramp_rate",
                docstring=f"The rate of the amplitude of the ramp pulse for {element_name}.",
                unit="V/s",
                parameter_class=ManualParameter,
                vals=Numbers(),
                initial_value=kwargs.get(f"{element_name}_ramp_rate", 0),
            )


class SpinEdge(Edge):
    """
    Spin edge implementation which connects two SpinElements.

    This edge implements some operations between the two SpinElements.
    """

    def __init__(
        self,
        parent_element_name: str,
        child_element_name: str,
        **kwargs,
    ) -> None:
        spin_init_data = kwargs.pop("spin_init", {})

        super().__init__(
            parent_element_name=parent_element_name,
            child_element_name=child_element_name,
            **kwargs,
        )

        self.add_submodule("spin_init", SpinInit(parent=self, name="spin_init", **spin_init_data))

    def generate_edge_config(self) -> dict[str, dict[str, OperationCompilationConfig]]:
        """
        Generate valid device config.

        Fills in the edges information to produce a valid device config for the
        quantify-scheduler making use of the
        :func:`~.circuit_to_device.compile_circuit_to_device_with_config_validation` function.
        """
        parent_name: str = self.parent_device_element.name
        child_name: str = self.child_device_element.name
        edge_op_config = {
            f"{self.name}": {
                "SpinInit": OperationCompilationConfig(
                    factory_func=spin_init_pulse,
                    factory_kwargs={
                        "square_duration": self.spin_init.square_duration(),
                        "ramp_diff": self.spin_init.ramp_diff(),
                        "parent_port": self.parent_device_element.ports.microwave(),
                        "parent_clock": f"{parent_name}.f_larmor",
                        "parent_square_amp": self.spin_init.parameters[
                            f"{parent_name}_square_amp"
                        ](),
                        "parent_ramp_amp": self.spin_init.parameters[f"{parent_name}_ramp_amp"](),
                        "parent_ramp_rate": self.spin_init.parameters[f"{parent_name}_ramp_rate"](),
                        "child_port": self.child_device_element.ports.microwave(),
                        "child_clock": f"{child_name}.f_larmor",
                        "child_square_amp": self.spin_init.parameters[f"{child_name}_square_amp"](),
                        "child_ramp_amp": self.spin_init.parameters[f"{child_name}_ramp_amp"](),
                        "child_ramp_rate": self.spin_init.parameters[f"{child_name}_ramp_rate"](),
                    },
                ),
            }
        }

        return edge_op_config
