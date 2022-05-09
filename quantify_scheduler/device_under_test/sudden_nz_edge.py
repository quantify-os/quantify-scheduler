# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
from typing import Dict, Any

from qcodes.instrument import InstrumentChannel
from qcodes.instrument.base import InstrumentBase
from qcodes.instrument.parameter import ManualParameter
from quantify_scheduler.backends.circuit_to_device import OperationCompilationConfig
from quantify_scheduler.helpers.validators import Numbers
from quantify_scheduler.device_under_test.edge import Edge


class CZ(InstrumentChannel):
    """
    Submodule containing parameters for performing a CZ operation
    """

    def __init__(self, parent: InstrumentBase, name: str, **kwargs: Any) -> None:
        super().__init__(parent=parent, name=name)
        self.add_parameter(
            name="amp_A",
            docstring=r"""The amplitude of the main square pulse.""",
            unit="V",
            parameter_class=ManualParameter,
            initial_value=0.5,
            vals=Numbers(min_value=0, max_value=1e12, allow_nan=True),
        )
        self.add_parameter(
            name="amp_B",
            docstring=r"""The scaling correction for the final sample of the first"""
            r""" square and first sample of the second square pulse.""",
            unit="V",
            parameter_class=ManualParameter,
            initial_value=0.5,
            vals=Numbers(min_value=0, max_value=1e12, allow_nan=True),
        )
        self.add_parameter(
            name="net_zero_A_scale",
            docstring=r"""amplitude scaling correction factor of the negative arm of"""
            r""" the net-zero pulse.""",
            unit="",
            parameter_class=ManualParameter,
            initial_value=0.95,
            vals=Numbers(min_value=0, max_value=1e12, allow_nan=True),
        )
        self.add_parameter(
            name="t_pulse",
            docstring=r"""The total duration of the two half square pulses.""",
            unit="s",
            parameter_class=ManualParameter,
            initial_value=2e-8,
            vals=Numbers(min_value=0, max_value=1e12, allow_nan=True),
        )
        self.add_parameter(
            name="t_phi",
            docstring=r"""The idling duration between the two half pulses.""",
            unit="s",
            parameter_class=ManualParameter,
            initial_value=2e-9,
            vals=Numbers(min_value=0, max_value=1e12, allow_nan=True),
        )
        self.add_parameter(
            name="t_integral_correction",
            docstring=r"""The duration in which any non-zero pulse amplitude needs to"""
            r""" be corrected.""",
            unit="s",
            parameter_class=ManualParameter,
            initial_value=1e-8,
            vals=Numbers(min_value=0, max_value=1e12, allow_nan=True),
        )
        self.add_parameter(
            name=f"{parent._parent_element_name}_phase_correction",
            docstring=r"""The phase correction for the parent qubit after the"""
            r""" SuddenNetZero operation has been performed.""",
            unit="degrees",
            parameter_class=ManualParameter,
            initial_value=0,
            vals=Numbers(min_value=0, max_value=1e12, allow_nan=True),
        )
        self.add_parameter(
            name=f"{parent._child_element_name}_phase_correction",
            docstring=r"""The phase correction for the child qubit after the"""
            r""" SuddenNetZero operation has been performed.""",
            unit="degrees",
            parameter_class=ManualParameter,
            initial_value=0,
            vals=Numbers(min_value=0, max_value=1e12, allow_nan=True),
        )


class SuddenNetZeroEdge(Edge):
    """
    An example Edge implementation which connects two BasicTransmonElements within a
    QuantumDevice. This edge implements the SuddenNetZero pulse from
    :cite:t:`negirneac_high_fidelity_2021` for the CZ operation between the two
    BasicTransmonElements.
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

        edge_op_config = {
            f"{self.name}": {
                "CZ": OperationCompilationConfig(
                    factory_func="quantify_scheduler.operations."
                    + "pulse_library.SuddenNetZeroPulse",
                    factory_kwargs={
                        "port": self.parent_device_element.ports.flux(),
                        "clock": "cl0.baseband",
                        "amp_A": self.cz.amp_A(),
                        "amp_B": self.cz.amp_B(),
                        "net_zero_A_scale": self.cz.net_zero_A_scale(),
                        "t_pulse": self.cz.t_pulse(),
                        "t_phi": self.cz.t_phi(),
                        "t_integral_correction": self.cz.t_integral_correction(),
                    },
                ),
            }
        }

        return edge_op_config
