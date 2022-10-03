# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch

from qcodes.instrument.base import Instrument
from quantify_scheduler.backends.circuit_to_device import DeviceCompilationConfig
from quantify_scheduler.device_under_test.device_element import DeviceElement


class Edge(Instrument):
    """
    This class encapsulates the connection information between DeviceElements in the
    QuantumDevice. It provides an interface for the QuantumDevice to generate the
    edge information for use in the device compilation step. See
    :class:`quantify_scheduler.device_under_test.composite_square_edge` for an example
    edge implementation.
    """

    def __init__(self, parent_element_name: str, child_element_name: str):
        edge_name = f"{parent_element_name}_{child_element_name}"
        self._parent_element_name = parent_element_name
        self._child_element_name = child_element_name
        super().__init__(name=edge_name)

    @property
    def parent_device_element(self):
        """
        The parent DeviceElement connected by the edge.
        """
        found_parent_element = self.find_instrument(
            name=self._parent_element_name, instrument_class=DeviceElement
        )
        return found_parent_element

    @property
    def child_device_element(self):
        """
        The child DeviceElement connected by the edge.
        """
        found_child_element = self.find_instrument(
            name=self._child_element_name, instrument_class=DeviceElement
        )
        return found_child_element

    def generate_edge_config(self) -> DeviceCompilationConfig:
        """
        Generates part of the device configuration specific to an edge connecting
        different device elements.

        This method is intended to be used when this object is part of a
        device object containing multiple elements.
        """
        raise NotImplementedError
