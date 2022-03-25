# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch

from qcodes.instrument.base import Instrument
from quantify_scheduler.backends.circuit_to_device import DeviceCompilationConfig


class DeviceElement(Instrument):
    """
    A device element is responsible for managing parameters of the device
    configuration responsible for compiling operations applied to that specific
    device element from the quantum-circuit to the quantum-device layer.
    """

    pass

    def generate_device_config(self) -> DeviceCompilationConfig:
        """
        Generates the device configuration
        """
        raise NotImplementedError
