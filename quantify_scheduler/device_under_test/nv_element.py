# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""
Device elements for NV centers. Currently only for the electronic qubit,
but could be extended for other qubits (eg. carbon qubit).
"""
from typing import Dict, Any

from qcodes.instrument import InstrumentModule
from qcodes.instrument.base import InstrumentBase
from qcodes.instrument.parameter import (
    ManualParameter,
    Parameter,
)
from qcodes.utils import validators
from quantify_scheduler.backends.circuit_to_device import (
    DeviceCompilationConfig,
    OperationCompilationConfig,
)
from quantify_scheduler.helpers.validators import Numbers
from quantify_scheduler.device_under_test.device_element import DeviceElement


# pylint: disable=too-few-public-methods
class _Ports(InstrumentModule):
    """
    Submodule containing the ports.
    """

    def __init__(self, parent: InstrumentBase, name: str, **kwargs: Any) -> None:
        super().__init__(parent=parent, name=name)

        self.microwave = Parameter(
            name="microwave",
            label="Name of microwave port",
            instrument=self,
            initial_cache_value=f"{parent.name}:mw",
            set_cmd=False,
        )
        """Name of the element's microwave port."""


# pylint: disable=too-few-public-methods
class _ClockFrequencies(InstrumentModule):
    """
    Submodule with clock frequencies specifying the transitions to address.
    """

    def __init__(self, parent: InstrumentBase, name: str, **kwargs: Any) -> None:
        super().__init__(parent=parent, name=name)

        self.f01 = ManualParameter(
            name="f01",
            label="Microwave frequency in resonance with transition between 0 and 1.",
            unit="Hz",
            instrument=self,
            initial_value=float("nan"),
            vals=Numbers(min_value=0, max_value=1e12, allow_nan=True),
        )
        """Microwave frequency to resonantly drive the electron spin state of a
        negatively charged diamond NV center from the 0-state to 1-state
        :cite:t:`DOHERTY20131`.
        """

        self.spec = ManualParameter(
            name="spec",
            label="Spectroscopy frequency",
            unit="Hz",
            instrument=self,
            initial_value=float("nan"),
            vals=Numbers(min_value=1e9, max_value=1e10, allow_nan=True),
        )
        """Parameter that is swept for a spectroscopy measurement. It does not track
        properties of the device element."""


# pylint: disable=too-few-public-methods
class _SpectroscopyOperation(InstrumentModule):
    """Submodule with parameters to convert the SpectroscopyOperation into a microwave
    pulse with a certain amplitude and duration for spin-state manipulation."""

    def __init__(self, parent: InstrumentBase, name: str, **kwargs: Any) -> None:
        super().__init__(parent=parent, name=name)

        self.amplitude = ManualParameter(
            name="amplitude",
            label="Amplitude of spectroscopy pulse",
            instrument=self,
            initial_value=float("nan"),
            unit="W",
        )
        """Amplitude of spectroscopy pulse"""

        self.duration = ManualParameter(
            name="duration",
            label="Duration of spectroscopy pulse",
            instrument=self,
            initial_value=15e-6,
            unit="s",
            vals=validators.Numbers(min_value=0, max_value=100e-6),
        )
        """Duration of the MW pulse."""


class _Reset(InstrumentModule):
    """
    Submodule containing parameters to run the spinpump laser with a square pulse
    to reset the NV to the $\ket{0}$ state.
    """

    def __init__(self, parent: InstrumentBase, name: str, **kwargs: Any) -> None:
        super().__init__(parent=parent, name=name)

        self.amplitude = ManualParameter(
            name="amplitude",
            instrument=self,
            initial_value=-0.5,
            unit="V",
            vals=validators.Numbers(min_value=-1.5, max_value=0),
        )
        """Amplitude of spin-pump pulse"""

        self.duration = ManualParameter(
            name="duration",
            instrument=self,
            initial_value=20e-6,
            unit="s",
            vals=validators.Numbers(min_value=10e-6, max_value=100e-6),
        )
        """Amplitude of spin-pump pulse"""


# pylint: disable=too-few-public-methods
class BasicElectronicNVElement(DeviceElement):
    """
    A device element representing an electronic qubit in an NV center.
    """

    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)

        self.add_submodule(
            "spectroscopy_operation",
            _SpectroscopyOperation(self, "spectroscopy_operation"),
        )
        self.add_submodule("ports", _Ports(self, "ports"))
        self.add_submodule("clock_freqs", _ClockFrequencies(self, "clock_freqs"))
        self.add_submodule("reset", _Reset(self, "reset"))

    def _generate_config(self) -> Dict[str, Dict[str, OperationCompilationConfig]]:
        """
        Generates part of the device configuration specific to a single qubit.

        This method is intended to be used when this object is part of a
        device object containing multiple elements.
        """
        qubit_config = {
            f"{self.name}": {
                "spectroscopy_operation": OperationCompilationConfig(
                    factory_func="quantify_scheduler.operations."
                    + "pulse_factories.nv_spec_pulse_mw",
                    factory_kwargs={
                        "duration": self.spectroscopy_operation.duration(),
                        "amplitude": self.spectroscopy_operation.amplitude(),
                        "port": self.ports.microwave(),
                        "clock": f"{self.name}.spec",
                    },
                ),
            }
        }
        return qubit_config

    def generate_device_config(self) -> DeviceCompilationConfig:
        """
        Generates a valid device config for the quantify-scheduler making use of the
        :func:`~.circuit_to_device.compile_circuit_to_device` function.

        This enables the settings of this qubit to be used in isolation.

        .. note:

            This config is only valid for single qubit experiments.
        """
        cfg_dict = {
            "backend": "quantify_scheduler.backends"
            ".circuit_to_device.compile_circuit_to_device",
            "elements": self._generate_config(),
            "clocks": {
                f"{self.name}.f01": self.clock_freqs.f01(),
                f"{self.name}.spec": self.clock_freqs.spec(),
            },
            "edges": {},
        }
        dev_cfg = DeviceCompilationConfig.parse_obj(cfg_dict)

        return dev_cfg
