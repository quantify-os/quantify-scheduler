# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
from typing import Dict, Any

from qcodes.instrument import InstrumentChannel
from qcodes.instrument.base import InstrumentBase
from qcodes.instrument.parameter import (
    InstrumentRefParameter,
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


class Ports(InstrumentChannel):
    """
    Submodule containing the ports.
    """

    def __init__(self, parent: InstrumentBase, name: str, **kwargs: Any) -> None:
        super().__init__(parent=parent, name=name)

        self.microwave = Parameter(
            "microwave",
            instrument=self,
            initial_cache_value=f"{parent.name}:mw",
            set_cmd=False,
        )
        """Name of the element's microwave port."""

        self.flux = Parameter(
            "flux",
            instrument=self,
            initial_cache_value=f"{parent.name}:fl",
            set_cmd=False,
        )
        """Name of the element's flux port."""

        self.readout = Parameter(
            "readout",
            instrument=self,
            initial_cache_value=f"{parent.name}:res",
            set_cmd=False,
        )
        """Name of the element's readout port."""


class ClocksFrequencies(InstrumentChannel):
    """
    Submodule containing the clock frequencies specifying the transitions to address.
    """

    def __init__(self, parent: InstrumentBase, name: str, **kwargs: Any) -> None:
        super().__init__(parent=parent, name=name)

        self.f01 = ManualParameter(
            "f01",
            instrument=self,
            label="Qubit frequency",
            unit="Hz",
            initial_value=float("nan"),
            vals=Numbers(min_value=0, max_value=1e12, allow_nan=True),
        )
        """Frequency of the 01 clock"""

        self.f12 = ManualParameter(
            "f12",
            instrument=self,
            label="Frequency of the |1>-|2> transition",
            unit="Hz",
            initial_value=float("nan"),
            vals=Numbers(min_value=0, max_value=1e12, allow_nan=True),
        )
        """Frequency of the 12 clock"""

        self.readout = ManualParameter(
            "readout",
            instrument=self,
            label="Readout frequency",
            unit="Hz",
            initial_value=float("nan"),
            vals=Numbers(min_value=0, max_value=1e12, allow_nan=True),
        )
        """Frequency of the ro clock. """


class IdlingReset(InstrumentChannel):
    """
    Submodule containing parameters for doing a reset by idling.
    """

    def __init__(self, parent: InstrumentBase, name: str, **kwargs: Any) -> None:
        super().__init__(parent=parent, name=name)

        self.duration = ManualParameter(
            "duration",
            instrument=self,
            initial_value=200e-6,
            unit="s",
            vals=validators.Numbers(min_value=0, max_value=1),
        )
        """Duration of the passive qubit reset (initialization by relaxation)."""


class RxyDRAG(InstrumentChannel):
    """
    Submodule containing parameters for performing an Rxy operation
    using a DRAG pulse.
    """

    def __init__(self, parent: InstrumentBase, name: str, **kwargs: Any) -> None:
        super().__init__(parent=parent, name=name)
        self.amp180 = ManualParameter(
            "amp180",
            instrument=self,
            label=r"$\pi-pulse amplitude$",
            initial_value=float("nan"),
            unit="V",
            vals=Numbers(min_value=-10, max_value=10, allow_nan=True),
        )
        r"""Amplitude required to perform a $\pi$ pulse."""

        self.motzoi = ManualParameter(
            "motzoi",
            instrument=self,
            initial_value=0,
            unit="",
            vals=validators.Numbers(min_value=-1, max_value=1),
        )
        """Ratio between the Gaussian Derivative (D) and Gaussian (G)
        components of the DRAG pulse."""

        self.duration = ManualParameter(
            "duration",
            instrument=self,
            initial_value=20e-9,
            unit="s",
            vals=validators.Numbers(min_value=0, max_value=1),
        )
        """Duration of the control pulse."""


class DispersiveMeasurement(InstrumentChannel):
    """
    Submodule containing parameters to perform a measurement using
    :func:`~quantify_scheduler.operations.measurement_factories.dispersive_measurement`
    """

    def __init__(self, parent: InstrumentBase, name: str, **kwargs: Any) -> None:
        super().__init__(parent=parent, name=name)

        pulse_types = validators.Enum("SquarePulse")
        self.pulse_type = ManualParameter(
            "pulse_type",
            instrument=self,
            initial_value="SquarePulse",
            vals=pulse_types,
        )
        """Envelope function that defines the shape of the readout pulse prior to
        modulation."""

        self.pulse_amp = ManualParameter(
            "pulse_amp",
            instrument=self,
            initial_value=0.25,
            unit="V",
            vals=validators.Numbers(min_value=0, max_value=2),
        )
        """Amplitude of the readout pulse."""

        self.pulse_duration = ManualParameter(
            "pulse_duration",
            instrument=self,
            initial_value=300e-9,
            unit="s",
            vals=validators.Numbers(min_value=0, max_value=1),
        )
        """Duration of the readout pulse."""

        self.acq_channel = ManualParameter(
            "acq_channel",
            instrument=self,
            initial_value=0,
            unit="#",
            vals=validators.Ints(min_value=0),
        )
        """Acquisition channel of to this device element."""

        self.acq_delay = ManualParameter(
            "acq_delay",
            instrument=self,
            initial_value=0,  # float("nan"),
            unit="s",
            # in principle the values should be a few 100 ns but the validator is here
            # only to protect against silly typos that lead to out of memory errors.
            vals=validators.Numbers(min_value=0, max_value=100e-6),
        )
        """Delay between the start of the readout pulse and the start of
        the acquisition. Note that some hardware backends do not support
        starting a pulse and the acquisition in the same clock cycle making 0
        delay an invalid value."""

        self.integration_time = ManualParameter(
            "integration_time",
            instrument=self,
            initial_value=1e-6,
            unit="s",
            # in principle the values should be a few us but the validator is here
            # only to protect against silly typos that lead to out of memory errors.
            vals=validators.Numbers(min_value=0, max_value=100e-6),
        )
        """Integration time for the readout acquisition."""

        self.reset_clock_phase = ManualParameter(
            "reset_clock_phase",
            instrument=self,
            initial_value=True,
            vals=validators.Bool(),
        )
        """The phase of the measurement clock will be reset by the
        control hardware at the start of each measurement if
        ``reset_clock_phase=True``."""

        ro_acq_weight_type_validator = validators.Enum("SSB")
        self.acq_weight_type = ManualParameter(
            "acq_weight_type",
            instrument=self,
            initial_value="SSB",
            vals=ro_acq_weight_type_validator,
        )


class BasicTransmonElement(DeviceElement):
    """
    A device element representing a single fixed-frequency transmon qubit coupled to a
    readout resonator.
    """

    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)

        self.add_submodule("reset", IdlingReset(self, "reset"))
        self.reset: IdlingReset
        """Submodule :class:`~.IdlingReset`."""
        self.add_submodule("rxy", RxyDRAG(self, "rxy"))
        self.rxy: RxyDRAG
        """Submodule :class:`~.RxyDRAG`."""
        self.add_submodule("measure", DispersiveMeasurement(self, "measure"))
        self.measure: DispersiveMeasurement
        """Submodule :class:`~.DispersiveMeasurement`."""
        self.add_submodule("ports", Ports(self, "ports"))
        self.ports: Ports
        """Submodule :class:`~.Ports`."""
        self.add_submodule("clock_freqs", ClocksFrequencies(self, "clock_freqs"))
        self.clock_freqs: ClocksFrequencies
        """Submodule :class:`~.ClocksFrequencies`."""

    def _generate_config(self) -> Dict[str, Dict[str, OperationCompilationConfig]]:
        """
        Generates part of the device configuration specific to a single qubit.

        This method is intended to be used when this object is part of a
        device object containing multiple elements.
        """
        qubit_config = {
            f"{self.name}": {
                "reset": OperationCompilationConfig(
                    factory_func="quantify_scheduler.operations."
                    + "pulse_library.IdlePulse",
                    factory_kwargs={
                        "duration": self.reset.duration(),
                    },
                ),
                # example of a pulse with a parametrized mapping, using a factory
                "Rxy": OperationCompilationConfig(
                    factory_func="quantify_scheduler.operations."
                    + "pulse_factories.rxy_drag_pulse",
                    factory_kwargs={
                        "amp180": self.rxy.amp180(),
                        "motzoi": self.rxy.motzoi(),
                        "port": self.ports.microwave(),
                        "clock": f"{self.name}.01",
                        "duration": self.rxy.duration(),
                    },
                    gate_info_factory_kwargs=[
                        "theta",
                        "phi",
                    ],  # the keys from the gate info to pass to the factory function
                ),
                # the measurement also has a parametrized mapping, and uses a
                # factory function.
                "measure": OperationCompilationConfig(
                    factory_func="quantify_scheduler.operations."
                    + "measurement_factories.dispersive_measurement",
                    factory_kwargs={
                        "port": self.ports.readout(),
                        "clock": f"{self.name}.ro",
                        "pulse_type": self.measure.pulse_type(),
                        "pulse_amp": self.measure.pulse_amp(),
                        "pulse_duration": self.measure.pulse_duration(),
                        "acq_delay": self.measure.acq_delay(),
                        "acq_duration": self.measure.integration_time(),
                        "acq_channel": self.measure.acq_channel(),
                        "acq_protocol_default": "SSBIntegrationComplex",
                        "reset_clock_phase": self.measure.reset_clock_phase(),
                    },
                    gate_info_factory_kwargs=["acq_index", "bin_mode", "acq_protocol"],
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
                f"{self.name}.01": self.clock_freqs.f01(),
                f"{self.name}.12": self.clock_freqs.f12(),
                f"{self.name}.ro": self.clock_freqs.readout(),
            },
            "edges": {},
        }
        dev_cfg = DeviceCompilationConfig.parse_obj(cfg_dict)

        return dev_cfg
