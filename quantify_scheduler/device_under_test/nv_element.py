from cgitb import reset
from typing import Dict, Any
from numpy import power

from qcodes.instrument.base import Instrument
from qcodes.instrument import InstrumentChannel
from qcodes.instrument.base import InstrumentBase
from quantify_core.utilities import deprecated
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
            name="microwave",
            instrument=self,
            initial_cache_value=f"{parent.name}:mw",
            set_cmd=False,
        )
        """Name of the element's microwave port."""

        self.optical = Parameter(
            name="optical",
            instrument=self,
            initial_cache_value=f"{parent.name}:opt",
            set_cmd=False,
        )
        """Name of the element's optical port."""

        self.readout = Parameter(
            name="readout",
            instrument=self,
            initial_cache_value=f"{parent.name}:pd",
            set_cmd=False,
        )
        """Name of the element's photon detector port."""

class ClocksFrequencies(InstrumentChannel):
    """
    Submodule containing the clock frequencies specifying the transitions to address.
    """

    def __init__(self, parent: InstrumentBase, name: str, **kwargs: Any) -> None:
        super().__init__(parent=parent, name=name)

        self.f01= ManualParameter(
            name="f01",
            label="Microwave frequency in resonance with |0> -> |1> transition.",
            unit="Hz",
            instrument=self,
            initial_value=float("nan"),
            vals=Numbers(min_value=0, max_value=1e12, allow_nan=True),
        )
        """Microwave frequency to resonantly drive the electron spin state of a
        negatively charged diamond NV center from |0> to |1> :cite:t:`DOHERTY20131`.
        """

        self.ge0= ManualParameter(
            name="ge0",
            label="Frequency of the |g>-|e> transition in spin state |0>",
            unit="Hz",
            instrument=self,
            initial_value=float("nan"),
            vals=Numbers(min_value=0, max_value=1e12, allow_nan=True),
        )
        """Resonance frequency of the spin conserving transition from ground
        state |g> to excited state |e> for singlet spin state |0> electrons
        :cite:t:`DOHERTY20131`."""

        self.ge1= ManualParameter(
            name="ge1",
            label="Frequency of the |g>-|e> transition in spin state |1>",
            unit="Hz",
            instrument=self,
            initial_value=float("nan"),
            vals=Numbers(min_value=0, max_value=1e12, allow_nan=True),
        )
        """Resonance frequency of the spin conserving transition from ground
        state |g> to excited state |e> for triplet spin state |1> electrons
        :cite:t:`DOHERTY20131`."""

        self.spec= ManualParameter(
            name="spec",
            label="Spectroscopy frequency",
            unit="Hz",
            instrument=self,
            initial_value=float("nan"),
            vals=Numbers(min_value=1e9, max_value=1e10, allow_nan=True),
        )
        """Parameter that is swept for a spectroscopy measurement. It does not track
        properties of the device element."""

class CRcheck(InstrumentChannel):
    """
    Submodule containing parameters to perform a charge resonance (CR) check on a
    electronic NV qubit.
    """
    def __init__(self, parent: InstrumentBase, name: str, **kwargs: Any) -> None:
        super().__init__(parent=parent, name=name)
        pulse_types = validators.Enum("SquarePulse")
        self.add_parameter(
            "pulse_type",
            docstring=(
                "Envelope function that defines the shape of "
                "the reset pulse."
            ),
            initial_value="SquarePulse",
            parameter_class=ManualParameter,
            vals=pulse_types,
        )
        self.add_parameter(
            "power",
            docstring="""Laser power for CR check.""",
            initial_value=float("nan"),
            unit="W",
            parameter_class=ManualParameter,
            vals=Numbers(min_value=0, max_value=1e-2, allow_nan=True),
        )
        self.add_parameter(
            "ge1",
            docstring="""Resonance wavelength from ground to excited state of spin
            state 1.""",
            initial_value=650e-9,
            unit="m",
            parameter_class=ManualParameter,
            vals=validators.Numbers(min_value=-1, max_value=1),
        )
        self.add_parameter(
            "duration",
            docstring="""Duration of the check pulse.""",
            initial_value=20e-9,
            unit="s",
            parameter_class=ManualParameter,
            vals=validators.Numbers(min_value=0, max_value=1),
        )
        self.add_parameter(
            "integration_time",
            docstring="Integration time for the readout acquisition.",
            initial_value=1e-6,
            unit="s",
            parameter_class=ManualParameter,
            vals=validators.Numbers(min_value=0, max_value=1),
        )
        self.add_parameter(
            "resonance_counts",
            label="Charge resonance counts",
            unit="#",
            parameter_class=ManualParameter,
            docstring="Integrated counts of the charge resonance check",
            initial_value=float("nan"),
            vals=Numbers(min_value=0, max_value=1e12, allow_nan=True),
        )
        self.add_parameter(
            "minimum_threshold",
            unit="1/s",
            parameter_class=ManualParameter,
            docstring="Minimum cps for successfull CR-check.",
            initial_value=float("nan"),
            vals=Numbers(min_value=0, max_value=1e12, allow_nan=True),
        )


class Reset(InstrumentChannel):
    """
    Submodule containing parameters to perform a spin state reset to |0>.
    """
    def __init__(self, parent: InstrumentBase, name: str, **kwargs: Any) -> None:
        super().__init__(parent=parent, name=name)
        pulse_types = validators.Enum("SquarePulse")
        self.add_parameter(
            "pulse_type",
            docstring=(
                "Envelope function that defines the shape of "
                "the reset pulse."
            ),
            initial_value="SquarePulse",
            parameter_class=ManualParameter,
            vals=pulse_types,
        )
        self.add_parameter(
            "power",
            docstring="""Laser power for spin state reset.""",
            initial_value=float("nan"),
            unit="W",
            parameter_class=ManualParameter,
            vals=Numbers(min_value=0, max_value=1e-2, allow_nan=True),
        )

        self.add_parameter(
            "duration",
            docstring="""Duration of the reset pulse.""",
            initial_value=20e-9,
            unit="s",
            parameter_class=ManualParameter,
            vals=validators.Numbers(min_value=0, max_value=1),
        )
        self.add_parameter(
            "reset_time",
            docstring="Illumination time for the spin state reset.",
            initial_value=1e-6,
            unit="s",
            parameter_class=ManualParameter,
            vals=validators.Numbers(min_value=0, max_value=1),
        )

class SpectroscopyPulseMW(InstrumentChannel): # SpectroscopyPulse
    """Submodule containing parameters run a spectroscopy pulse in the microwave range.
    """
    def __init__(self, parent: InstrumentBase, name: str, **kwargs: Any) -> None:
        super().__init__(parent=parent, name=name)

        self.amplitude = ManualParameter(
            name="amplitude",
            instrument=self,
            initial_value=float("nan"),
            unit="W",
            #vals=Numbers(min_value=0, max_value=1e-2, allow_nan=True),
        )
        """Microwave amplitude for spin state manipulation"""

        self.duration = ManualParameter(
            name="duration",
            instrument=self,
            initial_value=15e-6,
            unit="s",
            vals=validators.Numbers(min_value=0, max_value=100e-6),
        )
        """Duration of the MW pulse."""


class SpinStateMeasurement(InstrumentChannel):
    """
    Submodule containing parameters to perform a spin state measurement along the 
    z-axis.
    """
    def __init__(self, parent: InstrumentBase, name: str, **kwargs: Any) -> None:
        super().__init__(parent=parent, name=name)

        pulse_types = validators.Enum("SquarePulse")
        self.add_parameter(
            "pulse_type",
            docstring=(
                "Envelope function that defines the shape of "
                "the readout pulse prior to modulation."
            ),
            initial_value="SquarePulse",
            parameter_class=ManualParameter,
            vals=pulse_types,
        )

        self.add_parameter(
            "pulse_pow",
            docstring="Power amplitude of the photonic readout pulse.",
            initial_value=1e-6,
            unit="W",
            parameter_class=ManualParameter,
            vals=validators.Numbers(min_value=0, max_value=2),
        )
        self.add_parameter(
            "pulse_duration",
            docstring="Duration of the photonic readout pulse.",
            initial_value=300e-9,
            unit="s",
            parameter_class=ManualParameter,
            vals=validators.Numbers(min_value=0, max_value=1),
        )

        self.add_parameter(
            "acq_channel",
            docstring="Acquisition channel of to this device element.",
            initial_value=0,
            unit="#",
            parameter_class=ManualParameter,
            vals=validators.Ints(min_value=0),
        )

        self.add_parameter(
            "acq_delay",
            docstring="""Delay between the start of the readout pulse and the start of 
            the acquisition.""",
            initial_value=0,
            unit="s",
            parameter_class=ManualParameter,
            vals=validators.Numbers(min_value=0, max_value=1),
        )
        self.add_parameter(
            "integration_time",
            docstring="Integration time for the readout acquisition.",
            initial_value=1e-6,
            unit="s",
            parameter_class=ManualParameter,
            vals=validators.Numbers(min_value=0, max_value=1),
        )


class BasicElectronicNVElement(DeviceElement):
    """
    A device element representing a single fixed-frequency transmon qubit coupled to a
    readout resonator.
    """

    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)

        self.add_submodule("CRcheck", CRcheck(self, "CRcheck"))
        self.add_submodule("reset", Reset(self, "reset"))
        self.add_submodule("spectroscopy_pulse", SpectroscopyPulseMW(self, "spectroscopy_pulse"))
        self.add_submodule("measure", SpinStateMeasurement(self, "measure"))
        self.add_submodule("ports", Ports(self, "ports"))
        self.add_submodule("clock_freqs", ClocksFrequencies(self, "clock_freqs"))

    def _generate_config(self) -> Dict[str, Dict[str, OperationCompilationConfig]]:
        """
        Generates part of the device configuration specific to a single qubit.

        This method is intended to be used when this object is part of a
        device object containing multiple elements.
        """
        qubit_config = {
            f"{self.name}": {
                "CRcheck": OperationCompilationConfig(
                    factory_func= "quantify_scheduler.operations."
                    + "pulse_library.IdlePulse",
                    factory_kwargs={
                        "duration": self.reset.duration(),
                    }
                ),

                "reset": OperationCompilationConfig(
                    factory_func="quantify_scheduler.operations."
                    + "pulse_library.SquarePulse",
                    factory_kwargs={
                        "amp": self.reset.power(),
                        "duration": self.reset.duration(),
                        "port": self.ports.optical(),
                        "clock": f"{self.name}.ge0",
                    },
                ),

                "spectroscopy_pulse": OperationCompilationConfig(
                    factory_func="quantify_scheduler.operations."
                    + "pulse_factories.nv_spec_pulse_mw",
                    factory_kwargs={
                        "duration": self.spectroscopy_pulse.duration(),
                        "amplitude": self.spectroscopy_pulse.amplitude(),
                        "port": self.ports.microwave(),
                        "clock": f"{self.name}.f01",
                    },
                ),

                "measure": OperationCompilationConfig(
                    factory_func="quantify_scheduler.operations."
                    + "measurement_factories.dispersive_measurement",
                    factory_kwargs={
                        "port": self.ports.readout(),
                        "clock": f"{self.name}.ge1",
                        "pulse_type": self.measure.pulse_type(),
                        "pulse_amp": self.measure.pulse_pow(),
                        "pulse_duration": self.measure.pulse_duration(),
                        "acq_delay": self.measure.acq_delay(),
                        "acq_duration": self.measure.integration_time(),
                        "acq_channel": self.measure.acq_channel(),
                        "acq_protocol_default": "Trace",
                    },
                    gate_info_factory_kwargs=["acq_index", "bin_mode", "acq_protocol"],
                ),

                "CRcheck": OperationCompilationConfig(
                    factory_func= "quantify_scheduler.operations."
                    + "pulse_library.IdlePulse",
                    factory_kwargs={
                        "duration": self.reset.duration(),
                    }
                )
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
                f"{self.name}.ge0": self.clock_freqs.ge0(),
                f"{self.name}.ge1": self.clock_freqs.ge1(),
            },
            "edges": {},
        }
        dev_cfg = DeviceCompilationConfig.parse_obj(cfg_dict)

        return dev_cfg
