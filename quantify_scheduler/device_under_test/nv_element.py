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
from quantify_scheduler.helpers.validators import (
    _Durations,
    _Amplitudes,
    _NonNegativeFrequencies,
    _Delays,
)
from quantify_scheduler.device_under_test.device_element import DeviceElement
from quantify_scheduler.operations import (
    pulse_factories,
    pulse_library,
    measurement_factories,
)


# pylint: disable=too-few-public-methods
class Ports(InstrumentModule):
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
            vals=validators.Strings(),
        )
        """Name of the element's microwave port."""

        self.optical_control = Parameter(
            name="optical_control",
            label="Name of optical control port",
            instrument=self,
            initial_cache_value=f"{parent.name}:optical_control",
            set_cmd=False,
            vals=validators.Strings(),
        )
        """Port to control the device element with optical pulses."""

        self.optical_readout = Parameter(
            name="optical_readout",
            label="Name of optical readout port",
            instrument=self,
            initial_cache_value=f"{parent.name}:optical_readout",
            set_cmd=False,
            vals=validators.Strings(),
        )
        """Port to readout photons from the device element."""


# pylint: disable=too-few-public-methods
class ClockFrequencies(InstrumentModule):
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
            vals=_NonNegativeFrequencies(),
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
            vals=_NonNegativeFrequencies(),
        )
        """Parameter that is swept for a spectroscopy measurement. It does not track
        properties of the device element."""

        self.ge0 = ManualParameter(
            name="ge0",
            label="f_{ge0}",
            unit="Hz",
            instrument=self,
            initial_value=float("nan"),
            vals=_NonNegativeFrequencies(),
        )
        """Transition frequency from the m_s=0 state to the E_x,y state"""

        self.ge1 = ManualParameter(
            name="ge1",
            label="f_{ge1}",
            unit="Hz",
            instrument=self,
            initial_value=float("nan"),
            vals=_NonNegativeFrequencies(),
        )
        """Transition frequency from the m_s=+-1 state to any of the A_1, A_2, or
        E_1,2 states"""

        self.ionization = ManualParameter(
            name="ionization",
            label="Frequency of ionization laser",
            unit="Hz",
            instrument=self,
            initial_value=float("nan"),
            vals=_NonNegativeFrequencies(),
        )
        """Frequency of the green ionization laser for manipulation of the NVs charge state."""


# pylint: disable=too-few-public-methods
class SpectroscopyOperationHermiteMW(InstrumentModule):
    """Submodule with parameters to convert the SpectroscopyOperation into a hermite
    microwave pulse with a certain amplitude and duration for spin-state manipulation.

    The modulation frequency of the pulse is determined by the clock ``spec`` in
    :class:`~.ClockFrequencies`.
    """

    def __init__(self, parent: InstrumentBase, name: str, **kwargs: Any) -> None:
        super().__init__(parent=parent, name=name, **kwargs)

        self.amplitude = ManualParameter(
            name="amplitude",
            label="Amplitude of spectroscopy pulse",
            instrument=self,
            initial_value=float("nan"),
            unit="W",
            vals=_Amplitudes(),
        )
        """Amplitude of spectroscopy pulse"""

        self.duration = ManualParameter(
            name="duration",
            label="Duration of spectroscopy pulse",
            instrument=self,
            initial_value=15e-6,
            unit="s",
            vals=_Durations(),
        )
        """Duration of the MW pulse."""


class ResetSpinpump(InstrumentModule):
    r"""
    Submodule containing parameters to run the spinpump laser with a square pulse
    to reset the NV to the :math:`|0\rangle` state.
    """

    def __init__(self, parent: InstrumentBase, name: str, **kwargs: Any) -> None:
        super().__init__(parent=parent, name=name, **kwargs)

        self.amplitude = ManualParameter(
            name="amplitude",
            instrument=self,
            initial_value=float("nan"),
            unit="V",
            vals=_Amplitudes(),
        )
        """Amplitude of reset pulse"""

        self.duration = ManualParameter(
            name="duration",
            instrument=self,
            initial_value=50e-6,
            unit="s",
            vals=_Durations(),
        )
        """Duration of reset pulse"""


class Measure(InstrumentModule):
    r"""Submodule containing parameters to read out the spin state of the NV center.

    Excitation with a readout laser from the :math:`|0\rangle` to an excited state.
    Acquisition of photons when decaying back into the :math:`|0\rangle` state.
    """

    def __init__(self, parent: InstrumentBase, name: str, **kwargs: Any) -> None:
        super().__init__(parent=parent, name=name, **kwargs)

        self.pulse_amplitude = ManualParameter(
            name="pulse_amplitude",
            instrument=self,
            initial_value=float("nan"),
            unit="V",
            vals=_Amplitudes(),
        )
        """Amplitude of readout pulse"""

        self.pulse_duration = ManualParameter(
            name="pulse_duration",
            instrument=self,
            initial_value=20e-6,
            unit="s",
            vals=_Durations(),
        )
        """Readout pulse duration"""

        self.acq_duration = ManualParameter(
            name="acq_duration",
            instrument=self,
            initial_value=50e-6,
            unit="s",
            vals=_Durations(),
        )
        """
        Duration of the acquisition.
        """

        self.acq_delay = ManualParameter(
            name="acq_delay",
            instrument=self,
            initial_value=0,
            unit="s",
            vals=_Delays(),
        )
        """
        Delay between the start of the readout pulse and the start of the acquisition.
        """

        self.acq_channel = ManualParameter(
            name="acq_channel",
            instrument=self,
            initial_value=0,
            unit="#",
            vals=validators.Ints(min_value=0),
        )
        """
        Acquisition channel of this device element.
        """


class ChargeReset(InstrumentModule):
    """
    Submodule containing parameters to run an ionization laser square pulse to reset the NV in
    its negatively charged state.
    """

    def __init__(self, parent: InstrumentBase, name: str, **kwargs: Any) -> None:
        super().__init__(parent=parent, name=name)

        self.amplitude = ManualParameter(
            name="amplitude",
            instrument=self,
            initial_value=float("nan"),
            unit="V",
            vals=_Amplitudes(),
        )
        """Amplitude of charge reset pulse."""

        self.duration = ManualParameter(
            name="duration",
            instrument=self,
            initial_value=20e-6,
            unit="s",
            vals=_Durations(),
        )
        """Duration of the charge reset pulse."""


class CRCount(InstrumentModule):
    """
    Submodule containing parameters to run the ionization laser and the spin pump laser
    with a photon count to perform a charge and resonance count.
    """

    def __init__(self, parent: InstrumentBase, name: str, **kwargs: Any) -> None:
        super().__init__(parent=parent, name=name)

        self.readout_pulse_amplitude = ManualParameter(
            name="readout_pulse_amplitude",
            instrument=self,
            initial_value=float("nan"),
            unit="V",
            vals=_Amplitudes(),
        )
        """Amplitude of readout pulse"""

        self.spinpump_pulse_amplitude = ManualParameter(
            name="spinpump_pulse_amplitude",
            instrument=self,
            initial_value=float("nan"),
            unit="V",
            vals=_Amplitudes(),
        )
        """Amplitude of spin-pump pulse"""

        self.readout_pulse_duration = ManualParameter(
            name="readout_pulse_duration",
            instrument=self,
            initial_value=20e-6,
            unit="s",
            vals=_Durations(),
        )
        """Readout pulse duration"""

        self.spinpump_pulse_duration = ManualParameter(
            name="spinpump_pulse_duration",
            instrument=self,
            initial_value=20e-6,
            unit="s",
            vals=_Durations(),
        )
        """Readout pulse duration"""

        self.acq_duration = ManualParameter(
            name="acq_duration",
            instrument=self,
            initial_value=50e-6,
            unit="s",
            vals=_Durations(),
        )
        """
        Duration of the acquisition.
        """

        self.acq_delay = ManualParameter(
            name="acq_delay",
            instrument=self,
            initial_value=0,
            unit="s",
            vals=_Delays(),
        )
        """
        Delay between the start of the readout pulse and the start of the acquisition.
        """

        self.acq_channel = ManualParameter(
            name="acq_channel",
            instrument=self,
            initial_value=0,
            unit="#",
            vals=validators.Ints(min_value=0),
        )
        """
        Acquisition channel of this device element.
        """


class BasicElectronicNVElement(DeviceElement):
    """
    A device element representing an electronic qubit in an NV center.

    The submodules contain the necessary qubit parameters to translate higher-level
    operations into pulses. Please see the documentation of these classes.
    """

    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)

        self.add_submodule(
            "spectroscopy_operation",
            SpectroscopyOperationHermiteMW(self, "spectroscopy_operation"),
        )
        self.spectroscopy_operation: SpectroscopyOperationHermiteMW
        """Submodule :class:`~.SpectroscopyOperationHermiteMW`."""
        self.add_submodule("ports", Ports(self, "ports"))
        self.ports: Ports
        """Submodule :class:`~.Ports`."""
        self.add_submodule("clock_freqs", ClockFrequencies(self, "clock_freqs"))
        self.clock_freqs: ClockFrequencies
        """Submodule :class:`~.ClockFrequencies`."""
        self.add_submodule("reset", ResetSpinpump(self, "reset"))
        self.reset: ResetSpinpump
        """Submodule :class:`~.ResetSpinpump`."""
        self.add_submodule("charge_reset", ChargeReset(self, "charge_reset"))
        self.charge_reset: ChargeReset
        """Submodule :class:`~.ChargeReset`."""
        self.add_submodule("measure", Measure(self, "measure"))
        self.measure: Measure
        """Submodule :class:`~.Measure`."""
        self.add_submodule("cr_count", CRCount(self, "cr_count"))
        self.cr_count: CRCount
        """Submodule :class:`~.CRCount`."""

    def _generate_config(self) -> Dict[str, Dict[str, OperationCompilationConfig]]:
        """
        Generates part of the device configuration specific to a single qubit.

        This method is intended to be used when this object is part of a
        device object containing multiple elements.
        """
        qubit_config = {
            f"{self.name}": {
                "spectroscopy_operation": OperationCompilationConfig(
                    factory_func=pulse_factories.nv_spec_pulse_mw,
                    factory_kwargs={
                        "duration": self.spectroscopy_operation.duration(),
                        "amplitude": self.spectroscopy_operation.amplitude(),
                        "port": self.ports.microwave(),
                        "clock": f"{self.name}.spec",
                    },
                ),
                "reset": OperationCompilationConfig(
                    factory_func=pulse_library.SquarePulse,
                    factory_kwargs={
                        "duration": self.reset.duration(),
                        "amp": self.reset.amplitude(),
                        "port": self.ports.optical_control(),
                        "clock": f"{self.name}.ge1",
                    },
                ),
                "charge_reset": OperationCompilationConfig(
                    factory_func=pulse_library.SquarePulse,
                    factory_kwargs={
                        "duration": self.charge_reset.duration(),
                        "amp": self.charge_reset.amplitude(),
                        "port": self.ports.optical_control(),
                        "clock": f"{self.name}.ionization",
                    },
                ),
                "measure": OperationCompilationConfig(
                    factory_func=measurement_factories.optical_measurement,
                    factory_kwargs={
                        "pulse_amplitudes": [self.measure.pulse_amplitude()],
                        "pulse_durations": [self.measure.pulse_duration()],
                        "pulse_ports": [self.ports.optical_control()],
                        "pulse_clocks": [f"{self.name}.ge0"],
                        "acq_duration": self.measure.acq_duration(),
                        "acq_delay": self.measure.acq_delay(),
                        "acq_channel": self.measure.acq_channel(),
                        "acq_port": self.ports.optical_readout(),
                        "acq_clock": f"{self.name}.ge0",
                        "pulse_type": "SquarePulse",
                        "acq_protocol_default": "TriggerCount",
                    },
                    gate_info_factory_kwargs=["acq_index", "bin_mode", "acq_protocol"],
                ),
                "cr_count": OperationCompilationConfig(
                    factory_func=measurement_factories.optical_measurement,
                    factory_kwargs={
                        "pulse_amplitudes": [
                            self.cr_count.readout_pulse_amplitude(),
                            self.cr_count.spinpump_pulse_amplitude(),
                        ],
                        "pulse_durations": [
                            self.cr_count.readout_pulse_duration(),
                            self.cr_count.spinpump_pulse_duration(),
                        ],
                        "pulse_ports": [
                            self.ports.optical_control(),
                            self.ports.optical_control(),
                        ],
                        "pulse_clocks": [
                            f"{self.name}.ge0",
                            f"{self.name}.ge1",
                        ],
                        "acq_duration": self.cr_count.acq_duration(),
                        "acq_delay": self.cr_count.acq_delay(),
                        "acq_channel": self.cr_count.acq_channel(),
                        "acq_port": self.ports.optical_readout(),
                        "acq_clock": f"{self.name}.ge0",
                        "pulse_type": "SquarePulse",
                        "acq_protocol_default": "TriggerCount",
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
                f"{self.name}.f01": self.clock_freqs.f01(),
                f"{self.name}.spec": self.clock_freqs.spec(),
                f"{self.name}.ge0": self.clock_freqs.ge0(),
                f"{self.name}.ge1": self.clock_freqs.ge1(),
                f"{self.name}.ionization": self.clock_freqs.ionization(),
            },
            "edges": {},
        }
        dev_cfg = DeviceCompilationConfig.parse_obj(cfg_dict)

        return dev_cfg
