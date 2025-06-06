# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""
Device elements for NV centers.

Currently only for the electronic qubit,
but could be extended for other qubits (eg. carbon qubit).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

from qcodes.instrument import InstrumentModule
from qcodes.instrument.parameter import ManualParameter
from qcodes.utils import validators

from quantify_scheduler.backends.graph_compilation import (
    DeviceCompilationConfig,
    OperationCompilationConfig,
)
from quantify_scheduler.device_under_test.device_element import DeviceElement
from quantify_scheduler.device_under_test.transmon_element import (
    PulseCompensationModule,
)
from quantify_scheduler.enums import TimeRef, TimeSource
from quantify_scheduler.helpers.validators import (
    _Amplitudes,
    _Delays,
    _Durations,
    _Hashable,
    _NonNegativeFrequencies,
)
from quantify_scheduler.operations import (
    measurement_factories,
    pulse_factories,
    pulse_library,
)

if TYPE_CHECKING:
    from collections.abc import Hashable

    from qcodes.instrument.base import InstrumentBase


class Ports(InstrumentModule):
    """Submodule containing the ports."""

    def __init__(
        self,
        parent: InstrumentBase,
        name: str,
        *,
        microwave: str | None = None,
        optical_control: str | None = None,
        optical_readout: str | None = None,
    ) -> None:
        super().__init__(parent=parent, name=name)

        self.microwave = ManualParameter(
            name="microwave",
            label="Name of microwave port",
            instrument=self,
            initial_value=microwave or f"{parent.name}:mw",
            vals=validators.Strings(),
        )
        """Name of the element's microwave port."""

        self.optical_control = ManualParameter(
            name="optical_control",
            label="Name of optical control port",
            instrument=self,
            initial_value=optical_control or f"{parent.name}:optical_control",
            vals=validators.Strings(),
        )
        """Port to control the device element with optical pulses."""

        self.optical_readout = ManualParameter(
            name="optical_readout",
            label="Name of optical readout port",
            instrument=self,
            initial_value=optical_readout or f"{parent.name}:optical_readout",
            vals=validators.Strings(),
        )
        """Port to readout photons from the device element."""


class ClockFrequencies(InstrumentModule):
    """Submodule with clock frequencies specifying the transitions to address."""

    def __init__(
        self,
        parent: InstrumentBase,
        name: str,
        *,
        f01: float = math.nan,
        spec: float = math.nan,
        ge0: float = math.nan,
        ge1: float = math.nan,
        ionization: float = math.nan,
    ) -> None:
        super().__init__(parent=parent, name=name)

        self.f01 = ManualParameter(
            name="f01",
            label="Microwave frequency in resonance with transition between 0 and 1.",
            unit="Hz",
            instrument=self,
            initial_value=f01,
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
            initial_value=spec,
            vals=_NonNegativeFrequencies(),
        )
        """Parameter that is swept for a spectroscopy measurement. It does not track
        properties of the device element."""

        self.ge0 = ManualParameter(
            name="ge0",
            label="f_{ge0}",
            unit="Hz",
            instrument=self,
            initial_value=ge0,
            vals=_NonNegativeFrequencies(),
        )
        """Transition frequency from the m_s=0 state to the E_x,y state"""

        self.ge1 = ManualParameter(
            name="ge1",
            label="f_{ge1}",
            unit="Hz",
            instrument=self,
            initial_value=ge1,
            vals=_NonNegativeFrequencies(),
        )
        """Transition frequency from the m_s=+-1 state to any of the A_1, A_2, or
        E_1,2 states"""

        self.ionization = ManualParameter(
            name="ionization",
            label="Frequency of ionization laser",
            unit="Hz",
            instrument=self,
            initial_value=ionization,
            vals=_NonNegativeFrequencies(),
        )
        """Frequency of the green ionization laser for manipulation of the NVs charge state."""


class SpectroscopyOperationNV(InstrumentModule):
    """
    Convert the SpectroscopyOperation into a hermite, square, or gaussian microwave pulse.

    This class contains parameters with a certain amplitude and duration for
    spin-state manipulation.

    The modulation frequency of the pulse is determined by the clock ``spec`` in
    :class:`~.ClockFrequencies`.
    """

    def __init__(
        self,
        parent: InstrumentBase,
        name: str,
        *,
        amplitude: float = math.nan,
        duration: float = 8e-6,
        pulse_shape: Literal["SquarePulse", "SkewedHermitePulse", "GaussPulse"] = "SquarePulse",
    ) -> None:
        super().__init__(parent=parent, name=name)

        self.amplitude = ManualParameter(
            name="amplitude",
            label="Amplitude of spectroscopy pulse",
            instrument=self,
            initial_value=amplitude,
            unit="",
            vals=_Amplitudes(),
        )
        """Amplitude of spectroscopy pulse"""

        self.duration = ManualParameter(
            name="duration",
            label="Duration of spectroscopy pulse",
            instrument=self,
            initial_value=duration,
            unit="s",
            vals=_Durations(),
        )
        """Duration of the MW pulse."""

        self.pulse_shape = ManualParameter(
            name="pulse_shape",
            label="Shape of the pulse",
            instrument=self,
            initial_value=pulse_shape,
            unit="",
            vals=validators.Enum("SquarePulse", "SkewedHermitePulse", "GaussPulse"),
        )
        """Shape of the MW pulse."""


class ResetSpinpump(InstrumentModule):
    r"""
    Submodule containing parameters to run the spinpump laser with a square pulse.

    This should reset the NV to the :math:`|0\rangle` state.
    """

    def __init__(
        self,
        parent: InstrumentBase,
        name: str,
        amplitude: float = math.nan,
        duration: float = 50e-6,
    ) -> None:
        super().__init__(parent=parent, name=name)

        self.amplitude = ManualParameter(
            name="amplitude",
            instrument=self,
            initial_value=amplitude,
            unit="",
            vals=_Amplitudes(),
        )
        """Amplitude of reset pulse"""

        self.duration = ManualParameter(
            name="duration",
            instrument=self,
            initial_value=duration,
            unit="s",
            vals=_Durations(),
        )
        """Duration of reset pulse"""


class Measure(InstrumentModule):
    r"""
    Submodule containing parameters to read out the spin state of the NV center.

    Excitation with a readout laser from the :math:`|0\rangle` to an excited state.
    Acquisition of photons when decaying back into the :math:`|0\rangle` state.
    """

    def __init__(
        self,
        parent: InstrumentBase,
        name: str,
        *,
        pulse_amplitude: float = math.nan,
        pulse_duration: float = 20e-6,
        acq_duration: float = 50e-6,
        acq_delay: float = 0,
        acq_channel: Hashable = 0,
        time_source: TimeSource | str = TimeSource.FIRST,
        time_ref: TimeRef | str = TimeRef.START,
    ) -> None:
        super().__init__(parent=parent, name=name)

        self.pulse_amplitude = ManualParameter(
            name="pulse_amplitude",
            instrument=self,
            initial_value=pulse_amplitude,
            unit="",
            vals=_Amplitudes(),
        )
        """Amplitude of readout pulse"""

        self.pulse_duration = ManualParameter(
            name="pulse_duration",
            instrument=self,
            initial_value=pulse_duration,
            unit="s",
            vals=_Durations(),
        )
        """Readout pulse duration"""

        self.acq_duration = ManualParameter(
            name="acq_duration",
            instrument=self,
            initial_value=acq_duration,
            unit="s",
            vals=_Durations(),
        )
        """
        Duration of the acquisition.
        """

        self.acq_delay = ManualParameter(
            name="acq_delay",
            instrument=self,
            initial_value=acq_delay,
            unit="s",
            vals=_Delays(),
        )
        """
        Delay between the start of the readout pulse and the start of the acquisition.
        """

        self.acq_channel = ManualParameter(
            name="acq_channel",
            instrument=self,
            initial_value=acq_channel,
            unit="",
            vals=_Hashable(),
        )
        """
        Acquisition channel of this device element.
        """

        # Optional timetag-related parameters.

        self.time_source = ManualParameter(
            name="time_source",
            instrument=self,
            initial_value=time_source,
            unit="",
            vals=_Hashable(),
        )
        """
        Optional time source, in case the
        :class:`~quantify_scheduler.operations.acquisition_library.Timetag` acquisition
        protocols are used. Please see that protocol for more information.
        """

        self.time_ref = ManualParameter(
            name="time_ref",
            instrument=self,
            initial_value=time_ref,
            unit="",
            vals=_Hashable(),
        )
        """
        Optional time reference, in case
        :class:`~quantify_scheduler.operations.acquisition_library.Timetag` or
        :class:`~quantify_scheduler.operations.acquisition_library.TimetagTrace`
        acquisition protocols are used. Please see those protocols for more information.
        """


class ChargeReset(InstrumentModule):
    """
    Submodule containing parameters to run an ionization laser square pulse to reset the NV.

    After resetting, the qubit should be in its negatively charged state.
    """

    def __init__(
        self,
        parent: InstrumentBase,
        name: str,
        *,
        amplitude: float = math.nan,
        duration: float = 20e-6,
    ) -> None:
        super().__init__(parent=parent, name=name)

        self.amplitude = ManualParameter(
            name="amplitude",
            instrument=self,
            initial_value=amplitude,
            unit="",
            vals=_Amplitudes(),
        )
        """Amplitude of charge reset pulse."""

        self.duration = ManualParameter(
            name="duration",
            instrument=self,
            initial_value=duration,
            unit="s",
            vals=_Durations(),
        )
        """Duration of the charge reset pulse."""


class CRCount(InstrumentModule):
    """
    Submodule containing parameters to run the ionization laser and the spin pump laser.

    This uses a photon count to perform a charge and resonance count.
    """

    def __init__(
        self,
        parent: InstrumentBase,
        name: str,
        *,
        readout_pulse_amplitude: float = math.nan,
        spinpump_pulse_amplitude: float = math.nan,
        readout_pulse_duration: float = 20e-6,
        spinpump_pulse_duration: float = 20e-6,
        acq_duration: float = 50e-6,
        acq_delay: float = 0,
        acq_channel: Hashable = 0,
    ) -> None:
        super().__init__(parent=parent, name=name)

        self.readout_pulse_amplitude = ManualParameter(
            name="readout_pulse_amplitude",
            instrument=self,
            initial_value=readout_pulse_amplitude,
            unit="",
            vals=_Amplitudes(),
        )
        """Amplitude of readout pulse"""

        self.spinpump_pulse_amplitude = ManualParameter(
            name="spinpump_pulse_amplitude",
            instrument=self,
            initial_value=spinpump_pulse_amplitude,
            unit="",
            vals=_Amplitudes(),
        )
        """Amplitude of spin-pump pulse"""

        self.readout_pulse_duration = ManualParameter(
            name="readout_pulse_duration",
            instrument=self,
            initial_value=readout_pulse_duration,
            unit="s",
            vals=_Durations(),
        )
        """Readout pulse duration"""

        self.spinpump_pulse_duration = ManualParameter(
            name="spinpump_pulse_duration",
            instrument=self,
            initial_value=spinpump_pulse_duration,
            unit="s",
            vals=_Durations(),
        )
        """Readout pulse duration"""

        self.acq_duration = ManualParameter(
            name="acq_duration",
            instrument=self,
            initial_value=acq_duration,
            unit="s",
            vals=_Durations(),
        )
        """
        Duration of the acquisition.
        """

        self.acq_delay = ManualParameter(
            name="acq_delay",
            instrument=self,
            initial_value=acq_delay,
            unit="s",
            vals=_Delays(),
        )
        """
        Delay between the start of the readout pulse and the start of the acquisition.
        """

        self.acq_channel = ManualParameter(
            name="acq_channel",
            instrument=self,
            initial_value=acq_channel,
            unit="",
            vals=_Hashable(),
        )
        """
        Default acquisition channel of this device element.
        """


class RxyNV(InstrumentModule):
    """
    Submodule containing parameters to perform an Rxy operation
    using a Hermite or Gaussian pulse.
    """

    def __init__(
        self,
        parent: InstrumentBase,
        name: str,
        *,
        amp180: float = math.nan,
        skewness: float = 0,
        duration: float = 20e-9,
        pulse_shape: Literal["SkewedHermitePulse", "GaussPulse"] = "SkewedHermitePulse",
    ) -> None:
        super().__init__(parent=parent, name=name)

        self.amp180 = ManualParameter(
            name="amp180",
            instrument=self,
            initial_value=amp180,
            unit="",
            vals=_Amplitudes(),
        )
        r"""Amplitude of :math:`\pi` pulse."""

        self.skewness = ManualParameter(
            name="skewness",
            instrument=self,
            initial_value=skewness,
            unit="",
            vals=validators.Numbers(min_value=-1, max_value=1),
        )
        """ First-order amplitude to the Hermite pulse envelope."""

        self.duration = ManualParameter(
            name="duration",
            instrument=self,
            initial_value=duration,
            unit="s",
            vals=_Durations(),
        )
        """Duration of the pi pulse."""

        self.pulse_shape = ManualParameter(
            name="pulse_shape",
            label="Shape of the pulse",
            instrument=self,
            initial_value=pulse_shape,
            unit="",
            vals=validators.Enum("SkewedHermitePulse", "GaussPulse"),
        )
        """Shape of the pi pulse."""


class BasicElectronicNVElement(DeviceElement):
    """
    A device element representing an electronic qubit in an NV center.

    The submodules contain the necessary device element parameters to translate higher-level
    operations into pulses. Please see the documentation of these classes.

    .. admonition:: Examples

        Qubit parameters can be set through submodule attributes

        .. jupyter-execute::

            from quantify_scheduler import BasicElectronicNVElement

            device_element = BasicElectronicNVElement("q2")

            device_element.rxy.amp180(0.1)
            device_element.measure.pulse_amplitude(0.25)
            device_element.measure.pulse_duration(300e-9)
            device_element.measure.acq_delay(430e-9)
            device_element.measure.acq_duration(1e-6)
            ...


    """

    def __init__(self, name: str, **kwargs) -> None:
        submodules_to_add = {
            "spectroscopy_operation": SpectroscopyOperationNV,
            "ports": Ports,
            "clock_freqs": ClockFrequencies,
            "reset": ResetSpinpump,
            "charge_reset": ChargeReset,
            "measure": Measure,
            "pulse_compensation": PulseCompensationModule,
            "cr_count": CRCount,
            "rxy": RxyNV,
        }
        # the logic below is to support passing a dictionary to the constructor
        # e.g. `DeviceElement("q0", rxy={"amp180": 0.1})`. But we're planning to
        # remove this feature (SE-551).
        submodule_data = {sub_name: kwargs.pop(sub_name, {}) for sub_name in submodules_to_add}
        super().__init__(name, **kwargs)

        for sub_name, sub_class in submodules_to_add.items():
            self.add_submodule(
                sub_name,
                sub_class(parent=self, name=sub_name, **submodule_data.get(sub_name, {})),
            )

        self.spectroscopy_operation: SpectroscopyOperationNV
        """Submodule :class:`~.SpectroscopyOperationNV`."""
        self.ports: Ports
        """Submodule :class:`~.Ports`."""
        self.clock_freqs: ClockFrequencies
        """Submodule :class:`~.ClockFrequencies`."""
        self.reset: ResetSpinpump
        """Submodule :class:`~.ResetSpinpump`."""
        self.charge_reset: ChargeReset
        """Submodule :class:`~.ChargeReset`."""
        self.measure: Measure
        """Submodule :class:`~.Measure`."""
        self.pulse_compensation: PulseCompensationModule
        """Submodule :class:`~.PulseCompensationModule`."""
        self.cr_count: CRCount
        """Submodule :class:`~.CRCount`."""
        self.rxy: RxyNV
        """Submodule :class:`~.Rxy`."""

    def _generate_config(self) -> dict[str, dict[str, OperationCompilationConfig]]:
        """
        Generate part of the device configuration specific to a single qubit.

        This method is intended to be used when this object is part of a
        device object containing multiple elements.
        """
        device_element_config = {
            f"{self.name}": {
                "spectroscopy_operation": OperationCompilationConfig(
                    factory_func=pulse_factories.nv_spec_pulse_mw,
                    factory_kwargs={
                        "duration": self.spectroscopy_operation.duration(),
                        "amplitude": self.spectroscopy_operation.amplitude(),
                        "port": self.ports.microwave(),
                        "clock": f"{self.name}.spec",
                        "pulse_shape": self.spectroscopy_operation.pulse_shape(),
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
                        "acq_time_source": self.measure.time_source(),
                        "acq_time_ref": self.measure.time_ref(),
                        "pulse_type": "SquarePulse",
                        "acq_protocol_default": "TriggerCount",
                    },
                    gate_info_factory_kwargs=[
                        "acq_channel_override",
                        "acq_index",
                        "bin_mode",
                        "acq_protocol",
                    ],
                ),
                "pulse_compensation": OperationCompilationConfig(
                    factory_func=None,
                    factory_kwargs={
                        "port": self.ports.microwave(),
                        "clock": f"{self.name}.f_larmor",
                        "max_compensation_amp": self.pulse_compensation.max_compensation_amp(),
                        "time_grid": self.pulse_compensation.time_grid(),
                        "sampling_rate": self.pulse_compensation.sampling_rate(),
                    },
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
                    gate_info_factory_kwargs=[
                        "acq_channel_override",
                        "acq_index",
                        "bin_mode",
                        "acq_protocol",
                    ],
                ),
                "Rxy": OperationCompilationConfig(
                    factory_func=pulse_factories.rxy_pulse,
                    factory_kwargs={
                        "amp180": self.rxy.amp180(),
                        "skewness": self.rxy.skewness(),
                        "port": self.ports.microwave(),
                        "clock": f"{self.name}.spec",
                        "duration": self.rxy.duration(),
                        "pulse_shape": self.rxy.pulse_shape(),
                    },
                    gate_info_factory_kwargs=[
                        "theta",
                        "phi",
                    ],
                ),
            }
        }
        return device_element_config

    def generate_device_config(self) -> DeviceCompilationConfig:
        """
        Generate a valid device config for the quantify-scheduler.

        This makes use of the
        :func:`~.circuit_to_device.compile_circuit_to_device_with_config_validation` function.

        This enables the settings of this qubit to be used in isolation.

        .. note:

            This config is only valid for single qubit experiments.
        """
        cfg_dict = {
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
        dev_cfg = DeviceCompilationConfig.model_validate(cfg_dict)

        return dev_cfg
