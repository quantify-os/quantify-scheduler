# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""The module contains definitions related to spin qubit elements."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from qcodes.instrument import InstrumentChannel
from qcodes.instrument.parameter import (
    ManualParameter,
    Parameter,
)
from qcodes.utils import validators

from quantify_scheduler.backends.graph_compilation import (
    DeviceCompilationConfig,
    OperationCompilationConfig,
)
from quantify_scheduler.device_under_test.device_element import DeviceElement
from quantify_scheduler.device_under_test.transmon_element import (
    DispersiveMeasurement,
    IdlingReset,
    PulseCompensationModule,
    ReferenceMagnitude,
)
from quantify_scheduler.helpers.validators import Numbers
from quantify_scheduler.operations import (
    composite_factories,
    measurement_factories,
    pulse_factories,
    pulse_library,
)

if TYPE_CHECKING:
    from qcodes.instrument.base import InstrumentBase


class PortsChargeSensor(InstrumentChannel):
    """Submodule containing the ports."""

    def __init__(
        self,
        parent: InstrumentBase,
        name: str,
        *,
        gate: str | None = None,
        readout: str | None = None,
    ) -> None:
        super().__init__(parent=parent, name=name)

        self.gate = Parameter(
            name="gate",
            instrument=self,
            initial_cache_value=gate or f"{parent.name}:gt",
            set_cmd=False,
        )
        """Name of the element's ohmic gate port."""

        self.readout = Parameter(
            name="readout",
            instrument=self,
            initial_cache_value=readout or f"{parent.name}:res",
            set_cmd=False,
        )
        """Name of the element's readout port."""


class PortsSpin(PortsChargeSensor):
    """Submodule containing the ports."""

    def __init__(
        self,
        parent: InstrumentBase,
        name: str,
        *,
        microwave: str | None = None,
        gate: str | None = None,
        readout: str | None = None,
    ) -> None:
        super().__init__(parent=parent, name=name, gate=gate, readout=readout)

        self.microwave = Parameter(
            name="microwave",
            instrument=self,
            initial_cache_value=microwave or f"{parent.name}:mw",
            set_cmd=False,
        )
        """Name of the element's microwave port."""


class ClocksFrequenciesSensor(InstrumentChannel):
    """Submodule containing the clock frequencies specifying the transitions to address."""

    def __init__(
        self,
        parent: InstrumentBase,
        name: str,
        *,
        readout: float = math.nan,
    ) -> None:
        super().__init__(parent=parent, name=name)

        self.readout = ManualParameter(
            name="readout",
            instrument=self,
            label="Readout frequency",
            unit="Hz",
            initial_value=readout,
            vals=Numbers(min_value=0, max_value=1e12, allow_nan=True),
        )
        """Frequency of the ro clock. """


class ClocksFrequenciesSpin(ClocksFrequenciesSensor):
    """Submodule containing the clock frequencies specifying the transitions to address."""

    def __init__(
        self,
        parent: InstrumentBase,
        name: str,
        *,
        f_larmor: float = math.nan,
        readout: float = math.nan,
    ) -> None:
        super().__init__(parent=parent, name=name, readout=readout)

        self.f_larmor = ManualParameter(
            name="f_larmor",
            instrument=self,
            label="Larmor frequency",
            unit="Hz",
            initial_value=f_larmor,
            vals=Numbers(min_value=0, max_value=1e12, allow_nan=True),
        )
        """Larmor frequency for the spin device element"""


class RxyGaussian(InstrumentChannel):
    """
    Submodule containing parameters for performing an Rxy operation.

    The Rxy operation uses a Gaussian pulse.
    """

    def __init__(
        self,
        parent: InstrumentBase,
        name: str,
        *,
        amp180: float = math.nan,
        duration: float = 20e-9,
        reference_magnitude_dBm: float = math.nan,
        reference_magnitude_V: float = math.nan,
        reference_magnitude_A: float = math.nan,
    ) -> None:
        super().__init__(parent=parent, name=name)
        self.amp180 = ManualParameter(
            name="amp180",
            instrument=self,
            label=r"$\pi-pulse amplitude$",
            initial_value=amp180,
            unit="",
            vals=Numbers(min_value=-10, max_value=10, allow_nan=True),
        )
        r"""Amplitude required to perform a $\pi$ pulse."""

        self.duration = ManualParameter(
            name="duration",
            instrument=self,
            initial_value=duration,
            unit="s",
            vals=validators.Numbers(min_value=0, max_value=1),
        )
        """Duration of the control pulse."""

        self.add_submodule(
            name="reference_magnitude",
            submodule=ReferenceMagnitude(
                parent=self,
                name="reference_magnitude",
                dBm=reference_magnitude_dBm,
                V=reference_magnitude_V,
                A=reference_magnitude_A,
            ),
        )
        """Reference magnitude."""


class DispersiveMeasurementSpin(DispersiveMeasurement):
    """
    Submodule containing parameters to perform a measurement.

    The measurement that is performed is using
    :func:`~quantify_scheduler.operations.measurement_factories.dispersive_measurement_spin`.
    """

    def __init__(
        self, parent: InstrumentBase, name: str, *, gate_pulse_amp: float = 0, **kwargs
    ) -> None:
        super().__init__(parent=parent, name=name, **kwargs)

        self.gate_pulse_amp = ManualParameter(
            name="gate_pulse_amp",
            instrument=self,
            initial_value=gate_pulse_amp,
            unit="",
            vals=validators.Numbers(min_value=-1, max_value=1),
        )
        """Amplitude of the gate pulse."""


class BasicSpinElement(DeviceElement):
    """
    A device element representing a Loss–DiVincenzo Spin qubit.
    The element refers to the intrinsic spin-1/2 degree of freedom of
    individual electrons/holes trapped in quantum dots.
    The charge of the particle is coupled to a resonator.

    .. admonition:: Examples

        Qubit parameters can be set through submodule attributes

        .. jupyter-execute::

            from quantify_scheduler import BasicSpinElement

            device_element = BasicSpinElement("q1")

            device_element.rxy.amp180(0.1)
            device_element.measure.pulse_amp(0.25)
            device_element.measure.pulse_duration(300e-9)
            device_element.measure.acq_delay(430e-9)
            device_element.measure.integration_time(1e-6)
            ...


    Parameters
    ----------
    name
        The name of the spin element.
    kwargs
        Can be used to pass submodule initialization data by using submodule name
        as keyword and as argument a dictionary containing the submodule parameter
        names and their value.

    """

    #  TODO replace those submodules with proper parameters,
    #   and set the kwargs parameter type to `Unpack[InstrumentBaseKWArgs]`
    def __init__(self, name: str, **kwargs: Any) -> None:  # noqa: ANN401
        submodules_to_add = {
            "reset": IdlingReset,
            "rxy": RxyGaussian,
            "measure": DispersiveMeasurementSpin,
            "pulse_compensation": PulseCompensationModule,
            "ports": PortsSpin,
            "clock_freqs": ClocksFrequenciesSpin,
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

        self.reset: IdlingReset
        """Submodule :class:`~.IdlingReset`."""
        self.rxy: RxyGaussian
        """Submodule :class:`~.RxyGaussian`."""
        self.measure: DispersiveMeasurementSpin
        """Submodule :class:`~.DispersiveMeasurementSpin`."""
        self.pulse_compensation: PulseCompensationModule
        """Submodule :class:`~.PulseCompensationModule`."""
        self.ports: PortsSpin
        """Submodule :class:`~.PortsSpin`."""
        self.clock_freqs: ClocksFrequenciesSpin
        """Submodule :class:`~.ClocksFrequenciesSpin`."""

    def _generate_config(self) -> dict[str, dict[str, OperationCompilationConfig]]:
        """
        Generate part of the device configuration specific to a single qubit trapped in a quantum
        dot. A resonator to perform dispersive readout is attached to the gate to perform charge
        sensing.

        This method is intended to be used when this object is part of a
        device object containing multiple elements.
        """
        device_element_config = {
            f"{self.name}": {
                "reset": OperationCompilationConfig(
                    factory_func=pulse_library.IdlePulse,
                    factory_kwargs={
                        "duration": self.reset.duration(),
                    },
                ),
                # example of a pulse with a parametrized mapping, using a factory
                "Rxy": OperationCompilationConfig(
                    factory_func=pulse_factories.rxy_gauss_pulse,
                    factory_kwargs={
                        "amp180": self.rxy.amp180(),
                        "port": self.ports.microwave(),
                        "clock": f"{self.name}.f_larmor",
                        "duration": self.rxy.duration(),
                        "reference_magnitude": pulse_library.ReferenceMagnitude.from_parameter(
                            self.rxy.reference_magnitude
                        ),
                    },
                    gate_info_factory_kwargs=[
                        "theta",
                        "phi",
                    ],  # the keys from the gate info to pass to the factory function
                ),
                "Rz": OperationCompilationConfig(
                    factory_func=pulse_factories.phase_shift,
                    factory_kwargs={
                        "clock": f"{self.name}.f_larmor",
                    },
                    gate_info_factory_kwargs=[
                        "theta",
                    ],  # the keys from the gate info to pass to the factory function
                ),
                "H": OperationCompilationConfig(
                    factory_func=composite_factories.hadamard_as_y90z,
                    factory_kwargs={
                        "qubit": f"{self.name}",
                    },
                ),
                # the measurement also has a parametrized mapping, and uses a
                # factory function.
                "measure": OperationCompilationConfig(
                    factory_func=measurement_factories.dispersive_measurement_spin,
                    factory_kwargs={
                        "port": self.ports.readout(),
                        "clock": f"{self.name}.ro",
                        "gate_port": self.ports.gate(),
                        "pulse_type": self.measure.pulse_type(),
                        "pulse_amp": self.measure.pulse_amp(),
                        "gate_pulse_amp": self.measure.gate_pulse_amp(),
                        "pulse_duration": self.measure.pulse_duration(),
                        "acq_delay": self.measure.acq_delay(),
                        "acq_duration": self.measure.integration_time(),
                        "acq_channel": self.measure.acq_channel(),
                        "acq_protocol_default": "SSBIntegrationComplex",
                        "reset_clock_phase": self.measure.reset_clock_phase(),
                        "reference_magnitude": pulse_library.ReferenceMagnitude.from_parameter(
                            self.measure.reference_magnitude
                        ),
                        "acq_weights_a": self.measure.acq_weights_a(),
                        "acq_weights_b": self.measure.acq_weights_b(),
                        "acq_weights_sampling_rate": self.measure.acq_weights_sampling_rate(),
                        "acq_rotation": self.measure.acq_rotation(),
                        "acq_threshold": self.measure.acq_threshold(),
                        "num_points": self.measure.num_points(),
                        "freq": None,
                    },
                    gate_info_factory_kwargs=[
                        "acq_channel_override",
                        "acq_index",
                        "bin_mode",
                        "acq_protocol",
                        "feedback_trigger_label",
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
            }
        }
        return device_element_config

    def generate_device_config(self) -> DeviceCompilationConfig:
        """
        Generate a valid device config.

        The config will be used for the quantify-scheduler making use of the
        :func:`~.circuit_to_device.compile_circuit_to_device_with_config_validation` function.

        This enables the settings of this qubit to be used in isolation.

        .. note:

            This config is only valid for single qubit experiments.
        """
        cfg_dict = {
            "elements": self._generate_config(),
            "clocks": {
                f"{self.name}.f_larmor": self.clock_freqs.f_larmor(),
                f"{self.name}.ro": self.clock_freqs.readout(),
            },
            "edges": {},
        }
        dev_cfg = DeviceCompilationConfig.model_validate(cfg_dict)

        return dev_cfg


class ChargeSensor(DeviceElement):
    """
    A device element representing a Charge Senseor connected to a tank circuit to perform
    dispersive readoud.

    .. admonition:: Examples

        Sensor parameters can be set through submodule attributes

        .. jupyter-execute::

            from quantify_scheduler import ChargeSensor

            sensor = ChargeSensor("s1")

            sensor.measure.pulse_amp(0.25)
            sensor.measure.pulse_duration(300e-9)
            sensor.measure.acq_delay(430e-9)
            sensor.measure.integration_time(1e-6)
            ...

    Parameters
    ----------
    name
        The name of the spin element.
    kwargs
        Can be used to pass submodule initialization data by using submodule name
        as keyword and as argument a dictionary containing the submodule parameter
        names and their value.

    """

    def __init__(self, name: str, **kwargs: Any) -> None:  # noqa: ANN401
        submodules_to_add = {
            "measure": DispersiveMeasurementSpin,
            "pulse_compensation": PulseCompensationModule,
            "ports": PortsChargeSensor,
            "clock_freqs": ClocksFrequenciesSensor,
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

        """Submodule :class:`~.RxyGaussian`."""
        self.measure: DispersiveMeasurementSpin
        """Submodule :class:`~.DispersiveMeasurementSpin`."""
        self.pulse_compensation: PulseCompensationModule
        """Submodule :class:`~.PulseCompensationModule`."""
        self.ports: PortsChargeSensor
        """Submodule :class:`~.PortsSpin`."""
        self.clock_freqs: ClocksFrequenciesSensor
        """Submodule :class:`~.ClocksFrequenciesSpin`."""

    def _generate_config(self) -> dict[str, dict[str, OperationCompilationConfig]]:
        """
        Generate part of the device configuration specific to a single qubit.

        This method is intended to be used when this object is part of a
        device object containing multiple elements.
        """
        qubit_config = {
            f"{self.name}": {
                # the measurement also has a parametrized mapping, and uses a
                # factory function.
                "measure": OperationCompilationConfig(
                    factory_func=measurement_factories.dispersive_measurement_spin,
                    factory_kwargs={
                        "port": self.ports.readout(),
                        "clock": f"{self.name}.ro",
                        "gate_port": self.ports.gate(),
                        "pulse_type": self.measure.pulse_type(),
                        "pulse_amp": self.measure.pulse_amp(),
                        "gate_pulse_amp": self.measure.gate_pulse_amp(),
                        "pulse_duration": self.measure.pulse_duration(),
                        "acq_delay": self.measure.acq_delay(),
                        "acq_duration": self.measure.integration_time(),
                        "acq_channel": self.measure.acq_channel(),
                        "acq_protocol_default": "SSBIntegrationComplex",
                        "reset_clock_phase": self.measure.reset_clock_phase(),
                        "reference_magnitude": pulse_library.ReferenceMagnitude.from_parameter(
                            self.measure.reference_magnitude
                        ),
                        "acq_weights_a": self.measure.acq_weights_a(),
                        "acq_weights_b": self.measure.acq_weights_b(),
                        "acq_weights_sampling_rate": self.measure.acq_weights_sampling_rate(),
                        "acq_rotation": self.measure.acq_rotation(),
                        "acq_threshold": self.measure.acq_threshold(),
                        "num_points": self.measure.num_points(),
                        "freq": None,
                    },
                    gate_info_factory_kwargs=[
                        "acq_channel_override",
                        "acq_index",
                        "bin_mode",
                        "acq_protocol",
                        "feedback_trigger_label",
                    ],
                ),
                "pulse_compensation": OperationCompilationConfig(
                    factory_func=None,
                    factory_kwargs={
                        "port": self.ports.gate(),
                        "clock": "cl0.baseband",
                        "max_compensation_amp": self.pulse_compensation.max_compensation_amp(),
                        "time_grid": self.pulse_compensation.time_grid(),
                        "sampling_rate": self.pulse_compensation.sampling_rate(),
                    },
                ),
            }
        }
        return qubit_config

    def generate_device_config(self) -> DeviceCompilationConfig:
        """
        Generate a valid device config.

        The config will be used for the quantify-scheduler making use of the
        :func:`~.circuit_to_device.compile_circuit_to_device_with_config_validation` function.

        This enables the settings of this qubit to be used in isolation.

        .. note:

            This config is only valid for single qubit experiments.
        """
        cfg_dict = {
            "elements": self._generate_config(),
            "clocks": {
                f"{self.name}.ro": self.clock_freqs.readout(),
            },
            "edges": {},
        }
        dev_cfg = DeviceCompilationConfig.model_validate(cfg_dict)

        return dev_cfg
