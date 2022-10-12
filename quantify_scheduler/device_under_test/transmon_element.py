# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
from typing import Dict, Any

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

        self.add_parameter(
            "microwave",
            docstring=r"Name of the element's microwave port.",
            initial_cache_value=f"{parent.name}:mw",
            parameter_class=Parameter,
            set_cmd=False,
        )

        self.add_parameter(
            "flux",
            docstring=r"Name of the element's flux port.",
            initial_cache_value=f"{parent.name}:fl",
            parameter_class=Parameter,
            set_cmd=False,
        )

        self.add_parameter(
            "readout",
            docstring=r"Name of the element's readout port.",
            initial_cache_value=f"{parent.name}:res",
            parameter_class=Parameter,
            set_cmd=False,
        )


class ClocksFrequencies(InstrumentChannel):
    """
    Submodule containing the clock frequencies specifying the transitions to address.
    """

    def __init__(self, parent: InstrumentBase, name: str, **kwargs: Any) -> None:
        super().__init__(parent=parent, name=name)

        self.add_parameter(
            "f01",
            label="Qubit frequency",
            unit="Hz",
            parameter_class=ManualParameter,
            docstring=f"Frequency of the {parent.name}.01 clock",
            initial_value=float("nan"),
            vals=Numbers(min_value=0, max_value=1e12, allow_nan=True),
        )
        self.add_parameter(
            "f12",
            label="Frequency of the |1>-|2> transition",
            unit="Hz",
            initial_value=float("nan"),
            docstring=f"Frequency of the {parent.name}.12 clock",
            parameter_class=ManualParameter,
            vals=Numbers(min_value=0, max_value=1e12, allow_nan=True),
        )

        self.add_parameter(
            "readout",
            label="Readout frequency",
            unit="Hz",
            parameter_class=ManualParameter,
            docstring=f"Frequency of the {parent.name}.ro clock. ",
            initial_value=float("nan"),
            vals=Numbers(min_value=0, max_value=1e12, allow_nan=True),
        )


class IdlingReset(InstrumentChannel):
    """
    Submodule containing parameters for doing a reset by idling.
    """

    def __init__(self, parent: InstrumentBase, name: str, **kwargs: Any) -> None:
        super().__init__(parent=parent, name=name)

        self.add_parameter(
            "duration",
            docstring="""Duration of the passive qubit reset
            (initialization by relaxation).""",
            initial_value=200e-6,
            unit="s",
            parameter_class=ManualParameter,
            vals=validators.Numbers(min_value=0, max_value=1),
        )


class RxyDRAG(InstrumentChannel):
    """
    Submodule containing parameters for performing an Rxy operation
    using a DRAG pulse.
    """

    def __init__(self, parent: InstrumentBase, name: str, **kwargs: Any) -> None:
        super().__init__(parent=parent, name=name)
        self.add_parameter(
            "amp180",
            docstring=r"""Amplitude required to perform a $\pi$ pulse.""",
            label=r"$\pi-pulse amplitude$",
            initial_value=float("nan"),
            unit="V",
            parameter_class=ManualParameter,
            vals=Numbers(min_value=-10, max_value=10, allow_nan=True),
        )
        self.add_parameter(
            "motzoi",
            docstring="""Ratio between the Gaussian Derivative (D) and Gaussian (G)
            components of the DRAG pulse.""",
            initial_value=0,
            unit="",
            parameter_class=ManualParameter,
            vals=validators.Numbers(min_value=-1, max_value=1),
        )
        self.add_parameter(
            "duration",
            docstring="""Duration of the control pulse.""",
            initial_value=20e-9,
            unit="s",
            parameter_class=ManualParameter,
            vals=validators.Numbers(min_value=0, max_value=1),
        )


class DispersiveMeasurement(InstrumentChannel):
    """
    Submodule containing parameters to perform a measurement using
    :func:`~quantify_scheduler.operations.measurement_factories.dispersive_measurement`
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
            "pulse_amp",
            docstring="Amplitude of the readout pulse.",
            initial_value=0.25,
            unit="V",
            parameter_class=ManualParameter,
            vals=validators.Numbers(min_value=0, max_value=2),
        )
        self.add_parameter(
            "pulse_duration",
            docstring="Duration of the readout pulse.",
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
            docstring="Delay between the start of the readout pulse and the start of "
            + "the acquisition. Note that some hardware backends do not support "
            "starting a pulse and the acquisition in the same clock cycle making 0 "
            "delay an invalid value.",
            initial_value=0,  # float("nan"),
            unit="s",
            parameter_class=ManualParameter,
            # in principle the values should be a few 100 ns but the validator is here
            # only to protect against silly typos that lead to out of memory errors.
            vals=validators.Numbers(min_value=0, max_value=100e-6),
        )
        self.add_parameter(
            "integration_time",
            docstring="Integration time for the readout acquisition.",
            initial_value=1e-6,
            unit="s",
            parameter_class=ManualParameter,
            # in principle the values should be a few us but the validator is here
            # only to protect against silly typos that lead to out of memory errors.
            vals=validators.Numbers(min_value=0, max_value=100e-6),
        )

        ro_acq_weight_type_validator = validators.Enum("SSB")
        self.add_parameter(
            "acq_weight_type",
            initial_value="SSB",
            parameter_class=ManualParameter,
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
        self.add_submodule("rxy", RxyDRAG(self, "rxy"))
        self.add_submodule("measure", DispersiveMeasurement(self, "measure"))
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


@deprecated(
    "0.8",
    "Consider replacing with BasicTransmonElement or implementing a custom device element.",
)
class TransmonElement(DeviceElement):
    """
    A device element representing a single transmon coupled to a
    readout resonator.

    This object can be used to generate configuration files compatible with the
    :func:`~quantify_scheduler.compilation.add_pulse_information_transmon` function.
    """

    def __init__(self, name: str, **kwargs):
        """
        Initializes the parent class and adds
        :class:`~qcodes.instrument.parameter.Parameter` s /
        :class:`~qcodes.instrument.parameter.ManualParameter` s /
        :class:`~qcodes.instrument.parameter.InstrumentRefParameter` s to it.

        The list of all parameters and their latest (cached) value can be listed as
        follows:

        .. jupyter-execute::

            from quantify_scheduler.device_under_test import transmon_element
            q0 = transmon_element.TransmonElement("q0")

            q0.print_readable_snapshot()

        Parameters
        -----------
        name:
            The name of the transmon element.

        """
        super().__init__(name, **kwargs)
        # pylint: disable=fixme
        # TODO: create DeviceElement parent class and make instrument_coordinator
        # a parameter of that class, see issue quantify-scheduler#148
        self.add_parameter(
            "instrument_coordinator",
            initial_value=None,
            parameter_class=InstrumentRefParameter,
            vals=validators.Strings(),
        )
        self._add_device_parameters()

    def _add_device_parameters(self):
        self.add_parameter(
            "init_duration",
            docstring="""Duration of the passive qubit reset
            (initialization by relaxation).""",
            initial_value=200e-6,
            unit="s",
            parameter_class=ManualParameter,
            vals=validators.Numbers(min_value=0, max_value=1),
        )
        self.add_parameter(
            "mw_amp180",
            docstring=r"""Amplitude of the $\pi$ pulse
            (considering a pulse duration of `mw_pulse_duration`).""",
            label=r"$\pi-pulse amplitude$",
            initial_value=float("nan"),
            unit="V",
            parameter_class=ManualParameter,
            vals=Numbers(min_value=-10, max_value=10, allow_nan=True),
        )
        self.add_parameter(
            "mw_motzoi",
            docstring="""Ratio between the Gaussian Derivative (D) and Gaussian (G)
            components of the DRAG pulse.""",
            initial_value=0,
            unit="",
            parameter_class=ManualParameter,
            vals=validators.Numbers(min_value=-1, max_value=1),
        )
        self.add_parameter(
            "mw_pulse_duration",
            docstring=(
                "Duration of the pulses applied on the transmon's microwave port."
            ),
            initial_value=20e-9,
            unit="s",
            parameter_class=ManualParameter,
            vals=validators.Numbers(min_value=0, max_value=1),
        )
        self.add_parameter(
            "mw_ef_amp180",
            docstring=(
                "Amplitude of the pulse necessary to drive the |1>-|2> "
                "transition (considering a pulse duration of `mw_pulse_duration`)."
            ),
            unit="V",
            initial_value=float("nan"),
            parameter_class=ManualParameter,
            vals=Numbers(min_value=-10, max_value=10, allow_nan=True),
        )

        self.add_parameter(
            "mw_port",
            docstring=r"Name of the transmon's microwave port.",
            initial_cache_value=f"{self.name}:mw",
            parameter_class=Parameter,
            set_cmd=False,
        )

        self.add_parameter(
            "fl_port",
            docstring=r"Name of the transmon's flux port.",
            initial_cache_value=f"{self.name}:fl",
            parameter_class=Parameter,
            set_cmd=False,
        )

        self.add_parameter(
            "ro_port",
            docstring=r"Name of the transmon's readout resonator port.",
            initial_cache_value=f"{self.name}:res",
            parameter_class=Parameter,
            set_cmd=False,
        )

        self.add_parameter(
            "mw_01_clock",
            docstring=r"Name of the clock corresponding to the qubit frequency.",
            initial_cache_value=f"{self.name}.01",
            parameter_class=Parameter,
            set_cmd=False,
        )

        self.add_parameter(
            "mw_12_clock",
            docstring=(
                "Name of the clock corresponding to the |1>-|2> transition "
                "frequency."
            ),
            initial_cache_value=f"{self.name}.12",
            parameter_class=Parameter,
            set_cmd=False,
        )

        self.add_parameter(
            "ro_clock",
            docstring=r"Name of the readout resonator clock.",
            initial_cache_value=f"{self.name}.ro",
            parameter_class=Parameter,
            set_cmd=False,
        )

        self.add_parameter(
            "freq_01",
            label="Qubit frequency",
            unit="Hz",
            parameter_class=ManualParameter,
            initial_value=float("nan"),
            vals=Numbers(min_value=0, max_value=1e12, allow_nan=True),
        )
        self.add_parameter(
            "freq_12",
            label="Frequency of the |1>-|2> transition",
            unit="Hz",
            initial_value=float("nan"),
            parameter_class=ManualParameter,
            vals=Numbers(min_value=0, max_value=1e12, allow_nan=True),
        )

        self.add_parameter(
            "ro_freq",
            docstring="Frequency of the pulse sent to the readout resonator.",
            label="Readout frequency",
            unit="Hz",
            parameter_class=ManualParameter,
            initial_value=float("nan"),
            vals=Numbers(min_value=0, max_value=1e12, allow_nan=True),
        )
        self.add_parameter(
            "ro_pulse_amp",
            docstring="Amplitude of the readout pulse.",
            initial_value=0.5,
            unit="V",
            parameter_class=ManualParameter,
            vals=validators.Numbers(min_value=-10, max_value=10),
        )
        self.add_parameter(
            "ro_pulse_duration",
            docstring="Duration of the readout pulse.",
            initial_value=300e-9,
            unit="s",
            parameter_class=ManualParameter,
            vals=validators.Numbers(min_value=0, max_value=1),
        )

        pulse_types = validators.Enum("SquarePulse")
        self.add_parameter(
            "ro_pulse_type",
            docstring=(
                "Envelope function that defines the shape of "
                "the readout pulse prior to modulation."
            ),
            initial_value="SquarePulse",
            parameter_class=ManualParameter,
            vals=pulse_types,
        )

        self.add_parameter(
            "ro_pulse_delay",
            docstring="Delay before the execution of the readout pulse.",
            label="Readout pulse delay",
            initial_value=300e-9,
            unit="s",
            parameter_class=ManualParameter,
            vals=validators.Numbers(min_value=-1, max_value=1),
        )

        self.add_parameter(
            "ro_acq_channel",
            docstring="Channel corresponding to this qubit.",
            initial_value=0,
            unit="#",
            parameter_class=ManualParameter,
            vals=validators.Ints(min_value=0),
        )

        self.add_parameter(
            "ro_acq_delay",
            docstring="Delay between the readout pulse and acquisition.",
            initial_value=0,
            unit="s",
            parameter_class=ManualParameter,
            vals=validators.Numbers(min_value=-1, max_value=1),
        )
        self.add_parameter(
            "ro_acq_integration_time",
            docstring="Integration time for the readout acquisition.",
            initial_value=1e-6,
            unit="s",
            parameter_class=ManualParameter,
            vals=validators.Numbers(min_value=0, max_value=1),
        )
        self.add_parameter(
            "spec_pulse_duration",
            docstring="Duration of the qubit spectroscopy pulse.",
            initial_value=8e-6,
            unit="s",
            parameter_class=ManualParameter,
            vals=validators.Numbers(min_value=0, max_value=1),
        )
        self.add_parameter(
            "spec_pulse_frequency",
            docstring="Frequency of the qubit spectroscopy pulse.",
            initial_value=float("nan"),
            unit="Hz",
            parameter_class=ManualParameter,
            vals=Numbers(min_value=0, max_value=1e12, allow_nan=True),
        )
        self.add_parameter(
            "spec_pulse_amp",
            docstring="Amplitude of the qubit spectroscopy pulse.",
            initial_value=0.5,
            unit="V",
            parameter_class=ManualParameter,
            vals=validators.Numbers(min_value=0, max_value=1e12),
        )
        self.add_parameter(
            "spec_pulse_clock",
            initial_cache_value=f"{self.name}.01",
            parameter_class=Parameter,
            set_cmd=False,
        )
        acquisition_validator = validators.Enum("SSBIntegrationComplex", "Trace")
        self.add_parameter(
            "acquisition",
            docstring=(
                "Acquisition mode. Can take either the 'Trace' value, which "
                "yields a time trace of the data, or 'SSBIntegrationComplex', "
                "which yields integrated single-sideband demodulated "
                "data."
            ),
            initial_value="SSBIntegrationComplex",
            parameter_class=ManualParameter,
            vals=acquisition_validator,
        )
        ro_acq_weight_type_validator = validators.Enum("SSB")
        self.add_parameter(
            "ro_acq_weight_type",
            initial_value="SSB",
            parameter_class=ManualParameter,
            vals=ro_acq_weight_type_validator,
        )

    def generate_config(self) -> Dict[str, Dict[str, OperationCompilationConfig]]:
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
                        "duration": self.init_duration(),
                    },
                ),
                # example of a pulse with a parametrized mapping, using a factory
                "Rxy": OperationCompilationConfig(
                    factory_func="quantify_scheduler.operations."
                    + "pulse_factories.rxy_drag_pulse",
                    factory_kwargs={
                        "amp180": self.mw_amp180(),
                        "motzoi": self.mw_motzoi(),
                        "port": self.mw_port(),
                        "clock": self.mw_01_clock(),
                        "duration": self.mw_pulse_duration(),
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
                        "port": self.ro_port(),
                        "clock": self.ro_clock(),
                        "pulse_type": self.ro_pulse_type(),
                        "pulse_amp": self.ro_pulse_amp(),
                        "pulse_duration": self.ro_pulse_duration(),
                        "acq_delay": self.ro_acq_delay(),
                        "acq_duration": self.ro_acq_integration_time(),
                        "acq_channel": self.ro_acq_channel(),
                        "acq_protocol_default": "SSBIntegrationComplex",
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
            "elements": self.generate_config(),
            "clocks": {
                self.mw_01_clock(): self.freq_01(),
                self.mw_12_clock(): self.freq_12(),
                self.ro_clock(): self.ro_freq(),
            },
            "edges": {},
        }
        dev_cfg = DeviceCompilationConfig.parse_obj(cfg_dict)

        return dev_cfg
