# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
from typing import Dict, Any
from qcodes.instrument.base import Instrument
from qcodes.instrument.parameter import (
    ManualParameter,
    Parameter,
    InstrumentRefParameter,
)
from qcodes.utils import validators


class TransmonElement(Instrument):
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

            from quantify_scheduler.device_elements import transmon_element
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
            initial_value=200e-6,
            unit="s",
            parameter_class=ManualParameter,
            vals=validators.Numbers(min_value=0, max_value=1),
        )
        self.add_parameter(
            "mw_amp180",
            label=r"$\pi-pulse amplitude$",
            unit="V",
            parameter_class=ManualParameter,
            vals=validators.Numbers(min_value=-10, max_value=10),
        )
        self.add_parameter(
            "mw_motzoi",
            initial_value=0,
            unit="",
            parameter_class=ManualParameter,
            vals=validators.Numbers(min_value=0, max_value=1),
        )
        self.add_parameter(
            "mw_pulse_duration",
            initial_value=20e-9,
            unit="s",
            parameter_class=ManualParameter,
            vals=validators.Numbers(min_value=0, max_value=1),
        )
        self.add_parameter(
            "mw_ef_amp180",
            unit="V",
            parameter_class=ManualParameter,
            vals=validators.Numbers(min_value=-10, max_value=10),
        )

        self.add_parameter(
            "mw_port",
            initial_cache_value=f"{self.name}:mw",
            parameter_class=Parameter,
            set_cmd=False,
        )

        self.add_parameter(
            "fl_port",
            initial_cache_value=f"{self.name}:fl",
            parameter_class=Parameter,
            set_cmd=False,
        )

        self.add_parameter(
            "ro_port",
            initial_cache_value=f"{self.name}:res",
            parameter_class=Parameter,
            set_cmd=False,
        )

        self.add_parameter(
            "mw_01_clock",
            initial_cache_value=f"{self.name}.01",
            parameter_class=Parameter,
            set_cmd=False,
        )

        self.add_parameter(
            "mw_12_clock",
            initial_cache_value=f"{self.name}.12",
            parameter_class=Parameter,
            set_cmd=False,
        )

        self.add_parameter(
            "ro_clock",
            initial_cache_value=f"{self.name}.ro",
            parameter_class=Parameter,
            set_cmd=False,
        )

        self.add_parameter(
            "freq_01",
            label="Qubit frequency",
            unit="Hz",
            parameter_class=ManualParameter,
            vals=validators.Numbers(min_value=0, max_value=1e12),
        )
        self.add_parameter(
            "freq_12",
            label="Frequency of the 12 transition",
            unit="Hz",
            parameter_class=ManualParameter,
            vals=validators.Numbers(min_value=0, max_value=1e12),
        )

        self.add_parameter(
            "ro_freq",
            label="Readout frequency",
            unit="Hz",
            parameter_class=ManualParameter,
            vals=validators.Numbers(min_value=0, max_value=1e12),
        )
        self.add_parameter(
            "ro_pulse_amp",
            initial_value=0.5,
            unit="V",
            parameter_class=ManualParameter,
            vals=validators.Numbers(min_value=-10, max_value=10),
        )
        self.add_parameter(
            "ro_pulse_duration",
            initial_value=300e-9,
            unit="s",
            parameter_class=ManualParameter,
            vals=validators.Numbers(min_value=0, max_value=1),
        )

        pulse_types = validators.Enum("square")
        self.add_parameter(
            "ro_pulse_type",
            initial_value="square",
            parameter_class=ManualParameter,
            vals=pulse_types,
        )

        self.add_parameter(
            "ro_pulse_delay",
            label="Readout pulse delay",
            initial_value=300e-9,
            unit="s",
            parameter_class=ManualParameter,
            vals=validators.Numbers(min_value=0, max_value=1),
        )

        self.add_parameter(
            "ro_acq_delay",
            initial_value=0,
            unit="s",
            parameter_class=ManualParameter,
            vals=validators.Numbers(min_value=0, max_value=1),
        )
        self.add_parameter(
            "ro_acq_integration_time",
            initial_value=1e-6,
            unit="s",
            parameter_class=ManualParameter,
            vals=validators.Numbers(min_value=0, max_value=1),
        )
        self.add_parameter(
            "spec_pulse_duration",
            initial_value=8e-6,
            unit="s",
            parameter_class=ManualParameter,
            vals=validators.Numbers(min_value=0, max_value=1),
        )
        self.add_parameter(
            "spec_pulse_frequency",
            initial_value=4.715e9,
            unit="Hz",
            parameter_class=ManualParameter,
            vals=validators.Numbers(min_value=0, max_value=1e12),
        )
        self.add_parameter(
            "spec_pulse_amp",
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
        device_cfg_backend_validator = validators.Enum(
            "quantify_scheduler.compilation.add_pulse_information_transmon"
        )
        self.add_parameter(
            "device_cfg_backend",
            initial_value=(
                "quantify_scheduler.compilation.add_pulse_information_transmon"
            ),
            parameter_class=ManualParameter,
            vals=device_cfg_backend_validator,
        )

    def generate_config(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Generates part of the device configuration specific to a single qubit.

        This method is intended to be used when this object is part of a
        device object containing multiple elements.
        """
        qubit_config = {
            f"{self.name}": {
                "resources": {
                    "port_mw": self.mw_port(),
                    "port_ro": self.ro_port(),
                    "port_flux": self.fl_port(),
                    "clock_01": self.mw_01_clock(),
                    "clock_ro": self.ro_clock(),
                },
                "params": {
                    "acquisition": self.acquisition(),
                    "mw_freq": self.freq_01(),
                    "mw_amp180": self.mw_amp180(),
                    "mw_motzoi": self.mw_motzoi(),
                    "mw_duration": self.mw_pulse_duration(),
                    "mw_ef_amp180": self.mw_ef_amp180(),
                    "ro_freq": self.ro_freq(),
                    "ro_pulse_amp": self.ro_pulse_amp(),
                    "ro_pulse_type": self.ro_pulse_type(),
                    "ro_pulse_duration": self.ro_pulse_duration(),
                    "ro_acq_delay": self.ro_acq_delay(),
                    "ro_acq_integration_time": self.ro_acq_integration_time(),
                    "ro_acq_weight_type": self.ro_acq_weight_type(),
                    "init_duration": self.init_duration(),
                },
            }
        }
        return qubit_config

    def generate_device_config(self) -> Dict[str, Any]:
        """
        Generates a valid device config for the quantify-scheduler making use of the
        :func:`quantify_scheduler.compilation.add_pulse_information_transmon` function.

        This enables the settings of this qubit to be used in isolation.

        .. note:

            This config is only valid for single qubit experiments.
        """
        dev_cfg = {
            "backend": self.device_cfg_backend(),
            "qubits": self.generate_config(),
            "edges": {},
        }
        return dev_cfg
