from qcodes.instrument.base import Instrument
from qcodes.instrument.parameter import ManualParameter


class TransmonElement(Instrument):
    """
    A device element representing a single transmon coupled to a
    readout resonator.

    This object can be used to generate configuration files compatible with the
    :func:`quantify.scheduler.compilation.add_pulse_information_transmon` backend.

    """

    def __init__(self, name, **kw):
        super().__init__(name, **kw)
        self._add_device_parameters()

    def _add_device_parameters(self):
        """
        Adds parameters used to generate a device configuration file
        """

        self.add_parameter(
            "init_duration",
            initial_value=200e-6,
            unit="s",
            parameter_class=ManualParameter,
        )
        self.add_parameter(
            "mw_amp180",
            label=r"$\pi-pulse amplitude$",
            unit="V",
            parameter_class=ManualParameter,
        )
        self.add_parameter(
            "mw_motzoi", initial_value=0, unit="", parameter_class=ManualParameter
        )
        self.add_parameter(
            "mw_duration",
            initial_value=20e-9,
            unit="s",
            parameter_class=ManualParameter,
        )
        self.add_parameter("mw_ef_amp180", unit="V", parameter_class=ManualParameter)

        self.add_parameter(
            "mw_port", initial_value=f"{self.name}:mw", parameter_class=ManualParameter
        )

        self.add_parameter(
            "fl_port", initial_value=f"{self.name}:fl", parameter_class=ManualParameter
        )

        self.add_parameter(
            "ro_port", initial_value=f"{self.name}:res", parameter_class=ManualParameter
        )

        self.add_parameter(
            "freq_01",
            label="Qubit frequency",
            unit="Hz",
            parameter_class=ManualParameter,
        )

        self.add_parameter(
            "ro_freq",
            label="Readout frequency",
            unit="Hz",
            parameter_class=ManualParameter,
        )
        self.add_parameter(
            "ro_pulse_amp", initial_value=0.5, unit="V", parameter_class=ManualParameter
        )
        self.add_parameter(
            "ro_pulse_duration",
            initial_value=300e-9,
            unit="s",
            parameter_class=ManualParameter,
        )

        self.add_parameter(
            "ro_pulse_type", initial_value="square", parameter_class=ManualParameter
        )  # TODO add enum validator

        self.add_parameter(
            "ro_acq_delay", initial_value=0, unit="s", parameter_class=ManualParameter
        )
        self.add_parameter(
            "ro_acq_integration_time",
            initial_value=1e-6,
            unit="s",
            parameter_class=ManualParameter,
        )

    def generate_qubit_config(self) -> dict:
        """
        Generates part of the device configuration specific to a single qubit.

        This method is intended to be used when the this object is part of a
        device object containing multiple elements.
        """
        qubit_config = {
            f"{self.name}": {
                "resources": {
                    "port_mw": self.mw_port(),
                    "port_ro": self.ro_port(),
                    "port_flux": self.fl_port(),
                    "clock_01": f"{self.name}.01",  # defines a clock that tracks the 0-1 transition of the qubit
                    "clock_ro": f"{self.name}.ro",  # defines a clock that tracks the readout resonator
                },
                "params": {
                    "acquisition": "SSBIntegrationComplex",
                    "mw_freq": self.freq_01(),
                    "mw_amp180": self.mw_amp180(),
                    "mw_motzoi": self.mw_motzoi(),
                    "mw_duration": self.mw_duration(),
                    "mw_ef_amp180": self.mw_ef_amp180(),
                    "ro_freq": self.ro_freq(),
                    "ro_pulse_amp": self.ro_pulse_amp(),
                    "ro_pulse_type": self.ro_pulse_type(),
                    "ro_pulse_duration": self.ro_pulse_duration(),
                    "ro_acq_delay": self.ro_acq_delay(),
                    "ro_acq_integration_time": self.ro_acq_integration_time(),
                    "ro_acq_weigth_type": "SSB",
                    "init_duration": self.init_duration(),
                },
            }
        }
        return qubit_config

    def generate_device_config(self) -> dict:
        """
        Generates a valid device config for the quantify scheduler making use of the
        :func:`quantify.scheduler.compilation.add_pulse_information_transmon` backend.

        This enables the settings of this qubit to be used in isolation.

        .. note:

            This config is only valid for single qubit experiments.
        """
        dev_cfg = {
            "backend": "quantify.scheduler.compilation.add_pulse_information_transmon",
            "qubits": self.generate_qubit_config(),
            "edges": {},
        }
        return dev_cfg
