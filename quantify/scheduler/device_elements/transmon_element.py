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
        self.add_parameter("mw_amp180", unit="V", parameter_class=ManualParameter)
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
            "freq_01",
            label="Qubit frequency",
            unit="Hz",
            parameter_class=ManualParameter,
        )

    #         self.add_parameter("ro_freq", initial_value=200e-6, unit="s", parameter_class=ManualParameter)
    #         self.add_parameter("ro_pulse_amp", initial_value=200e-6, unit="s", parameter_class=ManualParameter)
    #         self.add_parameter("ro_pulse_type", initial_value=200e-6, unit="s", parameter_class=ManualParameter)
    #         self.add_parameter("ro_acq_delay", initial_value=200e-6, unit="s", parameter_class=ManualParameter)
    #         self.add_parameter("ro_acq_integration_time", initial_value=200e-6, unit="s", parameter_class=ManualParameter)

    # "ro_freq": 6.8979e9,
    #                 "ro_pulse_amp": 0.5,  # readout pulse amplitude (V)
    #                 "ro_pulse_type": "square",  # shape of waveform used for the readout pulse
    #                 "ro_pulse_duration": 200e-9,  # readout pulse duration (s)
    #                 "ro_acq_delay": 100e-9,  # time between sending readout pulse and acquisition (s)
    #                 "ro_acq_integration_time": 400e-9,  # Minimum wait time between iterations (s)
    #                 "ro_acq_weigth_type": "SSB",  # This is not used at the moment

    #         self.add_parameter("reset_duration", initial_value=200e-6, unit="s", parameter_class=ManualParameter)
    #         self.add_parameter("reset_duration", initial_value=200e-6, unit="s", parameter_class=ManualParameter)
    #         self.add_parameter("reset_duration", initial_value=200e-6, unit="s", parameter_class=ManualParameter)
    #         self.add_parameter("reset_duration", initial_value=200e-6, unit="s", parameter_class=ManualParameter)

    def generate_qubit_config(self) -> dict:
        """
        Generates part of the device configuration specific to a single qubit.

        This method is intended to be used when the this object is part of a
        device object containing multiple elements.
        """
        qubit_config = {
            f"{self.name}": {
                "resources": {
                    "port_mw": f"{self.name}.mw",  # defines what port to apply mw on
                    "port_ro": f"{self.name}.res",  # defines what port to apply readout pulses on
                    "port_flux": f"{self.name}.fl",  # defines what port to apply flux pulses on
                    "clock_01": f"{self.name}.01",  # defines a clock that tracks the 0-1 transition of the qubit
                    "clock_ro": f"{self.name}.ro",  # defines a clock that tracks the readout resonator
                },
                "params": {
                    "acquisition": "SSBIntegrationComplex",
                    "mw_freq": 5.55e9,  # this is the qubit frequency in Hz.
                    "mw_amp180": 1,  # This is not used at the moment
                    "mw_motzoi": 1,  # This is not used at the moment
                    "mw_duration": 40e-9,  # DRAG pulse duration
                    "mw_ef_amp180": 1,  # This is not used at the moment
                    "ro_freq": 6.8979e9,
                    "ro_pulse_amp": 0.5,  # readout pulse amplitude (V)
                    "ro_pulse_type": "square",  # shape of waveform used for the readout pulse
                    "ro_pulse_duration": 200e-9,  # readout pulse duration (s)
                    "ro_acq_delay": 100e-9,  # time between sending readout pulse and acquisition (s)
                    "ro_acq_integration_time": 400e-9,  # Minimum wait time between iterations (s)
                    "ro_acq_weigth_type": "SSB",  # This is not used at the moment
                    "init_duration": self.init_duration(),  # time before sending first pulse (s)
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
