# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Module that contains the RFSwitchToggle operation."""

from quantify_scheduler import Operation


class RFSwitchToggle(Operation):
    """
    Turn the RF complex output on for the given duration.
    The RF ports are on by default, make sure to set
    :attr:`~.quantify_scheduler.backends.types.qblox.RFDescription.rf_output_on`
    to `False` to turn them off.

    Parameters
    ----------
    duration
        Duration to turn the RF output on.
    port
        Name of the associated port.
    clock
        Name of the associated clock.
        For now the given port-clock combination must
        have a LO frequency defined in the hardware configuration.

    Examples
    --------
    Partial hardware configuration to turn the RF complex output off by default
    to be able to use this operation.

    .. code-block:: python

        hardware_compilation_config = {
            "config_type": QbloxHardwareCompilationConfig,
            "hardware_description": {
                "cluster0": {
                    "instrument_type": "Cluster",
                    "modules": {
                        "0": {"instrument_type": "QCM_RF", "rf_output_on": False},
                        "1": {"instrument_type": "QRM_RF", "rf_output_on": False},
                    },
                },
            },
        }

    """

    def __init__(
        self,
        duration: float,
        port: str,
        clock: str,
    ) -> None:
        super().__init__(name=self.__class__.__name__)
        self.data["pulse_info"] = [
            {
                "wf_func": None,
                "marker_pulse": True,  # This distinguishes MarkerPulse from other operations
                "t0": 0,
                "clock": clock,
                "port": port,
                "duration": duration,
            }
        ]
        self._update()

    def __str__(self) -> str:
        pulse_info = self.data["pulse_info"][0]
        return self._get_signature(pulse_info)
