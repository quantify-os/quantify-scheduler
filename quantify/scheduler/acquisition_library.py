# -----------------------------------------------------------------------------
# Description:    Library standard acquisition protocols for use with the quantify.scheduler.
# Repository:     https://gitlab.com/quantify-os/quantify-scheduler
# Copyright (C)   Qblox BV & Orange Quantum Systems Holding BV (2020-2021)
# -----------------------------------------------------------------------------
from typing import List
from quantify.scheduler.types import Operation

VALID_BIN_MODES = ("append", "average")


def _check_bin_mode_valid(bin_mode: str):
    """
    Raises exception if bin mode is not in VALID_BIN_MODES.

    Parameters
    ------------
    bin_mode
        The bin mode to check

    Raises
    ------------
    NotImplementedError
        bin_mode not in VALID_BIN_MODES
    """
    if bin_mode not in VALID_BIN_MODES:
        raise NotImplementedError(
            f"Bin mode {bin_mode} not implemented. Valid settings are {VALID_BIN_MODES}."
        )


class Trace(Operation):
    def __init__(
        self,
        duration: float,
        port: str,
        data_reg: int = 0,
        bin_mode: str = "append",
        t0: float = 0,
    ):
        """
        Measures a signal s(t). Only processing performed is rescaling and adding units based on a calibrated scale.
        Values are returned as a raw trace (numpy array of float datatype).

        Parameters
        ------------
        duration : float
            Duration of the acquisition in seconds.
        port : str
            Port of the acquisition.
        data_reg : int
            Data register in which the acquisition is stored.
        bin_mode : str
            Describes what is done when data is written to a register that already contains a value. Options are
            "append" which appends the result to the list ar "average" which stores the weighted average value of the
            new result and the old register value.

        """
        _check_bin_mode_valid(bin_mode)

        data = {
            "name": "Trace",
            "acquisition_info": [
                {
                    "duration": duration,
                    "t0": t0,
                    "port": port,
                    "data_reg": data_reg,
                    "bin_mode": bin_mode,
                    "protocol": "trace",
                }
            ],
        }
        super().__init__(name=data["name"], data=data)


class SSBIntegrationComplex(Operation):
    def __init__(
        self,
        duration: float,
        port: str,
        clock: str,
        data_reg: int = 0,
        bin_mode: str = "append",
        phase: float = 0,
        t0: float = 0,
    ):
        """
        A weighted integrated acquisition on a complex signal using a square window.

        Parameters
        ------------
        duration : float
            Duration of the acquisition in seconds.
        port : str
            Port of the acquisition.
        data_reg : int
            Data register in which the acquisition is stored.
        phase : float
            Phase of the pulse and acquisition in degrees.
        clock : str
            Clock used to demodulate acquisition.
        bin_mode : str
            Describes what is done when data is written to a register that already contains a value. Options are
            "append" which appends the result to the list ar "average" which stores the weighted average value of the
            new result and the old register value.

        """
        if phase != 0:
            # Because of how clock interfaces were changed.
            # FIXME: need to be able to add phases to the waveform separate from the clock.
            raise NotImplementedError("Non-zero phase not yet implemented")

        _check_bin_mode_valid(bin_mode)

        waveform_0 = {
            "func": "quantify.scheduler.waveforms.square",
            "amp": 1,
            "duration": duration,
        }
        waveform_1 = {
            "func": "quantify.scheduler.waveforms.square_complex",
            "amp": -1,
            "duration": duration,
        }

        data = {
            "name": "SSBIntegrationComplex",
            "acquisition_info": [
                {
                    "waveform_0": waveform_0,
                    "waveform_1": waveform_1,
                    "t0": t0,
                    "clock": clock,
                    "port": port,
                    "phase": phase,
                    "data_reg": data_reg,
                    "bin_mode": bin_mode,
                    "protocol": "weighted_integrated_complex",
                }
            ],
        }
        super().__init__(name=data["name"], data=data)


class WeightedIntegrationComplex(Operation):
    def __init__(
        self,
        weights_I: List[complex],
        weights_Q: List[complex],
        t: List[float],
        port: str,
        clock: str,
        interpolation: str = "linear",
        data_reg: int = 0,
        bin_mode: str = "append",
        phase: float = 0,
        t0: float = 0,
    ):
        """
        A weighted integrated acquisition on a complex signal using custom complex windows.

        Parameters
        ------------
        weights_I : List[complex]
            List of complex values used as weights on the incoming complex signal.
        weights_Q : List[complex]
            List of complex values used as weights on the incoming complex signal.
        t : List[float]
            Time value of each weight.
        port : str
            Port of the acquisition.
        data_reg : int
            Data register in which the acquisition is stored.
        phase : float
            Phase of the pulse and acquisition in degrees.
        clock : str
            Clock used to demodulate acquisition.
        bin_mode : str
            Describes what is done when data is written to a register that already contains a value. Options are
            "append" which appends the result to the list or "average" which stores the weighted average value of the
            new result and the old register value.

        """
        if phase != 0:
            # Because of how clock interfaces were changed.
            # FIXME: need to be able to add phases to the waveform separate from the clock.
            raise NotImplementedError("Non-zero phase not yet implemented")

        _check_bin_mode_valid(bin_mode)

        data = {
            "name": "NumericalWeightedIntegrationComplex",
            "acquisition_info": [
                {
                    "weights_I": weights_I,  # TODO add waveform function
                    "weights_Q": weights_Q,
                    "t": t,
                    "t0": t0,
                    "clock": clock,
                    "port": port,
                    "phase": phase,
                    "interpolation": interpolation,
                    "data_reg": data_reg,
                    "bin_mode": bin_mode,
                    "protocol": "weighted_integrated_complex",
                }
            ],
        }
        super().__init__(name=data["name"], data=data)
