# -----------------------------------------------------------------------------
# Description:    Library standard acquisition protocols for use with the quantify.scheduler.
# Repository:     https://gitlab.com/quantify-os/quantify-scheduler
# Copyright (C)   Qblox BV & Orange Quantum Systems Holding BV (2020-2021)
# -----------------------------------------------------------------------------
from typing import List, Dict, Any
from quantify.scheduler.types import Operation

VALID_BIN_MODES = ("append", "average")


def _check_bin_mode_valid(bin_mode: str):
    """Raises exception if bin mode is not in VALID_BIN_MODES."""
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
            "append" which appends the result to the list or "average" which stores the weighted average value of the
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


class WeightedIntegratedComplex(Operation):
    def __init__(
        self,
        waveform_i: Dict[str, Any],
        waveform_q: Dict[str, Any],
        port: str,
        clock: str,
        data_reg: int = 0,
        bin_mode: str = "append",
        phase: float = 0,
        t0: float = 0,
    ):
        """
        A weighted integrated acquisition on a complex signal using custom complex windows.

        :math:`\widetilde{I} = \int ( \mathfrak{R}(S(t))\cdot \mathfrak{R}(W_I(t))
        + \mathfrak{I}(S(t))\cdot \mathfrak{I}(W_I(t)) ) \mathrm{d}t`

        :math:`\widetilde{Q} = \int ( \mathfrak{R}(S(t))\cdot \mathfrak{R}(W_Q(t))
        + \mathfrak{I}(S(t))\cdot \mathfrak{I}(W_Q(t)) ) \mathrm{d}t`

        Parameters
        ------------
        waveform_i : Dict[str, Any]
            Dictionary with waveform function and parameters to be used as weights on the incoming complex signal.
        waveform_q : Dict[str, Any]
            Dictionary with waveform function and parameters to be used as weights on the incoming complex signal.
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

        waveforms = [waveform_i, waveform_q]
        data = {
            "name": "WeightedIntegrationComplex",
            "acquisition_info": [
                {
                    "waveforms": waveforms,
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


class SSBIntegrationComplex(WeightedIntegratedComplex):
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
        A weighted integrated acquisition on a complex signal using a square window for the acquisition weights.

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
            "append" which appends the result to the list or "average" which stores the weighted average value of the
            new result and the old register value.

        """
        waveforms_i = {
            "func": "quantify.scheduler.waveforms.square",
            "amp": 1,
            "duration": duration,
        }

        waveforms_q = {
            "func": "quantify.scheduler.waveforms.square",
            "amp": (0 - 1j),
            "duration": duration,
        }

        super().__init__(
            waveforms_i,
            waveforms_q,
            port=port,
            clock=clock,
            data_reg=data_reg,
            bin_mode=bin_mode,
            phase=phase,
            t0=t0,
        )
        self.data["name"] = "SSBIntegrationComplex"


class NumericalWeightedIntegrationComplex(WeightedIntegratedComplex):
    def __init__(
        self,
        weights_i: List[complex],
        weights_q: List[complex],
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
        Implementation of :class:`WeightedIntegratedComplex` that uses a parameterized waveform and interpolation as
        weights.

        Parameters
        ------------
        weights_i : List[complex]
            List of complex values used as weights on the incoming complex signal.
        weights_q : List[complex]
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
        waveforms_i = {
            "func": "scipy.interpolate.interp1d",
            "weights": weights_i,
            "t": t,
            "interpolation": interpolation,
        }
        waveforms_q = {
            "func": "scipy.interpolate.interp1d",
            "weights": weights_q,
            "t": t,
            "interpolation": interpolation,
        }
        super().__init__(
            waveforms_i,
            waveforms_q,
            port=port,
            clock=clock,
            data_reg=data_reg,
            bin_mode=bin_mode,
            phase=phase,
            t0=t0,
        )
        self.data["name"] = "NumericalWeightedIntegrationComplex"
