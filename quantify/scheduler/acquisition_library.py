# -----------------------------------------------------------------------------
# Description:    Library standard acquisition protocols for use with the quantify.scheduler.
# Repository:     https://gitlab.com/quantify-os/quantify-scheduler
# Copyright (C)   Qblox BV & Orange Quantum Systems Holding BV (2020-2021)
# -----------------------------------------------------------------------------
from typing import Any, Dict, List

from quantify.scheduler.enums import BinMode
from quantify.scheduler.types import Operation


class Trace(Operation):
    def __init__(
        self,
        duration: float,
        port: str,
        acq_channel: int = 0,
        acq_index: int = 0,
        bin_mode: BinMode = BinMode.APPEND,
        t0: float = 0,
    ):
        """
        Measure a signal s(t).

        Only processing performed is rescaling and adding units based on a calibrated scale.
        Values are returned as a raw trace (numpy array of float datatype).

        Parameters
        ------------
        duration : float
            Duration of the acquisition in seconds.
        port : str
            Port of the acquisition.
        acq_index : int
            Data register in which the acquisition is stored.
        bin_mode : BinMode
            Describes what is done when data is written to a register that already contains a value. Options are
            "append" which appends the result to the list or "average" which stores the weighted average value of the
            new result and the old register value.

        """
        data = {
            "name": "Trace",
            "acquisition_info": [
                {
                    "waveforms": [],
                    "duration": duration,
                    "t0": t0,
                    "port": port,
                    "acq_channel": acq_channel,
                    "acq_index": acq_index,
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
        acq_channel: int = 0,
        acq_index: int = 0,
        bin_mode: BinMode = BinMode.APPEND,
        phase: float = 0,
        t0: float = 0,
    ):
        r"""
        A weighted integrated acquisition on a complex signal using custom complex windows.

        .. math::

            \widetilde{I} = \int ( \mathrm{Re}(S(t))\cdot \mathrm{Re}(W_I(t)) +
            \mathrm{Im}(S(t))\cdot \mathrm{Im}(W_I(t)) ) \mathrm{d}t

        .. math::

            \widetilde{Q} = \int ( \mathrm{Re}(S(t))\cdot \mathrm{Re}(W_Q(t)) +
            \mathrm{Im}(S(t))\cdot \mathrm{Im}(W_Q(t)) ) \mathrm{d}t

        Parameters
        ------------
        waveform_i : Dict[str, Any]
            Dictionary with waveform function and parameters to be used as weights on the incoming complex signal.
        waveform_q : Dict[str, Any]
            Dictionary with waveform function and parameters to be used as weights on the incoming complex signal.
        port : str
            Port of the acquisition.
        acq_index : int
            Data register in which the acquisition is stored.
        phase : float
            Phase of the pulse and acquisition in degrees.
        clock : str
            Clock used to demodulate acquisition.
        bin_mode : BinMode
            Describes what is done when data is written to a register that already contains a value. Options are
            "append" which appends the result to the list or "average" which stores the weighted average value of the
            new result and the old register value.

        """
        if phase != 0:
            # Because of how clock interfaces were changed.
            # FIXME: need to be able to add phases to the waveform separate from the clock.
            raise NotImplementedError("Non-zero phase not yet implemented")

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
                    "acq_channel": acq_channel,
                    "acq_index": acq_index,
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
        acq_channel: int = 0,
        acq_index: int = 0,
        bin_mode: BinMode = BinMode.APPEND,
        phase: float = 0,
        t0: float = 0,
    ):
        """
        A weighted integrated acquisition on a complex signal using a square window for the acquisition weights.

        The signal is demodulated using the specified clock, and the square window then effectively specifies an
        integration window.

        Parameters
        ------------
        duration : float
            Duration of the acquisition in seconds.
        port : str
            Port of the acquisition.
        acq_index : int
            Data register in which the acquisition is stored.
        phase : float
            Phase of the pulse and acquisition in degrees.
        clock : str
            Clock used to demodulate acquisition.
        bin_mode : BinMode
            Describes what is done when data is written to a register that already contains a value. Options are
            "append" which appends the result to the list or "average" which stores the weighted average value of the
            new result and the old register value.

        """
        waveforms_i = {
            "port": port,
            "clock": clock,
            "t0": t0,
            "duration": duration,
            "wf_func": "quantify.scheduler.waveforms.square",
            "amp": 1,
        }

        waveforms_q = {
            "port": port,
            "clock": clock,
            "t0": t0,
            "duration": duration,
            "wf_func": "quantify.scheduler.waveforms.square",
            "amp": (0 - 1j),
        }

        super().__init__(
            waveforms_i,
            waveforms_q,
            port=port,
            clock=clock,
            acq_channel=acq_channel,
            acq_index=acq_index,
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
        acq_channel: int = 0,
        acq_index: int = 0,
        bin_mode: BinMode = BinMode.APPEND,
        phase: float = 0,
        t0: float = 0,
    ):
        """
        Measure using custom acquisition weights.

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
        acq_index : int
            Data register in which the acquisition is stored.
        phase : float
            Phase of the pulse and acquisition in degrees.
        clock : str
            Clock used to demodulate acquisition.
        bin_mode : BinMode
            Describes what is done when data is written to a register that already contains a value. Options are
            "append" which appends the result to the list or "average" which stores the weighted average value of the
            new result and the old register value.

        """
        waveforms_i = {
            "wf_func": "scipy.interpolate.interp1d",
            "weights": weights_i,
            "t": t,
            "interpolation": interpolation,
        }
        waveforms_q = {
            "wf_func": "scipy.interpolate.interp1d",
            "weights": weights_q,
            "t": t,
            "interpolation": interpolation,
        }
        super().__init__(
            waveforms_i,
            waveforms_q,
            port=port,
            clock=clock,
            acq_channel=acq_channel,
            acq_index=acq_index,
            bin_mode=bin_mode,
            phase=phase,
            t0=t0,
        )
        self.data["name"] = "NumericalWeightedIntegrationComplex"
