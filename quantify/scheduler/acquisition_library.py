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
        Creates a new instance of Trace.
        The Trace acquisition protocol measures a signal s(t).

        Only processing performed is rescaling and adding
        units based on a calibrated scale. Values are returned
        as a raw trace (numpy array of float datatype).

        Parameters
        ----------
        port : str
            The acquisition port.
        duration : float
            The acquisition duration in seconds.
        acq_channel : int, optional
            [description], by default 0
        acq_index : int, optional
            The data register in which the acquisition is stored, by default 0
        bin_mode : BinMode, optional
            Describes what is done when data is written to a register that already contains a value. Options are
            "append" which appends the result to the list or "average" which stores the weighted average value of the
            new result and the old register value, by default BinMode.APPEND
        t0 : float, optional
            The acquisition start time in seconds, by default 0
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
        duration: float,
        acq_channel: int = 0,
        acq_index: int = 0,
        bin_mode: BinMode = BinMode.APPEND,
        phase: float = 0,
        t0: float = 0,
    ):
        """
        Creates a new instance of WeightedIntegratedComplex.
        Weighted integration acquisition protocol on a
        complex signal in a custom complex window.

        A weighted integrated acquisition on a complex
        signal using custom complex windows.

        Weights are applied as:

        .. math::

            \widetilde{I} = \int ( \mathrm{Re}(S(t))\cdot \mathrm{Re}(W_I(t)) +
            \mathrm{Im}(S(t))\cdot \mathrm{Im}(W_I(t)) ) \mathrm{d}t

        .. math::

            \widetilde{Q} = \int ( \mathrm{Re}(S(t))\cdot \mathrm{Re}(W_Q(t)) +
            \mathrm{Im}(S(t))\cdot \mathrm{Im}(W_Q(t)) ) \mathrm{d}t

        Parameters
        ----------
        waveform_i : Dict[str, Any]
            The complex imaginary waveform used as integration weights.
        weights_q : Dict[str, Any]
            The complex real waveform used as integration weights.
        port : str
            The acquisition port.
        clock : str
            The clock used to demodulate the acquisition.
        duration : float
            The acquisition duration in seconds.
        acq_channel : int, optional
            [description], by default 0
        acq_index : int, optional
            The data register in which the acquisition is stored, by default 0
        bin_mode : BinMode, optional
            Describes what is done when data is written to a register that already contains a value. Options are
            "append" which appends the result to the list or "average" which stores the weighted average value of the
            new result and the old register value, by default BinMode.APPEND
        phase : float, optional
            The phase of the pulse and acquisition in degrees, by default 0
        t0 : float, optional
            The acquisition start time in seconds, by default 0

        Raises
        ------
        NotImplementedError
        """
        if phase != 0:
            # Because of how clock interfaces were changed.
            # FIXME: need to be able to add phases to the waveform separate from the clock.
            raise NotImplementedError("Non-zero phase not yet implemented")

        data = {
            "name": "WeightedIntegrationComplex",
            "acquisition_info": [
                {
                    "waveforms": [waveform_i, waveform_q],
                    "t0": t0,
                    "clock": clock,
                    "port": port,
                    "duration": duration,
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
        port: str,
        clock: str,
        duration: float,
        acq_channel: int = 0,
        acq_index: int = 0,
        bin_mode: BinMode = BinMode.APPEND,
        phase: float = 0,
        t0: float = 0,
    ):
        """
        Creates a new instance of SSBIntegrationComplex.
        Single Sideband Integration acquisition protocol
        with complex results.

        A weighted integrated acquisition on a complex
        signal using a square window for the acquisition
        weights.

        The signal is demodulated using the specified clock, and the square window then effectively specifies an
        integration window.

        Parameters
        ----------
        port : str
            The acquisition port.
        clock : str
            The clock used to demodulate the acquisition.
        duration : float
            The acquisition duration in seconds.
        acq_channel : int, optional
            [description], by default 0
        acq_index : int, optional
            The data register in which the acquisition is stored, by default 0
        bin_mode : BinMode, optional
            Describes what is done when data is written to a register that already contains a value. Options are
            "append" which appends the result to the list or "average" which stores the weighted average value of the
            new result and the old register value, by default BinMode.APPEND
        phase : float, optional
            The phase of the pulse and acquisition in degrees, by default 0
        t0 : float, optional
            The acquisition start time in seconds, by default 0
        """
        waveform_i = {
            "port": port,
            "clock": clock,
            "t0": t0,
            "duration": duration,
            "wf_func": "quantify.scheduler.waveforms.square",
            "amp": 1,
        }

        waveform_q = {
            "port": port,
            "clock": clock,
            "t0": t0,
            "duration": duration,
            "wf_func": "quantify.scheduler.waveforms.square",
            "amp": (0 + 1j),
        }

        super().__init__(
            waveform_i,
            waveform_q,
            port,
            clock,
            duration,
            acq_channel,
            acq_index,
            bin_mode,
            phase,
            t0,
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
        Creates a new instance of NumericalWeightedIntegrationComplex.
        NumericalWeightedIntegrationComplex inherits from
        :class:`WeightedIntegratedComplex` that uses parameterized
        waveforms and interpolation as integration weights.

        Parameters
        ----------
        weights_i : List[complex]
            The list of complex imaginary values used as weights on
            the incoming complex signal.
        weights_q : List[complex]
            The list of complex real values used as weights on
            the incoming complex signal.
        t : List[float]
            The time values of each weight.
        port : str
            The acquisition port.
        clock : str
            The clock used to demodulate the acquisition.
        interpolation : str, optional
            [description], by default "linear"
        acq_channel : int, optional
            [description], by default 0
        acq_index : int, optional
            The data register in which the acquisition is stored, by default 0
        bin_mode : BinMode, optional
            Describes what is done when data is written to a register that already contains a value. Options are
            "append" which appends the result to the list or "average" which stores the weighted average value of the
            new result and the old register value, by default BinMode.APPEND
        phase : float, optional
            The phase of the pulse and acquisition in degrees, by default 0
        t0 : float, optional
            The acquisition start time in seconds, by default 0
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
        duration = t[-1] - t[0]
        super().__init__(
            waveforms_i,
            waveforms_q,
            port,
            clock,
            duration,
            acq_channel,
            acq_index,
            bin_mode,
            phase,
            t0,
        )
        self.data["name"] = "NumericalWeightedIntegrationComplex"
