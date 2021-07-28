# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
"""Standard acquisition protocols for use with the quantify_scheduler."""
from typing import Any, Dict, List, Optional, Union

import numpy as np
from quantify_scheduler.enums import BinMode
from quantify_scheduler.types import Operation


class Trace(Operation):
    """The Trace acquisition protocol measures a signal s(t)."""

    def __init__(
        self,
        duration: float,
        port: str,
        acq_channel: int = 0,
        acq_index: int = 0,
        bin_mode: Union[BinMode, str] = BinMode.APPEND,
        t0: float = 0,
        data: Optional[dict] = None,
    ):
        """
        Creates a new instance of Trace.
        The Trace acquisition protocol measures a signal s(t).

        Only processing performed is rescaling and adding
        units based on a calibrated scale. Values are returned
        as a raw trace (numpy array of float datatype). Length of
        this array depends on the sampling rate of the acquisition
        device.

        Parameters
        ----------
        port :
            The acquisition port.
        duration :
            The acquisition duration in seconds.
        acq_channel :
            The data channel in which the acquisition is stored, is by default 0.
            Describes the "where" information of the  measurement, which typically
            corresponds to a qubit idx.
        acq_index :
            The data register in which the acquisition is stored, by default 0.
            Describes the "when" information of the measurement, used to label or
            tag individual measurements in a large circuit. Typically corresponds
            to the setpoints of a schedule (e.g., tau in a T1 experiment).
        bin_mode :
            Describes what is done when data is written to a register that already
            contains a value. Options are "append" which appends the result to the
            list or "average" which stores the weighted average value of the
            new result and the old register value, by default BinMode.APPEND
        t0 :
            The acquisition start time in seconds, by default 0
        data :
            The operation's dictionary, by default None
            Note: if the data parameter is not None all other parameters are
            overwritten using the contents of data.
        """

        if data is None:
            if not isinstance(duration, float):
                duration = float(duration)
            if isinstance(bin_mode, str):
                bin_mode = BinMode(bin_mode)

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

    def __str__(self) -> str:
        acq_info = self.data["acquisition_info"][0]
        return self._get_signature(acq_info)


class WeightedIntegratedComplex(Operation):
    """
    Weighted integration acquisition protocol on a
    complex signal in a custom complex window.
    """

    def __init__(
        self,
        waveform_a: Dict[str, Any],
        waveform_b: Dict[str, Any],
        port: str,
        clock: str,
        duration: float,
        acq_channel: int = 0,
        acq_index: int = 0,
        bin_mode: Union[BinMode, str] = BinMode.APPEND,
        phase: float = 0,
        t0: float = 0,
        data: Optional[dict] = None,
    ):
        r"""
        Creates a new instance of WeightedIntegratedComplex.
        Weighted integration acquisition protocol on a
        complex signal in a custom complex window.

        A weighted integrated acquisition on a complex
        signal using custom complex windows.

        Weights are applied as:

        .. math::

            \widetilde{A} = \int ( \mathrm{Re}(S(t))\cdot \mathrm{Re}(W_A(t)) +
            \mathrm{Im}(S(t))\cdot \mathrm{Im}(W_A(t)) ) \mathrm{d}t

        .. math::

            \widetilde{B} = \int ( \mathrm{Re}(S(t))\cdot \mathrm{Re}(W_B(t)) +
            \mathrm{Im}(S(t))\cdot \mathrm{Im}(W_B(t)) ) \mathrm{d}t

        Parameters
        ----------
        waveform_a :
            The complex waveform used as integration weights :math:`A(t)`.
        waveform_b :
            The complex waveform used as integration weights :math:`B(t)`.
        port :
            The acquisition port.
        clock :
            The clock used to demodulate the acquisition.
        duration :
            The acquisition duration in seconds.
        acq_channel :
            The data channel in which the acquisition is stored, by default 0.
            Describes the "where" information of the  measurement, which typically
            corresponds to a qubit idx.
        acq_index :
            The data register in which the acquisition is stored, by default 0.
            Describes the "when" information of the measurement, used to label or
            tag individual measurements in a large circuit. Typically corresponds
            to the setpoints of a schedule (e.g., tau in a T1 experiment).
        bin_mode :
            Describes what is done when data is written to a register that already
            contains a value. Options are "append" which appends the result to the
            list or "average" which stores the weighted average value of the
            new result and the old register value, by default BinMode.APPEND
        phase :
            The phase of the pulse and acquisition in degrees, by default 0
        t0 :
            The acquisition start time in seconds, by default 0
        data :
            The operation's dictionary, by default None
            Note: if the data parameter is not None all other parameters are
            overwritten using the contents of data.

        Raises
        ------
        NotImplementedError
        """
        if phase != 0:
            # Because of how clock interfaces were changed.
            raise NotImplementedError("Non-zero phase not yet implemented")

        if data is None:
            data = {
                "name": "WeightedIntegrationComplex",
                "acquisition_info": [
                    {
                        "waveforms": [waveform_a, waveform_b],
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

    def __str__(self) -> str:
        acq_info = self.data["acquisition_info"][0]
        return self._get_signature(acq_info)


class SSBIntegrationComplex(WeightedIntegratedComplex):
    def __init__(
        self,
        port: str,
        clock: str,
        duration: float,
        acq_channel: int = 0,
        acq_index: int = 0,
        bin_mode: Union[BinMode, str] = BinMode.APPEND,
        phase: float = 0,
        t0: float = 0,
        data: Optional[dict] = None,
    ):
        """
        Creates a new instance of SSBIntegrationComplex. Single Sideband
        Integration acquisition protocol with complex results.

        A weighted integrated acquisition on a complex signal using a
        square window for the acquisition weights.

        The signal is demodulated using the specified clock, and the
        square window then effectively specifies an integration window.

        Parameters
        ----------
        port :
            The acquisition port.
        clock :
            The clock used to demodulate the acquisition.
        duration :
            The acquisition duration in seconds.
        acq_channel :
            The data channel in which the acquisition is stored, by default 0.
            Describes the "where" information of the  measurement, which typically
            corresponds to a qubit idx.
        acq_index :
            The data register in which the acquisition is stored, by default 0.
            Describes the "when" information of the measurement, used to label or
            tag individual measurements in a large circuit. Typically corresponds
            to the setpoints of a schedule (e.g., tau in a T1 experiment).
        bin_mode :
            Describes what is done when data is written to a register that already
            contains a value. Options are "append" which appends the result to the
            list or "average" which stores the weighted average value of the
            new result and the old register value, by default BinMode.APPEND
        phase :
            The phase of the pulse and acquisition in degrees, by default 0
        t0 :
            The acquisition start time in seconds, by default 0
        data :
            The operation's dictionary, by default None
            Note: if the data parameter is not None all other parameters are
            overwritten using the contents of data.
        """
        waveform_i = {
            "port": port,
            "clock": clock,
            "t0": t0,
            "duration": duration,
            "wf_func": "quantify_scheduler.waveforms.square",
            "amp": 1,
        }

        waveform_q = {
            "port": port,
            "clock": clock,
            "t0": t0,
            "duration": duration,
            "wf_func": "quantify_scheduler.waveforms.square",
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
            data,
        )
        self.data["name"] = "SSBIntegrationComplex"


class NumericalWeightedIntegrationComplex(WeightedIntegratedComplex):
    def __init__(
        self,
        weights_a: List[complex],
        weights_b: List[complex],
        t: List[float],
        port: str,
        clock: str,
        interpolation: str = "linear",
        acq_channel: int = 0,
        acq_index: int = 0,
        bin_mode: Union[BinMode, str] = BinMode.APPEND,
        phase: float = 0,
        t0: float = 0,
        data: Optional[dict] = None,
    ):
        r"""
        Creates a new instance of NumericalWeightedIntegrationComplex.
        NumericalWeightedIntegrationComplex inherits from
        :class:`WeightedIntegratedComplex` that uses parameterized
        waveforms and interpolation as integration weights.

        Weights are applied as:

        .. math::

            \widetilde{A} = \int ( \mathrm{Re}(S(t))\cdot \mathrm{Re}(W_A(t)) +
            \mathrm{Im}(S(t))\cdot \mathrm{Im}(W_A(t)) ) \mathrm{d}t

        .. math::

            \widetilde{B} = \int ( \mathrm{Re}(S(t))\cdot \mathrm{Re}(W_B(t)) +
            \mathrm{Im}(S(t))\cdot \mathrm{Im}(W_B(t)) ) \mathrm{d}t

        Parameters
        ----------
        weights_a :
            The list of complex values used as weights :math:`A(t)` on
            the incoming complex signal.
        weights_b :
            The list of complex values used as weights :math:`B(t)` on
            the incoming complex signal.
        t :
            The time values of each weight.
        port :
            The acquisition port.
        clock :
            The clock used to demodulate the acquisition.
        interpolation :
            The type of interpolation to use, by default "linear". This argument is
            passed to :obj:`~scipy.interpolate.interp1d`.
        acq_channel :
            The data channel in which the acquisition is stored, by default 0.
            Describes the "where" information of the  measurement, which typically
            corresponds to a qubit idx.
        acq_index :
            The data register in which the acquisition is stored, by default 0.
            Describes the "when" information of the measurement, used to label or
            tag individual measurements in a large circuit. Typically corresponds
            to the setpoints of a schedule (e.g., tau in a T1 experiment).
        bin_mode :
            Describes what is done when data is written to a register that already
            contains a value. Options are "append" which appends the result to the
            list or "average" which stores the weighted average value of the
            new result and the old register value, by default BinMode.APPEND
        phase :
            The phase of the pulse and acquisition in degrees, by default 0
        t0 :
            The acquisition start time in seconds, by default 0
        data :
            The operation's dictionary, by default None
            Note: if the data parameter is not None all other parameters are
            overwritten using the contents of data.
        """
        waveforms_a = {
            "wf_func": "scipy.interpolate.interp1d",
            "weights": weights_a,
            "t": t,
            "interpolation": interpolation,
        }
        waveforms_b = {
            "wf_func": "scipy.interpolate.interp1d",
            "weights": weights_b,
            "t": t,
            "interpolation": interpolation,
        }
        duration = t[-1] - t[0]
        super().__init__(
            waveforms_a,
            waveforms_b,
            port,
            clock,
            duration,
            acq_channel,
            acq_index,
            bin_mode,
            phase,
            t0,
            data,
        )
        self.data["name"] = "NumericalWeightedIntegrationComplex"

    def __str__(self) -> str:
        acq_info = self.data["acquisition_info"][0]
        weights_a = np.array2string(
            acq_info["waveforms"][0]["weights"], separator=", ", precision=9
        )
        weights_b = np.array2string(
            acq_info["waveforms"][1]["weights"], separator=", ", precision=9
        )
        t = np.array2string(acq_info["waveforms"][0]["t"], separator=", ", precision=9)
        port = acq_info["port"]
        clock = acq_info["clock"]
        interpolation = acq_info["waveforms"][0]["interpolation"]
        acq_channel = acq_info["acq_channel"]
        acq_index = acq_info["acq_index"]
        bin_mode = acq_info["bin_mode"].value
        phase = acq_info["phase"]
        t0 = acq_info["t0"]

        return (
            f"{self.__class__.__name__}(weights_a={weights_a}, weights_b={weights_b}, "
            f"t={t}, port='{port}', clock='{clock}', interpolation='{interpolation}', "
            f"acq_channel={acq_channel}, acq_index={acq_index}, bin_mode='{bin_mode}', "
            f"phase={phase}, t0={t0})"
        )
