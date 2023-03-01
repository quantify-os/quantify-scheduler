# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
# pylint: disable=too-many-arguments
"""Standard acquisition protocols for use with the quantify_scheduler."""

from typing import Any, Dict, List, Optional, Union
import warnings

import numpy as np

from quantify_scheduler import Operation
from quantify_scheduler.enums import BinMode


class AcquisitionOperation(Operation):  # pylint: disable=too-many-ancestors
    """
    This class is used to help differentiate an acquisition operation from the regular
    operations. This enables us to use
    :func:`~.visualization.pulse_diagram.plot_acquisition_operations` to highlight
    acquisition pulses in the pulse diagrams.
    """


class Trace(AcquisitionOperation):  # pylint: disable=too-many-ancestors
    """The Trace acquisition protocol measures a signal s(t)."""

    def __init__(
        self,
        duration: float,
        port: str,
        clock: str,
        acq_channel: int = 0,
        acq_index: int = 0,
        bin_mode: Union[BinMode, str] = BinMode.AVERAGE,
        t0: float = 0,
        data: Optional[dict] = None,
    ) -> None:
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
        clock :
            The clock used to demodulate the acquisition.
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
            The operation's dictionary, by default None\n
            Note: if the data parameter is not None all other parameters are
            overwritten using the contents of data.\n
            Deprecated: support for the data argument will be dropped in
            quantify-scheduler >= 0.13.0. Please consider updating the data
            dictionary after initialization.
        """
        if data is None:
            if not isinstance(duration, float):
                duration = float(duration)
            if isinstance(bin_mode, str):
                bin_mode = BinMode(bin_mode)

            super().__init__(name="Trace")
            self.data.update(
                {
                    "name": "Trace",
                    "acquisition_info": [
                        {
                            "waveforms": [],
                            "duration": duration,
                            "t0": t0,
                            "port": port,
                            "clock": clock,
                            "acq_channel": acq_channel,
                            "acq_index": acq_index,
                            "bin_mode": bin_mode,
                            "protocol": "Trace",
                            "acq_return_type": np.ndarray,
                        }
                    ],
                }
            )
            self._update()
        else:
            warnings.warn(
                "Support for the data argument will be dropped in"
                "quantify-scheduler >= 0.13.0.\n"
                "Please consider updating the data "
                "dictionary after initialization.",
                FutureWarning,
            )
            super().__init__(name=data["name"], data=data)

    def __str__(self) -> str:
        acq_info = self.data["acquisition_info"][0]
        return self._get_signature(acq_info)


class WeightedIntegratedComplex(
    AcquisitionOperation
):  # pylint: disable=too-many-ancestors
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
    ) -> None:
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
            The operation's dictionary, by default None\n
            Note: if the data parameter is not None all other parameters are
            overwritten using the contents of data.\n
            Deprecated: support for the data argument will be dropped in
            quantify-scheduler >= 0.13.0. Please consider updating the data
            dictionary after initialization.

        Raises
        ------
        NotImplementedError
        """
        if phase != 0:
            # Because of how clock interfaces were changed.
            raise NotImplementedError("Non-zero phase not yet implemented")

        if data is None:
            super().__init__(name="WeightedIntegratedComplex")
            self.data.update(
                {
                    "name": "WeightedIntegratedComplex",
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
                            "protocol": "WeightedIntegratedComplex",
                            "acq_return_type": complex,
                        }
                    ],
                }
            )
            self._update()
            # certain fields are required in the acquisition data
            if "acq_return_type" not in self.data["acquisition_info"][0]:
                self.data["acquisition_info"][0]["acq_return_type"] = complex
                self.data["acquisition_info"][0][
                    "protocol"
                ] = "WeightedIntegratedComplex"
        else:
            warnings.warn(
                "Support for the data argument will be dropped in"
                "quantify-scheduler >= 0.13.0.\n"
                "Please consider updating the data "
                "dictionary after initialization.",
                FutureWarning,
            )
            if "acq_return_type" not in data["acquisition_info"][0]:
                data["acquisition_info"][0]["acq_return_type"] = complex
                data["acquisition_info"][0]["protocol"] = "WeightedIntegratedComplex"

            super().__init__(name=data["name"], data=data)

    def __str__(self) -> str:
        acq_info = self.data["acquisition_info"][0]
        return self._get_signature(acq_info)


class SSBIntegrationComplex(AcquisitionOperation):  # pylint: disable=too-many-ancestors
    """
    This class implements a SingleSideBand Integration acquisition protocol with
    complex results.
    """

    def __init__(
        self,
        port: str,
        clock: str,
        duration: float,
        acq_channel: int = 0,
        acq_index: int = 0,
        bin_mode: Union[BinMode, str] = BinMode.AVERAGE,
        phase: float = 0,
        t0: float = 0,
        data: Optional[dict] = None,
    ) -> None:
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
            new result and the old register value, by default BinMode.AVERAGE
        phase :
            The phase of the pulse and acquisition in degrees, by default 0
        t0 :
            The acquisition start time in seconds, by default 0
        data :
            The operation's dictionary, by default None\n
            Note: if the data parameter is not None all other parameters are
            overwritten using the contents of data.\n
            Deprecated: support for the data argument will be dropped in
            quantify-scheduler >= 0.13.0. Please consider updating the data
            dictionary after initialization.
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

        if phase != 0:
            # Because of how clock interfaces were changed.
            raise NotImplementedError("Non-zero phase not yet implemented")

        if data is None:
            super().__init__(name="SSBIntegrationComplex")
            self.data.update(
                {
                    "name": "SSBIntegrationComplex",
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
                            "acq_return_type": complex,
                            "protocol": "SSBIntegrationComplex",
                        }
                    ],
                }
            )
            self._update()
            # certain fields are required in the acquisition data
            if "acq_return_type" not in self.data["acquisition_info"][0]:
                self.data["acquisition_info"][0]["acq_return_type"] = complex
                self.data["acquisition_info"][0]["protocol"] = "SSBIntegrationComplex"
        else:
            warnings.warn(
                "Support for the data argument will be dropped in"
                "quantify-scheduler >= 0.13.0.\n"
                "Please consider updating the data "
                "dictionary after initialization.",
                FutureWarning,
            )
            if "acq_return_type" not in data["acquisition_info"][0]:
                data["acquisition_info"][0]["acq_return_type"] = complex
                data["acquisition_info"][0]["protocol"] = "SSBIntegrationComplex"
            super().__init__(name=data["name"], data=data)

    def __str__(self) -> str:
        acq_info = self.data["acquisition_info"][0]
        return self._get_signature(acq_info)


class NumericalWeightedIntegrationComplex(
    WeightedIntegratedComplex
):  # pylint: disable=too-many-ancestors
    """
    Implements a WeightedIntegratedComplex class using parameterized waveforms and
    interpolation as the integration weights.
    """

    def __init__(
        self,
        weights_a: Union[List[complex], np.ndarray],
        weights_b: Union[List[complex], np.ndarray],
        t: Union[List[float], np.ndarray],
        port: str,
        clock: str,
        interpolation: str = "linear",
        acq_channel: int = 0,
        acq_index: int = 0,
        bin_mode: Union[BinMode, str] = BinMode.APPEND,
        phase: float = 0,
        t0: float = 0,
        data: Optional[dict] = None,
    ) -> None:
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
            The operation's dictionary, by default None\n
            Note: if the data parameter is not None all other parameters are
            overwritten using the contents of data.\n
            Deprecated: support for the data argument will be dropped in
            quantify-scheduler >= 0.13.0. Please consider updating the data
            dictionary after initialization.
        """
        if not isinstance(weights_a, np.ndarray):
            weights_a = np.array(weights_a)
        if not isinstance(weights_b, np.ndarray):
            weights_b = np.array(weights_b)
        if not isinstance(t, np.ndarray):
            t = np.array(t)

        waveforms_a = {
            "wf_func": "quantify_scheduler.waveforms.interpolated_complex_waveform",
            "samples": weights_a,
            "t_samples": t,
            "interpolation": interpolation,
        }
        waveforms_b = {
            "wf_func": "quantify_scheduler.waveforms.interpolated_complex_waveform",
            "samples": weights_b,
            "t_samples": t,
            "interpolation": interpolation,
        }
        duration = t[-1] - t[0]
        if data is not None:
            warnings.warn(
                "Support for the data argument will be dropped in"
                "quantify-scheduler >= 0.13.0.\n"
                "Please consider updating the data "
                "dictionary after initialization.",
                FutureWarning,
            )
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
        self._update()
        self.data["name"] = "NumericalWeightedIntegrationComplex"

    def __str__(self) -> str:
        acq_info = self.data["acquisition_info"][0]
        weights_a = np.array2string(
            acq_info["waveforms"][0]["samples"], separator=", ", precision=9
        )
        weights_b = np.array2string(
            acq_info["waveforms"][1]["samples"], separator=", ", precision=9
        )
        t = np.array2string(
            acq_info["waveforms"][0]["t_samples"], separator=", ", precision=9
        )
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

    def __repr__(self) -> str:
        return str(self)


class TriggerCount(AcquisitionOperation):  # pylint: disable=too-many-ancestors
    """Trigger counting acquisition protocol returning an integer."""

    def __init__(
        self,
        port: str,
        clock: str,
        duration: float,
        acq_channel: int = 0,
        acq_index: int = 0,
        bin_mode: Union[BinMode, str] = BinMode.APPEND,
        t0: float = 0,
    ) -> None:
        """
        Creates a new instance of TriggerCount, trigger counting
        acquisition protocol, returning an integer.

        The trigger acquisition mode is used to measure how
        many times the trigger level is surpassed. The level is set
        in the hardware configuration.

        Parameters
        ----------
        port
            The acquisition port.
        clock
            The clock used to demodulate the acquisition.
        duration
            The acquisition duration in seconds.
        acq_channel
            The data channel in which the acquisition is stored, by default 0.
            Describes the "where" information of the measurement, which typically
            corresponds to a qubit idx.
        acq_index
            The data register in which the acquisition is stored, by default 0.
            Describes the "when" information of the measurement, used to label or
            tag individual measurements in a large circuit. Typically corresponds
            to the setpoints of a schedule (e.g., tau in a T1 experiment).
        bin_mode
            Describes what is done when data is written to a register that already
            contains a value. Options are "append" which appends the result to the
            list or "average" which stores the count value of the
            new result and the old register value, by default BinMode.APPEND
        t0
            The acquisition start time in seconds, by default 0
        """

        super().__init__(name="TriggerCount")
        self.data.update(
            {
                "name": "TriggerCount",
                "acquisition_info": [
                    {
                        "waveforms": [],
                        "t0": t0,
                        "clock": clock,
                        "port": port,
                        "duration": duration,
                        "acq_channel": acq_channel,
                        "acq_index": acq_index,
                        "bin_mode": bin_mode,
                        "acq_return_type": int,
                        "protocol": "TriggerCount",
                    }
                ],
            }
        )
        self._update()

    def __str__(self) -> str:
        acq_info = self.data["acquisition_info"][0]
        return self._get_signature(acq_info)
