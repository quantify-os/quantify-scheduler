# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
# pylint: disable=too-many-arguments
"""Standard acquisition protocols for use with the quantify_scheduler."""

from typing import Any, Dict, List, Optional, Sequence, Union
import warnings

import numpy as np

import quantify_scheduler.backends.qblox.constants as qblox_constants
from quantify_scheduler import Operation
from quantify_scheduler.enums import BinMode


class AcquisitionOperation(Operation):  # pylint: disable=too-many-ancestors
    """
    Acquisition operations for highlighting in pulse diagrams.

    This class is used to help differentiate an acquisition operation from the regular
    operations. This enables us to use
    :func:`~.quantify_scheduler.schedules._visualization.pulse_diagram.plot_acquisition_operations`
    to highlight acquisition pulses in the pulse diagrams.
    """


class Trace(AcquisitionOperation):  # pylint: disable=too-many-ancestors
    """
    The Trace acquisition protocol measures a signal s(t).

    Only processing performed is rescaling and adding
    units based on a calibrated scale. Values are returned
    as a raw trace (numpy array of float datatype). Length of
    this array depends on the sampling rate of the acquisition
    device.

    Parameters
    ----------
    port
        The acquisition port.
    clock
        The clock used to demodulate the acquisition.
    duration
        The acquisition duration in seconds.
    acq_channel
        The data channel in which the acquisition is stored, is by default 0.
        Describes the "where" information of the  measurement, which typically
        corresponds to a qubit idx.
    acq_index
        The data register in which the acquisition is stored, by default 0.
        Describes the "when" information of the measurement, used to label or
        tag individual measurements in a large circuit. Typically corresponds
        to the setpoints of a schedule (e.g., tau in a T1 experiment).
    bin_mode
        Describes what is done when data is written to a register that already
        contains a value. Options are "append" which appends the result to the
        list or "average" which stores the weighted average value of the
        new result and the old register value, by default BinMode.APPEND.
    t0
        The acquisition start time in seconds, by default 0.
    """

    def __init__(
        self,
        duration: float,
        port: str,
        clock: str,
        acq_channel: int = 0,
        acq_index: int = 0,
        bin_mode: Union[BinMode, str] = BinMode.AVERAGE,
        t0: float = 0,
    ) -> None:
        if not isinstance(duration, float):
            duration = float(duration)
        if isinstance(bin_mode, str):
            bin_mode = BinMode(bin_mode)

        super().__init__(name=self.__class__.__name__)
        self.data["acquisition_info"] = [
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
        ]
        self._update()

    def __str__(self) -> str:
        acq_info = self.data["acquisition_info"][0]
        return self._get_signature(acq_info)


class WeightedIntegratedComplex(
    AcquisitionOperation
):  # pylint: disable=too-many-ancestors
    r"""
    Weighted integration acquisition protocol on a complex signal.

    Weights are applied as:

    .. math::

        \widetilde{A} = \int ( \mathrm{Re}(S(t))\cdot \mathrm{Re}(W_A(t)) +
        \mathrm{Im}(S(t))\cdot \mathrm{Im}(W_A(t)) ) \mathrm{d}t

    .. math::

        \widetilde{B} = \int ( \mathrm{Re}(S(t))\cdot \mathrm{Re}(W_B(t)) +
        \mathrm{Im}(S(t))\cdot \mathrm{Im}(W_B(t)) ) \mathrm{d}t

    Parameters
    ----------
    waveform_a
        The complex waveform used as integration weights :math:`A(t)`.
    waveform_b
        The complex waveform used as integration weights :math:`B(t)`.
    port
        The acquisition port.
    clock
        The clock used to demodulate the acquisition.
    duration
        The acquisition duration in seconds.
    acq_channel
        The data channel in which the acquisition is stored, by default 0.
        Describes the "where" information of the  measurement, which typically
        corresponds to a qubit idx.
    acq_index
        The data register in which the acquisition is stored, by default 0.
        Describes the "when" information of the measurement, used to label or
        tag individual measurements in a large circuit. Typically corresponds
        to the setpoints of a schedule (e.g., tau in a T1 experiment).
    bin_mode
        Describes what is done when data is written to a register that already
        contains a value. Options are "append" which appends the result to the
        list or "average" which stores the weighted average value of the
        new result and the old register value, by default BinMode.APPEND.
    phase
        The phase of the pulse and acquisition in degrees, by default 0.
    t0
        The acquisition start time in seconds, by default 0.

    Raises
    ------
    NotImplementedError
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
    ) -> None:
        if phase != 0:
            raise NotImplementedError("Non-zero phase not yet implemented")

        super().__init__(name=self.__class__.__name__)
        self.data["acquisition_info"] = [
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
        ]
        self._update()
        # certain fields are required in the acquisition data
        if "acq_return_type" not in self.data["acquisition_info"][0]:
            self.data["acquisition_info"][0]["acq_return_type"] = complex
            self.data["acquisition_info"][0]["protocol"] = "WeightedIntegratedComplex"

    def __str__(self) -> str:
        acq_info = self.data["acquisition_info"][0]
        return self._get_signature(acq_info)


class SSBIntegrationComplex(AcquisitionOperation):  # pylint: disable=too-many-ancestors
    """
    Single sideband integration acquisition protocol with complex results.

    A weighted integrated acquisition on a complex signal using a
    square window for the acquisition weights.

    The signal is demodulated using the specified clock, and the
    square window then effectively specifies an integration window.

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
        Describes the "where" information of the  measurement, which typically
        corresponds to a qubit idx.
    acq_index
        The data register in which the acquisition is stored, by default 0.
        Describes the "when" information of the measurement, used to label or
        tag individual measurements in a large circuit. Typically corresponds
        to the setpoints of a schedule (e.g., tau in a T1 experiment).
    bin_mode
        Describes what is done when data is written to a register that already
        contains a value. Options are "append" which appends the result to the
        list or "average" which stores the weighted average value of the
        new result and the old register value, by default BinMode.AVERAGE.
    phase
        The phase of the pulse and acquisition in degrees, by default 0.
    t0
        The acquisition start time in seconds, by default 0.
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
    ) -> None:
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
            raise NotImplementedError("Non-zero phase not yet implemented")

        super().__init__(name=self.__class__.__name__)
        self.data["acquisition_info"] = [
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
        ]
        self._update()
        # certain fields are required in the acquisition data
        if "acq_return_type" not in self.data["acquisition_info"][0]:
            self.data["acquisition_info"][0]["acq_return_type"] = complex
            self.data["acquisition_info"][0]["protocol"] = "SSBIntegrationComplex"

    def __str__(self) -> str:
        acq_info = self.data["acquisition_info"][0]
        return self._get_signature(acq_info)


class ThresholdedAcquisition(AcquisitionOperation):
    """
    Acquisition protocol allowing to control rotation and threshold.

    This acquisition protocol is similar to the :class:`~.SSBIntegrationComplex`
    acquisition protocol, but the complex result is now rotated and thresholded
    to produce a "0" or a "1", as controlled by the parameters for rotation
    angle `<qubit>.measure.acq_rotation` and threshold value
    `<qubit>.measure.acq_threshold` in the device configuration (see example
    below).

    The rotation angle and threshold value for each qubit can be set through
    the device configuration.

    .. admonition:: Note

        Thresholded acquisition is currently only supported by the Qblox
        backend.

    .. admonition:: Examples

        .. jupyter-execute::

            from quantify_scheduler import Schedule
            from quantify_scheduler.device_under_test.transmon_element import BasicTransmonElement
            from quantify_scheduler.operations.acquisition_library import ThresholdedAcquisition

            # set up qubit
            qubit = BasicTransmonElement("q0")
            qubit.clock_freqs.readout(8.0e9)

            # set rotation and threshold value
            rotation, threshold = 20, -0.1
            qubit.measure.acq_rotation(rotation)
            qubit.measure.acq_threshold(threshold)

            # basic schedule
            schedule = Schedule("thresholded acquisition")
            schedule.add(ThresholdedAcquisition(port="q0:res", clock="q0.ro", duration=1e-6))


    Parameters
    ----------
    port : str
        The acquisition port.
    clock : str
        The clock used to demodulate the acquisition.
    duration : float
        The acquisition duration in seconds.
    acq_channel : int
        The data channel in which the acquisition is stored, by default 0.
        Describes the "where" information of the  measurement, which
        typically corresponds to a qubit idx.
    acq_index : int
        The data register in which the acquisition is stored, by default 0.
        Describes the "when" information of the measurement, used to label
        or tag individual measurements in a large circuit. Typically
        corresponds to the setpoints of a schedule (e.g., tau in a T1
        experiment).
    bin_mode : BinMode or str
        Describes what is done when data is written to a register that
        already contains a value. Options are "append" which appends the
        result to the list or "average" which stores the weighted average
        value of the new result and the old register value, by default
        BinMode.AVERAGE.
    phase : float
        The phase of the pulse and acquisition in degrees, by default 0.
    t0 : float
        The acquisition start time in seconds, by default 0.
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
    ) -> None:
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
            raise NotImplementedError("Non-zero phase not yet implemented")

        super().__init__(name=self.__class__.__name__)
        self.data["acquisition_info"] = [
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
                "acq_return_type": int,
                "protocol": "ThresholdedAcquisition",
                # The following are set during _compile_circuit_to_device
                "acq_threshold": None,
                "acq_rotation": None,
                "integration_length": None,
            },
        ]
        self._update()

    def __str__(self) -> str:
        acq_info = self.data["acquisition_info"][0]
        return self._get_signature(acq_info)


class NumericalWeightedIntegrationComplex(
    WeightedIntegratedComplex
):  # pylint: disable=too-many-ancestors
    r"""
    Subclass of WeightedIntegratedComplex with parameterized waveforms as weights.

    A WeightedIntegratedComplex class using parameterized waveforms and
    interpolation as the integration weights.

    Weights are applied as:

    .. math::

        \widetilde{A} = \int ( \mathrm{Re}(S(t))\cdot \mathrm{Re}(W_A(t)) +
        \mathrm{Im}(S(t))\cdot \mathrm{Im}(W_A(t)) ) \mathrm{d}t

    .. math::

        \widetilde{B} = \int ( \mathrm{Re}(S(t))\cdot \mathrm{Re}(W_B(t)) +
        \mathrm{Im}(S(t))\cdot \mathrm{Im}(W_B(t)) ) \mathrm{d}t

    Parameters
    ----------
    port
        The acquisition port.
    clock
        The clock used to demodulate the acquisition.
    weights_a
        The list of complex values used as weights :math:`A(t)` on
        the incoming complex signal.
    weights_b
        The list of complex values used as weights :math:`B(t)` on
        the incoming complex signal.
    weights_sampling_rate
        The rate with which the weights have been sampled, in Hz. By default equal
        to the Qblox backend sampling rate. Note that during hardware compilation,
        the weights will be resampled with the sampling rate supported by the target
        hardware.
    t
        The time values of each weight. This parameter is deprecated in favor of
        ``weights_sampling_rate``. If a value is provided for ``t``, the
        ``weights_sampling_rate`` parameter will be ignored.
    interpolation
        The type of interpolation to use, by default "linear". This argument is
        passed to :obj:`~scipy.interpolate.interp1d`.
    acq_channel
        The data channel in which the acquisition is stored, by default 0.
        Describes the "where" information of the  measurement, which typically
        corresponds to a qubit idx.
    acq_index
        The data register in which the acquisition is stored, by default 0.
        Describes the "when" information of the measurement, used to label or
        tag individual measurements in a large circuit. Typically corresponds
        to the setpoints of a schedule (e.g., tau in a T1 experiment).
    bin_mode
        Describes what is done when data is written to a register that already
        contains a value. Options are "append" which appends the result to the
        list or "average" which stores the weighted average value of the
        new result and the old register value, by default BinMode.APPEND.
    phase
        The phase of the pulse and acquisition in degrees, by default 0.
    t0
        The acquisition start time in seconds, by default 0.
    """

    def __init__(
        self,
        port: str,
        clock: str,
        weights_a: Union[List[complex], np.ndarray],
        weights_b: Union[List[complex], np.ndarray],
        weights_sampling_rate: float = qblox_constants.SAMPLING_RATE,
        t: Optional[Union[List[float], np.ndarray]] = None,
        interpolation: str = "linear",
        acq_channel: int = 0,
        acq_index: int = 0,
        bin_mode: Union[BinMode, str] = BinMode.APPEND,
        phase: float = 0,
        t0: float = 0,
    ) -> None:
        if t is not None:
            warnings.warn(
                "Support for the 't' argument will be dropped in quantify-scheduler >= "
                "0.17.0. Please use 'weights_sampling_rate' instead.",
                FutureWarning,
            )
            if not _is_increasing_at_constant_rate(t):
                raise ValueError(
                    "The NumericalWeightedIntegrationComplex protocol requires that "
                    "the 't' argument has a length larger than 1 and increases at a "
                    "constant rate"
                )
            t_samples = np.array(t)
            weights_sampling_rate = 1 / (t_samples[1] - t_samples[0])
        else:
            t_samples = np.arange(len(weights_a)) / weights_sampling_rate

        weights_a = np.array(weights_a)
        weights_b = np.array(weights_b)

        waveforms_a = {
            "wf_func": "quantify_scheduler.waveforms.interpolated_complex_waveform",
            "samples": weights_a,
            "t_samples": t_samples,
            "interpolation": interpolation,
        }
        waveforms_b = {
            "wf_func": "quantify_scheduler.waveforms.interpolated_complex_waveform",
            "samples": weights_b,
            "t_samples": t_samples,
            "interpolation": interpolation,
        }
        duration = len(t_samples) / weights_sampling_rate

        super().__init__(
            waveform_a=waveforms_a,
            waveform_b=waveforms_b,
            port=port,
            clock=clock,
            duration=duration,
            acq_channel=acq_channel,
            acq_index=acq_index,
            bin_mode=bin_mode,
            phase=phase,
            t0=t0,
        )
        self.data["name"] = self.__class__.__name__
        self._update()

    def __str__(self) -> str:
        acq_info = self.data["acquisition_info"][0]
        weights_a = np.array2string(
            acq_info["waveforms"][0]["samples"], separator=", ", precision=9
        )
        weights_b = np.array2string(
            acq_info["waveforms"][1]["samples"], separator=", ", precision=9
        )
        t_samples = acq_info["waveforms"][0]["t_samples"]
        weights_sampling_rate = 1 / (t_samples[1] - t_samples[0])
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
            f"{weights_sampling_rate=}, {port=}, {clock=}, {interpolation=}, "
            f"{acq_channel=}, {acq_index=}, {bin_mode=}, {phase=}, {t0=})"
        )

    def __repr__(self) -> str:
        return str(self)


def _is_increasing_at_constant_rate(array: Sequence[float]) -> bool:
    """
    Checks whether the array is increasing at a constant rate.

    An array with size 2 is assumed to be increasing at a constant rate.

    .. admonition:: Examples

        .. jupyter-execute::
            :hide-code:

            from quantify_scheduler.operations.acquisition_library import (
                _is_increasing_at_constant_rate
            )

        .. jupyter-execute::

            assert _is_increasing_at_constant_rate([1,2,3,4]) is True
            assert _is_increasing_at_constant_rate([1,2,4]) is False
            assert _is_increasing_at_constant_rate([4,3,2,1]) is False
            assert _is_increasing_at_constant_rate([1,1,1]) is False
            assert _is_increasing_at_constant_rate([2,1]) is False
            assert _is_increasing_at_constant_rate([1]) is False
    """
    if len(array) < 2:
        return False
    diff = np.diff(array)
    is_constant_rate = np.all(np.isclose(diff, diff[0], atol=1e-10))
    is_increasing = diff[0] > 0
    return bool(is_constant_rate and is_increasing)


class TriggerCount(AcquisitionOperation):  # pylint: disable=too-many-ancestors
    """
    Trigger counting acquisition protocol returning an integer.

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
        new result and the old register value, by default BinMode.APPEND.
    t0
        The acquisition start time in seconds, by default 0.
    """

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
        if bin_mode == BinMode.AVERAGE and acq_index != 0:
            # In average mode the count distribution is measured,
            # and currently we do not support multiple indices for this,
            # or starting the counting from a predefined count number.
            raise NotImplementedError(
                "Using nonzero acq_index is not yet implemented for AVERAGE bin mode for "
                "the trigger count protocol"
            )

        super().__init__(name=self.__class__.__name__)
        self.data["acquisition_info"] = [
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
        ]
        self._update()

    def __str__(self) -> str:
        acq_info = self.data["acquisition_info"][0]
        return self._get_signature(acq_info)
