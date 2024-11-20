# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch

"""Standard acquisition protocols for use with the quantify_scheduler."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from quantify_core.utilities import deprecated
from quantify_scheduler.enums import BinMode, TimeRef, TimeSource
from quantify_scheduler.operations.operation import Operation
from quantify_scheduler.resources import DigitalClockResource


class Acquisition(Operation):
    """
    An operation representing data acquisition at the quantum-device abstraction layer.

    An Acquisition must consist of (at least) an AcquisitionProtocol specifying how the
    acquired signal is to be processed, and an AcquisitionChannel and AcquisitionIndex
    specifying where the acquired data is to be stored in the RawDataset.


    N.B. This class helps differentiate an acquisition operation from the regular
    operations. This enables us to use
    :func:`~.quantify_scheduler.schedules._visualization.pulse_diagram.plot_acquisition_operations`
    to highlight acquisition pulses in the pulse diagrams.
    """


@deprecated("1.0", Acquisition)
class AcquisitionOperation(Acquisition):
    """Deprecated alias."""

    pass


class Trace(Acquisition):
    """
    The Trace acquisition protocol measures a signal s(t).

    Only processing performed is rescaling and adding
    units based on a calibrated scale. Values are returned
    as a raw trace (numpy array of float datatype). Length of
    this array depends on the sampling rate of the acquisition
    device.

    .. important::

        The exact duration of this operation, and the possible bin modes may depend on
        the control hardware. Please consult your hardware vendor's :ref:`Reference
        guide` for more information.

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
        Describes what is done when data is written to a memory location that already
        contains values. Which bin mode can be used for Trace acquisitions may depend on
        the hardware. ``BinMode.AVERAGE``, the default, works on most hardware. This bin
        mode stores the weighted average value of the new result and the old values.
        ``BinMode.FIRST`` is used for hardware where only the result of the first
        acquisition in a Schedule is stored, e.g. for a Trace acquisition with Qblox QTM
        modules.
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
        bin_mode: BinMode | str = BinMode.AVERAGE,
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


class WeightedIntegratedSeparated(Acquisition):
    r"""
    Weighted integration acquisition protocol where two sets weights
    are applied separately to the real and imaginary parts
    of the signal.

    Weights are applied as:

    .. math::

        \widetilde{A} = \int \mathrm{Re}(S(t))\cdot W_A(t) \mathrm{d}t

    .. math::

        \widetilde{B} = \int \mathrm{Im}(S(t))\cdot W_B(t) \mathrm{d}t

    Parameters
    ----------
    waveform_a
        The complex waveform used as integration weights :math:`W_A(t)`.
    waveform_b
        The complex waveform used as integration weights :math:`W_B(t)`.
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
        waveform_a: dict[str, Any],
        waveform_b: dict[str, Any],
        port: str,
        clock: str,
        duration: float,
        acq_channel: int = 0,
        acq_index: int = 0,
        bin_mode: BinMode | str = BinMode.APPEND,
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
                "protocol": "WeightedIntegratedSeparated",
                "acq_return_type": complex,
            }
        ]
        self._update()
        # certain fields are required in the acquisition data
        if "acq_return_type" not in self.data["acquisition_info"][0]:
            self.data["acquisition_info"][0]["acq_return_type"] = complex
            self.data["acquisition_info"][0]["protocol"] = "WeightedIntegratedSeparated"

    def __str__(self) -> str:
        acq_info = self.data["acquisition_info"][0]
        return self._get_signature(acq_info)


class SSBIntegrationComplex(Acquisition):
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
        bin_mode: BinMode | str = BinMode.AVERAGE,
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


class ThresholdedAcquisition(Acquisition):
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
    feedback_trigger_label : str
        The label corresponding to the feedback trigger, which is mapped by the
        compiler to a feedback trigger address on hardware, by default None.

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
        bin_mode: BinMode | str = BinMode.AVERAGE,
        feedback_trigger_label: str | None = None,
        phase: float = 0,
        t0: float = 0,
        acq_rotation: float = 0,
        acq_threshold: float = 0,
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
                "acq_return_type": np.int32,
                "protocol": "ThresholdedAcquisition",
                "feedback_trigger_label": feedback_trigger_label,
                "acq_threshold": acq_threshold,
                "acq_rotation": acq_rotation,
            },
        ]
        self._update()

    def __str__(self) -> str:
        acq_info = self.data["acquisition_info"][0]
        return self._get_signature(acq_info)


class NumericalSeparatedWeightedIntegration(WeightedIntegratedSeparated):
    r"""
    Subclass of :class:`~WeightedIntegratedSeparated` with parameterized waveforms as weights.

    A WeightedIntegratedSeparated class using parameterized waveforms and
    interpolation as the integration weights.

    Weights are applied as:

    .. math::

        \widetilde{A} = \int \mathrm{Re}(S(t)\cdot W_A(t) \mathrm{d}t

    .. math::

        \widetilde{B} = \int \mathrm{Im}(S(t))\cdot W_B(t) \mathrm{d}t

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
        to 1 GHz. Note that during hardware compilation, the weights will be resampled
        with the sampling rate supported by the target hardware.
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
        weights_a: list[complex] | np.ndarray,
        weights_b: list[complex] | np.ndarray,
        weights_sampling_rate: float = 1e9,
        interpolation: str = "linear",
        acq_channel: int = 0,
        acq_index: int = 0,
        bin_mode: BinMode | str = BinMode.APPEND,
        phase: float = 0,
        t0: float = 0,
    ) -> None:
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
        self.data["acquisition_info"][0]["protocol"] = "NumericalSeparatedWeightedIntegration"
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


class NumericalWeightedIntegration(NumericalSeparatedWeightedIntegration):
    """
    Subclass of :class:`~NumericalSeparatedWeightedIntegration` returning a complex number.

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
        to 1 GHz. Note that during hardware compilation, the weights will be resampled
        with the sampling rate supported by the target hardware.
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
        weights_a: list[complex] | np.ndarray,
        weights_b: list[complex] | np.ndarray,
        weights_sampling_rate: float = 1e9,
        interpolation: str = "linear",
        acq_channel: int = 0,
        acq_index: int = 0,
        bin_mode: BinMode | str = BinMode.APPEND,
        phase: float = 0,
        t0: float = 0,
    ) -> None:
        super().__init__(
            port=port,
            clock=clock,
            weights_a=weights_a,
            weights_b=weights_b,
            weights_sampling_rate=weights_sampling_rate,
            interpolation=interpolation,
            acq_channel=acq_channel,
            acq_index=acq_index,
            bin_mode=bin_mode,
            phase=phase,
            t0=t0,
        )
        self.data["acquisition_info"][0]["protocol"] = "NumericalWeightedIntegration"
        self._update()


class TriggerCount(Acquisition):
    """
    Trigger counting acquisition protocol returning an integer.

    The trigger acquisition mode is used to measure how
    many times the trigger level is surpassed. The level is set
    in the hardware configuration.

    .. important::

        The exact duration of this operation, and the possible bin modes may depend on
        the control hardware. Please consult your hardware vendor's :ref:`Reference
        guide` for more information.

    Parameters
    ----------
    port
        The acquisition port.
    clock
        The clock used to demodulate the acquisition.
    duration
        The duration of the operation in seconds.
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
        list or "distribution" which stores the count value of the
        new result and the old register value, by default BinMode.APPEND.
    t0
        The acquisition start time in seconds, by default 0.
    fine_start_delay
        Delays the start of the acquisition by the given amount in seconds. Does
        not delay the start time of the operation in the schedule. If the
        hardware supports it, this parameter can be used to shift the
        acquisition window by a small amount of time, independent of the
        hardware instruction timing grid. Currently only implemented for Qblox
        QTM modules, which allow only positive values for this parameter. By
        default 0.
    fine_end_delay
        Delays the end of the pulse by the given amount in seconds. Does not
        delay the end time of the operation in the schedule. If the hardware
        supports it, this parameter can be used to shift the acquisition window
        by a small amount of time, independent of the hardware instruction
        timing grid. Currently only implemented for Qblox QTM modules, which
        allow only positive values for this parameter. By default 0.

    """

    def __init__(
        self,
        port: str,
        clock: str,
        duration: float,
        acq_channel: int = 0,
        acq_index: int = 0,
        bin_mode: BinMode | str = BinMode.APPEND,
        t0: float = 0,
        fine_start_delay: float = 0,
        fine_end_delay: float = 0,
    ) -> None:
        if bin_mode == BinMode.AVERAGE:
            warnings.warn(
                (
                    f"{bin_mode} is deprecated for the TriggerCount acquisition protocol, "
                    f"and will be removed in quantify-scheduler>=0.24.0. "
                    f"Use {BinMode.DISTRIBUTION} instead, which has the same effect."
                ),
                FutureWarning,
            )
            bin_mode = BinMode.DISTRIBUTION
        if bin_mode == BinMode.DISTRIBUTION and acq_index != 0:
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
                "fine_start_delay": fine_start_delay,
                "fine_end_delay": fine_end_delay,
            }
        ]
        self._update()

    def __str__(self) -> str:
        acq_info = self.data["acquisition_info"][0]
        return self._get_signature(acq_info)


class TimetagTrace(Acquisition):
    """
    The TimetagTrace acquisition protocol records timetags within an acquisition window.

    .. important::

        The exact duration of this operation, and the possible bin modes may depend on
        the control hardware. Please consult your hardware vendor's :ref:`Reference
        guide` for more information.

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
        contains a value. Only "BinMode.APPEND" is available at the moment; this option
        concatenates timetag results with the same acquisition channel and index.
    time_ref
        Selects the time reference that the timetag is recorded in relation to. String
        enumeration, one of:

        * start (default): record relative to the start of the window.
        * end: record relative to the end of the window. Note that this always yields a
          negative timetag.
        * first: syntactic sugar for first#, where # is the current channel.
        * timestamp: record relative to the timestamp marked using the ``Timestamp`` operation.
    t0
        The acquisition start time in seconds, by default 0.
    fine_start_delay
        Delays the start of the acquisition by the given amount in seconds. Does
        not delay the start time of the operation in the schedule. If the
        hardware supports it, this parameter can be used to shift the
        acquisition window by a small amount of time, independent of the
        hardware instruction timing grid. Currently only implemented for Qblox
        QTM modules, which allow only positive values for this parameter. By
        default 0.
    fine_end_delay
        Delays the end of the pulse by the given amount in seconds. Does not
        delay the end time of the operation in the schedule. If the hardware
        supports it, this parameter can be used to shift the acquisition window
        by a small amount of time, independent of the hardware instruction
        timing grid. Currently only implemented for Qblox QTM modules, which
        allow only positive values for this parameter. By default 0.

    """

    def __init__(
        self,
        duration: float,
        port: str,
        clock: str = DigitalClockResource.IDENTITY,
        acq_channel: int = 0,
        acq_index: int = 0,
        bin_mode: BinMode | str = BinMode.APPEND,
        time_ref: TimeRef | str = TimeRef.START,
        t0: float = 0,
        fine_start_delay: float = 0,
        fine_end_delay: float = 0,
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
                # time_source is not settable, because all timetags will be returned
                # instead of just one. It is fixed to TimeSource.FIRST
                # because the instrument coordinator uses this to calculate the
                # relative timestamps.
                "time_source": TimeSource.FIRST,
                "time_ref": time_ref,
                "protocol": "TimetagTrace",
                "acq_return_type": np.ndarray,
                "fine_start_delay": fine_start_delay,
                "fine_end_delay": fine_end_delay,
            }
        ]
        self._update()

    def __str__(self) -> str:
        acq_info = self.data["acquisition_info"][0]
        return self._get_signature(acq_info)


class Timetag(Acquisition):
    """
    Acquire a single timetag per acquisition index.

    .. important::

        The exact duration of this operation, and the possible bin modes may depend on
        the control hardware. Please consult your hardware vendor's :ref:`Reference
        guide` for more information.

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
        new result and the old register value, by default BinMode.APPEND.
    time_source
        Selects the timetag data source for this acquisition type. String enumeration,
        one of:

        * ``first`` (default): record the first timetag in the window.
        * ``second``: record the second timetag in the window. Can be used to measure
          pulse distance when combined with first as reference.
        * ``last``: record the last timetag in the window.
    time_ref
        Selects the time reference that the timetag is recorded in relation to. String
        enumeration, one of:

        * ``start`` (default): record relative to the start of the window.
        * ``end``: record relative to the end of the window. Note that this always
          yields a negative timetag.
        * ``first``: record relative to the first timetag in the window.
        * ``timestamp``: record relative to the timestamp marked using the
          :class:`~quantify_scheduler.operations.pulse_library.Timestamp` operation.
    t0
        The acquisition start time in seconds, by default 0.
    fine_start_delay
        Delays the start of the acquisition by the given amount in seconds. Does
        not delay the start time of the operation in the schedule. If the
        hardware supports it, this parameter can be used to shift the
        acquisition window by a small amount of time, independent of the
        hardware instruction timing grid. Currently only implemented for Qblox
        QTM modules, which allow only positive values for this parameter. By
        default 0.
    fine_end_delay
        Delays the end of the pulse by the given amount. Does not delay the end
        time of the operation in the schedule. If the hardware supports it, this
        parameter can be used to shift the acquisition window by a small amount
        of time, independent of the hardware instruction timing grid. Currently
        only implemented for Qblox QTM modules, which allow only positive values
        for this parameter. By default 0.

    """

    def __init__(
        self,
        duration: float,
        port: str,
        clock: str = DigitalClockResource.IDENTITY,
        acq_channel: int = 0,
        acq_index: int = 0,
        bin_mode: BinMode | str = BinMode.APPEND,
        time_source: TimeSource | str = TimeSource.FIRST,
        time_ref: TimeRef | str = TimeRef.START,
        t0: float = 0,
        fine_start_delay: float = 0,
        fine_end_delay: float = 0,
    ) -> None:
        super().__init__(name=self.__class__.__name__)

        if isinstance(time_source, str):
            time_source = TimeSource(time_source)
        if isinstance(time_ref, str):
            time_ref = TimeRef(time_ref)

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
                "time_source": time_source,
                "time_ref": time_ref,
                "acq_return_type": float,
                "protocol": "Timetag",
                "fine_start_delay": fine_start_delay,
                "fine_end_delay": fine_end_delay,
            }
        ]
        self._update()

    def __str__(self) -> str:
        acq_info = self.data["acquisition_info"][0]
        return self._get_signature(acq_info)
