# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Module containing definitions related to stitched pulses."""
from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np

from quantify_scheduler.backends.qblox import constants
from quantify_scheduler.backends.qblox.helpers import generate_waveform_data
from quantify_scheduler.helpers.deprecation import deprecated_arg_alias
from quantify_scheduler.operations.operation import Operation
from quantify_scheduler.operations.pulse_library import (
    NumericalPulse,
    ReferenceMagnitude,
    VoltageOffset,
)


class StitchedPulse(Operation):
    """
    A pulse composed of multiple operations that together constitute a waveform.

    This class can be used to construct arbitrarily long
    waveforms by stitching together pulses with optional changes in offset in
    between.

    Parameters
    ----------
    name : str or None, optional
        An optional name for the pulse.
    pulse_info : list[Any] or None, optional
        A list containing the pulses that are part of the StitchedPulse. By default
        None.
    """

    def __init__(
        self,
        name: str | None = None,
        pulse_info: list[Any] | None = None,
    ) -> None:
        pulse_info = pulse_info or []
        super().__init__(name=name or self.__class__.__name__)
        self.data["pulse_info"] = pulse_info
        self._update()

    def __str__(self) -> str:
        return (
            f"StitchedPulse(name='{self.data['name']}', pulse_info="
            f"{self.data['pulse_info']})"
        )

    def add_pulse(self, pulse_operation: Operation) -> None:
        """
        Adds pulse_info of pulse_operation Operation to this Operation.

        Parameters
        ----------
        pulse_operation : Operation
            an operation containing pulse_info.

        Raises
        ------
        ValueError
            When the operation's port and/or clock do not match those of the previously
            added SitchedPulse components.
        """
        if not self._pulse_and_clock_match(pulse_operation["pulse_info"]):
            raise ValueError(
                "All ports and clocks of a StitchedPulse's components must be equal."
            )

        super().add_pulse(pulse_operation)

    def _pulse_and_clock_match(self, operation_info: list[dict[str, Any]]) -> bool:
        """Check operation's port and clock match with StitchedPulse or if it's empty."""
        if len(self.data["pulse_info"]) == 0:
            return True

        port = self["pulse_info"][0]["port"]
        clock = self["pulse_info"][0]["clock"]

        for pulse_info in operation_info:
            if pulse_info["port"] != port or pulse_info["clock"] != clock:
                return False
        return True


def convert_to_numerical_pulse(
    operation: Operation,
    scheduled_at: float = 0.0,
    sampling_rate: float = 1e9,
) -> Operation:
    """
    Convert an operation with pulses and voltage offsets to a :class:`~.NumericalPulse`.

    If the operation also contains gate_info and/or
    acquisition_info, the original operation type is returned with only the
    pulse_info replaced by that of a NumericalPulse.

    The :class:`~.StitchedPulse`, and possibly other pulses, can contain descriptions
    for DC voltage offsets in the pulse info list. Not all functionality in
    :mod:`quantify_scheduler` supports such operations. For the cases where DC offset
    operations are not supported, this function can be used to convert the operation to
    a :class:`~.NumericalPulse`.

    Parameters
    ----------
    operation : Operation
        The operation to be converted.
    scheduled_at : float, optional
        The scheduled play time of the operation. The resulting NumericalPulse can be
        sampled from this time until the end of its duration. By default 0.0.
    sampling_rate : float, optional
        The rate with which to sample the input operation. By default 1e9 (1 GHz).

    Returns
    -------
    converted_operation : Operation
        The converted operation containing the sampled pulse information of the input
        operation. If the original operation only contained pulse_info and no
        gate_info or acquisition_info, a NumericalPulse is returned. Otherwise,
        the input type is returned with the pulse_info replaced by that of a
        NumericalPulse.
    """
    if not operation.valid_pulse:
        return operation

    pulse_t0: float = min(p_i["t0"] for p_i in operation["pulse_info"])
    # Round to nanoseconds, to avoid rounding errors.
    timestamps = np.round(
        pulse_t0
        + np.arange(round(operation.duration * sampling_rate) + 1) / sampling_rate,
        decimals=9,
    )

    waveform = np.zeros_like(timestamps).astype(complex)

    # First set all offsets
    for pulse_info in sorted(operation["pulse_info"], key=lambda inf: inf["t0"]):
        if "offset_path_I" not in pulse_info or "offset_path_Q" not in pulse_info:
            continue

        t0 = round(pulse_info["t0"], 9)

        if math.isclose(pulse_info["duration"], 0, abs_tol=5e-10):
            # Offset operations with "0" duration at the end of the pulse are
            # possible; we ignore these.
            if round(operation.duration, 9) == t0:
                continue
            time_idx = np.where(timestamps >= t0)
        else:
            t1 = round(pulse_info["t0"] + pulse_info["duration"], 9)
            time_idx = np.where((timestamps >= t0) & (timestamps < t1))
        waveform[time_idx] = (
            pulse_info["offset_path_I"] + 1j * pulse_info["offset_path_Q"]
        )

    # Then add the pulses
    for pulse_info in operation["pulse_info"]:
        if not pulse_info["wf_func"]:
            continue
        waveform_data = generate_waveform_data(
            data_dict=pulse_info,
            sampling_rate=sampling_rate,
        )
        t0, t1 = round(pulse_info["t0"], 9), round(
            pulse_info["t0"] + pulse_info["duration"], 9
        )
        time_idx = np.where((timestamps >= t0) & (timestamps < t1))
        waveform[time_idx] += waveform_data

    num_pulse = NumericalPulse(
        samples=waveform,
        t_samples=timestamps + scheduled_at,
        port=operation["pulse_info"][0]["port"],
        clock=operation["pulse_info"][0]["clock"],
        t0=pulse_t0,
    )
    num_pulse["name"] = operation["name"]
    if operation.valid_acquisition or operation.valid_gate:
        converted_op = deepcopy(operation)  # Do not modify the original operation
        converted_op.data["pulse_info"] = num_pulse.data["pulse_info"]
        return converted_op
    else:
        return num_pulse


@dataclass
class _VoltageOffsetInfo:
    path_I: float
    path_Q: float
    t0: float
    duration: float | None = None
    reference_magnitude: ReferenceMagnitude | None = None


class StitchedPulseBuilder:
    """
    Incrementally construct a StitchedPulse using pulse and offset operations.

    Parameters
    ----------
    port : str or None, optional
        Port of the stitched pulse. This can also be added later through
        :meth:`~.set_port`. By default None.
    clock : str or None, optional
        Clock used to modulate the stitched pulse. This can also be added later
        through :meth:`~.set_clock`. By default None.
    t0 : float, optional
        Time in seconds when to start the pulses relative to the start time
        of the Operation in the Schedule. This can also be added later through
        :meth:`~.set_t0`. By default None.
    """

    def __init__(
        self,
        name: str | None = None,
        port: str | None = None,
        clock: str | None = None,
        t0: float = 0.0,
    ) -> None:
        self._name = name or StitchedPulse.__name__
        self._port = port
        self._clock = clock
        self._t0 = t0
        self._pulses: list[Operation] = []
        self._offsets: list[_VoltageOffsetInfo] = []

    def set_port(self, port: str) -> StitchedPulseBuilder:
        """
        Set the port for all parts of the StitchedPulse.

        Parameters
        ----------
        port : str
            Port of the stitched pulse.

        Returns
        -------
        StitchedPulseBuilder
        """
        self._port = port
        return self

    def set_clock(self, clock: str) -> StitchedPulseBuilder:
        """
        Set the clock for all parts of the StitchedPulse.

        Parameters
        ----------
        clock : str
            Clock used to modulate the stitched pulse.

        Returns
        -------
        StitchedPulseBuilder
        """
        self._clock = clock
        return self

    def set_t0(self, t0: float) -> StitchedPulseBuilder:
        """
        Set the start time of the whole StitchedPulse.

        Parameters
        ----------
        t0 : float
            Time in seconds when to start the pulses relative to the start time
            of the Operation in the Schedule.

        Returns
        -------
        StitchedPulseBuilder
        """
        self._t0 = t0
        return self

    def add_pulse(
        self,
        pulse: Operation,
        append: bool = True,
    ) -> StitchedPulseBuilder:
        """
        Add an Operation to the StitchedPulse that is a valid pulse.

        Parameters
        ----------
        pulse : Operation
            The Operation to add.
        append : bool, optional
            Specifies whether to append the operation to the end of the StitchedPulse,
            or to insert it at a time relative to the start of the StitchedPulse,
            specified by the pulse's t0 attribute. By default True.

        Returns
        -------
        StitchedPulseBuilder

        Raises
        ------
        RuntimeError
            If the Operation is not a pulse.
        """
        if pulse.valid_acquisition:
            raise RuntimeError(
                "Cannot add acquisition to StitchedPulse. Please add it directly to "
                "the schedule instead."
            )
        if pulse.valid_gate:
            raise RuntimeError(
                "Cannot add gate to StitchedPulse. Please add it directly to the "
                "schedule instead."
            )
        if len(pulse["logic_info"]) > 0:
            raise RuntimeError(
                "Cannot add logic element to StitchedPulse. Please add it directly to "
                "the schedule instead."
            )
        if pulse.has_voltage_offset:
            raise RuntimeError(
                "Cannot use this method to add a voltage offset. Please use "
                "`add_voltage_offset` instead."
            )

        pulse = deepcopy(pulse)  # we will modify it
        if append:
            for pulse_info in pulse["pulse_info"]:
                pulse_info["t0"] += self.operation_end
        self._pulses.append(pulse)
        return self

    @deprecated_arg_alias("0.20.0", path_0="path_I", path_1="path_Q")
    def add_voltage_offset(
        self,
        path_I: float,
        path_Q: float,
        duration: float | None = None,
        rel_time: float = 0.0,
        append: bool = True,
        min_duration: float = constants.MIN_TIME_BETWEEN_OPERATIONS * 1e-9,
        reference_magnitude: ReferenceMagnitude | None = None,
    ) -> StitchedPulseBuilder:
        """
        Add a DC voltage offset to the StitchedPulse.

        .. note::

            If the voltage offset is not zero at the end of the StitchedPulse, the
            StitchedPulseBuilder will automatically add a zero voltage offset operation
            at the end. In that case, the StitchedPulse operation cannot be used at the
            very end of a Schedule or a control-flow block. For more information, please
            see :ref:`sec-qblox-offsets-long-voltage-offsets`.

        Parameters
        ----------
        path_I : float
            The offset on path I of the sequencer.
        path_Q : float
            The offset on path Q of the sequencer.
        duration : float or None, optional
            Specifies how long to maintain the offset. The StitchedPulseBuilder will add
            a zero voltage offset operation after the specified duration. If set to
            None, the offset voltage offset will hold until the end of the
            StitchedPulse. By default None.
        rel_time : float, optional
            Specifies when to set the offset, relative to the current end of the
            StitchedPulse (if ``append = True``), or to the start of the StitchedPulse
            (if ``append = False``). By default 0.0.
        append : bool, optional
            Specifies whether to append the operation to the end of the StitchedPulse,
            or to insert it at a time relative to the start of the StitchedPulse,
            specified by the the rel_time argument. By default True.
        min_duration : float, optional
            The minimal duration of the voltage offset. By default equal to the grid
            time of Qblox modules.
        reference_magnitude : optional
            Scaling value and unit for the unitless amplitude. Uses settings in
            hardware config if not provided.

        Returns
        -------
        StitchedPulseBuilder

        Raises
        ------
        ValueError
            If the duration is specified and not at least ``min_duration``.
        RuntimeError
            If the offset overlaps in time with a previously added offset.
        """
        if append:
            rel_time += self.operation_end

        if duration is not None and duration < min_duration:
            raise ValueError(
                f"Minimum duration of a voltage offset is {min_duration} ns"
            )

        offset = _VoltageOffsetInfo(
            path_I=path_I,
            path_Q=path_Q,
            t0=rel_time,
            duration=duration,
            reference_magnitude=reference_magnitude,
        )
        if self._overlaps_with_existing_offsets(offset):
            raise RuntimeError(
                "Tried to add offset that overlaps with existing offsets in the "
                "StitchedPulse."
            )

        self._offsets.append(offset)
        return self

    @property
    def operation_end(self) -> float:
        """
        Determine the end time of an operation based on its pulses and offsets.

        For pulses, the end time is calculated as the start time (`t0`) plus the pulse
        duration. For offsets, it uses the start time (`t0`) and, if provided, adds the
        duration. If no duration is specified for an offset, it assumes a default value
        of 0.0.

        Returns
        -------
        :
            The maximum end time considering all pulses and offsets.
        """
        max_from_pulses: float = (
            0.0
            if len(self._pulses) == 0
            else max(
                pulse_info["t0"] + pulse_info["duration"]
                for op in self._pulses
                for pulse_info in op.data["pulse_info"]
            )
        )
        max_from_offsets: float = (
            0.0
            if len(self._offsets) == 0
            else max(offs.t0 + (offs.duration or 0.0) for offs in self._offsets)
        )
        return max(max_from_pulses, max_from_offsets)

    def _distribute_port_clock(self) -> None:
        if self._port is None:
            raise RuntimeError("No port is defined.")
        if self._clock is None:
            raise RuntimeError("No clock is defined.")
        for op in self._pulses:
            for pulse_info in op.data["pulse_info"]:
                pulse_info["port"] = self._port
                pulse_info["clock"] = self._clock

    def _distribute_t0(self) -> None:
        for op in self._pulses:
            for pulse_info in op.data["pulse_info"]:
                pulse_info["t0"] += self._t0

    def _build_voltage_offset_operations(self) -> list[VoltageOffset]:
        """
        Add offset instructions that reset any offset that had a specified duration.

        If an offset was added without a duration, it is assumed that its duration
        should be until the end of the StitchedPulse, and any following offsets that
        _do_ have a duration will be reset to this value. Otherwise, offsets with a
        duration will be reset to 0.

        At the end of the StitchedPulse, the offset will be reset to 0.

        An offset does not need to be reset, if at the end of its duration, another
        offset instruction starts.

        This method requires the port and clock to have been set.
        """
        if len(self._offsets) == 0:
            return []

        def create_operation_from_info(info: _VoltageOffsetInfo) -> VoltageOffset:
            return VoltageOffset(
                offset_path_I=info.path_I,
                offset_path_Q=info.path_Q,
                port=self._port,  # type: ignore (_distribute_port_clock runs first)
                clock=self._clock,  # type: ignore
                t0=info.t0,
                reference_magnitude=info.reference_magnitude,
            )

        offset_ops: list[VoltageOffset] = []
        offset_infos = sorted(
            self._offsets,
            key=lambda op: op.t0,
        )
        background = (0.0, 0.0)
        for i, offset_info in enumerate(offset_infos):
            offset_ops.append(create_operation_from_info(offset_info))

            if offset_info.duration is None:
                # If no duration was specified, this offset should hold until the end of
                # the StitchedPulse.
                background = (
                    offset_info.path_I,
                    offset_info.path_Q,
                )
                continue

            this_end = offset_info.t0 + (offset_info.duration or 0.0)
            if math.isclose(this_end, self.operation_end):
                background = (0.0, 0.0)
            # Reset if the next offset's start does not overlap with the current
            # offset's end, or if the current offset is the last one
            if i + 1 >= len(self._offsets) or not math.isclose(
                self._offsets[i + 1].t0, this_end
            ):
                offset_ops.append(
                    create_operation_from_info(
                        _VoltageOffsetInfo(
                            background[0],
                            background[1],
                            t0=this_end,
                            reference_magnitude=offset_info.reference_magnitude,
                        )
                    )
                )

        # If this wasn't done yet, add a reset to 0 at the end of the StitchedPulse
        if not (math.isclose(background[0], 0) and math.isclose(background[1], 0)):
            offset_ops.append(
                create_operation_from_info(
                    _VoltageOffsetInfo(0.0, 0.0, t0=self.operation_end)
                )
            )

        return offset_ops

    def _overlaps_with_existing_offsets(self, offset: _VoltageOffsetInfo) -> bool:
        offsets = self._offsets[:]
        offsets.append(offset)
        offsets.sort(key=lambda op: op.t0)
        for i, offs in enumerate(offsets[:-1]):
            next_start = offsets[i + 1].t0
            this_end = offs.t0 + (offs.duration or 0.0)
            if next_start < this_end:
                return True
        return False

    def build(self) -> StitchedPulse:
        """
        Build the StitchedPulse.

        Returns
        -------
        StitchedPulse
        """
        self._distribute_port_clock()
        offsets = self._build_voltage_offset_operations()
        self._distribute_t0()
        stitched_pulse = StitchedPulse(self._name)
        for op in self._pulses + offsets:
            stitched_pulse.add_pulse(op)
        return stitched_pulse
