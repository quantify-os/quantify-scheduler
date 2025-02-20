# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Classes for handling pulses."""

from __future__ import annotations

import logging
from abc import ABC
from typing import TYPE_CHECKING, Any

import numpy as np

from quantify_scheduler.backends.qblox import constants, helpers, q1asm_instructions
from quantify_scheduler.backends.qblox.enums import ChannelMode
from quantify_scheduler.backends.qblox.operation_handling.base import IOperationStrategy
from quantify_scheduler.backends.types.qblox import (
    ClusterModuleDescription,
    RFDescription,
    StaticAnalogModuleProperties,
)
from quantify_scheduler.helpers.waveforms import normalize_waveform_data

if TYPE_CHECKING:
    from quantify_scheduler.backends.qblox.qasm_program import (
        QASMProgram,
    )
    from quantify_scheduler.backends.types import qblox as types

logger = logging.getLogger(__name__)


class PulseStrategyPartial(IOperationStrategy, ABC):
    """
    Contains the logic shared between all the pulses.

    Parameters
    ----------
    operation_info
        The operation info that corresponds to this pulse.
    channel_name
        Specifies the channel identifier of the hardware config (e.g. `complex_output_0`).

    """

    _amplitude_path_I: float | None  # noqa: N815  (mixed case)
    _amplitude_path_Q: float | None  # noqa: N815  (mixed case)

    def __init__(self, operation_info: types.OpInfo, channel_name: str) -> None:
        self._pulse_info: types.OpInfo = operation_info
        self.channel_name = channel_name

    @property
    def operation_info(self) -> types.OpInfo:
        """Property for retrieving the operation info."""
        return self._pulse_info


class GenericPulseStrategy(PulseStrategyPartial):
    """
    Default class for handling pulses.

    No assumptions are made with regards to the pulse shape and no optimizations
    are done.

    Parameters
    ----------
    operation_info
        The operation info that corresponds to this pulse.
    channel_name
        Specifies the channel identifier of the hardware config (e.g. `complex_output_0`).

    """

    def __init__(self, operation_info: types.OpInfo, channel_name: str) -> None:
        super().__init__(
            operation_info=operation_info,
            channel_name=channel_name,
        )

        self._amplitude_path_I: float | None = None
        self._amplitude_path_Q: float | None = None

        self._waveform_index0: int | None = None
        self._waveform_index1: int | None = None

        self._waveform_len: int | None = None

    def generate_data(self, wf_dict: dict[str, Any]) -> None:
        """
        Generates the data and adds them to the ``wf_dict`` (if not already present).

        In complex mode (e.g. ``complex_output_0``), the NCO produces real-valued data
        (:math:`I_\\text{IF}`) on sequencer path_I and imaginary data (:math:`Q_\\text{IF}`)
        on sequencer path_Q.

        .. math::
            \\underbrace{\\begin{bmatrix}
            \\cos\\omega t & -\\sin\\omega t \\\\
            \\sin\\omega t & \\phantom{-}\\cos\\omega t \\end{bmatrix}}_\\text{NCO}
            \\begin{bmatrix}
            I \\\\
            Q \\end{bmatrix} =
            \\begin{bmatrix}
            I \\cdot \\cos\\omega t - Q \\cdot\\sin\\omega t \\\\
            I \\cdot \\sin\\omega t + Q \\cdot\\cos\\omega t \\end{bmatrix}
            \\begin{matrix}
            \\ \\text{(path_I)} \\\\
            \\ \\text{(path_Q)} \\end{matrix}
            =
            \\begin{bmatrix}
            I_\\text{IF} \\\\
            Q_\\text{IF} \\end{bmatrix}


        In real mode (e.g. ``real_output_0``), the NCO produces :math:`I_\\text{IF}` on
        path_I


        .. math::
            \\underbrace{\\begin{bmatrix}
            \\cos\\omega t & -\\sin\\omega t \\\\
            \\sin\\omega t & \\phantom{-}\\cos\\omega t \\end{bmatrix}}_\\text{NCO}
            \\begin{bmatrix}
            I \\\\
            Q \\end{bmatrix}  =
            \\begin{bmatrix}
            I \\cdot \\cos\\omega t - Q \\cdot\\sin\\omega t\\\\
             - \\end{bmatrix}
            \\begin{matrix}
            \\ \\text{(path_I)} \\\\
            \\ \\text{(path_Q)} \\end{matrix}
            =
            \\begin{bmatrix}
            I_\\text{IF} \\\\
            - \\end{bmatrix}


        Note that the fields marked with `-` represent waveforms that are not relevant
        for the mode.


        Parameters
        ----------
        wf_dict
            The dictionary to add the waveform to. N.B. the dictionary is modified in
            function.

        Raises
        ------
        ValueError
            Data is complex (has an imaginary component), but the channel_name is not
            set as complex (e.g. ``complex_output_0``).

        """  # noqa: D301
        op_info = self.operation_info
        waveform_data = helpers.generate_waveform_data(
            op_info.data, sampling_rate=constants.SAMPLING_RATE
        )
        waveform_data, amp_real, amp_imag = normalize_waveform_data(waveform_data)
        self._waveform_len = len(waveform_data)

        if np.any(np.iscomplex(waveform_data)) and ChannelMode.COMPLEX not in self.channel_name:
            raise ValueError(
                f"Complex valued {str(op_info)} detected but the sequencer"
                f" is not expecting complex input. This can be caused by "
                f"attempting to play complex valued waveforms on an output"
                f" marked as real.\n\nException caused by {repr(op_info)}."
            )

        def non_null(amp: float) -> bool:
            return abs(amp) >= 2 / constants.IMMEDIATE_SZ_GAIN

        idx_real = (
            helpers.add_to_wf_dict_if_unique(wf_dict=wf_dict, waveform=waveform_data.real)
            if non_null(amp_real)
            else None
        )
        idx_imag = (
            helpers.add_to_wf_dict_if_unique(wf_dict=wf_dict, waveform=waveform_data.imag)
            if non_null(amp_imag)
            else None
        )

        self._waveform_index0, self._waveform_index1 = idx_real, idx_imag
        self._amplitude_path_I, self._amplitude_path_Q = amp_real, amp_imag

    def insert_qasm(self, qasm_program: QASMProgram) -> None:
        """
        Add the assembly instructions for the Q1 sequence processor that corresponds to
        this pulse.

        Parameters
        ----------
        qasm_program
            The QASMProgram to add the assembly instructions to.

        """
        if qasm_program.time_last_pulse_triggered is not None and (
            qasm_program.elapsed_time - qasm_program.time_last_pulse_triggered
            < constants.MIN_TIME_BETWEEN_OPERATIONS
        ):

            raise ValueError(
                f"Attempting to start an operation at t="
                f"{qasm_program.elapsed_time} ns, while the last operation was "
                f"started at t={qasm_program.time_last_pulse_triggered} ns. "
                f"Please ensure a minimum interval of "
                f"{constants.MIN_TIME_BETWEEN_OPERATIONS} ns between "
                f"operations.\n\nError caused by operation:\n"
                f"{repr(self.operation_info)}."
            )
        qasm_program.time_last_pulse_triggered = qasm_program.elapsed_time

        # Only emit play command if at least one path has a signal
        # else update parameters as there might still be some lingering
        # from for example a voltage offset.
        index0 = self._waveform_index0
        index1 = self._waveform_index1
        if index0 is None and index1 is None:
            qasm_program.emit(
                q1asm_instructions.UPDATE_PARAMETERS,
                constants.MIN_TIME_BETWEEN_OPERATIONS,
                comment=f"{self.operation_info.name} has too low amplitude to be played, "
                f"updating parameters instead",
            )
        else:
            assert self._amplitude_path_I is not None
            assert self._amplitude_path_Q is not None
            qasm_program.set_gain_from_amplitude(
                self._amplitude_path_I, self._amplitude_path_Q, self.operation_info
            )
            # If a channel doesn't have an index (index0 or index1 is None) means,
            # that for that channel we do not want to play any waveform;
            # it's also ensured in this case, that the gain is set to 0 for that channel;
            # but, the Q1ASM program needs a waveform index for both channels,
            # so we set the other waveform's index in this case as a dummy
            qasm_program.emit(
                q1asm_instructions.PLAY,
                index0 if (index0 is not None) else index1,
                index1 if (index1 is not None) else index0,
                constants.MIN_TIME_BETWEEN_OPERATIONS,  # N.B. the waveform keeps playing
                comment=f"play {self.operation_info.name} ({self._waveform_len} ns)",
            )
        qasm_program.elapsed_time += constants.MIN_TIME_BETWEEN_OPERATIONS


class DigitalOutputStrategy(PulseStrategyPartial, ABC):
    """
    Interface class for :class:`MarkerPulseStrategy` and :class:`DigitalPulseStrategy`.

    Both classes work very similarly, since they are both strategy classes for the
    `~quantify_scheduler.operations.pulse_library.MarkerPulse`. The
    ``MarkerPulseStrategy`` is for the QCM/QRM modules, and the ``DigitalPulseStrategy``
    for the QTM.
    """

    def generate_data(self, wf_dict: dict[str, Any]) -> None:
        """Returns None as no waveforms are generated in this strategy."""
        pass


class MarkerPulseStrategy(DigitalOutputStrategy):
    """If this strategy is used a digital pulse is played on the corresponding marker."""

    def __init__(
        self,
        operation_info: types.OpInfo,
        channel_name: str,
        module_options: ClusterModuleDescription,
    ) -> None:
        super().__init__(operation_info, channel_name)
        self.module_options = module_options

    def insert_qasm(self, qasm_program: QASMProgram) -> None:
        """
        Inserts the QASM instructions to play the marker pulse.
        Note that for RF modules the first two bits of set_mrk
        are used as switches for the RF outputs.

        Parameters
        ----------
        qasm_program
            The QASMProgram to add the assembly instructions to.

        """
        hw_properties = qasm_program.static_hw_properties
        if not isinstance(hw_properties, StaticAnalogModuleProperties):
            raise TypeError(
                f"Marker Operations are only supported for analog modules, "
                f"not for instrument {self.module_options.instrument_type}."
            )
        if self.channel_name not in hw_properties.channel_name_to_digital_marker:
            raise ValueError(
                f"Unable to set markers on channel '{self.channel_name}' for "
                f"instrument {hw_properties.instrument_type} "
                f"and operation {self.operation_info.name}. "
                f"Supported channels: {list(hw_properties.channel_name_to_digital_marker.keys())}"
            )
        marker = hw_properties.channel_name_to_digital_marker[self.channel_name]
        default_marker = 0
        if (
            isinstance(self.module_options, RFDescription)
            and self.module_options.rf_output_on
            and hw_properties.default_markers
        ):
            default_marker = hw_properties.default_markers[self.channel_name]
            if marker | default_marker == default_marker:  # Marker has no effect
                raise RuntimeError(
                    "Attempting to turn on an RF output on a module where "
                    "`rf_output_on` is set to True (the default value). \n"
                    "Turning the RF output on with an RFSwitchToggle Operation "
                    "has no effect. \n"
                    "Please set `rf_output_on` to False for this module "
                    "in the hardware configuration."
                )

        if self.operation_info.data["enable"]:
            marker |= default_marker
            qasm_program.emit(
                q1asm_instructions.SET_MARKER,
                marker,
                comment=f"set markers to {marker} (marker pulse)",
            )
        else:
            qasm_program.emit(
                q1asm_instructions.SET_MARKER,
                default_marker,
                comment=f"set markers to {default_marker} (default, marker pulse)",
            )


class DigitalPulseStrategy(DigitalOutputStrategy):
    """
    If this strategy is used a digital pulse is played
    on the corresponding digital output channel.
    """

    def insert_qasm(self, qasm_program: QASMProgram) -> None:
        """
        Inserts the QASM instructions to play the marker pulse.
        Note that for RF modules the first two bits of set_mrk
        are used as switches for the RF outputs.

        Parameters
        ----------
        qasm_program
            The QASMProgram to add the assembly instructions to.

        """
        if ChannelMode.DIGITAL not in self.channel_name:
            port = self.operation_info.data.get("port")
            clock = self.operation_info.data.get("clock")

            raise ValueError(
                f"{self.__class__.__name__} can only be used with a "
                f"digital channel. Please make sure that "
                f"'digital' keyword is included in the channel_name in the hardware configuration "
                f"for port-clock combination '{port}-{clock}' "
                f"(current channel_name is '{self.channel_name}')."
                f"Operation causing exception: {self.operation_info}"
            )

        if self.operation_info.data["enable"]:
            fine_delay = helpers.convert_qtm_fine_delay_to_int(
                self.operation_info.data.get("fine_start_delay", 0)
            )
        else:
            fine_delay = helpers.convert_qtm_fine_delay_to_int(
                self.operation_info.data.get("fine_end_delay", 0)
            )
        qasm_program.emit(
            q1asm_instructions.SET_DIGITAL,
            int(self.operation_info.data["enable"]),
            1,  # Mask. Reserved for future use, set to 1.
            fine_delay,
        )
