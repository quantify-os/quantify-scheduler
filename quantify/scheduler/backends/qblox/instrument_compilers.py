# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
"""Compiler classes for Qblox backend."""
from __future__ import annotations

from typing import Dict, Any

from quantify.scheduler.backends.qblox.compiler_abc import (
    InstrumentCompiler,
    PulsarSequencerBase,
    PulsarBase,
)

from quantify.scheduler.backends.types.qblox import OpInfo, LOSettings


class LocalOscillator(InstrumentCompiler):
    """
    Implementation of an `InstrumentCompiler` that compiles for a generic LO. The main
    difference between this class and the other compiler classes is that it doesn't take
    pulses and acquisitions.
    """

    def __init__(self, name: str, total_play_time: float, hw_mapping: Dict[str, Any]):
        """
        Constructor for a local oscillator compiler.

        Parameters
        ----------
        name
            QCoDeS name of the device it compiles for.
        total_play_time
            Total time execution of the schedule should go on for. This parameter is
            used to ensure that the different devices, potentially with different clock
            rates, can work in a synchronized way when performing multiple executions of
            the schedule.
        lo_freq
            LO frequency it needs to be set to. Either this is passed to the constructor
            or set later in the compilation process, in case the LO frequency is not
            initially given and needs to be calculated.
        """
        super().__init__(name, total_play_time, hw_mapping)
        self._settings = LOSettings.from_mapping(hw_mapping)

    @property
    def frequency(self) -> float:
        """
        Getter for the frequency.

        Returns
        -------
        :
            The current frequency.
        """
        return self._settings.lo_freq

    @frequency.setter
    def frequency(self, value: float):
        """
        Sets the lo frequency for this device if no frequency is specified, but raises
        an exception otherwise.

        Parameters
        ----------
        value
            The frequency to set it to.

        Raises
        -------
        ValueError
            Occurs when a frequency has been previously set and attempting to set the
            frequency to a different value than what it is currently set to. This would
            indicate an invalid configuration in the hardware mapping.
        """
        if self._settings.lo_freq is not None:
            if value != self._settings.lo_freq:
                raise ValueError(
                    f"Attempting to set LO {self.name} to frequency {value}, "
                    f"while it has previously already been set to "
                    f"{self._settings.lo_freq}!"
                )
        self._settings.lo_freq = value

    def compile(self, repetitions: int = 1) -> Dict[str, Any]:
        """
        Compiles the program for the LO control stack component.

        Parameters
        ----------
        repetitions
            Number of times execution the schedule is repeated.

        Returns
        -------
        :
            Dictionary containing all the information the cs component needs to set the
            parameters appropriately.
        """
        return self._settings.to_dict()


# ---------- pulsar sequencer classes ----------


class QCMSequencer(PulsarSequencerBase):
    """
    Subclass of Pulsar_sequencer_base that is meant to implement all the parts that are
    specific to a Pulsar QCM sequencer.
    """

    awg_output_volt = 2.5
    """Voltage range of the awg output paths."""


class QRMSequencer(PulsarSequencerBase):
    """
    Subclass of Pulsar_sequencer_base that is meant to implement all the parts that are
    specific to a Pulsar QRM sequencer.
    """

    awg_output_volt = 0.5
    """Voltage range of the awg output paths."""


# pylint: disable=invalid-name
class Pulsar_QCM(PulsarBase):
    """
    Pulsar QCM specific implementation of the pulsar compiler.
    """

    sequencer_type = QCMSequencer
    max_sequencers: int = 2

    def _distribute_data(self):
        """
        Distributes the pulses and acquisitions assigned to this pulsar over the
        different sequencers based on their portclocks. Overrides the function of the
        same name in the superclass to raise an exception in case it attempts to
        distribute acquisitions, since this is not supported by the pulsar QCM.

        Raises
        ------
        RuntimeError
            Pulsar_QCM._acquisitions is not empty.
        """
        if len(self._acquisitions) > 0:
            raise RuntimeError(
                f"Attempting to add acquisitions to {self.__class__} {self.name}, "
                f"which is not supported by hardware."
            )
        super()._distribute_data()

    def add_acquisition(self, port: str, clock: str, acq_info: OpInfo):
        """
        Raises an exception when called since the pulsar QCM does not support
        acquisitions.

        Parameters
        ----------
        port
            The port the pulse needs to be sent to.
        clock
            The clock for modulation of the pulse. Can be a BasebandClock.
        acq_info
            Data structure containing all the information regarding this specific
            acquisition operation.

        Raises
        ------
        RuntimeError
            Always.
        """
        raise RuntimeError(
            f"Pulsar QCM {self.name} does not support acquisitions. "
            f"Attempting to add acquisition {repr(acq_info)} "
            f"on port {port} with clock {clock}."
        )


# pylint: disable=invalid-name
class Pulsar_QRM(PulsarBase):
    """
    Pulsar QRM specific implementation of the pulsar compiler.
    """

    sequencer_type = QRMSequencer
    """The type of the sequencer."""
    max_sequencers: int = 1
    """Maximum number of sequencer available in the instrument."""
