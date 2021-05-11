# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
"""Compiler classes for Qblox backend."""
from __future__ import annotations

from typing import Dict, Any, Optional

from quantify.scheduler.backends.qblox.compiler_abc import (
    InstrumentCompiler,
    PulsarSequencerBase,
    PulsarBase,
)

from quantify.scheduler.backends.types.qblox import OpInfo


class LocalOscillator(InstrumentCompiler):
    """
    Implementation of an `InstrumentCompiler` that compiles for a generic LO. The main
    difference between this class and the other compiler classes is that it doesn't take
    pulses and acquisitions.
    """

    def __init__(
        self,
        name: str,
        total_play_time: float,
        lo_freq: Optional[int] = None,
    ):
        """
        Constructor for a local oscillator compiler.

        Parameters
        ----------
        name:
            QCoDeS name of the device it compiles for.
        total_play_time:
            Total time execution of the schedule should go on for. This parameter is
            used to ensure that the different devices, potentially with different clock
            rates, can work in a synchronized way when performing multiple executions of
            the schedule.
        lo_freq:
            LO frequency it needs to be set to. Either this is passed to the constructor
            or set later in the compilation process, in case the LO frequency is not
            initially given and needs to be calculated.
        """
        super().__init__(name, total_play_time)
        self._lo_freq = lo_freq

    def assign_frequency(self, freq: float):
        """
        Sets the lo frequency for this device if no frequency is specified, but raises
        an exception otherwise.

        Parameters
        ----------
        freq:
            The frequency to set it to.

        Returns
        -------

        Raises
        -------
        ValueError
            Occurs when a frequency has been previously set and attempting to set the
            frequency to a different value than what it is currently set to. This would
            indicate an invalid configuration in the hardware mapping.
        """
        if self._lo_freq is not None:
            if freq != self._lo_freq:
                raise ValueError(
                    f"Attempting to set LO {self.name} to frequency {freq}, "
                    f"while it has previously already been set to {self._lo_freq}!"
                )
        self._lo_freq = freq

    @property
    def frequency(self) -> float:
        """
        Getter for the frequency.

        Returns
        -------
        :
            The current frequency
        """
        return self._lo_freq

    def compile(self, repetitions: int = 1) -> Dict[str, Any]:
        """
        Compiles the program for the LO control stack component.

        Parameters
        ----------
        repetitions:
            Number of times execution the schedule is repeated

        Returns
        -------
        :
            Dictionary containing all the information the cs component needs to set the
            parameters appropriately.
        """
        return {"lo_freq": self._lo_freq}


# ---------- pulsar sequencer classes ----------


class QCMSequencer(PulsarSequencerBase):
    """
    Subclass of Pulsar_sequencer_base that is meant to implement all the parts that are
    specific to a Pulsar QCM sequencer.

    Attributes
    ----------
    awg_output_volt:
        Voltage range of the awg output paths.
    """

    awg_output_volt = 2.5


class QRMSequencer(PulsarSequencerBase):
    """
    Subclass of Pulsar_sequencer_base that is meant to implement all the parts that are
    specific to a Pulsar QRM sequencer.

    Attributes
    ----------
    awg_output_volt:
        Voltage range of the awg output paths.
    """

    awg_output_volt = 0.5


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
            Pulsar_QCM._acquisitions is not empty
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
        port:
            The port the pulse needs to be sent to.
        clock:
            The clock for modulation of the pulse. Can be a BasebandClock.
        acq_info:
            Data structure containing all the information regarding this specific
            acquisition operation.

        Returns
        -------

        Raises
        ------
        RuntimeError
            Always
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
    max_sequencers: int = 1
