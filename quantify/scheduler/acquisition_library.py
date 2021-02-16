# -----------------------------------------------------------------------------
# Description:    Library standard acquisition protocols for use with the quantify.scheduler.
# Repository:     https://gitlab.com/quantify-os/quantify-scheduler
# Copyright (C)   Qblox BV & Orange Quantum Systems Holding BV (2020-2021)
# -----------------------------------------------------------------------------
from quantify.scheduler.types import Operation
from quantify.scheduler.resources import BasebandClockResource


class TraceAcquisition(Operation):
    def __init__(
        self,
        duration: float,
        acq_index: int,
        port: str,
        t0: float = 0,
    ):
        """
        A time-domain aqcuisition.

        Parameters
        ------------
        duration : float
            Duration of the aquisition in seconds.
        port : str
            Port of the acquisition.
        acq_index : int
            Bin in which the acquisition is stored. If multiple acquisitions with the same acq_index are used in the same experiment, their results will be averaged.

        """
        data = {
            "name": "TraceAcquisition",
            "trace_info": [
                {
                    "duration": duration,
                    "t0": t0,
                    "port": port,
                    "acq_index": acq_index,
                }
            ],
        }
        super().__init__(name=data["name"], data=data)


class VectorAcquisition(Operation):
    def __init__(
        self,
        duration_pulse: float,
        duration_acq: float,
        acq_delay: float,
        acq_index: int,
        amp: float,
        port: str,
        clock: str = None,
        phase: float = 0,
        t0: float = 0,
    ):
        """
        A two-port vector network measurement. A modulated square-pulse is played on the output and captured and integrated on the input to determine IQ values

        Parameters
        ------------
        duration_pulse : float
            Duration of the pulse in seconds.
        duration_acq : float
            Duration of the acquisition in seconds.
        acq_delay: float
            Delay between start of pulse and start of acquisition in seconds.
        amp: float
            amplitude of the pulse in Volt.
        port : str
            Port of the acquisition.
        acq_index : int
            Bin in which the acquisition is stored. If multiple acquisitions with the same acq_index are used in the same experiment, their results will be averaged.
        phase : float
            Phase of the pulse and acquisition in degrees.
        clock : str
            Clock used to modulate the pulse and demodulate acquisition.
        """
        if phase != 0:
            # Because of how clock interfaces were changed.
            # FIXME: need to be able to add phases to the waveform separate from the clock.
            raise NotImplementedError

        data = {
            "name": "VectorAcquisition",
            "pulse_info": [
                {
                    "wf_func": "quantify.scheduler.waveforms.square",
                    "amp": amp,
                    "duration": duration_pulse,
                    "t0": t0,
                    "clock": clock,
                    "phase": phase,
                    "port": port,
                }
            ],
            "acquisition_weights_info": [
                {
                    "wf_func": "quantify.scheduler.waveforms.square",
                    "amp": 1,
                    "duration": duration_acq,
                    "t0": t0 + acq_delay,
                    "clock": clock,
                    "port": port,
                    "phase": phase,
                    "acq_index": acq_index,
                }
            ],
        }
        super().__init__(name=data["name"], data=data)