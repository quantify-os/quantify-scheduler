# -----------------------------------------------------------------------------
# Description:    Library standard measurement protocols for use with the quantify.scheduler.
# Repository:     https://gitlab.com/quantify-os/quantify-scheduler
# Copyright (C)   Qblox BV & Orange Quantum Systems Holding BV (2020-2021)
# -----------------------------------------------------------------------------
from quantify.scheduler.types import Operation
from quantify.scheduler.resources import BasebandClockResource


class WeightedAcquisition(Operation):
    def __init__(
        self,
        duration_pulse: float,
        duration_acq: float,
        acq_delay: float,
        acq_index: int,
        port: str,
        clock: str,
        amp: float,
        rot_angle: float = 0,
        threshold: float = 0,
        store_trace: bool = False,
        store_IQ: bool = True,
        store_threshold: bool = False,
        phase: float = 0,
        t0: float = 0,
    ):
        """
        A two-channel modulated acquisition.

        Parameters
        ------------
        duration : float
            Duration of the aquisition in seconds.
        port : str
            Port of the pulse, must be capable of playing a complex waveform.
        phase : float
            Phase of the pulse in degrees.
        clock : str
            Clock used to modulate the pulse.
        """
        if phase != 0:
            # Because of how clock interfaces were changed.
            # FIXME: need to be able to add phases to the waveform separate from the clock.
            raise NotImplementedError

        data = {
            "name": "SquareAcquisition",
            "pulse_info": [
                {
                    "wf_func": "quantify.scheduler.waveforms.square",
                    "amp": amp,
                    "duration": duration_pulse,
                    "t0": t0,
                    "clock": clock,
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
                    "acq_index": acq_index,
                    "store_trace": store_trace,
                    "store_IQ": store_IQ,
                    "store_threshold": store_threshold,
                    "rot_angle": rot_angle,
                    "threshold": threshold,
                }
            ],
        }
        super().__init__(name=data["name"], data=data)