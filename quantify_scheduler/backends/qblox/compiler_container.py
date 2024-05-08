# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Contains the compiler container class."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from quantify_scheduler.backends.qblox import constants
from quantify_scheduler.backends.qblox.instrument_compilers import (
    ClusterCompiler,
    LocalOscillatorCompiler,
)
from quantify_scheduler.helpers.schedule import get_total_duration

if TYPE_CHECKING:
    from quantify_scheduler import Schedule
    from quantify_scheduler.backends.qblox.compiler_abc import InstrumentCompiler


class CompilerContainer:
    """
    Container class that holds all the compiler objects for the individual instruments.

    This class serves to allow all the possible compilation steps that involve multiple
    devices at the same time, such as calculating the modulation frequency for a device
    with a separate local oscillator from a clock that is defined at the schedule level.

    It is recommended to construct this object using the ``from_hardware_cfg`` factory
    method.


    Parameters
    ----------
    schedule
        The schedule to be compiled.
    """

    def __init__(self, schedule: Schedule):
        self.total_play_time = get_total_duration(schedule)
        """
        The total duration of the schedule in absolute time this class will be
         compiling.
        """
        self.resources = schedule.resources
        """
        The resources attribute of the schedule. Used for getting the information
         from the clocks.
        """
        self.clusters: dict[str, ClusterCompiler] = {}
        """Cluster compiler instances managed by this container instance."""
        self.local_oscillators: dict[str, LocalOscillatorCompiler] = {}
        """Local oscillator compiler instances managed by this container instance."""

    def prepare(self):
        """
        Prepares all the instrument compilers contained in the class,
        by running their respective :code:`prepare` methods.
        """
        for compiler in self.instrument_compilers.values():
            compiler.prepare()

    @property
    def instrument_compilers(self) -> dict[str, InstrumentCompiler]:
        """The compilers for the individual instruments."""
        return {**self.clusters, **self.local_oscillators}

    def compile(self, debug_mode: bool, repetitions: int) -> dict[str, Any]:
        """
        Performs the compilation for all the individual instruments.

        Parameters
        ----------
        debug_mode
            Debug mode can modify the compilation process,
            so that debugging of the compilation process is easier.
        repetitions
            Amount of times to perform execution of the schedule.

        Returns
        -------
        :
            Dictionary containing all the compiled programs for each instrument. The key
            refers to the name of the instrument that the program belongs to.
        """
        compiled_schedule: dict[str, Any] = {}

        for name, compiler in self.clusters.items():
            if (
                compiled_instrument_program := compiler.compile(
                    debug_mode=debug_mode, repetitions=repetitions
                )
            ) is not None:
                compiled_schedule[name] = compiled_instrument_program

        for name, compiler in self.local_oscillators.items():
            if (
                compiled_instrument_program := compiler.compile(
                    debug_mode=debug_mode, repetitions=repetitions
                )
            ) is not None:
                if constants.GENERIC_IC_COMPONENT_NAME not in compiled_schedule:
                    compiled_schedule[constants.GENERIC_IC_COMPONENT_NAME] = {}
                compiled_schedule[constants.GENERIC_IC_COMPONENT_NAME].update(
                    compiled_instrument_program
                )

        return compiled_schedule

    def _add_cluster(
        self,
        name: str,
        instrument_cfg: dict[str, Any],
        latency_corrections: dict[str, float] | None = None,
        distortion_corrections: dict[int, Any] | None = None,
    ) -> None:
        self.clusters[name] = ClusterCompiler(
            parent=self,
            name=name,
            total_play_time=self.total_play_time,
            instrument_cfg=instrument_cfg,
            latency_corrections=latency_corrections,
            distortion_corrections=distortion_corrections,
        )

    def _add_local_oscillator(self, name: str, instrument_cfg: dict[str, Any]) -> None:
        self.local_oscillators[name] = LocalOscillatorCompiler(
            parent=self,
            name=name,
            total_play_time=self.total_play_time,
            instrument_cfg=instrument_cfg,
        )

    @classmethod
    def from_hardware_cfg(
        cls, schedule: Schedule, hardware_cfg: dict
    ) -> CompilerContainer:
        """
        Factory method for the CompilerContainer. This is the preferred way to use the
        CompilerContainer class.

        Parameters
        ----------
        schedule
            The schedule to pass to the constructor.
        hardware_cfg
            The hardware config.
        """
        distortion_corrections = hardware_cfg.get("distortion_corrections", {})
        latency_corrections = hardware_cfg.get("latency_corrections", {})

        composite = cls(schedule)
        for instrument_name, instrument_cfg in hardware_cfg.items():
            if (
                not isinstance(instrument_cfg, dict)
                or "instrument_type" not in instrument_cfg
            ):
                continue

            instrument_type = instrument_cfg["instrument_type"]

            instrument_distortion_corrections = None

            if instrument_type == "Cluster":
                instrument_distortion_corrections = distortion_corrections

            if instrument_type == "Cluster":
                composite._add_cluster(
                    name=instrument_name,
                    instrument_cfg=instrument_cfg,
                    latency_corrections=latency_corrections,
                    distortion_corrections=instrument_distortion_corrections,
                )
            elif instrument_type == "LocalOscillator":
                composite._add_local_oscillator(
                    name=instrument_name, instrument_cfg=instrument_cfg
                )
            else:
                raise ValueError(
                    f"{instrument_type} is not a known compiler type. Expected either a "
                    "'Cluster' or a 'LocalOscillator'."
                )

        return composite
