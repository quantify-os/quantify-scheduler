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
from quantify_scheduler.helpers.schedule import _extract_port_clocks_used
from quantify_scheduler.operations.control_flow_library import ControlFlowOperation
from quantify_scheduler.schedules.schedule import ScheduleBase

if TYPE_CHECKING:
    from quantify_scheduler import Schedule
    from quantify_scheduler.backends.qblox.compiler_abc import InstrumentCompiler
    from quantify_scheduler.backends.qblox_backend import (
        QbloxHardwareCompilationConfig,
        _ClusterCompilationConfig,
        _LocalOscillatorCompilationConfig,
    )
    from quantify_scheduler.operations.operation import Operation
    from quantify_scheduler.resources import Resource


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

    def __init__(self, schedule: Schedule) -> None:
        self.total_play_time = schedule.get_schedule_duration() / schedule.repetitions
        """
        The total duration of a single repetition of the schedule.
        """
        self.resources = _extract_all_resources(schedule)
        """
        The resources attribute of the schedule. Used for getting the information
         from the clocks.
        """
        self.clusters: dict[str, ClusterCompiler] = {}
        """Cluster compiler instances managed by this container instance."""
        self.local_oscillators: dict[str, LocalOscillatorCompiler] = {}
        """Local oscillator compiler instances managed by this container instance."""

    def prepare(self) -> None:
        """
        Prepares all the instrument compilers contained in the class,
        by running their respective :code:`prepare` methods.
        """
        for lo_compiler in self.local_oscillators.values():
            lo_compiler.prepare()
        for cluster_compiler in self.clusters.values():
            cluster_compiler.prepare(
                external_los=self.local_oscillators, schedule_resources=self.resources
            )

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
        instrument_cfg: _ClusterCompilationConfig,
    ) -> None:
        self.clusters[name] = ClusterCompiler(
            name=name,
            total_play_time=self.total_play_time,
            instrument_cfg=instrument_cfg,
        )

    def _add_local_oscillator(
        self, name: str, instrument_cfg: _LocalOscillatorCompilationConfig
    ) -> None:
        self.local_oscillators[name] = LocalOscillatorCompiler(
            name=name,
            total_play_time=self.total_play_time,
            instrument_cfg=instrument_cfg,
        )

    @classmethod
    def from_hardware_cfg(
        cls, schedule: Schedule, hardware_cfg: QbloxHardwareCompilationConfig
    ) -> CompilerContainer:
        """
        Factory method for the CompilerContainer. This is the preferred way to use the
        CompilerContainer class.

        Parameters
        ----------
        schedule
            The schedule to pass to the constructor.
        hardware_cfg
            The hardware compilation config.

        """
        composite = cls(schedule)
        instrument_configs = hardware_cfg._extract_instrument_compilation_configs(
            _extract_port_clocks_used(schedule)
        )

        for (
            instrument_name,
            cfg,
        ) in instrument_configs.items():

            instrument_type = cfg.hardware_description.instrument_type

            if instrument_type == "Cluster":
                composite._add_cluster(
                    name=instrument_name,
                    instrument_cfg=cfg,
                )
            elif instrument_type == "LocalOscillator":
                composite._add_local_oscillator(
                    name=instrument_name,
                    instrument_cfg=cfg,
                )
            else:
                raise ValueError(
                    f"{instrument_type} is not a known compiler type. Expected either a "
                    "'Cluster' or a 'LocalOscillator'."
                )

        return composite


def _extract_all_resources(operation: Operation | ScheduleBase) -> dict[str, Resource]:
    if isinstance(operation, ScheduleBase):
        resources: dict[str, Resource] = operation.resources
        for inner_operation in operation.operations.values():
            resources.update(_extract_all_resources(inner_operation))
        return resources
    elif isinstance(operation, ControlFlowOperation):
        return _extract_all_resources(operation.body)
    else:
        return {}
