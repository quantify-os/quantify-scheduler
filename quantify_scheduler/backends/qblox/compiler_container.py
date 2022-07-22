# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Contains the compiler container class."""

from __future__ import annotations

from typing import Any, Dict, Optional, Set, Union

from quantify_scheduler import Schedule
from quantify_scheduler.backends.qblox import constants
from quantify_scheduler.backends.qblox import instrument_compilers as compiler_classes
from quantify_scheduler.helpers.importers import import_python_object_from_string
from quantify_scheduler.helpers.schedule import get_total_duration


class CompilerContainer:
    """
    Container class that holds all the compiler objects for the individual instruments.

    This class serves to allow all the possible compilation steps that involve multiple
    devices at the same time, such as calculating the modulation frequency for a device
    with a separate local oscillator from a clock that is defined at the schedule level.

    It is recommended to construct this object using the ``from_hardware_cfg`` factory
    method.
    """

    def __init__(self, schedule: Schedule):
        """
        Constructor for the instrument container.

        Parameters
        ----------
        schedule
            The schedule to be compiled.
        """
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
        self.instrument_compilers = {}
        """The compilers for the individual instruments."""
        self.generics: Set[str] = set()
        """Set of generic instruments in the setup."""

    def prepare(self):
        """
        Prepares all the instrument compilers contained in the class,
        by running their respective :code:`prepare` methods.
        """
        for compiler in self.instrument_compilers.values():
            compiler.prepare()

    def compile(self, repetitions: int) -> Dict[str, Any]:
        """
        Performs the compilation for all the individual instruments.

        Parameters
        ----------
        repetitions
            Amount of times to perform execution of the schedule.

        Returns
        -------
        :
            Dictionary containing all the compiled programs for each instrument. The key
            refers to the name of the instrument that the program belongs to.
        """

        # for now name is hardcoded, but should be read from config.
        compiled_schedule = {}
        for name, compiler in self.instrument_compilers.items():
            compiled_instrument_program = compiler.compile(repetitions=repetitions)

            if compiled_instrument_program is not None:
                if name in self.generics:
                    if constants.GENERIC_IC_COMPONENT_NAME not in compiled_schedule:
                        compiled_schedule[constants.GENERIC_IC_COMPONENT_NAME] = {}
                    compiled_schedule[constants.GENERIC_IC_COMPONENT_NAME].update(
                        compiled_instrument_program
                    )
                else:
                    compiled_schedule[name] = compiled_instrument_program
        return compiled_schedule

    def add_instrument_compiler(
        self,
        name: str,
        instrument_type: Union[str, type],
        instrument_cfg: Dict[str, Any],
        latency_corrections: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Adds an instrument compiler to the container.

        Parameters
        ----------
        name
            Name of the instrument.
        instrument_type
            A reference to the compiler class. Can either be passed as string or a
            direct reference.
        instrument_cfg
            The hardware config dict for this specific instrument.
        latency_corrections
            Dict containing the delays for each port-clock combination. This is
            specified in the top layer of hardware config.
        """
        if isinstance(instrument_type, str):
            if instrument_type in compiler_classes.COMPILER_MAPPING:
                instrument_type = compiler_classes.COMPILER_MAPPING[instrument_type]
            else:
                instrument_type = import_python_object_from_string(instrument_type)

        if isinstance(instrument_type, type):
            if instrument_type is compiler_classes.LocalOscillator:
                compiler = compiler_classes.LocalOscillator(
                    parent=self,
                    name=name,
                    total_play_time=self.total_play_time,
                    hw_mapping=instrument_cfg,
                )
                self.generics.add(name)
            else:
                compiler = instrument_type(
                    parent=self,
                    name=name,
                    total_play_time=self.total_play_time,
                    hw_mapping=instrument_cfg,
                    latency_corrections=latency_corrections,
                )

            self.instrument_compilers[name] = compiler
        else:
            raise ValueError(
                f"{instrument_type} is not a valid compiler. {self.__class__} "
                f"expects either a string or a type. But {type(instrument_type)} was "
                f"passed."
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
        composite = cls(schedule)
        for instrument_name, instrument_cfg in hardware_cfg.items():
            if (
                not isinstance(instrument_cfg, dict)
                or "instrument_type" not in instrument_cfg
            ):
                continue

            instrument_type = instrument_cfg["instrument_type"]
            latency_corrections = hardware_cfg.get("latency_corrections", {})

            composite.add_instrument_compiler(
                name=instrument_name,
                instrument_type=instrument_type,
                instrument_cfg=instrument_cfg,
                latency_corrections=latency_corrections,
            )
        return composite
