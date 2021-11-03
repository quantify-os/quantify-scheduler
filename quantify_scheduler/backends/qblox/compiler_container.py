# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
"""Contains the compiler container class."""

from __future__ import annotations

from typing import Any, Dict, Union

from quantify_core.utilities import general

from quantify_scheduler import Schedule
from quantify_scheduler.backends.qblox import instrument_compilers as compiler_classes
from quantify_scheduler.helpers.schedule import get_total_duration


class CompilerContainer:
    """
    Container class that holds all the compiler objects for the individual instruments.

    This class serves to allow all the possible compilation steps that involve multiple
    devices at the same time, such as calculating the modulation frequency for a device
    with a separate local oscillator from a clock that is defined at the schedule level.

    It is recommended to construct this object using the `from_mapping` factory method.
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
        for compiler in self.instrument_compilers.values():
            compiler.prepare()

        compiled_schedule = {}
        for name, compiler in self.instrument_compilers.items():
            compiled_instrument_program = compiler.compile(repetitions=repetitions)

            if compiled_instrument_program is not None:
                compiled_schedule[name] = compiled_instrument_program
        return compiled_schedule

    def add_instrument_compiler(
        self, name: str, instrument: Union[str, type], mapping: Dict[str, Any]
    ) -> None:
        """
        Adds an instrument compiler to the container.

        Parameters
        ----------
        name
            Name of the instrument.
        instrument
            A reference to the compiler class. Can either be passed as string or a
            direct reference.
        mapping
            The hardware mapping dict for this specific instrument.
        """
        if isinstance(instrument, type):
            self._add_from_type(name, instrument, mapping)
        elif isinstance(instrument, str):
            self._add_from_str(name, instrument, mapping)
        else:
            raise ValueError(
                f"{instrument} is not a valid compiler. {self.__class__} "
                f"expects either a string or a type. But {type(instrument)} was "
                f"passed."
            )

    def _add_from_str(
        self, name: str, instrument: str, mapping: Dict[str, Any]
    ) -> None:
        """
        Adds the instrument compiler from a string.

        Parameters
        ----------
        name
            Name of the Instrument.
        instrument
            The string that specifies the type of the compiler.
        mapping
            Hardware mapping for this instrument.
        """
        if instrument in compiler_classes.COMPILER_MAPPING:
            compiler: type = compiler_classes.COMPILER_MAPPING[instrument]
            self.add_instrument_compiler(name, compiler, mapping)
        else:
            self._add_from_path(name, instrument, mapping)

    def _add_from_path(
        self, name: str, instrument: str, mapping: Dict[str, Any]
    ) -> None:
        """
        Adds the instrument compiler from a path.

        Parameters
        ----------
        name
            Name of the Instrument.
        instrument
            The string that specifies the path to the type of the compiler.
            E.g., ``"my_module.MyInstrument"``.
        mapping
            Hardware mapping for this instrument.
        """
        # pylint: disable=fixme
        # TODO It seems to work for classes too. See quantify-core!232
        compiler: type = general.import_func_from_string(instrument)
        self.add_instrument_compiler(name, compiler, mapping)

    def _add_from_type(
        self, name: str, instrument: type, mapping: Dict[str, Any]
    ) -> None:
        """
        Adds the instrument compiler from a type.

        Parameters
        ----------
        name
            Name of the Instrument.
        instrument
            The type of the compiler.
        mapping
            Hardware mapping for this instrument.
        """
        compiler = instrument(self, name, self.total_play_time, mapping)
        self.instrument_compilers[name] = compiler

    @classmethod
    def from_mapping(cls, schedule: Schedule, mapping: dict) -> CompilerContainer:
        """
        Factory method for the CompilerContainer. This is the preferred way to use the
        CompilerContainer class.

        Parameters
        ----------
        schedule
            The schedule to pass to the constructor.
        mapping
            The hardware mapping.
        """
        composite = cls(schedule)
        for instr_name, instr_cfg in mapping.items():
            if not isinstance(instr_cfg, dict):
                continue

            device_type = instr_cfg["instrument_type"]
            composite.add_instrument_compiler(
                instr_name, device_type, mapping[instr_name]
            )

        return composite
