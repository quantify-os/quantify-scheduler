# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
"""Contains the compiler container class."""

from __future__ import annotations
from typing import Dict, Any, Union

from quantify.scheduler import types
from quantify.scheduler.helpers.schedule import get_total_duration


from quantify.scheduler.backends.qblox import instrument_compilers
from quantify.scheduler.backends.qblox.compiler_abc import InstrumentCompiler


class CompilerContainer:
    def __init__(self, schedule: types.Schedule):
        self.total_play_time = get_total_duration(schedule)
        self.resources = schedule.resources
        self.instrument_compilers: Dict[str, InstrumentCompiler] = dict()

    def compile(self, repetitions):
        compiled_schedule = dict()
        for name, compiler in self.instrument_compilers.items():
            compiled_dev_program = compiler.compile(repetitions=repetitions)

            if compiled_dev_program is not None:
                compiled_schedule[name] = compiled_dev_program
        return compiled_schedule

    def add_instrument_compiler(
        self, name: str, instrument: Union[str, type], mapping: Dict[str, Any]
    ):
        if isinstance(instrument, type):
            self._add_from_type(name, instrument, mapping)
        elif isinstance(instrument, str):
            self._add_from_str(name, instrument, mapping)

    def _add_from_str(self, name: str, instrument: str, mapping: Dict[str, Any]):
        compiler: type = getattr(instrument_compilers, instrument)
        self._add_from_type(name, compiler, mapping)

    def _add_from_type(self, name: str, instrument: type, mapping: Dict[str, Any]):
        compiler = instrument(self, name, self.total_play_time, mapping)
        self.instrument_compilers[name] = compiler

    @classmethod
    def from_mapping(cls, schedule: types.Schedule, mapping: dict) -> CompilerContainer:
        composite = cls(schedule)
        for instr_name, instr_cfg in mapping.items():
            if not isinstance(instr_cfg, dict):
                continue

            device_type = instr_cfg["instrument_type"]
            composite.add_instrument_compiler(
                instr_name, device_type, mapping[instr_name]
            )

        return composite
