"""
.. list-table::
    :header-rows: 1
    :widths: auto

    * - Import alias
      - Target
    * - :class:`.Schedule`
      - :class:`!quantify_scheduler.schedules.schedule.Schedule`
    * - :class:`.Schedulable`
      - :class:`!quantify_scheduler.schedules.schedule.Schedulable`
    * - :class:`.CompiledSchedule`
      - :class:`!quantify_scheduler.schedules.schedule.CompiledSchedule`
    * - :class:`.Operation`
      - :class:`!quantify_scheduler.operations.operation.Operation`
    * - :class:`.Resource`
      - :class:`!quantify_scheduler.Resource`
"""
from ._version import __version__, __version_tuple__

from . import structure
from .operations.operation import Operation
from .resources import Resource
from .schedules.schedule import CompiledSchedule, Schedulable, Schedule

__all__ = ["Schedule", "CompiledSchedule", "Operation", "Resource", "structure"]
