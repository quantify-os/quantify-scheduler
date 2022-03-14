"""
.. list-table::
    :header-rows: 1
    :widths: auto

    * - Import alias
      - Maps to
    * - :class:`!quantify_scheduler.schedules.schedule.Schedule`
      - :class:`.Schedule`
    * - :class:`!quantify_scheduler.operations.operation.Operation`
      - :class:`.Operation`
    * - :class:`!quantify_scheduler.schedules.schedule.CompiledSchedule`
      - :class:`.CompiledSchedule`
    * - :class:`!quantify_scheduler.Resource`
      - :class:`.Resource`
"""

__version__ = "0.6.0"


from . import structure
from .operations.operation import Operation
from .resources import Resource
from .schedules.schedule import CompiledSchedule, Schedule, Schedulable

__all__ = ["Schedule", "CompiledSchedule", "Operation", "Resource", "structure"]
