"""
.. list-table::
    :header-rows: 1
    :widths: auto

    * - Import alias
      - Maps to
    * - :class:`!quantify_scheduler.Schedule`
      - :class:`.Schedule`
    * - :class:`!quantify_scheduler.Operation`
      - :class:`.Operation`
    * - :class:`!quantify_scheduler.CompiledSchedule`
      - :class:`.CompiledSchedule`
    * - :class:`!quantify_scheduler.Resource`
      - :class:`.Resource`
"""

__version__ = "0.5.0"


from .types import Schedule, Operation, CompiledSchedule
from .resources import Resource

# Commented out because it messes up Sphinx and sphinx extensions
# __all__ = ["Schedule", "CompiledSchedule", "Operation", "Resource"]
