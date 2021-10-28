# Commented out, likely not necessary anymore since quantify core and scheduler do not
# share the same namespace anymore. See also https://pymotw.com/3/pkgutil/
# __path__ = __import__("pkgutil").extend_path(__path__, __name__)
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


from .resources import Resource
from .types import CompiledSchedule, Operation, Schedule

# Commented out because it messes up Sphinx and sphinx extensions
# __all__ = ["Schedule", "CompiledSchedule", "Operation", "Resource"]
