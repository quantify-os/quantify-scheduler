"""
.. list-table::
    :header-rows: 1
    :widths: auto

    * - Import alias
      - Maps to
    * - :class:`!quantify_scheduler.instrument_coordinator.InstrumentCoordinator`
      - :class:`.InstrumentCoordinator`
    * - :class:`!quantify_scheduler.instrument_coordinator.ZIInstrumentCoordinator`
      - :class:`.ZIInstrumentCoordinator`
"""
from .instrument_coordinator import InstrumentCoordinator, ZIInstrumentCoordinator

# Commented out because it messes up Sphinx and sphinx extensions
# __all__ = ["InstrumentCoordinator", "ZIInstrumentCoordinator"]
