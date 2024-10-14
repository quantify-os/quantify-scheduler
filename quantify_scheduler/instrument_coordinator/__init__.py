"""
.. list-table::
    :header-rows: 1
    :widths: auto

    * - Import alias
      - Target
    * - :class:`.InstrumentCoordinator`
      - :class:`!quantify_scheduler.instrument_coordinator.InstrumentCoordinator`
    * - :class:`.ZIInstrumentCoordinator`
      - :class:`!quantify_scheduler.instrument_coordinator.ZIInstrumentCoordinator`
"""

from .instrument_coordinator import InstrumentCoordinator, ZIInstrumentCoordinator

__all__ = ["InstrumentCoordinator", "ZIInstrumentCoordinator"]
