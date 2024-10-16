"""
.. list-table::
    :header-rows: 1
    :widths: auto

    * - Import alias
      - Target
    * - :class:`.QuantumDevice`
      - :class:`!quantify_scheduler.QuantumDevice`
    * - :class:`.Schedule`
      - :class:`!quantify_scheduler.Schedule`
    * - :class:`.Resource`
      - :class:`!quantify_scheduler.Resource`
    * - :class:`.ClockResource`
      - :class:`!quantify_scheduler.ClockResource`
    * - :class:`.BasebandClockResource`
      - :class:`!quantify_scheduler.BasebandClockResource`
    * - :class:`.DigitalClockResource`
      - :class:`!quantify_scheduler.DigitalClockResource`
    * - :class:`.Operation`
      - :class:`!quantify_scheduler.Operation`
    * - :obj:`.structure`
      - :obj:`!quantify_scheduler.structure`
    * - :class:`.ScheduleGettable`
      - :class:`!quantify_scheduler.ScheduleGettable`
    * - :class:`.BasicElectronicNVElement`
      - :class:`!quantify_scheduler.BasicElectronicNVElement`
    * - :class:`.BasicSpinElement`
      - :class:`!quantify_scheduler.BasicSpinElement`
    * - :class:`.BasicTransmonElement`
      - :class:`!quantify_scheduler.BasicTransmonElement`
    * - :class:`.CompositeSquareEdge`
      - :class:`!quantify_scheduler.CompositeSquareEdge`
    * - :class:`.InstrumentCoordinator`
      - :class:`!quantify_scheduler.InstrumentCoordinator`
    * - :class:`.GenericInstrumentCoordinatorComponent`
      - :class:`!quantify_scheduler.GenericInstrumentCoordinatorComponent`
    * - :class:`.SerialCompiler`
      - :class:`!quantify_scheduler.SerialCompiler`
    * - :class:`.MockLocalOscillator`
      - :class:`!quantify_scheduler.MockLocalOscillator`
"""

from . import structure
from ._version import __version__
from .backends import SerialCompiler
from .device_under_test import (
    BasicElectronicNVElement,
    BasicSpinElement,
    BasicTransmonElement,
    CompositeSquareEdge,
    QuantumDevice,
)
from .gettables import ScheduleGettable
from .helpers.mock_instruments import MockLocalOscillator
from .instrument_coordinator import InstrumentCoordinator
from .instrument_coordinator.components.generic import (
    GenericInstrumentCoordinatorComponent,
)
from .operations import Operation
from .resources import (
    BasebandClockResource,
    ClockResource,
    DigitalClockResource,
    Resource,
)
from .schedules import Schedule

__all__ = [
    "QuantumDevice",
    "Schedule",
    "Resource",
    "Operation",
    "ClockResource",
    "BasebandClockResource",
    "DigitalClockResource",
    "structure",
    "ScheduleGettable",
    "BasicElectronicNVElement",
    "BasicSpinElement",
    "BasicTransmonElement",
    "CompositeSquareEdge",
    "InstrumentCoordinator",
    "GenericInstrumentCoordinatorComponent",
    "SerialCompiler",
    "MockLocalOscillator",
]
