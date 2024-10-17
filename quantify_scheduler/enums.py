# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Enums for quantify-scheduler."""

try:
    from enum import StrEnum, unique  # type: ignore
except ImportError:
    from enum import Enum, unique

    class StrEnum(Enum):
        """Enum that can be directly serialized to string."""

        def __str__(self) -> str:
            return self.value


@unique
class BinMode(StrEnum):  # type: ignore
    """
    Describes how to handle `Acquisitions` that write to the same `AcquisitionIndex`.

    A BinMode is a property of an `AcquisitionChannel` that describes how to
    handle multiple
    :class:`~quantify_scheduler.operations.acquisition_library.Acquisition` s
    that write data to the same `AcquisitionIndex` on a channel.

    The most common use-case for this is when iterating over multiple
    repetitions of a :class:`~quantify_scheduler.schedules.schedule.Schedule`
    When the BinMode is set to `APPEND` new entries will be added as a list
    along the `repetitions` dimension.

    When the BinMode is set to `AVERAGE` the outcomes are averaged together
    into one value.

    Note that not all `AcquisitionProtocols` and backends support all possible
    BinModes. For more information, please see the :ref:`sec-acquisition-protocols`
    reference guide and some of the Qblox-specific :ref:`acquisition details
    <sec-qblox-acquisition-details>`.
    """

    APPEND = "append"
    AVERAGE = "average"
    FIRST = "first"
    DISTRIBUTION = "distribution"
    # N.B. in principle it is possible to specify other behaviours for
    # BinMode such as `SUM` or `OVERWRITE` but these are not
    # currently supported by any backend.


class TimeSource(StrEnum):  # type: ignore
    """
    Selects the timetag data source for timetag (trace) acquisitions.

    See :class:`~quantify_scheduler.operations.acquisition_library.Timetag` and
    :class:`~quantify_scheduler.operations.acquisition_library.TimetagTrace`.
    """

    FIRST = "first"
    SECOND = "second"
    LAST = "last"


class TimeRef(StrEnum):  # type: ignore
    """
    Selects the event that counts as a time reference (i.e. t=0) for timetags.

    See :class:`~quantify_scheduler.operations.acquisition_library.Timetag` and
    :class:`~quantify_scheduler.operations.acquisition_library.TimetagTrace`.
    """

    START = "start"
    END = "end"
    FIRST = "first"
    TIMESTAMP = "timestamp"
