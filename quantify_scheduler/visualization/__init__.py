"""
This module is deprecated. Visualization tools are now accessed from :py:mod:`quantify_scheduler.schedules._visualization` (see example in links),
through alias methods of the :class:`.ScheduleBase` class.
"""

import warnings

warnings.warn(
    "This module is deprecated, please use the visualization methods provided in `quantify_scheduler.schedules.ScheduleBase` instead.",
    FutureWarning,
    stacklevel=2,
)
