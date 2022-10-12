# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
r"""
Module containing ProfiledScheduleGettable, a subclass of ScheduleGettable
used for profiling.

.. warning::

    The ProfiledScheduleGettable is currently only tested to support Qblox hardware.
"""
import json
import logging
import os
import time

import numpy as np
import matplotlib.pyplot as plt

from qcodes import Instrument

from quantify_scheduler.gettables import ScheduleGettable
from quantify_scheduler.instrument_coordinator import InstrumentCoordinator

logger = logging.getLogger(__name__)


def profiler(func):
    """Decorator that reports the execution time."""

    def wrap(self, *args, **kwargs):
        start = time.time()
        result = func(self, *args, **kwargs)
        end = time.time()
        if func.__name__ not in self.profile:
            self.profile[func.__name__] = []
        self.profile[func.__name__].append(end - start)
        return result

    return wrap


class ProfiledInstrumentCoordinator(InstrumentCoordinator):
    """
    This subclass implements a profiling tool to log timing results.
    """

    def __init__(self, name: str, parent_ic: InstrumentCoordinator):
        self.profile = {"schedule": []}
        super().__init__(name, add_default_generic_icc=False)
        self.parent_ic = parent_ic
        self.plot = None

    @profiler
    def add_component(
        self,
        component,
    ) -> None:
        self.parent_ic.add_component(component)

    @profiler
    def prepare(
        self,
        compiled_schedule,
    ) -> None:
        self.profile["schedule"].append(compiled_schedule.get_schedule_duration())
        self.parent_ic.prepare(compiled_schedule)

    @profiler
    def start(self):
        self.parent_ic.start()

    @profiler
    def stop(self, allow_failure=False):
        self.parent_ic.stop()

    @profiler
    def retrieve_acquisition(self):
        self.parent_ic.retrieve_acquisition()

    @profiler
    def wait_done(self, timeout_sec: int = 10):
        self.parent_ic.wait_done(timeout_sec)


class ProfiledScheduleGettable(ScheduleGettable):
    """
    Subclass to overwite the initialize method, in order to include
    compilation in the profiling.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.profile = {}
        self.plot = None

        # overwrite linked IC to a profiled IC
        self.instr_coordinator = (
            self.quantum_device.instr_instrument_coordinator.get_instr()
        )
        self.profiled_instr_coordinator = ProfiledInstrumentCoordinator(
            name="profiled_ic", parent_ic=self.instr_coordinator
        )
        self.quantum_device.instr_instrument_coordinator(
            self.profiled_instr_coordinator.name
        )

    @profiler
    def _compile(self, sched):
        """Overwrite compile step for profiling."""
        super()._compile(sched)

    def close(self):
        """Cleanup new profiling instruments to avoid future conflicts."""
        self.profile.update(self.profiled_instr_coordinator.profile)
        self.quantum_device.instr_instrument_coordinator(self.instr_coordinator.name)
        prof_ic = Instrument.find_instrument("profiled_ic")
        Instrument.close(prof_ic)

    def log_profile(  # pylint: disable=too-many-arguments
        self,
        obj=None,
        path="profiling_logs",
        filename=None,
        indent: int = 4,
        separators=None,
    ):
        """Store profiling logs to json file."""
        if not obj:
            obj = self.profile
        if not separators:
            separators = (",", ": ")
        if filename:
            if not os.path.exists(path):
                os.makedirs(path)

            file_path = os.path.join(path, filename)
            with open(file_path, "w", encoding="utf-8") as file:
                json.dump(obj, file, indent=indent, separators=separators)

        return self.profile

    def plot_profile(self, path=None, filename="average_runtimes.pdf"):
        """Create barplot of accumulated profiling data."""
        profile = self.profile
        time_ax = list(profile.keys())
        num_keys = len(time_ax)
        x_pos = np.arange(num_keys)
        means = [np.mean(x) for x in profile.values()]
        error = [np.std(x) for x in profile.values()]
        fig, ax = plt.subplots(figsize=(9, 6))

        color = ["r", "b", "c", "m", "k", "g", "y"][:num_keys]
        ax.bar(x_pos, means, yerr=error, color=color)
        ax.bar(num_keys, means[0], color=color[0])
        for i in range(1, num_keys):
            ax.bar(num_keys, means[i], color=color[i], bottom=sum(means[:i]))
        time_ax.append("total")
        ax.set_xticks(np.append(x_pos, num_keys))
        ax.set_xticklabels(time_ax)
        self.plot = (fig, ax)
        plt.ylabel("runtime [s]")
        plt.title("Average runtimes")

        file_path = os.path.join(path, filename) if path else filename
        fig.savefig(file_path)
