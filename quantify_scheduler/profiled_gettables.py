#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 14:05:01 2022

Profiled gettable
@author: koen
"""

from quantify_scheduler.compilation import qcompile
from quantify_scheduler.gettables import ScheduleGettable, _evaluate_parameter_dict
from quantify_scheduler.instrument_coordinator import InstrumentCoordinator

import logging
import time
import numpy as np
from datetime import datetime
import os
import json
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

def profiler(func):
    '''Decorator that reports the execution time.'''

    def wrap(self, *args, **kwargs):
        start = time.time()
        result = func(self, *args, **kwargs)
        end = time.time()
        if not self.profile.get(func.__name__):
            self.profile[func.__name__] = []
        self.profile[func.__name__].append(end - start)
        return result
    return wrap


class ProfiledInstrumentCoordinator(InstrumentCoordinator):
    """
    This subclass implements a profiling tool to log the timing results.
    """

    def __init__(self, name: str, parentinstrumentcoordinator):
        self.profile = {"schedule": []}
        super().__init__(name, add_default_generic_icc=False)
        self.parentIC = parentinstrumentcoordinator

    def _get_schedule_time(self, compiled_schedule):
        op_len = [op["pulse_info"][0]["duration"]
                  for op in compiled_schedule["operation_dict"].values()]
        schedule_time = sum(op_len)
        self.profile["schedule"].append(schedule_time)

    @profiler
    def add_component(self, component,
                      ) -> None:
        self.parentIC.add_component(component)

    @profiler
    def prepare(self, compiled_schedule,
                ) -> None:
        self._get_schedule_time(compiled_schedule)
        self.parentIC.prepare(compiled_schedule)

    @profiler
    def start(self):
        self.parentIC.start()

    @profiler
    def stop(self, allow_failure=False):
        self.parentIC.stop()

    @profiler
    def retrieve_acquisition(self):
        self.parentIC.retrieve_acquisition()

    @profiler
    def wait_done(self):
        self.parentIC.wait_done()


class ProfiledGettable(ScheduleGettable):
    """
    Subclass to overwite the initialize method, in order to include
    compilation in the profiling.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.profile = {"compile": []}

        # overwrite linked IC to a profiled IC
        instr_coordinator = self.quantum_device.instr_instrument_coordinator.get_instr()
        self.profiled_instr_coordinator = ProfiledInstrumentCoordinator(
            "profiled_IC", instr_coordinator)
        self.quantum_device.instr_instrument_coordinator("profiled_IC")

    def _compile(self, sched):
        start = time.time()
        self._compiled_schedule = qcompile(
            schedule=sched,
            device_cfg=self.quantum_device.generate_device_config(),
            hardware_cfg=self.quantum_device.generate_hardware_config(),
        )
        stop = time.time()
        self.profile["compile"].append(stop - start)
        
    def log_profiles(
        self, path="profiling_logs/profiling_log{}.json".format(
            datetime.now().strftime("%m%d%H%M"))):
        """
        Store time logs to json file.

        """
        profile = self.profile.copy()
        profile.update(self.profiled_instr_coordinator.profile)

        if not os.path.exists("profiling_logs"):
            os.makedirs("profiling_logs")
        with open(path, 'w') as f:
            json.dump(profile, f, indent=4, separators=(',', ': '))

    def plot_profile(self):
        profile = self.profile.copy()
        profile.update(self.profiled_instr_coordinator.profile)
        time_ax = list(profile.keys())
        profile_keys = len(time_ax)
        x_pos = np.arange(profile_keys)
        means = [np.mean(x)
                 for x in profile.values()]
        error = [np.std(x)
                 for x in profile.values()]
        fig, ax = plt.subplots(figsize=(9, 6))
        color = ['xkcd:bright blue', 'xkcd:sky blue', 'xkcd:sea blue',
                 'xkcd:turquoise blue', 'xkcd:aqua', 'xkcd:cyan']
        color = color[:profile_keys]
        ax.bar(
            x_pos,
            means,
            yerr=error,
            align='center',
            color=color,
            ecolor='black',
            capsize=10)
        ax.bar(profile_keys, means[0], color=color[0])

        for i in range(1, profile_keys):
            ax.bar(profile_keys, means[i],
                   color=color[i], bottom=sum(means[:i]))
        time_ax.append('total')
        ax.set_xticks(np.append(x_pos, profile_keys))
        ax.set_xticklabels(time_ax)
        plt.ylabel("runtime [s]")
        plt.title("Average runtimes")
        plt.savefig("average_runtimes.pdf")
