# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""
Qblox Visualization Module.

This module, part of the Qblox backend system, is dedicated to creating and
managing visual and User Interface (UI) elements essential for representing
compiled instructions and other relevant data.
"""

from __future__ import annotations

import re
from contextlib import suppress
from dataclasses import asdict, is_dataclass
from typing import Any

import ipywidgets
import pandas as pd
from columnar import columnar
from IPython.display import display
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator


def _display_dict(settings: dict[str, Any]) -> None:
    df = pd.DataFrame([settings]).T  # noqa: PD901
    df.columns = ["value"]
    df.columns.name = "setting"  # type: ignore
    display(df)


def _display_compiled_instructions(
    data: dict[Any, Any], parent_tab_name: str | None = None
) -> ipywidgets.Tab | None:
    """
    Display compiled instructions in a tabulated format.

    This function creates an interactive table, rendering and displaying
    compiled instructions along with other relevant data, allowing for a
    structured and user-friendly representation.

    In addition, it provides formatting specific for Qblox-specific sequencer
    programs, waveforms, and settings.

    .. admonition:: Note

        This function is tailored for :attr:`~.CompiledSchedule.compiled_instructions`
        but works with any nested dictionary.

    .. admonition:: Example

        .. jupyter-execute::
            :hide-code:

            from quantify_scheduler.backends import SerialCompiler
            from quantify_scheduler.device_under_test.quantum_device import QuantumDevice
            from quantify_scheduler.device_under_test.transmon_element import BasicTransmonElement
            from quantify_scheduler.operations.gate_library import (
                Measure,
                Reset,
                X,
                Y,
            )
            from quantify_scheduler.schedules.schedule import Schedule
            from quantify_scheduler.schemas.examples import utils

            compiler = SerialCompiler("compiler")

            q0 = BasicTransmonElement("q0")
            q4 = BasicTransmonElement("q4")

            for qubit in [q0, q4]:
                qubit.rxy.amp180(0.115)
                qubit.rxy.motzoi(0.1)
                qubit.clock_freqs.f01(7.3e9)
                qubit.clock_freqs.f12(7.0e9)
                qubit.clock_freqs.readout(8.0e9)
                qubit.measure.acq_delay(100e-9)

            quantum_device = QuantumDevice(name="quantum_device")
            quantum_device.add_element(q0)
            quantum_device.add_element(q4)

            device_config = quantum_device.generate_device_config()
            hardware_config = utils.load_json_example_scheme(
                "qblox_hardware_config_transmon.json"
            )
            quantum_device.hardware_config(hardware_config)

            compilation_config = quantum_device.generate_compilation_config()
            compiler = SerialCompiler("compiler")
            compiler.quantum_device = quantum_device

        .. jupyter-execute::

            schedule = Schedule("demo compiled instructions")
            schedule.add(Reset("q0", "q4"))
            schedule.add(X("q0"))
            schedule.add(Y("q4"))
            schedule.add(Measure("q0", acq_channel=0, acq_protocol='ThresholdedAcquisition'))
            schedule.add(Measure("q4", acq_channel=1, acq_protocol='ThresholdedAcquisition'))

            comp_schedule = compiler.compile(schedule)
            comp_schedule.compiled_instructions

    Parameters
    ----------
    data : dict
        A dictionary containing the compiled instructions and related data. The
        keys are strings representing tab names and the values are dictionaries
        containing the respective instruction data.

    parent_tab_name : str, optional
        A string representing the name of the parent tab in the user interface.
        If not specified, the function will use a default parent tab name.

    Returns
    -------
    widgets.Tab or None
        A Tab widget containing the structured representation of compiled
        instructions if the input data is not empty, otherwise None.

    """
    tab = ipywidgets.Tab()
    children: list[ipywidgets.Tab | ipywidgets.Output | None] = []
    titles: list[str] = []

    non_dict_values = {}

    for key, value in data.items():
        # modules that have unused sequencers are unused and filtered out.
        with suppress(AttributeError):
            if value.sequencers == {}:
                continue

        tab_name = str(key)
        child_tab = ipywidgets.Output()

        if is_dataclass(value):
            value = asdict(value)  # noqa: PLW2901 (overriding for loop variable) # pyright: ignore
        if isinstance(value, dict):
            if parent_tab_name == "waveforms":
                with child_tab:
                    tab_name = str(value["index"])
                    plt.plot(value["data"])
                    plt.xlabel("time [ns]")
                    plt.ylabel("amplitude")
                    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
                    plt.show()

            else:
                child_tab = _display_compiled_instructions(value, parent_tab_name=str(key))

            if child_tab:
                children.append(child_tab)
                titles.append(tab_name)
        else:
            non_dict_values[key] = value

    if non_dict_values:
        out = ipywidgets.Output()
        with out:
            if parent_tab_name == "sequence":
                titles.append("program")
                program = non_dict_values.get("program", "N/A")
                pattern = r"^(?P<label>\S+)?\s*(?P<cmd>\S+)?\s*(?P<args>\S+)?\s*(?P<comment>#.*)?$"
                matches = re.finditer(pattern, program, re.MULTILINE)

                result = []
                for match in matches:
                    row = [
                        match.group("label") or "",
                        match.group("cmd") or "",
                        match.group("args") or "",
                        match.group("comment") or "",
                    ]
                    result.append(row)
                print(columnar(result, headers=None, no_borders=True))
            else:
                tab_name = (
                    "settings"
                    if parent_tab_name and parent_tab_name.startswith("seq")
                    else "other values"
                )
                titles.append(tab_name)
                _display_dict(non_dict_values)
            children.append(out)

    tab.children = children
    for index, title in enumerate(titles):
        tab_name = "other" if title == "generic" else str(title)
        tab.set_title(index, tab_name)

    if titles:
        return tab
