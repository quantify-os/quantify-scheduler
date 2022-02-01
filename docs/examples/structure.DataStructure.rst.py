# ---
# jupyter:
#   jupytext:
#     cell_markers: \"\"\"
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---
# %% [raw]
"""
Initializing a custom data structure:
"""

# %%
rst_conf = {"jupyter_execute_options": [":hide-code:"]}

# pylint: disable=duplicate-code
# pylint: disable=pointless-string-statement
# pylint: disable=too-few-public-methods
# pylint: disable=wrong-import-order
# pylint: disable=wrong-import-position


from quantify_scheduler.structure import DataStructure

# %%


class ExampleClockResource(DataStructure):
    """An example clock resource. It has name, frequency and phase. By default phase
    is zero.

    Parameters
    ----------
    name
        Clock name
    freq
        Clock frequency [Hz]
    phase
        Clock phase, by default 0 [rad].
    """

    name: str
    freq: float
    phase: float = 0


clock = ExampleClockResource(name="q0.01", freq=7e9)
print(str(clock))
print(repr(clock))

# %%

# %% [raw]

"""
This data structure can be used within other data structures:
"""

# %%


class ExamplePulse(DataStructure):
    """..."""

    amplitude: float
    duration: float
    clock: ExampleClockResource


pulse = ExamplePulse(amplitude=0.1, duration=2e-8, clock=clock)
print(str(pulse))
print(repr(pulse))

# %% [raw]
"""
Serialization, deserialization and comparison are provided from scratch:
"""
# %%
pulse_json = pulse.json()
print(pulse_json)
pulse2 = ExamplePulse.parse_raw(pulse_json)
assert pulse == pulse2

# %% [raw]
#
# User may implement custom validators:
#
# %%
rst_conf = {"jupyter_execute_options": [":raises:"]}

from pydantic import validator


class ScheduledExamplePulse(DataStructure):
    """..."""

    pulse: ExamplePulse
    start: float

    @validator("start")
    def _ensure_4ns_grid(cls, value):  # pylint: disable=no-self-argument,no-self-use
        if value % 4e-9 > 1e-12:
            raise ValueError("Start must be on a 4 ns grid due to hardware limitations")

        return value


# This works fine
scheduled_pulse = ScheduledExamplePulse(pulse=pulse, start=8e-9)
# This raises a ValidationError
scheduled_pulse = ScheduledExamplePulse(pulse=pulse, start=9e-9)

# %% [raw]

"""
See `pydantic documentation <https://pydantic-docs.helpmanual.io/>`_ for more usage
examples.
"""
