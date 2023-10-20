Initializing a custom data structure:


.. jupyter-execute::
    :hide-code:

    # pylint: disable=duplicate-code
    # pylint: disable=pointless-string-statement
    # pylint: disable=too-few-public-methods
    # pylint: disable=wrong-import-order
    # pylint: disable=wrong-import-position


    from quantify_scheduler.structure import DataStructure

    import warnings

    warnings.filterwarnings("ignore", category=DeprecationWarning)



.. jupyter-execute::



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



This data structure can be used within other data structures:


.. jupyter-execute::



    class ExamplePulse(DataStructure):
        """..."""

        amplitude: float
        duration: float
        clock: ExampleClockResource


    pulse = ExamplePulse(amplitude=0.1, duration=2e-8, clock=clock)
    print(str(pulse))
    print(repr(pulse))



Serialization, deserialization and comparison are provided from scratch:


.. jupyter-execute::

    pulse_json = pulse.model_dump_json()
    print(pulse_json)
    pulse2 = ExamplePulse.model_validate_json(pulse_json)
    assert pulse == pulse2




User may implement custom validators:



.. jupyter-execute::
    :raises:

    from pydantic import field_validator


    class ScheduledExamplePulse(DataStructure):
        """..."""

        pulse: ExamplePulse
        start: float

        @field_validator("start")
        def _ensure_4ns_grid(cls, value):  # pylint: disable=no-self-argument,no-self-use
            if value % 4e-9 > 1e-12:
                raise ValueError("Start must be on a 4 ns grid due to hardware limitations")

            return value


    # This works fine
    scheduled_pulse = ScheduledExamplePulse(pulse=pulse, start=8e-9)
    # This raises a ValidationError
    scheduled_pulse = ScheduledExamplePulse(pulse=pulse, start=9e-9)



See `pydantic documentation <https://docs.pydantic.dev/>`_ for more usage
examples.
