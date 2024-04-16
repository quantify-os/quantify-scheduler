# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Module containing pydantic validators."""
import numpy as np
from typing import Hashable

from qcodes.utils import validators
from qcodes.utils.validators import numbertypes, Validator


# this is a custom qcodes Numbers validator that allows for nan values.
class Numbers(validators.Numbers):
    """
    A custom qcodes Numbers validator that allows for nan values.

    Requires a number  of type int, float, numpy.integer or numpy.floating.

    Parameters
    ----------
    min_value:
        Minimal value allowed, default -inf.
    max_value:
        Maximal value allowed, default inf.
    allow_nan:
        if nan values are allowed, default False.

    Raises
    ------
    TypeError: If min or max value not a number. Or if min_value is
        larger than the max_value.
    """

    def __init__(
        self,
        min_value: numbertypes = -np.inf,
        max_value: numbertypes = np.inf,
        allow_nan: bool = False,
    ) -> None:
        super().__init__(min_value, max_value)
        self._allow_nan = allow_nan

    def validate(self, value: numbertypes, context: str = "") -> None:
        """
        Validate if number else raises error.

        Parameters
        ----------
        value:
            A number.
        context:
            Context for validation.

        Raises
        ------
        TypeError: If not int or float.
        ValueError: If number is not between the min and the max value.
        """
        if not isinstance(value, self.validtypes):
            raise TypeError(f"{repr(value)} is not an int or float; {context}")

        if self._allow_nan and np.isnan(value):
            # return early as the next statement will otherwise trigger
            return

        if not (self._min_value <= value <= self._max_value):
            raise ValueError(
                "{} is invalid: must be between "
                "{} and {} inclusive; {}".format(
                    repr(value), self._min_value, self._max_value, context
                )
            )


class _Durations(Numbers):
    """Validator used for durations. It allows all numbers greater than or equal to 0."""

    def __init__(
        self,
    ) -> None:
        super().__init__(min_value=0, allow_nan=False)


class _Amplitudes(Numbers):
    """Validator used for amplitudes. It allows all numbers and nan."""

    def __init__(
        self,
    ) -> None:
        super().__init__(allow_nan=True)


class _NonNegativeFrequencies(Numbers):
    """Validator used for frequencies. It allows positive numbers and nan."""

    def __init__(
        self,
    ) -> None:
        super().__init__(min_value=0, allow_nan=True)


class _Delays(Numbers):
    """Validator used for delays. It allows all numbers."""

    def __init__(
        self,
    ) -> None:
        super().__init__(allow_nan=False)


class _Hashable(Validator[Hashable]):
    """Validator used for hashables."""

    def __init__(self) -> None:
        self._valid_values = (0, "str")

    def validate(self, value: Hashable, context: str = "") -> None:
        """
        Validates if hashable else raises error.

        Parameters
        ----------
        value
            Value to validate
        context
            Context for validation.

        Raises
        ------
        TypeError
            If value is not hashable.
        """
        if not isinstance(value, Hashable):
            raise TypeError(f"{value!r} is not Hashable; {context}")

    def __repr__(self) -> str:
        return "<Hashable>"
