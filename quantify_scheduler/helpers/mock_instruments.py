# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Module containing Mock Instruments."""

from qcodes.instrument.instrument import Instrument
from qcodes.instrument.parameter import ManualParameter
from qcodes.utils import validators


class MockLocalOscillator(Instrument):
    """
    A class representing a dummy Local Oscillator.

    Parameters
    ----------
    name
        QCoDeS name of the intstrument.

    """

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._add_qcodes_parameters_dummy()

    def _add_qcodes_parameters_dummy(self) -> None:
        """Use to fake communications."""
        self.status = ManualParameter(
            "status",
            initial_value=False,
            vals=validators.Bool(),
            docstring="turns the output on/off",
            instrument=self,
        )

        self.frequency = ManualParameter(
            "frequency",
            label="Frequency",
            unit="Hz",
            initial_value=7e9,
            docstring="The RF Frequency in Hz",
            vals=validators.Numbers(),
            instrument=self,
        )

        self.power = ManualParameter(
            "power",
            label="Power",
            unit="dBm",
            initial_value=15.0,
            vals=validators.Numbers(min_value=-60.0, max_value=20.0),
            docstring="Signal power in dBm",
            instrument=self,
        )
