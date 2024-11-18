# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""
Module containing the HardwareConfig object.

Extends ManualParameter to add methods to load from/to file and reload.
Note: ManualParameter might be refactored out at some point in the future.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from qcodes.instrument.parameter import ManualParameter

if TYPE_CHECKING:
    from quantify_scheduler import QuantumDevice


class HardwareConfig(ManualParameter):
    """
    The input dictionary used to generate a valid HardwareCompilationConfig.
    This configures the compilation from the quantum-device layer to the control-hardware layer.

    Parameters
    ----------
    configuration
        A dictionary with the hardware configuration.

    """

    def __init__(
        self, configuration: dict | None = None, instrument: QuantumDevice | None = None
    ) -> None:
        super().__init__(
            "hardware_config",
            instrument=instrument,
            initial_value=configuration,
            docstring=(
                "The input dictionary used to generate a valid HardwareCompilationConfig "
                "using quantum_device.generate_hardware_compilation_config(). This configures "
                "the compilation from the quantum-device layer to the control-hardware layer."
            ),
        )

    def load_from_json_file(self, file_path: str | Path) -> None:
        """
        Reload the object's configuration from a file.
        Updates the object's data using the contents of the file.

        Parameters
        ----------
        file_path
            The path to the file to reload from.

        Raises
        ------
        FileNotFoundError
            If the provided file path does not exist.
        IOError
            If an I/O error occurs during file reading.

        """
        with Path(file_path).open("r") as file:
            self.set(json.load(file))

    def write_to_json_file(self, file_path: str | Path) -> None:
        """
        Write the current configuration to a specified file.
        If the file does not exist, it is created.
        The data is written in JSON format, and an indentation of 2.

        Parameters
        ----------
        file_path
            The path to the file where data will be written.

        Raises
        ------
        ValueError
            If neither a file path is provided nor a previously known file path exists.
        IOError
            If an I/O error occurs during file creation or writing.

        """
        with Path(file_path).open("w") as file:
            json.dump(self.get(), file, ensure_ascii=False, indent=2)
