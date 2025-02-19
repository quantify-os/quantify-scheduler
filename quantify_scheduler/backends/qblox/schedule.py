# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Qblox backend specific schedule classes and associated utilities."""

from collections import UserDict
from typing import Any

from IPython.display import display

from quantify_scheduler.backends.qblox.visualization import _display_compiled_instructions
from quantify_scheduler.helpers.importers import export_python_object_to_path_string


class CompiledInstructions(UserDict):
    """
    Create an interactive table that represents the compiled instructions.

    When displayed in an interactive environment such as a jupyter notebook, the
    dictionary is displayed as an interactive table (if supported by the
    backend), otherwise is displayed as a regular dictionary. Each key from the
    compiled instructions can be retrieved with the usual ``[key]`` and
    ``.get(key)`` syntax. A raw dictionary can also be obtained via the
    ``.data`` attribute.

    These values typically contain a combination of sequence files, waveform
    definitions, and parameters to configure on the instrument.

    See examples below as well.

    .. admonition:: Examples

        .. admonition:: Example

            .. jupyter-execute::
                :hide-code:

                from quantify_scheduler import (
                    BasicTransmonElement,
                    QuantumDevice,
                    SerialCompiler,
                    Schedule,
                )
                from quantify_scheduler.operations import (
                    Measure,
                    Reset,
                    X,
                    Y,
                )
                from quantify_scheduler.schemas.examples import utils
                from qcodes.instrument import Instrument

                Instrument.close_all()

                q0 = BasicTransmonElement("q0")
                q4 = BasicTransmonElement("q4")

                for qubit in [q0, q4]:
                    qubit.rxy.amp180(0.115)
                    qubit.rxy.motzoi(0.1)
                    qubit.clock_freqs.f01(7.3e9)
                    qubit.clock_freqs.f12(7.0e9)
                    qubit.clock_freqs.readout(8.0e9)
                    qubit.measure.acq_delay(100e-9)

                quantum_device = QuantumDevice(name="quantum_device0")
                quantum_device.add_element(q0)
                quantum_device.add_element(q4)

                hardware_config = utils.load_json_example_scheme(
                    "qblox_hardware_config_transmon.json"
                )
                quantum_device.hardware_config(hardware_config)

                compiler = SerialCompiler("compiler")
                compiler.quantum_device = quantum_device

            .. jupyter-execute::

                schedule = Schedule("demo compiled instructions")
                schedule.add(Reset("q0", "q4"))
                schedule.add(X("q0"))
                schedule.add(Y("q4"))
                schedule.add(Measure("q0", acq_channel=0, acq_protocol="ThresholdedAcquisition"))
                schedule.add(Measure("q4", acq_channel=1, acq_protocol="ThresholdedAcquisition"))

                compiled_schedule = compiler.compile(schedule)
                compiled_instructions = compiled_schedule.compiled_instructions
                compiled_instructions


        .. admonition:: CompiledInstructions behave like dictionaries


            .. jupyter-execute::

                compiled_instructions["cluster0"]["cluster0_module4"]["sequencers"]["seq0"].integration_length_acq

    Parameters
    ----------
    compiled_instructions: dict
        Instructions in a dictionary form that are sent to the hardware.

    """

    # note for developers: We're inheriting from UserDict so that an instance of
    # CompiledInstructions behaves as a regular dictionary with all the
    # __get_item__ and similar methods.
    def __init__(
        self,
        compiled_instructions: dict[Any, Any],
    ) -> None:
        self.data = compiled_instructions
        """The raw compiled instructions in a dictionary form."""

    def _ipython_display_(self) -> None:
        """Generate interactive table when running in jupyter notebook."""
        tab = _display_compiled_instructions(self.data)
        display(tab)

    def __repr__(self) -> str:
        return str(self.data)

    def __setstate__(self, state: dict[str, dict]) -> None:
        self.data = state["data"]

    def __getstate__(self) -> dict[str, Any]:
        return {
            "deserialization_type": export_python_object_to_path_string(self.__class__),
            "data": self.data,
        }
