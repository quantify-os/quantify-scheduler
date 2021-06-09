# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
"""Module containing the main ControlStack Component."""
from __future__ import annotations

from typing import Any, Dict, Optional
from collections import OrderedDict

from qcodes.instrument.base import Instrument
from qcodes.station import Station
from quantify.scheduler.controlstack.components import base


class ControlStack(Station):
    """
    The ControlStack class is a collection of ControlStack components.

    Each component in the collection is a QCoDeS
    :class:`~qcodes.instrument.base.Instrument` required to execute the experiment.

    This class provides a high level interface to:

    1. Arm instruments with sequence programs,
       waveforms and other settings.
    2. Start and stop the components
    3. Get the results.
    """

    components: Dict[str, base.AbstractControlStackComponent]

    def get_component(self, name: str) -> base.AbstractControlStackComponent:
        """
        Returns the ControlStack component by name.

        Parameters
        ----------
        name :
            The QCoDeS instrument name.

        Returns
        -------
        :
            The ControlStack component.

        Raises
        ------
        KeyError
            If key `name` is not present in `self.components`.
        """
        if name not in self.components:
            raise KeyError(f"Device '{name}' not added to '{self.__class__.__name__}'")
        return self.components[name]

    def add_component(
        self,
        component: base.AbstractControlStackComponent,
        name: Optional[str] = None,
        update_snapshot: bool = True,
    ) -> str:
        """
        Adds a ControlStack component to the ControlStack collection.

        Parameters
        ----------
        component
            The control stack component to add.
        name
            Optionally set the name of this component.
        update_snapshot
            Immediately update the snapshot of each component as it is added to the
            Station.

        Returns
        -------
        :
            The QCoDeS name assigned to this component, the name might be changed to
            make it unique among previously added components.

        Raises
        ------
        TypeError
            If `component` is not an instance of `AbstractControlStackComponent` or
            `Instrument`.
        """
        if not isinstance(component, (base.AbstractControlStackComponent, Instrument)):
            raise TypeError(
                (
                    "Expected AbstractControlStackComponent and Instrument for "
                    f"component argument, instead got {type(component)}"
                )
            )
        return super().add_component(
            component, name=name, update_snapshot=update_snapshot
        )

    def prepare(
        self,
        options: Dict[str, Any],
    ) -> None:
        """
        Prepares each ControlStack component in the list by name with
        parameters.

        The parameters such as sequence programs, waveforms and other
        settings are used to arm the instrument so it is ready to execute
        the experiment.

        Parameters
        ----------
        options
            The arguments per ControlStack component required to arm the
            instrument.

        Raises
        ------
        KeyError
            Undefined component name.
        """
        for instrument_name, args in options.items():
            self.get_component(instrument_name).prepare(args)

    def start(self) -> None:
        """
        Start all of the ControlStack components.

        The components are started in the order in which they were added
        to the ControlStack.
        """
        for instrument_name in self.components:
            self.get_component(instrument_name).start()

    def stop(self) -> None:
        """
        Stops all ControlStack Components.

        The components are stopped in the order in which they were added
        to the ControlStack.
        """
        for instrument_name in self.components:
            self.get_component(instrument_name).stop()

    def retrieve_acquisition(self) -> Dict[str, Any]:
        """
        Retrieves the latest acquisition results of ControlStack components
        with acquisition capabilities.

        Returns
        -------
        :
            The acquisition data per ControlStack component.
        """
        acq_dict = OrderedDict()
        for instrument_name in self.components:
            acq = self.get_component(instrument_name).retrieve_acquisition()
            if acq is not None:
                acq_dict[instrument_name] = acq
        return acq_dict
