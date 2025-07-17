# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""
Utility class for dynamically allocating
Qblox acquisition indices and bins and for Qblox sequencers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Union

from quantify_scheduler.backends.qblox import constants

if TYPE_CHECKING:
    from collections.abc import Hashable

    from quantify_scheduler.backends.types.common import ThresholdedTriggerCountMetadata

QbloxAcquisitionIndex = int


@dataclass
class QbloxAcquisitionIndexBin:
    """Qblox acquisition index and QBlox acquisition bin."""

    index: QbloxAcquisitionIndex
    """Qblox acquisition index."""
    bin: int
    """
    Qblox acquisition bin.
    For average bin mode, this is the bin where the data is stored.
    For append bin mode, this is first bin where data is stored,
    for each loop and repetition cycle, the data is consecutively stored.
    """
    stride: int
    """
    Stride.
    Only used for acquisitions within a loop (not schedule repetitions).
    Defines what's the stride between each repetitions of the schedule for the data.

    The assumption is that for an append bin mode operation
    with loops and schedule repetitions there is only one register;
    the register's inner iteration first goes through the loop,
    and then the schedule repetitions.
    """
    thresholded_trigger_count_metadata: ThresholdedTriggerCountMetadata | None
    """
    Thresholded trigger count metadata.
    Only applicable for ThresholdedTriggerCount,
    and only on QRM, QRM-RF, QRC.
    On QTM, this is unused, threshold calculations are on the hardware.
    """


QbloxAcquisitionBinMapping = dict[int, QbloxAcquisitionIndexBin]
"""
Binned type acquisition hardware mapping.

Each value maps the acquisition index to a hardware bin,
which is specified by the Qblox acquisition index, and the Qblox acquisition bin.
"""


QbloxAcquisitionHardwareMapping = Union[
    QbloxAcquisitionBinMapping,
    QbloxAcquisitionIndex,
]
"""
Type for all type acquisition hardware mapping.

This is a union of types, because the exact mapping type depends on the protocol.
"""


class QbloxAcquisitionIndexManager:
    """
    Utility class that keeps track of all the reserved indices, bins for a sequencer.

    Each acquisition channel is mapped to a unique Qblox acquisition index.
    For binned acquisitions, each new allocation request reserves
    the Qblox acquisition bins in order (incrementing the bin index by one).
    For trace and ttl and other acquisitions, the whole Qblox acquisition index is reserved,
    there, the bin index has no relevance.
    """

    def __init__(self) -> None:
        self._acq_declaration_dict: dict[str, Any] = {}
        """
        Acquisition declaration dictionary.

        This data is used in :class:`qblox_instruments.qcodes_drivers.Sequencer`
        `sequence` parameter's `"acquisitions"`.
        """
        self._acq_hardware_mapping_binned: dict[Hashable, QbloxAcquisitionBinMapping] = {}
        """
        Acquisition hardware mapping for binned acquisitions.
        """
        self._acq_hardware_mapping_not_binned: dict[Hashable, QbloxAcquisitionIndex] = {}
        """
        Acquisition hardware mapping for not binned acquisitions.
        """
        self._next_free_qblox_acq_bin: list[int] = [0] * constants.NUMBER_OF_QBLOX_ACQ_INDICES
        """
        Maps each Qblox acquisition index to the next free (unreserved) bin.
        """
        self._acq_channel_to_qblox_acq_index: dict[Hashable, int] = {}
        """
        Maps each acquisition channel to the
        Qblox acquisition index it uses.
        """
        self._trace_allocated: bool = False
        """
        Specifying whether a Trace or TimetagTrace have already been allocated.
        """

    def _number_of_free_qblox_bins(self, qblox_acq_index: int) -> int:
        return constants.NUMBER_OF_QBLOX_ACQ_BINS - self._next_free_qblox_acq_bin[qblox_acq_index]

    def _next_qblox_acq_index_with_all_free_bins(self) -> int | None:
        for i in range(constants.NUMBER_OF_QBLOX_ACQ_INDICES):
            if self._next_free_qblox_acq_bin[i] == 0:
                return i
        return None

    def _reserve_qblox_acq_bins(
        self,
        number_of_indices: int,
        qblox_acq_index: int,
        acq_channel: Hashable,
        acq_indices: list[int] | None,
        thresholded_trigger_count_metadata: ThresholdedTriggerCountMetadata | None,
        repetitions: int,
    ) -> int:
        """
        Reserves the Qblox acquisition bin with the parameters.
        This function already assumes that the bin is free, not yet used.

        Note, `number_of_indices` must be equal to the length of `acq_indices` if not `None`.

        Parameters
        ----------
        number_of_indices
            Number of indices to reserve.
        qblox_acq_index
            Qblox acquisition index to be used.
        acq_channel
            Acquisition channel.
        acq_indices
            Acquisition index.
            If `None`, it has no corresponding acquisition index (for example Trace acquisition).
        thresholded_trigger_count_metadata
            Thresholded trigger count metadata. If not applicable, `None`.
        repetitions
            Repetitions of the schedule for append bin mode; otherwise 1.

        Returns
        -------
            The starting Qblox acquisition bin.

        """
        next_free_qblox_bin: int = self._next_free_qblox_acq_bin[qblox_acq_index]

        if acq_indices is None:
            if acq_channel in self._acq_hardware_mapping_binned:
                raise ValueError(
                    f"QbloxAcquisitionIndexManager conflicting type of acquisitions for "
                    f"{acq_channel=} and {qblox_acq_index=}."
                )
            self._acq_hardware_mapping_not_binned[acq_channel] = qblox_acq_index
        else:
            assert len(acq_indices) == number_of_indices
            if acq_channel in self._acq_hardware_mapping_not_binned:
                raise ValueError(
                    f"QbloxAcquisitionIndexManager conflicting type of acquisitions for "
                    f"{acq_channel=} and {qblox_acq_index=}."
                )
            new_qblox_acq_bins = range(next_free_qblox_bin, next_free_qblox_bin + number_of_indices)
            new_qblox_bin_mappings: QbloxAcquisitionBinMapping = {
                i: QbloxAcquisitionIndexBin(
                    index=qblox_acq_index,
                    bin=qblox_bin,
                    stride=number_of_indices,
                    thresholded_trigger_count_metadata=thresholded_trigger_count_metadata,
                )
                for (i, qblox_bin) in zip(acq_indices, new_qblox_acq_bins)
            }
            if acq_channel not in self._acq_hardware_mapping_binned:
                self._acq_hardware_mapping_binned[acq_channel] = new_qblox_bin_mappings
            else:
                self._acq_hardware_mapping_binned[acq_channel].update(new_qblox_bin_mappings)

        self._next_free_qblox_acq_bin[qblox_acq_index] = (
            next_free_qblox_bin + number_of_indices * repetitions
        )

        if str(qblox_acq_index) not in self._acq_declaration_dict:
            self._acq_declaration_dict[str(qblox_acq_index)] = {
                "num_bins": 0,
                "index": qblox_acq_index,
            }
        self._acq_declaration_dict[str(qblox_acq_index)]["num_bins"] += (
            number_of_indices * repetitions
        )

        return next_free_qblox_bin

    def allocate_bins(
        self,
        acq_channel: Hashable,
        acq_indices: list[int] | int,
        sequencer_name: str,
        thresholded_trigger_count_metadata: ThresholdedTriggerCountMetadata | None,
        repetitions: int | None,
    ) -> tuple[int, int]:
        """
        Allocates len(acq_indices) number of Qblox acquisition bins.

        Parameters
        ----------
        acq_channel
            Acquisition channel.
        acq_indices
            Acquisition index.
            If `None`, it has no corresponding acquisition index (for example Trace acquisition).
        sequencer_name
            Sequencer name.
        thresholded_trigger_count_metadata
            Thresholded trigger count metadata. If not applicable, `None`.
        repetitions
            Repetitions of the schedule when using append bin mode.

        Returns
        -------
            The Qblox acquisition index, and the Qblox acquisition bin offset as integers.

        Raises
        ------
        IndexError
            When the QbloxAcquisitionBinManager runs out of bins to allocate.

        """
        if isinstance(acq_indices, int):
            acq_indices = [acq_indices]
        if repetitions is None:
            repetitions = 1

        qblox_acq_index: int | None = self._acq_channel_to_qblox_acq_index.get(acq_channel)
        if qblox_acq_index is None:
            qblox_acq_index = self._next_qblox_acq_index_with_all_free_bins()
        else:
            qblox_acq_index = self._acq_channel_to_qblox_acq_index[acq_channel]

        requested_number_of_indices: int = len(acq_indices)

        if (qblox_acq_index is None) or (
            requested_number_of_indices > self._number_of_free_qblox_bins(qblox_acq_index)
        ):
            raise IndexError(
                f"Out of Qblox acquisition bins. "
                f"The schedule requested too many Qblox acquisition bins "
                f"for the sequencer {sequencer_name}."
            )

        qblox_acq_bin_offset: int = self._reserve_qblox_acq_bins(
            number_of_indices=requested_number_of_indices,
            qblox_acq_index=qblox_acq_index,
            acq_channel=acq_channel,
            acq_indices=acq_indices,
            thresholded_trigger_count_metadata=thresholded_trigger_count_metadata,
            repetitions=repetitions,
        )
        self._acq_channel_to_qblox_acq_index[acq_channel] = qblox_acq_index

        return qblox_acq_index, qblox_acq_bin_offset

    def allocate_qblox_index(self, acq_channel: Hashable, sequencer_name: str) -> int:
        """
        Allocates a whole Qblox acquisition index for ttl, other acquisition
        for the given acquisition channel.

        Parameters
        ----------
        acq_channel
            Acquisition channel.
        sequencer_name
            Sequencer name.

        Returns
        -------
            The Qblox acquisition index.

        Raises
        ------
        IndexError
            When the QbloxAcquisitionBinManager runs out of acquisition indices to allocate.

        """
        if acq_channel in self._acq_channel_to_qblox_acq_index:
            return self._acq_channel_to_qblox_acq_index[acq_channel]

        qblox_acq_index: int | None = self._next_qblox_acq_index_with_all_free_bins()
        if qblox_acq_index is None:
            raise IndexError(
                f"Out of Qblox acquisition bins. "
                f"The schedule requested too many Qblox acquisition bins "
                f"for the sequencer {sequencer_name}."
            )

        requested_number_of_indices: int = self._number_of_free_qblox_bins(qblox_acq_index)

        self._reserve_qblox_acq_bins(
            number_of_indices=requested_number_of_indices,
            qblox_acq_index=qblox_acq_index,
            acq_channel=acq_channel,
            acq_indices=None,
            thresholded_trigger_count_metadata=None,
            repetitions=1,
        )
        self._acq_channel_to_qblox_acq_index[acq_channel] = qblox_acq_index

        return qblox_acq_index

    def allocate_trace(self, acq_channel: Hashable, sequencer_name: str) -> tuple[int, int]:
        """
        Allocates a whole Qblox acquisition index for trace
        for the given acquisition channel.

        Parameters
        ----------
        acq_channel
            Acquisition channel.
        sequencer_name
            Sequencer name.

        Returns
        -------
            The Qblox acquisition index, and the Qblox acquisition bin offset as integers.

        Raises
        ------
        IndexError
            When the QbloxAcquisitionBinManager runs out of acquisition indices to allocate.

        """
        if acq_channel in self._acq_channel_to_qblox_acq_index:
            return self._acq_channel_to_qblox_acq_index[acq_channel], 0
        elif self._trace_allocated:
            raise RuntimeError(
                f"Only one acquisition channel per port-clock can be specified, "
                f"if the 'Trace' acquisition protocol is used. "
                f"Attempted to compile for acquisition channel '{acq_channel}' "
                f"on sequencer '{sequencer_name}'."
            )

        qblox_acq_index: int | None = self._next_qblox_acq_index_with_all_free_bins()
        if qblox_acq_index is None:
            raise IndexError(
                f"Out of Qblox acquisition bins. "
                f"The schedule requested too many Qblox acquisition bins "
                f"for the sequencer {sequencer_name}."
            )

        requested_number_of_indices: int = 1

        self._reserve_qblox_acq_bins(
            number_of_indices=requested_number_of_indices,
            qblox_acq_index=qblox_acq_index,
            acq_channel=acq_channel,
            acq_indices=None,
            thresholded_trigger_count_metadata=None,
            repetitions=1,
        )
        self._acq_channel_to_qblox_acq_index[acq_channel] = qblox_acq_index

        self._trace_allocated = True

        return qblox_acq_index, 0

    def allocate_timetagtrace(
        self,
        acq_channel: Hashable,
        acq_indices: list[int],
        sequencer_name: str,
        repetitions: int,
    ) -> tuple[int, int]:
        """
        Allocates a whole Qblox acquisition index for TimetagTrace
        for the given acquisition channel.

        Parameters
        ----------
        acq_channel
            Acquisition channel.
        acq_indices
            Acquisition index.
        sequencer_name
            Sequencer name.
        repetitions
            Repetitions of the schedule.

        Returns
        -------
            The Qblox acquisition index, and the Qblox acquisition bin offset as integers.

        Raises
        ------
        IndexError
            When the QbloxAcquisitionBinManager runs out of acquisition indices to allocate.
        RuntimeError
            When there have already been an other trace acquisition allocated.

        """
        if (acq_channel not in self._acq_channel_to_qblox_acq_index) and (
            not self._trace_allocated
        ):
            qblox_acq_index: int | None = self._next_qblox_acq_index_with_all_free_bins()
            if qblox_acq_index is None:
                raise IndexError(
                    f"Out of Qblox acquisition bins. "
                    f"The schedule requested too many Qblox acquisition bins "
                    f"for the sequencer {sequencer_name}."
                )
        elif acq_channel in self._acq_channel_to_qblox_acq_index:
            qblox_acq_index = self._acq_channel_to_qblox_acq_index[acq_channel]
        else:
            raise RuntimeError(
                f"Only one acquisition channel per port-clock can be specified, "
                f"if the 'TimetagTrace' acquisition protocol is used. "
                f"Attempted to compile for acquisition channel '{acq_channel}' "
                f"on sequencer '{sequencer_name}'."
            )

        requested_number_of_indices: int = len(acq_indices)

        if (qblox_acq_index is None) or (
            requested_number_of_indices > self._number_of_free_qblox_bins(qblox_acq_index)
        ):
            raise IndexError(
                f"Out of Qblox acquisition bins. "
                f"The schedule requested too many Qblox acquisition bins "
                f"for the sequencer {sequencer_name}."
            )

        qblox_acq_bin_offset: int = self._reserve_qblox_acq_bins(
            number_of_indices=requested_number_of_indices,
            qblox_acq_index=qblox_acq_index,
            acq_channel=acq_channel,
            acq_indices=acq_indices,
            thresholded_trigger_count_metadata=None,
            repetitions=repetitions,
        )
        self._acq_channel_to_qblox_acq_index[acq_channel] = qblox_acq_index

        self._trace_allocated = True

        return qblox_acq_index, qblox_acq_bin_offset

    def acq_declaration_dict(self) -> dict[str, Any]:
        """
        Returns the acquisition declaration dict, which is needed for the qblox-instruments.
        This data is used in :class:`qblox_instruments.qcodes_drivers.Sequencer`
        `sequence` parameter's `"acquisitions"`.

        Returns
        -------
            The acquisition declaration dict.

        """
        return self._acq_declaration_dict

    def acq_hardware_mapping(self) -> dict[Hashable, QbloxAcquisitionHardwareMapping]:
        """
        Returns the acquisition hardware mapping, which is needed for
        quantify instrument coordinator to figure out which hardware index, bin needs
        to be mapped to which output acquisition data.

        Returns
        -------
            The acquisition hardware mapping.

        """
        return self._acq_hardware_mapping_binned | self._acq_hardware_mapping_not_binned
