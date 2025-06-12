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

QbloxAcquisitionIndex = int


@dataclass
class QbloxAcquisitionIndexBin:
    """Qblox acquisition index and QBlox acquisition bin."""

    index: QbloxAcquisitionIndex
    bin: int


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
        self._not_allowed_multiple_on_channel: set[Hashable] = set()
        """
        Specifies which acquisition channels do not allow
        multiple acquisitions on the same acquisition channel.
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
        number_of_bins: int,
        qblox_acq_index: int,
        acq_channel: Hashable,
        acq_indices: list[int] | None = None,
    ) -> int:
        """
        Reserves the Qblox acquisition bin with the parameters.
        This function already assumes that the bin is free, not yet used.

        Note, `number_of_bins` must be equal to the length of `acq_indices` if not `None`.

        Parameters
        ----------
        number_of_bins
            Number of bins to reserve.
        qblox_acq_index
            Qblox acquisition index to be used.
        acq_channel
            Acquisition channel.
        acq_indices
            Acquisition index.
            If `None`, it has no corresponding acquisition index (for example Trace acquisition).

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
            assert len(acq_indices) == number_of_bins
            if acq_channel in self._acq_hardware_mapping_not_binned:
                raise ValueError(
                    f"QbloxAcquisitionIndexManager conflicting type of acquisitions for "
                    f"{acq_channel=} and {qblox_acq_index=}."
                )
            new_qblox_acq_bins = range(next_free_qblox_bin, next_free_qblox_bin + number_of_bins)
            new_qblox_bin_mappings: QbloxAcquisitionBinMapping = {
                i: QbloxAcquisitionIndexBin(index=qblox_acq_index, bin=qblox_bin)
                for (i, qblox_bin) in zip(acq_indices, new_qblox_acq_bins)
            }
            if acq_channel not in self._acq_hardware_mapping_binned:
                self._acq_hardware_mapping_binned[acq_channel] = new_qblox_bin_mappings
            else:
                self._acq_hardware_mapping_binned[acq_channel].update(new_qblox_bin_mappings)

        self._next_free_qblox_acq_bin[qblox_acq_index] = next_free_qblox_bin + number_of_bins

        if str(qblox_acq_index) not in self._acq_declaration_dict:
            self._acq_declaration_dict[str(qblox_acq_index)] = {
                "num_bins": 0,
                "index": qblox_acq_index,
            }
        self._acq_declaration_dict[str(qblox_acq_index)]["num_bins"] += number_of_bins

        return next_free_qblox_bin

    def allocate_bins(
        self, acq_channel: Hashable, acq_indices: list[int] | int, sequencer_name: str
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

        qblox_acq_index: int | None = self._acq_channel_to_qblox_acq_index.get(acq_channel)
        if qblox_acq_index is None:
            qblox_acq_index = self._next_qblox_acq_index_with_all_free_bins()
        else:
            qblox_acq_index = self._acq_channel_to_qblox_acq_index[acq_channel]

        requested_number_of_bins: int = len(acq_indices)

        if (qblox_acq_index is None) or (
            requested_number_of_bins > self._number_of_free_qblox_bins(qblox_acq_index)
        ):
            raise IndexError(
                f"Out of Qblox acquisition bins. "
                f"The schedule requested too many Qblox acquisition bins "
                f"for the sequencer {sequencer_name}."
            )

        qblox_acq_bin_offset: int = self._reserve_qblox_acq_bins(
            number_of_bins=requested_number_of_bins,
            qblox_acq_index=qblox_acq_index,
            acq_channel=acq_channel,
            acq_indices=acq_indices,
        )
        self._acq_channel_to_qblox_acq_index[acq_channel] = qblox_acq_index

        return qblox_acq_index, qblox_acq_bin_offset

    def allocate_qblox_index(
        self, acq_channel: Hashable, sequencer_name: str, allow_multiple_on_channel: bool
    ) -> int:
        """
        Allocates a whole Qblox acquisition index for ttl, trace or other acquisition
        for the given acquisition channel.

        Parameters
        ----------
        acq_channel
            Acquisition channel.
        sequencer_name
            Sequencer name.
        allow_multiple_on_channel
            True if and only if we allow multiple acquisitions on the same acquisition channel.

        Returns
        -------
            The Qblox acquisition index.

        Raises
        ------
        IndexError
            When the QbloxAcquisitionBinManager runs out of acquisition indices to allocate.

        """
        if not allow_multiple_on_channel:
            self._not_allowed_multiple_on_channel.add(acq_channel)

        if acq_channel in self._acq_channel_to_qblox_acq_index:
            if acq_channel in self._not_allowed_multiple_on_channel:
                raise ValueError(
                    f"Multiple acquisition allocations for the same "
                    f"acquisition channels is not allowed for acquisition channel {acq_channel} "
                    f"for the sequencer {sequencer_name}."
                )
            return self._acq_channel_to_qblox_acq_index[acq_channel]

        qblox_acq_index: int | None = self._next_qblox_acq_index_with_all_free_bins()
        if qblox_acq_index is None:
            raise IndexError(
                f"Out of Qblox acquisition bins. "
                f"The schedule requested too many Qblox acquisition bins "
                f"for the sequencer {sequencer_name}."
            )

        requested_number_of_bins: int = self._number_of_free_qblox_bins(qblox_acq_index)

        self._reserve_qblox_acq_bins(
            number_of_bins=requested_number_of_bins,
            qblox_acq_index=qblox_acq_index,
            acq_channel=acq_channel,
            acq_indices=None,
        )
        self._acq_channel_to_qblox_acq_index[acq_channel] = qblox_acq_index

        return qblox_acq_index

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
