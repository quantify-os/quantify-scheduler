# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Tests for the InstrumentCompiler subclasses."""

import pytest

from quantify_scheduler.backends.qblox import constants
from quantify_scheduler.backends.qblox.qblox_acq_index_manager import (
    QbloxAcquisitionBinMapping,
    QbloxAcquisitionIndexBin,
    QbloxAcquisitionIndexManager,
)


def test_allocate_mixed_acquisitions():
    qblox_acq_index_manager = QbloxAcquisitionIndexManager()

    qblox_acq_index, qblox_acq_bin = qblox_acq_index_manager.allocate_bins("ch0", 2, "seq0")
    assert (qblox_acq_index, qblox_acq_bin) == (0, 0)
    qblox_acq_index = qblox_acq_index_manager.allocate_qblox_index("ch_trace", "seq0", False)
    assert qblox_acq_index == 1
    qblox_acq_index, qblox_acq_bin = qblox_acq_index_manager.allocate_bins("ch1", 6, "seq0")
    assert (qblox_acq_index, qblox_acq_bin) == (2, 0)

    qblox_acq_index, qblox_acq_bin = qblox_acq_index_manager.allocate_bins(
        "ch0", [5, 6, 7, 8], "seq0"
    )
    assert (qblox_acq_index, qblox_acq_bin) == (0, 1)

    qblox_acq_index, qblox_acq_bin = qblox_acq_index_manager.allocate_bins("ch1", [7, 8, 9], "seq0")
    assert (qblox_acq_index, qblox_acq_bin) == (2, 1)

    qblox_acq_index, qblox_acq_bin = qblox_acq_index_manager.allocate_bins(
        "ch2", list(range(constants.NUMBER_OF_QBLOX_ACQ_BINS - 3)), "seq0"
    )
    assert (qblox_acq_index, qblox_acq_bin) == (3, 0)

    qblox_acq_index, qblox_acq_bin = qblox_acq_index_manager.allocate_bins("ch0", 9, "seq0")
    assert (qblox_acq_index, qblox_acq_bin) == (0, 5)

    qblox_acq_index = qblox_acq_index_manager.allocate_qblox_index("ch3", "seq0", True)
    assert qblox_acq_index == 4

    expected_acq_declaration_dict = {
        "0": {"num_bins": 6, "index": 0},
        "1": {"num_bins": 4096, "index": 1},
        "2": {"num_bins": 4, "index": 2},
        "3": {"num_bins": 4093, "index": 3},
        "4": {"num_bins": 4096, "index": 4},
    }
    assert qblox_acq_index_manager.acq_declaration_dict() == expected_acq_declaration_dict

    expected_acq_hardware_mapping = {
        "ch0": {
            2: QbloxAcquisitionIndexBin(0, 0),
            5: QbloxAcquisitionIndexBin(0, 1),
            6: QbloxAcquisitionIndexBin(0, 2),
            7: QbloxAcquisitionIndexBin(0, 3),
            8: QbloxAcquisitionIndexBin(0, 4),
            9: QbloxAcquisitionIndexBin(0, 5),
        },
        "ch1": {
            6: QbloxAcquisitionIndexBin(2, 0),
            7: QbloxAcquisitionIndexBin(2, 1),
            8: QbloxAcquisitionIndexBin(2, 2),
            9: QbloxAcquisitionIndexBin(2, 3),
        },
        "ch2": {
            i: QbloxAcquisitionIndexBin(3, i) for i in range(constants.NUMBER_OF_QBLOX_ACQ_BINS - 3)
        },
        "ch3": 4,
        "ch_trace": 1,
    }
    assert qblox_acq_index_manager.acq_hardware_mapping() == expected_acq_hardware_mapping


def test_out_of_bins():
    qblox_acq_index_manager = QbloxAcquisitionIndexManager()

    # quantify_scheduler.backends.qblox.constants.NUMBER_OF_QBLOX_ACQ_BINS
    max_bins = 4096

    with pytest.raises(
        IndexError,
        match="Out of Qblox acquisition bins. "
        "The schedule requested too many Qblox acquisition bins for the sequencer seq0.",
    ):
        qblox_acq_index_manager.allocate_bins("ch1", list(range(max_bins + 1)), "seq0")


def test_out_of_qblox_acq_indices():
    qblox_acq_index_manager = QbloxAcquisitionIndexManager()

    # quantify_scheduler.backends.qblox.constants.NUMBER_OF_QBLOX_ACQ_INDICES
    max_qblox_acq_indices = 32

    for i in range(max_qblox_acq_indices):
        qblox_acq_index_manager.allocate_bins(f"ch{i}", 0, "seq0")

    with pytest.raises(
        IndexError,
        match="Out of Qblox acquisition bins. "
        "The schedule requested too many Qblox acquisition bins for the sequencer seq0.",
    ):
        qblox_acq_index_manager.allocate_bins("ch_extra", 1, "seq0")


def test_out_of_qblox_acq_indices_qblox_index_trace():
    qblox_acq_index_manager = QbloxAcquisitionIndexManager()

    # quantify_scheduler.backends.qblox.constants.NUMBER_OF_QBLOX_ACQ_INDICES
    max_qblox_acq_indices = 32

    for i in range(max_qblox_acq_indices):
        qblox_acq_index_manager.allocate_qblox_index(i, "seq0", False)

    with pytest.raises(
        IndexError,
        match="Out of Qblox acquisition bins. "
        "The schedule requested too many Qblox acquisition bins for the sequencer seq0.",
    ):
        qblox_acq_index_manager.allocate_qblox_index("ch1", "seq0", False)


def test_multiple_trace_raises():
    qblox_acq_index_manager = QbloxAcquisitionIndexManager()

    qblox_acq_index_manager.allocate_qblox_index(3, "seq0", False)

    with pytest.raises(
        ValueError,
        match="Multiple acquisition allocations for the same "
        "acquisition channels is not allowed for acquisition channel 3 "
        "for the sequencer seq0.",
    ):
        qblox_acq_index_manager.allocate_qblox_index(3, "seq0", True)
