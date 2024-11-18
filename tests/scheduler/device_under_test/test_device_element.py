import math
from unittest import mock

import numpy as np
import pytest

from quantify_scheduler import (
    BasicElectronicNVElement,
    BasicSpinElement,
    BasicTransmonElement,
)


@pytest.fixture
def transmon_element_with_params(mock_setup_basic_transmon_with_standard_params):
    qubit = mock_setup_basic_transmon_with_standard_params["q0"]
    qubit.measure.acq_delay(123e-9)
    yield qubit


class TestDeviceElement:
    @pytest.mark.parametrize(
        "element_type",
        [BasicTransmonElement, BasicElectronicNVElement, BasicSpinElement],
    )
    def test_device_element_creation(self, element_type):
        element = element_type("qubit")
        assert isinstance(element, element_type)

    def test_device_element_deserialization(self, transmon_element_with_params):
        original_qubit = transmon_element_with_params
        original_qubit_json = original_qubit.to_json()
        original_snapshot = original_qubit.snapshot()
        original_qubit.close()

        new_qubit = BasicTransmonElement.from_json(original_qubit_json)
        new_snapshot = new_qubit.snapshot()

        self.normalize_snapshot(original_snapshot)
        self.normalize_snapshot(new_snapshot)

        # Ensure that the modified parameter is conserved upon deserialization.
        assert new_qubit.measure.acq_delay() == 123e-9
        # Ensure that [almost] the entire device is reconstructed correctly.
        assert original_snapshot == new_snapshot

    def test_device_element_deserialization_from_file(self, transmon_element_with_params, tmp_path):
        original_qubit = transmon_element_with_params
        original_qubit_json_file = original_qubit.to_json_file(str(tmp_path))
        original_snapshot = original_qubit.snapshot()
        original_qubit.close()

        new_qubit = BasicTransmonElement.from_json_file(original_qubit_json_file)
        new_snapshot = new_qubit.snapshot()

        self.normalize_snapshot(original_snapshot)
        self.normalize_snapshot(new_snapshot)

        # Ensure that the modified parameter is conserved upon deserialization.
        assert new_qubit.measure.acq_delay() == 123e-9
        # Ensure that [almost] the entire device is reconstructed correctly.
        assert original_snapshot == new_snapshot

    @classmethod
    def normalize_snapshot(cls, snap: dict) -> None:
        """
        Normalize a test snapshot in-place to allow for "soft" comparisons.
        Sets the following parameter to `mock.ANY`, which passes any equality check:
        - "ts", which is missing from serialization and is restored using `now()`, thus failing
          assertions on clock drift;
        - "value" or "raw_value", which are often `math.nan`, a non-comparable value.
        It also converts `np.ndarray` values to builtin `list`, to avoid a `ValueError` thrown
        by numpy on comparison.
        """
        for k, v in snap.items():
            if k == "ts" or (isinstance(v, float) and math.isnan(v)):
                snap[k] = mock.ANY
            elif isinstance(v, np.ndarray):
                snap[k] = list(v)
            elif isinstance(v, dict):
                cls.normalize_snapshot(snap[k])
