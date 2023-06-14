import numpy as np

from quantify_scheduler.structure import NDArray, DataStructure


def test_ndarray():
    class Model(DataStructure):
        a_string: str
        an_array: NDArray

    instance = Model(a_string="foo", an_array=[1 + 1j, 101])  # type: ignore
    assert isinstance(instance.an_array, NDArray)
    assert instance.an_array.dtype == np.complex_

    serialized = instance.json()
    deserialized = Model.parse_raw(serialized)

    np.testing.assert_equal(instance.an_array, deserialized.an_array)
