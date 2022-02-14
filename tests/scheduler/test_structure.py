# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=too-few-public-methods
from quantify_scheduler.structure import DataStructure


def test_serializable():
    class ADevice(DataStructure):
        a_field: int
        another_field: str = "Foo"

    class ADeviceSet(DataStructure):
        a_device: ADevice
        a_param: str

    a_device_set = ADeviceSet(a_device=ADevice(a_field=42), a_param="towel")
    serialized = a_device_set.json()
    deserialzed = a_device_set.parse_raw(serialized)
    assert a_device_set == deserialzed
