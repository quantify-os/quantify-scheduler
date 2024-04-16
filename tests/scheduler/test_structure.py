# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch


from quantify_scheduler.structure import DataStructure


def test_serializable():
    class ADevice(DataStructure):
        a_field: int
        another_field: str = "Foo"

    class ADeviceSet(DataStructure):
        a_device: ADevice
        a_param: str

    a_device_set = ADeviceSet(a_device=ADevice(a_field=42), a_param="towel")
    serialized = a_device_set.model_dump_json()
    deserialzed = a_device_set.model_validate_json(serialized)
    assert a_device_set == deserialzed
