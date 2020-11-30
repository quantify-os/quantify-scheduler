from quantify.scheduler.resources import PortResource, ClockResource


def test_PortResource():
    # port associated with qubit
    port = PortResource('q0:mw')
    assert port.data['name'] == 'q0:mw'

    # port 5
    port = PortResource('p5')
    assert port.data['name'] == 'p5'


def test_ClockResource():
    # clock associated with qubit
    port = ClockResource('q0:cl:01')
    assert port.data['name'] == 'q0:cl:01'

    # clock 3
    port = ClockResource('cl3')
    assert port.data['name'] == 'cl3'


# def test_pulsar_address():
#     for res in [Pulsar_QRM_sequencer, Pulsar_QCM_sequencer]:
#         dev = res("my_device.bananas", seq_idx=0)
#         assert dev.data['name'] == "my_device.bananas"
#         assert dev.data['instrument_name'] == "my_device"
