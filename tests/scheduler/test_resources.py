from quantify.scheduler.resources import PortResource, ClockResource, BasebandClockResource


def test_PortResource():
    # port associated with qubit
    port = PortResource('q0:mw')
    assert port.data['name'] == 'q0:mw'

    # port 5
    port = PortResource('p5')
    assert port.data['name'] == 'p5'


def test_ClockResource():
    # clock associated with qubit
    clock = ClockResource('q0:cl:01', freq=6.5e9, phase=23.9)
    assert clock.data['name'] == 'q0:cl:01'
    assert clock.data['freq'] == 6.5e9
    assert clock.data['phase'] == 23.9

    # clock 3
    clock = ClockResource('cl3', freq=4.5e9)
    assert clock.data['name'] == 'cl3'
    assert clock.data['freq'] == 4.5e9
    assert clock.data['phase'] == 0


def test_BasebandClockResource():
    # clock associated with qubit
    clock = BasebandClockResource('baseband')
    assert clock.data['name'] == 'baseband'
    assert clock.data['freq'] == 0


# def test_pulsar_address():
#     for res in [Pulsar_QRM_sequencer, Pulsar_QCM_sequencer]:
#         dev = res("my_device.bananas", seq_idx=0)
#         assert dev.data['name'] == "my_device.bananas"
#         assert dev.data['instrument_name'] == "my_device"
