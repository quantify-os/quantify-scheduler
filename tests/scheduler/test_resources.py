from quantify.scheduler.resources import Pulsar_QCM_sequencer, Pulsar_QRM_sequencer


def test_pulsar_address():
    for res in [Pulsar_QRM_sequencer, Pulsar_QCM_sequencer]:
        dev = res("my_device.bananas", seq_idx=0)
        assert dev.data['name'] == "my_device.bananas"
        assert dev.data['instrument_name'] == "my_device"
