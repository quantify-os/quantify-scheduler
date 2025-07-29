from qblox_instruments import Cluster, ClusterType
from qblox_instruments.qcodes_drivers.time import Polarity, SyncRef

from quantify_scheduler.backends.types.qblox import ExternalTriggerSyncSettings
from quantify_scheduler.instrument_coordinator.components.qblox import ClusterComponent
from quantify_scheduler.schedules.schedule import AcquisitionChannelsData


def test_sync_on_external_trigger_success_cmm(mocker):
    cluster = Cluster("cluster", dummy_cfg={0: ClusterType._CLUSTER_MM})
    mocker.patch.object(cluster.time, "sync_ext_trigger")
    component = ClusterComponent(cluster)
    # TODO more settings
    component.prepare(
        {
            "settings": {
                "reference_source": "internal",
                "sync_on_external_trigger": ExternalTriggerSyncSettings(
                    slot=0, channel=1
                ).to_dict(),
            }
        }
    )
    component.start()

    cluster.time.sync_ext_trigger.assert_called_once_with(
        slot=0,
        channel=1,
        trigger_timestamp=0,
        timeout=1.0,
        format="s",
        edge_polarity=Polarity.RISING_EDGE,
        sync_ref=SyncRef.OFF,
    )


def test_sync_on_external_trigger_success_qtm(mocker):
    cluster = Cluster("cluster", dummy_cfg={10: ClusterType.CLUSTER_QTM})
    mocker.patch.object(cluster.time, "sync_ext_trigger")
    mocker.patch.object(cluster.module10.io_channel0.analog_threshold, "set")
    mocker.patch.object(cluster.module10.io_channel0.mode, "set")
    component = ClusterComponent(cluster)
    # TODO more settings
    component.prepare(
        {
            "settings": {
                "reference_source": "internal",
                "sync_on_external_trigger": ExternalTriggerSyncSettings(
                    slot=10, channel=1, input_threshold=0.5
                ).to_dict(),
            }
        }
    )
    component.start()

    cluster.time.sync_ext_trigger.assert_called_once_with(
        slot=10,
        channel=1,
        trigger_timestamp=0,
        timeout=1.0,
        format="s",
        edge_polarity=Polarity.RISING_EDGE,
        sync_ref=SyncRef.OFF,
    )
    cluster.module10.io_channel0.analog_threshold.set.assert_called_once_with(0.5)
    cluster.module10.io_channel0.mode.set.assert_called_once_with("input")
