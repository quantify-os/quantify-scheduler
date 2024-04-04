# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Helpers for Qblox dummy device."""

from qblox_instruments import SequencerStates

from quantify_scheduler.instrument_coordinator.components.qblox import ClusterComponent


def start_dummy_cluster_armed_sequencers(cluster_component: ClusterComponent) -> None:
    """
    Starting all armed sequencers in a dummy cluster.

    Starting all armed sequencers via Cluster.start_sequencer() doesn't yet
    work with dummy acquisition data (verified it does work on hardware).
    Hence, we need still need to call start_sequencer() for all sequencers separately.
    TODO: qblox_instruments.ieee488_2.cluster_dummy_transport.ClusterDummyTransport
    See SE-441.
    """
    for module in cluster_component._cluster_modules.values():
        for idx in range(module._hardware_properties.number_of_sequencers):
            status = module.instrument.get_sequencer_status(idx)
            if status.state is SequencerStates.ARMED:
                module.instrument.start_sequencer(idx)
