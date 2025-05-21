import pytest

from quantify_scheduler.backends.qblox.instrument_compilers import ClusterCompiler
from quantify_scheduler.backends.qblox_backend import ChannelPath, _ClusterCompilationConfig
from quantify_scheduler.backends.types.qblox import (
    ClusterDescription,
    DigitizationThresholds,
    ExternalTriggerSyncSettings,
    QbloxHardwareOptions,
    QRMDescription,
    QTMDescription,
)


def test_validate_sync_on_external_trigger_cmm_success():
    compiler = ClusterCompiler(
        name="cluster",
        total_play_time=1e-6,
        instrument_cfg=_ClusterCompilationConfig(
            hardware_description=ClusterDescription(
                instrument_type="Cluster",
                ref="internal",
                sync_on_external_trigger=ExternalTriggerSyncSettings(
                    slot=0,
                    channel=1,
                ),
            ),
            hardware_options=QbloxHardwareOptions(),
            parent_config_version="v0.1",
        ),
    )

    compiler._validate_external_trigger_sync()


def test_validate_sync_on_external_trigger_qtm_wrong_module_type():
    module_idx = 10
    compiler = ClusterCompiler(
        name="cluster",
        total_play_time=1e-6,
        instrument_cfg=_ClusterCompilationConfig(
            hardware_description=ClusterDescription(
                instrument_type="Cluster",
                ref="internal",
                modules={module_idx: QRMDescription(instrument_type="QRM")},
                sync_on_external_trigger=ExternalTriggerSyncSettings(
                    slot=module_idx,
                    channel=1,
                ),
            ),
            hardware_options=QbloxHardwareOptions(),
            parent_config_version="v0.1",
        ),
    )

    with pytest.raises(
        ValueError,
        match=f"Slot {module_idx} specified in the `sync_on_external_trigger` settings "
        "contains a module that is not a QTM or CMM. External trigger synchronization "
        "only works with these two module types.",
    ):
        compiler._validate_external_trigger_sync()


def test_validate_sync_on_external_trigger_doubly_defined_channel_wrong_type():
    module_idx = 10
    channel_idx = 0
    path = f"cluster.module{module_idx}.digital_output_{channel_idx}"
    compiler = ClusterCompiler(
        name="cluster",
        total_play_time=1e-6,
        instrument_cfg=_ClusterCompilationConfig(
            hardware_description=ClusterDescription(
                instrument_type="Cluster",
                ref="internal",
                modules={module_idx: QTMDescription(instrument_type="QTM")},
                sync_on_external_trigger=ExternalTriggerSyncSettings(
                    slot=module_idx,
                    channel=channel_idx + 1,
                ),
            ),
            hardware_options=QbloxHardwareOptions(),
            portclock_to_path={"some:port-some.clock": ChannelPath.from_path(path)},
            parent_config_version="v0.1",
        ),
    )

    with pytest.raises(
        ValueError,
        match=f"Slot {module_idx} channel {channel_idx} is present in the "
        f"connectivity as {path}, which is not a 'digital_input'.",
    ):
        compiler._validate_external_trigger_sync()


def test_validate_sync_on_external_trigger_doubly_defined_channel_correct_type():
    module_idx = 10
    path = f"cluster.module{module_idx}.digital_input_0"
    compiler = ClusterCompiler(
        name="cluster",
        total_play_time=1e-6,
        instrument_cfg=_ClusterCompilationConfig(
            hardware_description=ClusterDescription(
                instrument_type="Cluster",
                ref="internal",
                modules={module_idx: QTMDescription(instrument_type="QTM")},
                sync_on_external_trigger=ExternalTriggerSyncSettings(
                    slot=module_idx,
                    channel=1,
                    input_threshold=1.0,
                ),
            ),
            hardware_options=QbloxHardwareOptions(),
            portclock_to_path={"some:port-some.clock": ChannelPath.from_path(path)},
            parent_config_version="v0.1",
        ),
    )

    compiler._validate_external_trigger_sync()


def test_validate_sync_on_external_trigger_doubly_defined_channel_diff_threshold():
    module_idx = 10
    path = f"cluster.module{module_idx}.digital_input_0"
    compiler = ClusterCompiler(
        name="cluster",
        total_play_time=1e-6,
        instrument_cfg=_ClusterCompilationConfig(
            hardware_description=ClusterDescription(
                instrument_type="Cluster",
                ref="internal",
                modules={module_idx: QTMDescription(instrument_type="QTM")},
                sync_on_external_trigger=ExternalTriggerSyncSettings(
                    slot=module_idx,
                    channel=1,
                    input_threshold=1.0,
                ),
            ),
            hardware_options=QbloxHardwareOptions(
                digitization_thresholds={
                    "some:port-some.clock": DigitizationThresholds(analog_threshold=0.5)
                }
            ),
            portclock_to_path={"some:port-some.clock": ChannelPath.from_path(path)},
            parent_config_version="v0.1",
        ),
    )

    with pytest.raises(
        ValueError,
        match=f"Channel {path} has an associated 'analog_threshold="
        "0.5' "
        "which is different from 'sync_on_external_trigger.input_threshold="
        "1.0'",
    ):
        compiler._validate_external_trigger_sync()


def test_validate_sync_on_external_trigger_doubly_defined_channel_no_threshold():
    module_idx = 10
    path = f"cluster.module{module_idx}.digital_input_0"
    compiler = ClusterCompiler(
        name="cluster",
        total_play_time=1e-6,
        instrument_cfg=_ClusterCompilationConfig(
            hardware_description=ClusterDescription(
                instrument_type="Cluster",
                ref="internal",
                modules={module_idx: QTMDescription(instrument_type="QTM")},
                sync_on_external_trigger=ExternalTriggerSyncSettings(
                    slot=module_idx,
                    channel=1,
                ),
            ),
            hardware_options=QbloxHardwareOptions(),
            portclock_to_path={"some:port-some.clock": ChannelPath.from_path(path)},
            parent_config_version="v0.1",
        ),
    )

    with pytest.raises(
        ValueError,
        match=f"No input threshold was set for {path}. Please specify an input "
        "threshold, either via the 'sync_on_external_trigger' settings or the "
        "hardware options.",
    ):
        compiler._validate_external_trigger_sync()
