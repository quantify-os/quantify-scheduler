"""Module to prepare instruments and quantum device."""

from __future__ import annotations

import contextlib
import json
import os
import pathlib
from pathlib import Path
from typing import TYPE_CHECKING

from qblox_instruments import Cluster, ClusterType
from qcodes.instrument import find_or_create_instrument

from quantify_core.measurement.control import MeasurementControl
from quantify_scheduler import InstrumentCoordinator, QuantumDevice
from quantify_scheduler.backends.types.qblox import ClusterDescription
from quantify_scheduler.instrument_coordinator.components.qblox import ClusterComponent

if TYPE_CHECKING:
    from quantify_scheduler.backends.qblox_backend import QbloxHardwareCompilationConfig

DEFAULT_QUANTUM_DEVICE: QuantumDevice | None = None

STR_TO_DUMMY_TYPE = {
    "QCM": ClusterType.CLUSTER_QCM,
    "QRM": ClusterType.CLUSTER_QRM,
    "QCM_RF": ClusterType.CLUSTER_QCM_RF,
    "QRM_RF": ClusterType.CLUSTER_QRM_RF,
    "QTM": ClusterType.CLUSTER_QTM,
    "QDM": ClusterType.CLUSTER_QDM,
}


def initialize_hardware(
    quantum_device: QuantumDevice | None = None, ip: str | None = None
) -> tuple[MeasurementControl, InstrumentCoordinator, Cluster]:
    """
    Initialize MeasurementControl and InstrumentCoordinator from QuantumDevice.

    Parameters
    ----------
    quantum_device : QuantumDevice | None, optional
        target quantum device, by default None
    ip : str | None, optional
        ip address of the qblox cluster. Will use a dummy cluster if None, by default None
    live_plotting : bool, optional
        wether live plotting should be enabled, by default False

    Returns
    -------
    tuple[MeasurementControl, InstrumentCoordinator]

    Raises
    ------
    ValueError
        Neither QuantumDevice nor global default are provided.

    """
    if quantum_device is None:
        if DEFAULT_QUANTUM_DEVICE is None:
            raise ValueError("Either provide a QuantumDevice or set the global default")
        else:
            quantum_device = DEFAULT_QUANTUM_DEVICE
    config: QbloxHardwareCompilationConfig = quantum_device.generate_hardware_compilation_config()  # type: ignore
    all_instruments = []
    for name, instrument in config.hardware_description.items():
        if isinstance(instrument, ClusterDescription):
            if ip:
                instrument_ip = ip
            else:
                instrument_ip = getattr(instrument, "ip", None)

            if instrument_ip:
                dummy_config = None
            else:
                dummy_config = {
                    int(slot): STR_TO_DUMMY_TYPE[module.instrument_type]
                    for slot, module in instrument.modules.items()
                }
            cluster = find_or_create_instrument(
                Cluster,
                recreate=True,
                name=name,
                identifier=instrument_ip,
                dummy_cfg=dummy_config,
                debug=True,
            )
            all_instruments.append(cluster)

    meas_ctrl = find_or_create_instrument(MeasurementControl, recreate=True, name="meas_ctrl")
    quantum_device.instr_measurement_control(meas_ctrl.name)
    ic = find_or_create_instrument(InstrumentCoordinator, recreate=True, name="ic")
    quantum_device.instr_instrument_coordinator(ic.name)

    for cluster in all_instruments:
        # Add cluster to instrument coordinator
        ic_cluster = ClusterComponent(cluster)
        with contextlib.suppress(ValueError):
            ic.add_component(ic_cluster)

    return (meas_ctrl, ic, cluster)


def set_up_config() -> tuple[str | None, dict, Path | str]:
    """Provide the location of config file before runing an ip of the cluster."""
    ip = os.environ.get("CLUSTER_IP")
    hw_config_path = os.environ.get("HARDWARE_CONFIG_PATH")
    qdevice_path = os.environ.get("QUANTUM_DEVICE_CONFIG_PATH")
    p_root = pathlib.Path(__file__).parent.resolve()

    if hw_config_path is None:
        hw_config_path = p_root / "configs/config_file_with_each_modules.json"
    if qdevice_path is None:
        qdevice_path = p_root / "devices/transmon_device_2q.json"
    with open(hw_config_path) as hw_cfg_json_file:
        hardware_cfg = json.load(hw_cfg_json_file)
    return ip, hardware_cfg, qdevice_path
