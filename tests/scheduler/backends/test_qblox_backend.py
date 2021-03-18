import os
import inspect
import json
import tempfile
import pytest

from qcodes.instrument.base import Instrument

from quantify.data.handling import set_datadir

from quantify.scheduler.types import Schedule
from quantify.scheduler.gate_library import Reset, Measure, Rxy, X
from quantify.scheduler.pulse_library import SquarePulse, DRAGPulse, RampPulse
from quantify.scheduler.resources import ClockResource
from quantify.scheduler.compilation import qcompile, determine_absolute_timing

from quantify.scheduler.backends.qblox_backend import hardware_compile

import quantify.scheduler.schemas.examples as es

esp = inspect.getfile(es)

cfg_f = os.path.abspath(os.path.join(esp, "..", "transmon_test_config.json"))
with open(cfg_f, "r") as f:
    DEVICE_CFG = json.load(f)

map_f = os.path.abspath(os.path.join(esp, "..", "qblox_test_mapping.json"))
with open(map_f, "r") as f:
    HARDWARE_MAPPING = json.load(f)


try:
    from pulsar_qcm.pulsar_qcm import pulsar_qcm_dummy
    from pulsar_qrm.pulsar_qrm import pulsar_qrm_dummy

    PULSAR_ASSEMBLER = True
except ImportError:
    PULSAR_ASSEMBLER = False


@pytest.fixture
def dummy_pulsars():
    if PULSAR_ASSEMBLER:
        _pulsars = []
        for qcm in ["qcm0", "qcm1"]:
            _pulsars.append(pulsar_qcm_dummy(qcm))
        for qrm in ["qrm0", "qrm1"]:
            _pulsars.append(pulsar_qrm_dummy(qrm))
    else:
        _pulsars = []

    # ensures a temporary datadir is used which is excluded from git
    tmp_dir = tempfile.TemporaryDirectory()
    set_datadir(tmp_dir.name)
    yield _pulsars

    # teardown
    for instr_name in list(Instrument._all_instruments):
        try:
            inst = Instrument.find_instrument(instr_name)
            inst.close()
        except KeyError:
            pass


def test_simple_compile():
    sched = Schedule("pulse_only_experiment")
    sched.add(
        DRAGPulse(
            G_amp=0.7,
            D_amp=-0.2,
            phase=90,
            port="q0:mw",
            duration=20e-9,
            clock="q0.01",
            t0=4e-9,
        )
    )
    sched.add(RampPulse(amp=0.5, duration=28e-9, port="q0:mw", clock="q0.01"))
    # Clocks need to be manually added at this stage.
    sched.add_resources([ClockResource("q0.01", freq=5e9)])
    determine_absolute_timing(sched)
    qcompile(sched, DEVICE_CFG, HARDWARE_MAPPING)
