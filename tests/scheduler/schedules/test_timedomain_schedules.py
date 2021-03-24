import inspect
import os
import json
import tempfile
import quantify.scheduler.schemas.examples as es
from quantify.scheduler.schedules import timedomain_schedules as ts
from quantify.scheduler.compilation import determine_absolute_timing, qcompile
from quantify.data.handling import set_datadir

tmp_dir = tempfile.TemporaryDirectory()
set_datadir(tmp_dir.name)

esp = inspect.getfile(es)

cfg_f = os.path.abspath(os.path.join(esp, "..", "transmon_test_config.json"))
with open(cfg_f, "r") as f:
    DEVICE_CFG = json.load(f)

map_f = os.path.abspath(os.path.join(esp, "..", "qblox_test_mapping.json"))
with open(map_f, "r") as f:
    HARDWARE_MAPPING = json.load(f)



def test_rabi_schedule():
    pass