# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
import json
import os.path

import pytest

from quantify_scheduler.device_under_test.hardware_config import HardwareConfig

# Hardware config to test against,
# might fail the tests once validation is added, then just change this
hardware_config = {"a": 4}
hardware_config2 = {"b": 6}


class TestWriteToFile:

    def test_file_does_not_exist(self, tmp_path):
        file_path = tmp_path / "tmp.json"
        assert not os.path.exists(file_path)
        config = HardwareConfig(hardware_config)
        config.write_to_json_file(file_path)
        assert os.path.exists(file_path)
        with file_path.open("r") as file:
            assert json.load(file) == hardware_config

    def test_file_does_exist(self, tmp_path):
        file_path = tmp_path / "tmp.json"
        path = file_path
        assert not path.exists()
        path.touch()
        assert path.exists()
        config = HardwareConfig(hardware_config)
        config.write_to_json_file(file_path)
        assert os.path.exists(file_path)
        with path.open("r") as file:
            assert json.load(file) == hardware_config

    def test_file_not_json_extension(self, tmp_path):
        file_path = tmp_path / "not_json.py"
        assert not os.path.exists(file_path)
        config = HardwareConfig(hardware_config)
        config.write_to_json_file(file_path)
        assert os.path.exists(file_path)
        with file_path.open("r") as file:
            assert json.load(file) == hardware_config

    def test_file_is_a_directory(self, tmp_path):
        config = HardwareConfig(hardware_config)
        with pytest.raises(IsADirectoryError):
            config.write_to_json_file(tmp_path)

    def test_filename_is_saved(self, tmp_path):
        file_path = tmp_path / "tmp.json"
        config = HardwareConfig()
        config.write_to_json_file(file_path)
        with file_path.open("r") as file:
            assert json.load(file) is None
        config.set(hardware_config)
        config.write_to_json_file(file_path)
        with file_path.open("r") as file:
            assert json.load(file) == hardware_config


class TestLoadFromFile:

    def test_file_does_not_exist(self, tmp_path):
        file_path = tmp_path / "tmp.json"
        assert not os.path.exists(file_path)
        config = HardwareConfig()
        with pytest.raises(FileNotFoundError):
            config.load_from_json_file(file_path)

    def test_file_does_exist(self, tmp_path):
        file_path = tmp_path / "tmp.json"
        config1 = HardwareConfig()
        config2 = HardwareConfig(hardware_config)
        config2.write_to_json_file(file_path)
        assert config1.get() is None
        config1.load_from_json_file(file_path)
        assert config1.get() == hardware_config

    def test_file_is_a_directory(self, tmp_path):
        config = HardwareConfig(hardware_config)
        with pytest.raises(IsADirectoryError):
            config.load_from_json_file(tmp_path)
