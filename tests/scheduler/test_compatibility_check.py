import pytest

from quantify_scheduler.compatibility_check import check_zhinst_compatibility


def mock_ge(self, other):
    other_major, other_minor = other[:2]
    if self.major > other_major:
        return True
    elif self.major == other_major:
        return self.minor >= other_minor
    else:
        return False


def test_incompatible_python_version(mocker):
    version_info_mock = mocker.MagicMock()
    version_info_mock.major = 3
    version_info_mock.minor = 10
    version_info_mock.__ge__ = mock_ge
    mocker.patch("sys.version_info", version_info_mock)

    with pytest.raises(RuntimeError) as exc_info:
        check_zhinst_compatibility()
    assert "The zhinst backend is only compatible with Python 3.8 and Python 3.9" in str(
        exc_info.value
    )


def test_missing_zhinst_module(mocker):
    version_info_mock = mocker.MagicMock()
    version_info_mock.major = 3
    version_info_mock.minor = 9
    version_info_mock.__ge__ = mock_ge
    mocker.patch("sys.version_info", version_info_mock)

    mocker.patch("importlib.util.find_spec", return_value=None)

    with pytest.raises(ModuleNotFoundError) as exc_info:
        check_zhinst_compatibility()
    assert "Please install the zhinst backend" in str(exc_info.value)
