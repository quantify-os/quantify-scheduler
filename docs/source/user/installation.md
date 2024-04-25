---
file_format: mystnb
kernelspec:
    name: python3

---
```{highlight} shell
```

# Installation

## Stable release

To install `quantify-scheduler` follow the {ref}`installation guide of quantify-core <quantify-core:installation>`.

(zhinst-backend-install)=
## Optional zhinst backend

The `zhinst` backend is not installed by default. To install this optional dependency, please run

```console
$ pip install quantify-scheduler[zhinst]
```

### additional requirements

The zhinst backend is currently compatible with Python versions `3.8` and `3.9`. When importing anything from the `zhinst` backend, an error is raised when either the backend is not installed, or when using an incorrect Python version:

```{code-cell} ipython3
---
mystnb:
  remove_code_source: true
---

from unittest.mock import patch, MagicMock

with patch("importlib.util.find_spec", return_value=None):
    try:
        from quantify_scheduler.backends.zhinst import settings
    except ModuleNotFoundError as error:
        print(f"{error.__class__.__name__}: {error}")
```

```{code-cell} ipython3
---
mystnb:
  remove_code_source: true
---

def mock_ge(self, other):
    other_major, other_minor = other[:2]
    if self.major > other_major:
        return True
    elif self.major == other_major:
        return self.minor >= other_minor
    else:
        return False

version_info_mock = MagicMock()
version_info_mock.major = 3
version_info_mock.minor = 10
version_info_mock.__ge__ = mock_ge

with patch("sys.version_info", version_info_mock):
    try:
        from quantify_scheduler.backends.zhinst import settings
    except RuntimeError as error:
        print(f"{error.__class__.__name__}: {error}")
```



## Update to the latest version

To update to the latest version

```console
$ pip install --upgrade quantify-scheduler
```

## From sources

The sources for `quantify-scheduler` can be downloaded from the [GitLab repo](https://gitlab.com/quantify-os/quantify-scheduler).

You can clone the public repository:

```console
$ git clone git@gitlab.com:quantify-os/quantify-scheduler.git
$ # or if you prefer to use https:
$ # git clone https://gitlab.com/quantify-os/quantify-scheduler.git/
```

Once you have a copy of the source, you can install it with:

```console
$ python -m pip install --upgrade .
```

## Setting up for local development

In order to develop the code locally, the package can be installed in the "editable mode" with the `-e` flag. `[dev]` optional requirement set will pull all (necessary and recommended) development requirements:

```console
$ python -m pip install -e ".[dev]"
```

Contributions are very welcome! To set up an environment for local development see the instructions in the {ref}`installation guide of quantify-core <quantify-core:installation>`. You only need to replace `quantify-core` with `quantify-scheduler` in the provided commands.

If you need any help reach out to us by [creating a new issue](https://gitlab.com/quantify-os/quantify-scheduler/-/issues).

## Jupyter and Plotly

`quantify-scheduler` uses the [ploty] graphing framework for some components, which can require some additional set-up
to run with a Jupyter environment - please see [this page for details.]

[ploty]: https://plotly.com/
[this page for details.]: https://plotly.com/python/getting-started/
