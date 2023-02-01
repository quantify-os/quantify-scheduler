```{highlight} shell
```

# Installation

## Stable release

To install Quantify-Scheduler follow the {doc}`installation guide of quantify-core <quantify-core:installation>`.

## Update to latest version

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

Contributions are very welcome! To setup a an environment for local development see the instructions in the {doc}`installation guide of quantify-core <quantify-core:installation>`. You only need to replace {code}`quantify-core` with {code}`quantify-scheduler` in the provided commands.

If you need any help reach out to us by [creating a new issue](https://gitlab.com/quantify-os/quantify-scheduler/-/issues).

## Jupyter and plotly

Quantify-scheduler uses the [ploty] graphing framework for some components, which can require some additional set-up
to run with a Jupyter environment - please see [this page for details.]

[ploty]: https://plotly.com/
[this page for details.]: https://plotly.com/python/getting-started/
