# quantify-scheduler

[![Slack](https://img.shields.io/badge/slack-chat-green.svg)](https://quantify-os.org/slack.html#sec-slack)
[![Pipelines](https://gitlab.com/quantify-os/quantify-scheduler/badges/main/pipeline.svg)](https://gitlab.com/quantify-os/quantify-scheduler/-/pipelines/)
[![PyPi](https://img.shields.io/pypi/v/quantify-scheduler.svg)](https://pypi.org/project/quantify-scheduler)
[![Code Quality](https://app.codacy.com/project/badge/Grade/0c9cf5b6eb5f47ffbd2bb484d555c7e3)](https://app.codacy.com/gl/quantify-os/quantify-scheduler/dashboard?utm_source=gitlab.com&amp;utm_medium=referral&amp;utm_content=quantify-os/quantify-scheduler&amp;utm_campaign=Badge_Grade)
[![Coverage](https://app.codacy.com/project/badge/Coverage/0c9cf5b6eb5f47ffbd2bb484d555c7e3)](https://app.codacy.com/gl/quantify-os/quantify-scheduler/dashboard?utm_source=gitlab.com&amp;utm_medium=referral&amp;utm_content=quantify-os/quantify-scheduler&amp;utm_campaign=Badge_Coverage)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://gitlab.com/quantify-os/quantify-scheduler/-/raw/main/LICENSE)
[![Code style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Unitary Fund](https://img.shields.io/badge/Supported%20By-UNITARY%20FUND-brightgreen.svg?style=flat)](https://unitary.fund)
[![Documentation](https://img.shields.io/badge/documentation-grey)](https://quantify-os.org/docs/quantify-scheduler)

![Quantify logo](https://gitlab.com/quantify-os/quantify-scheduler/-/raw/main/docs/source/images/QUANTIFY_LANDSCAPE.svg)

Quantify is a Python-based data acquisition framework focused on Quantum Computing and
solid-state physics experiments.
The framework consists of [quantify-core](https://pypi.org/project/quantify-core/) ([git](https://gitlab.com/quantify-os/quantify-core/) | [docs](https://quantify-os.org/docs/quantify-core/))
and [quantify-scheduler](https://pypi.org/project/quantify-scheduler/) ([git](https://gitlab.com/quantify-os/quantify-scheduler/) | [docs](https://quantify-os.org/docs/quantify-scheduler/)).
It is built on top of [QCoDeS](https://microsoft.github.io/Qcodes/)
and is a spiritual successor of [PycQED](https://github.com/DiCarloLab-Delft/PycQED_py3).

`quantify-scheduler` is a Python module for writing quantum programs featuring a hybrid gate-pulse control model with explicit timing control.
This control model allows quantum gate and pulse-level descriptions to be combined in a clearly defined and hardware-agnostic way.
`quantify-scheduler` is designed to allow experimentalists to easily define complex experiments. It produces synchronized pulse schedules
that are distributed to control hardware, after compiling these schedules into control-hardware specific executable programs.

## Hardware/driver compatibility

**Qblox**

| quantify-scheduler |                      qblox-instruments                       |                               Cluster firmware                                |
|--------------------|:------------------------------------------------------------:|:-----------------------------------------------------------------------------:|
| v0.21.0            | [0.13.0](https://pypi.org/project/qblox-instruments/0.12.0/) | [0.8.0](https://gitlab.com/qblox/releases/cluster_releases/-/releases/v0.8.0) |
| v0.21.0            | [0.12.0](https://pypi.org/project/qblox-instruments/0.12.0/) | [0.7.0](https://gitlab.com/qblox/releases/cluster_releases/-/releases/v0.7.0) |
| v0.20.0            | [0.13.0](https://pypi.org/project/qblox-instruments/0.12.0/) | [0.8.0](https://gitlab.com/qblox/releases/cluster_releases/-/releases/v0.8.0) |
| v0.20.0            | [0.12.0](https://pypi.org/project/qblox-instruments/0.12.0/) | [0.7.0](https://gitlab.com/qblox/releases/cluster_releases/-/releases/v0.7.0) |
| v0.19.0            | [0.12.0](https://pypi.org/project/qblox-instruments/0.12.0/) | [0.7.0](https://gitlab.com/qblox/releases/cluster_releases/-/releases/v0.7.0) |
| v0.18.2            | [0.11.2](https://pypi.org/project/qblox-instruments/0.11.2/) | [0.6.2](https://gitlab.com/qblox/releases/cluster_releases/-/releases/v0.6.2) |
|                    | [0.11.1](https://pypi.org/project/qblox-instruments/0.11.1/) | [0.6.1](https://gitlab.com/qblox/releases/cluster_releases/-/releases/v0.6.1) |
|                    | [0.11.0](https://pypi.org/project/qblox-instruments/0.11.0/) | [0.6.0](https://gitlab.com/qblox/releases/cluster_releases/-/releases/v0.6.0) |
| v0.18.1            | [0.11.2](https://pypi.org/project/qblox-instruments/0.11.2/) | [0.6.2](https://gitlab.com/qblox/releases/cluster_releases/-/releases/v0.6.2) |
|                    | [0.11.1](https://pypi.org/project/qblox-instruments/0.11.1/) | [0.6.1](https://gitlab.com/qblox/releases/cluster_releases/-/releases/v0.6.1) |
|                    | [0.11.0](https://pypi.org/project/qblox-instruments/0.11.0/) | [0.6.0](https://gitlab.com/qblox/releases/cluster_releases/-/releases/v0.6.0) |
| v0.18.0            | [0.11.2](https://pypi.org/project/qblox-instruments/0.11.2/) | [0.6.2](https://gitlab.com/qblox/releases/cluster_releases/-/releases/v0.6.2) |
|                    | [0.11.1](https://pypi.org/project/qblox-instruments/0.11.1/) | [0.6.1](https://gitlab.com/qblox/releases/cluster_releases/-/releases/v0.6.1) |
|                    | [0.11.0](https://pypi.org/project/qblox-instruments/0.11.0/) | [0.6.0](https://gitlab.com/qblox/releases/cluster_releases/-/releases/v0.6.0) |
| v0.17.1            | [0.11.2](https://pypi.org/project/qblox-instruments/0.11.2/) | [0.6.2](https://gitlab.com/qblox/releases/cluster_releases/-/releases/v0.6.2) |
|                    | [0.11.1](https://pypi.org/project/qblox-instruments/0.11.1/) | [0.6.1](https://gitlab.com/qblox/releases/cluster_releases/-/releases/v0.6.1) |
|                    | [0.11.0](https://pypi.org/project/qblox-instruments/0.11.0/) | [0.6.0](https://gitlab.com/qblox/releases/cluster_releases/-/releases/v0.6.0) |
| v0.17.0            | [0.11.2](https://pypi.org/project/qblox-instruments/0.11.2/) | [0.6.2](https://gitlab.com/qblox/releases/cluster_releases/-/releases/v0.6.2) |
|                    | [0.11.1](https://pypi.org/project/qblox-instruments/0.11.1/) | [0.6.1](https://gitlab.com/qblox/releases/cluster_releases/-/releases/v0.6.1) |
|                    | [0.11.0](https://pypi.org/project/qblox-instruments/0.11.0/) | [0.6.0](https://gitlab.com/qblox/releases/cluster_releases/-/releases/v0.6.0) |
| v0.16.1            | [0.11.2](https://pypi.org/project/qblox-instruments/0.11.2/) | [0.6.2](https://gitlab.com/qblox/releases/cluster_releases/-/releases/v0.6.2) |
|                    | [0.11.1](https://pypi.org/project/qblox-instruments/0.11.1/) | [0.6.1](https://gitlab.com/qblox/releases/cluster_releases/-/releases/v0.6.1) |
|                    | [0.11.0](https://pypi.org/project/qblox-instruments/0.11.0/) | [0.6.0](https://gitlab.com/qblox/releases/cluster_releases/-/releases/v0.6.0) |
| v0.16.0            | [0.11.2](https://pypi.org/project/qblox-instruments/0.11.2/) | [0.6.2](https://gitlab.com/qblox/releases/cluster_releases/-/releases/v0.6.2) |
|                    | [0.11.1](https://pypi.org/project/qblox-instruments/0.11.1/) | [0.6.1](https://gitlab.com/qblox/releases/cluster_releases/-/releases/v0.6.1) |
|                    | [0.11.0](https://pypi.org/project/qblox-instruments/0.11.0/) | [0.6.0](https://gitlab.com/qblox/releases/cluster_releases/-/releases/v0.6.0) |
| v0.15.0            | [0.10.x](https://pypi.org/project/qblox-instruments/0.10.0/) | [0.5.0](https://gitlab.com/qblox/releases/cluster_releases/-/releases/v0.5.0) |
|                    |  [0.9.0](https://pypi.org/project/qblox-instruments/0.9.0/)  | [0.4.0](https://gitlab.com/qblox/releases/cluster_releases/-/releases/v0.4.0) |
| v0.14.0            | [0.10.x](https://pypi.org/project/qblox-instruments/0.10.0/) | [0.5.0](https://gitlab.com/qblox/releases/cluster_releases/-/releases/v0.5.0) |
|                    |  [0.9.0](https://pypi.org/project/qblox-instruments/0.9.0/)  | [0.4.0](https://gitlab.com/qblox/releases/cluster_releases/-/releases/v0.4.0) |
| v0.13.0            | [0.10.x](https://pypi.org/project/qblox-instruments/0.10.0/) | [0.5.0](https://gitlab.com/qblox/releases/cluster_releases/-/releases/v0.5.0) |
|                    |  [0.9.0](https://pypi.org/project/qblox-instruments/0.9.0/)  | [0.4.0](https://gitlab.com/qblox/releases/cluster_releases/-/releases/v0.4.0) |

**Zurich Instruments**
- `zhinst==21.8.20515`, `zhinst-qcodes==0.1.4`, `zhinst-toolkit==0.1.5`


## Overview and Community

For a general overview of Quantify and connecting to its open-source community, see [quantify-os.org](https://quantify-os.org/).
Quantify is maintained by the Quantify Consortium consisting of Qblox and Orange Quantum Systems.

[<img src="https://gitlab.com/quantify-os/quantify-scheduler/-/raw/main/docs/source/images/Qblox_logo.svg" alt="Qblox logo" width=200px/>](https://www.qblox.com)
&nbsp;
&nbsp;
&nbsp;
&nbsp;
[<img src="https://gitlab.com/quantify-os/quantify-scheduler/-/raw/main/docs/source/images/OQS_logo_with_text.svg" alt="Orange Quantum Systems logo" width=200px/>](https://orangeqs.com)

&nbsp;

The software is free to use under the conditions specified in the [license](https://gitlab.com/quantify-os/quantify-scheduler/-/raw/main/LICENSE).
