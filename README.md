# quantify-scheduler

[![Slack](https://img.shields.io/badge/slack-chat-green.svg)](https://quantify-hq.slack.com/join/shared_invite/zt-1nd78r4e9-rbWdna53cW4DO_YbtMhVuA)
[![Pipelines](https://gitlab.com/quantify-os/quantify-scheduler/badges/main/pipeline.svg)](https://gitlab.com/quantify-os/quantify-scheduler/-/pipelines/)
[![PyPi](https://img.shields.io/pypi/v/quantify-scheduler.svg)](https://pypi.org/project/quantify-scheduler)
[![Code Quality](https://app.codacy.com/project/badge/Grade/0c9cf5b6eb5f47ffbd2bb484d555c7e3)](https://www.codacy.com/gl/quantify-os/quantify-scheduler/dashboard?utm_source=gitlab.com&amp;utm_medium=referral&amp;utm_content=quantify-os/quantify-scheduler&amp;utm_campaign=Badge_Grade)
[![Coverage](https://app.codacy.com/project/badge/Coverage/0c9cf5b6eb5f47ffbd2bb484d555c7e3)](https://www.codacy.com/gl/quantify-os/quantify-scheduler/dashboard?utm_source=gitlab.com&amp;utm_medium=referral&amp;utm_content=quantify-os/quantify-scheduler&amp;utm_campaign=Badge_Coverage)
[![Documentation Status](https://readthedocs.com/projects/quantify-quantify-scheduler/badge/?version=latest&token=ed6fdbf228e1369eacbeafdbad464f6de927e5dfb3a8e482ad0adcbea76fe74c)](https://quantify-quantify-scheduler.readthedocs-hosted.com)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://gitlab.com/quantify-os/quantify-scheduler/-/raw/main/LICENSE)
[![Code style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Unitary Fund](https://img.shields.io/badge/Supported%20By-UNITARY%20FUND-brightgreen.svg?style=flat)](https://unitary.fund)

![Quantify logo](https://orangeqs.com/logos/QUANTIFY_LANDSCAPE.svg)

Quantify is a python based data acquisition platform focused on Quantum Computing and solid-state physics experiments.
It is built on top of [QCoDeS](https://qcodes.github.io/Qcodes/) and is a spiritual successor of [PycQED](https://github.com/DiCarloLab-Delft/PycQED_py3).
Quantify currently consists of [quantify-core](https://pypi.org/project/quantify-core/) and [quantify-scheduler](https://pypi.org/project/quantify-scheduler/).

Take a look at the [latest documentation for quantify-scheduler](https://quantify-quantify-scheduler.readthedocs-hosted.com/) or use the switch at the bottom of the left panel to read the documentation for older releases.

Quantify-scheduler is a python module for writing quantum programs featuring a hybrid gate-pulse control model with explicit timing control.
The control model allows quantum gate- and pulse-level descriptions to be combined in a clearly defined and hardware-agnostic way.
Quantify-scheduler is designed to allow experimentalists to easily define complex experiments, and produces synchronized pulse schedules to be distributed to control hardware.


## About

Quantify-scheduler is maintained by The Quantify consortium consisting of Qblox and Orange Quantum Systems.


[<img src="https://cdn.sanity.io/images/ostxzp7d/production/f9ab429fc72aea1b31c4b2c7fab5e378b67d75c3-132x31.svg" alt="Qblox logo" width=200px/>](https://www.qblox.com)
&nbsp;
&nbsp;
&nbsp;
&nbsp;
[<img src="https://orangeqs.com/OQS_logo_with_text.svg" alt="Orange Quantum Systems logo" width=200px/>](https://orangeqs.com)

&nbsp;

The software is free to use under the conditions specified in the [license](https://gitlab.com/quantify-os/quantify-scheduler/-/raw/main/LICENSE).
