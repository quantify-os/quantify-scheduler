# About Quantify-scheduler

```{image} /images/QUANTIFY_LANDSCAPE.svg
:class: only-light
```
```{image} /images/QUANTIFY_LANDSCAPE_DM.svg
:class: only-dark
```

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
