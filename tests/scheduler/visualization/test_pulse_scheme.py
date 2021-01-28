import matplotlib.pyplot as plt
import quantify.scheduler.visualization.pulse_scheme as pls
import pytest
from quantify.scheduler import Schedule
from quantify.scheduler.gate_library import Reset, Measure, Rxy
from quantify.scheduler.compilation import qcompile
import inspect
import json
import os
import quantify.scheduler.schemas.examples as es

esp = inspect.getfile(es)
cfg_f = os.path.abspath(os.path.join(esp, "..", "transmon_test_config.json"))
with open(cfg_f, "r") as f:
    DEVICE_CFG = json.load(f)


cm = 1 / 2.54  # inch to cm conversion


@pytest.mark.mpl_image_compare(style="default", savefig_kwargs={"dpi": 300})
def test_plot_pulses_single_q():
    fig, ax = pls.new_pulse_fig((7 * cm, 3 * cm))

    # Plot pulses
    p1 = pls.mwPulse(ax, 0, width=1.5, label="$X_{\\pi/2}$")
    p2 = pls.ramZPulse(ax, p1, width=2.5, sep=1.5)

    # Add some arrows and labeling
    pls.interval(ax, p1, p1 + 1.5, height=1.7, label="$T_\\mathsf{p}$")
    pls.interval(
        ax, p1, p2 + 0.5, height=-0.6, label_height=-0.5, label="$\\tau$", vlines=False
    )

    # Adjust plot range to fit the whole figure
    ax.set_ylim(-1.2, 2.5)
    return fig


@pytest.mark.mpl_image_compare(style="default", savefig_kwargs={"dpi": 300})
def test_plot_pulses_n_q():
    # Two-qubit pulse scheme (Grover's algorithm)
    fig = plt.figure(figsize=(9 * cm, 5 * cm))
    labHeight = 1.25

    ax1 = pls.new_pulse_subplot(fig, 211)
    p1 = pls.mwPulse(ax1, 0, label="$G_0$", label_height=labHeight)
    p2 = pls.fluxPulse(ax1, p1, label="CZ")
    p3 = pls.mwPulse(ax1, p2, label="$Y_{\\pi/2}$", label_height=labHeight)
    p4 = pls.fluxPulse(ax1, p3, label="CZ")
    pls.mwPulse(ax1, p4, label="$Y_{\\pi/2}$", label_height=labHeight)

    ax1.text(-0.5, 0, "$Q_0$", va="center", ha="right")

    ax2 = pls.new_pulse_subplot(fig, 212, sharex=ax1, sharey=ax1)
    pls.mwPulse(ax2, 0, label="$G_1$", label_height=labHeight)
    pls.mwPulse(ax2, p2, label="$Y_{\\pi/2}$", label_height=labHeight)
    pls.mwPulse(ax2, p4, label="$Y_{\\pi/2}$", label_height=labHeight)

    ax2.text(-0.5, 0, "$Q_1$", va="center", ha="right")

    fig.subplots_adjust(left=0.07, top=0.9, hspace=0.1)
    return fig


# todo, verification of this, probably requires some horrible selenium malarkey
def test_pulse_diagram_plotly():
    sched = Schedule("Test schedule")

    # define the resources
    q0, q1 = ("q0", "q1")
    sched.add(Reset(q0, q1))
    sched.add(Rxy(90, 0, qubit=q0))
    # sched.add(operation=CZ(qC=q0, qT=q1)) # not implemented in config
    sched.add(Rxy(theta=90, phi=0, qubit=q0))
    sched.add(Measure(q0, q1), label="M0")
    # pulse information is added
    sched = qcompile(sched, DEVICE_CFG, None)

    # It should be possible to generate this visualization after compilation
    _ = pls.pulse_diagram_plotly(sched)
    # and with auto labels
    _ = pls.pulse_diagram_plotly(sched)
