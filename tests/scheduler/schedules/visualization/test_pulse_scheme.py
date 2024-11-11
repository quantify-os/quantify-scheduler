import matplotlib.pyplot as plt
import pytest
from matplotlib.figure import Figure

import quantify_scheduler.schedules._visualization.pulse_scheme as pls

cm = 1 / 2.54  # inch to cm conversion


@pytest.mark.mpl_image_compare(style="default", savefig_kwargs={"dpi": 300})
def test_plot_pulses_single_q() -> Figure:
    """
    Generates figure for testing
    """
    fig, ax = pls.new_pulse_fig((7 * cm, 3 * cm))

    # Plot pulses
    p1 = pls.mw_pulse(ax, 0, width=1.5, label="$X_{\\pi/2}$")
    p2 = pls.ram_Z_pulse(ax, p1, width=2.5, sep=1.5)

    # Add some arrows and labeling
    pls.interval(ax, p1, p1 + 1.5, height=1.7, label="$T_\\mathsf{p}$")
    pls.interval(ax, p1, p2 + 0.5, height=-0.6, label_height=-0.5, label="$\\tau$", vlines=False)

    # Adjust plot range to fit the whole figure
    ax.set_ylim(-1.2, 2.5)
    return fig


@pytest.mark.mpl_image_compare(style="default", savefig_kwargs={"dpi": 300})
def test_plot_pulses_n_q() -> Figure:
    """
    Generates figure for testing
    """

    # Two-qubit pulse scheme (Grover's algorithm)
    fig = plt.figure(figsize=(9 * cm, 5 * cm))
    lab_height = 1.25

    ax1 = pls.new_pulse_subplot(fig, 211)
    p1 = pls.mw_pulse(ax1, 0, label="$G_0$", label_height=lab_height)
    p2 = pls.flux_pulse(ax1, p1, label="CZ")
    p3 = pls.mw_pulse(ax1, p2, label="$Y_{\\pi/2}$", label_height=lab_height)
    p4 = pls.flux_pulse(ax1, p3, label="CZ")
    pls.mw_pulse(ax1, p4, label="$Y_{\\pi/2}$", label_height=lab_height)

    ax1.text(-0.5, 0, "$Q_0$", va="center", ha="right")

    ax2 = pls.new_pulse_subplot(fig, 212, sharex=ax1, sharey=ax1)
    pls.mw_pulse(ax2, 0, label="$G_1$", label_height=lab_height)
    pls.mw_pulse(ax2, p2, label="$Y_{\\pi/2}$", label_height=lab_height)
    pls.mw_pulse(ax2, p4, label="$Y_{\\pi/2}$", label_height=lab_height)

    ax2.text(-0.5, 0, "$Q_1$", va="center", ha="right")

    fig.subplots_adjust(left=0.07, top=0.9, hspace=0.1)
    return fig
