# -----------------------------------------------------------------------------
# Description:    Plotting functions used in the visualization backend of the sequencer.
# Repository:     https://gitlab.com/qblox/packages/software/quantify/
# Copyright (C) Qblox BV & Orange Quantum Systems Holding BV (2020)
# -----------------------------------------------------------------------------
import quantify.scheduler.visualization.pulse_scheme as ps
from quantify.scheduler.compilation import _determine_absolute_timing
from quantify.utilities.general import import_func_from_string


def gate_box(ax, time: float, qubit_idxs: list, tex: str, **kw):
    """
    A box for a single gate containing a label.
    """
    for qubit_idx in qubit_idxs:
        ps.box_text(ax, x0=time, y0=qubit_idx, text=tex, fillcolor='C0', w=.8, h=.5, **kw)


def meter(ax, time: float, qubit_idxs: list, tex: str, **kw):
    """
    A simple meter to depict a measurement.
    """
    for qubit_idx in qubit_idxs:
        ps.meter(ax, x0=time, y0=qubit_idx, fillcolor='C4', y_offs=0, w=.8, h=.5, **kw)


def cnot(ax, time: float, qubit_idxs: list, tex: str, **kw):
    """
    Markers to denote a CNOT gate between two qubits.
    """
    ax.plot([time, time], qubit_idxs, marker='o', markersize=15, color='C1')
    ax.plot([time], qubit_idxs[1], marker='+', markersize=12, color='white')


def cz(ax, time: float, qubit_idxs: list, tex: str, **kw):
    """
    Markers to denote a CZ gate between two qubits.
    """
    ax.plot([time, time], qubit_idxs, marker='o', markersize=15, color='C1')


def reset(ax, time: float, qubit_idxs: list, tex: str, **kw):
    """
    A broken line to denote qubit initialization.
    """
    for qubit_idx in qubit_idxs:
        ps.box_text(ax, x0=time, y0=qubit_idx, text=tex, color='white', fillcolor='white', w=.4, h=.5, **kw)


def circuit_diagram_matplotlib(schedule, figsize=None):
    """
    Creates a circuit diagram visualization of a schedule using matplotlib.

    For this visualization backend to work, the schedule must contain a value for `abs_time` for each element in the
    timing_constraints.

    Parameters
    ----------
    schedule : :class:`~quantify.scheduler.types.Schedule`
        the schedule to render.
    figsize : tuple
        matplotlib figsize.
    Returns
    -------
    tuple
        - matplotlib figure object.
        - matplotlib axis object.
    """
    schedule = _determine_absolute_timing(schedule, 'ideal')

    qubits = set()
    for _, op in schedule.operations.items():
        if op.valid_gate:
            qubits.update(op.data['gate_info']['qubits'])
    qubit_map = {}
    for idx, qubit in enumerate(sorted(qubits)):
        qubit_map[qubit] = idx

    if figsize is None:
        figsize = (10, len(qubit_map))
    f, ax = ps.new_pulse_fig(figsize=figsize)
    ax.set_title(schedule.data['name'])
    ax.set_aspect('equal')

    ax.set_ylim(-.5, len(qubit_map)-.5)
    for q in qubits:
        ax.axhline(qubit_map[q], color='.75')
    # plot the qubit names on the y-axis
    ax.set_yticks(list(qubit_map.values()))
    ax.set_yticklabels(qubit_map.keys())

    total_duration = 0
    for t_constr in schedule.timing_constraints:
        op = schedule.operations[t_constr['operation_hash']]
        time = t_constr['abs_time']
        total_duration = total_duration if total_duration > time else time

        if op.valid_gate:
            plot_func = import_func_from_string(op['gate_info']['plot_func'])
            idxs = [qubit_map[q] for q in op['gate_info']['qubits']]
            plot_func(ax, time=time, qubit_idxs=idxs, tex=op['gate_info']['tex'])
        elif op.valid_pulse:
            plot_func = import_func_from_string('quantify.scheduler.visualization.circuit_diagram.gate_box')
            idxs = []
            for pulse in op['pulse_info']:
                resolved = False
                for sub_addr in pulse['channel'].split(":"):
                    if sub_addr in qubit_map:
                        idxs.append(qubit_map[sub_addr])
                        resolved = True
                if not resolved:
                    raise ValueError("Could not resolve the address of pulse {} on channel {}"
                                     .format(op.name, op['channel']))
            time = t_constr['abs_time']
            plot_func(ax, time=time, qubit_idxs=idxs, tex='Pulse')
        else:
            raise ValueError("Unknown operation")

    ax.set_xlim(-1, total_duration + 1)
    return f, ax
