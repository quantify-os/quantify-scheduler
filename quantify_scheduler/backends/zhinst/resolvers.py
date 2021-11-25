# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch

from __future__ import annotations

from typing import Tuple

import numpy as np
from zhinst import qcodes

from quantify_scheduler.backends.zhinst import helpers as zi_helpers


def monitor_acquisition_resolver(
    uhfqa: qcodes.UHFQA, monitor_nodes: Tuple[str, str]
) -> np.ndarray:
    """
    Returns complex value of UHFQA Monitor nodes.

    This acquisition resolver corresponds to measuring a time trace of the input on the
    I channel (input 1) and Q channel (input 2).

    Parameters
    ----------
    uhfqa
    monitor_nodes
    """
    (node_i, node_q) = monitor_nodes
    results_i = zi_helpers.get_value(uhfqa, node_i)
    results_q = zi_helpers.get_value(uhfqa, node_q)
    return results_i + 1j * results_q


def result_acquisition_resolver(
    uhfqa: qcodes.UHFQA, result_nodes: Tuple[str, str]
) -> np.ndarray:
    """
    Returns complex value of UHFQA Result nodes.

    Note that it needs two nodes to return a complex valued result.
    For optimal weights one can ignore the imaginary part.

    Parameters
    ----------
    uhfqa
    result_nodes
    """
    vals_node0 = zi_helpers.get_value(uhfqa, result_nodes[0])
    vals_node1 = zi_helpers.get_value(uhfqa, result_nodes[1])

    # the ZI API keeps the contributions of both weight functions separate
    # here we combine them so they correspond to the I and Q components.
    vals_i = vals_node0.real + vals_node0.imag
    vals_q = vals_node1.real + vals_node1.imag

    results = vals_i + 1j * vals_q

    return results
