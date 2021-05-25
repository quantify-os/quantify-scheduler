# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
# pylint: disable=invalid-name
"""Standard gateset for use with the quantify.scheduler."""
from typing import Tuple, Union

import numpy as np

from .types import Operation


# pylint: disable=too-many-ancestors
class Rxy(Operation):
    # pylint: disable=line-too-long
    """
    A single qubit rotation around an axis in the equator of the Bloch sphere.


    This operation can be represented by the following unitary:

    .. math::

        \\mathsf {R}_{xy} \\left(\\theta, \\varphi\\right) = \\begin{bmatrix}
        \\textrm {cos}(\\theta /2) & -ie^{-i\\varphi }\\textrm {sin}(\\theta /2)
        \\\\ -ie^{i\\varphi }\\textrm {sin}(\\theta /2) & \\textrm {cos}(\\theta /2) \\end{bmatrix}

    """

    def __init__(self, theta: float, phi: float, qubit: str):
        """
        A single qubit rotation around an axis in the equator of the Bloch sphere.

        Parameters
        ----------
        theta : float
            rotation angle in degrees
        phi : float
            phase of the rotation axis
        qubit : str
            the target qubit
        """
        name = "Rxy({:.2f}, {:.2f}) {}".format(theta, phi, qubit)
        self.qubit = qubit
        self._theta = theta
        self._phi = phi

        theta_r = np.deg2rad(theta)
        phi_r = np.deg2rad(phi)

        # not all operations have a valid unitary description
        # (e.g., measure and init)
        unitary = np.array(
            [
                [
                    np.cos(theta_r / 2),
                    -1j * np.exp(-1j * phi_r) * np.sin(theta_r / 2),
                ],
                [
                    -1j * np.exp(-1j * phi_r) * np.sin(theta_r / 2),
                    np.cos(theta_r / 2),
                ],
            ]
        )

        tex = r"$R_{xy}^{" + "{:.0f}, {:.0f}".format(theta, phi) + "}$"
        data = {
            "name": name,
            "gate_info": {
                "unitary": unitary,
                "tex": tex,
                "plot_func": "quantify.scheduler.visualization.circuit_diagram.gate_box",
                "qubits": [qubit],
                "operation_type": "Rxy",
                "theta": theta,
                "phi": phi,
            },
        }
        super().__init__(name, data=data)

    def __repr__(self):
        return f'Rxy({self._theta}, {self._phi}, "{self.qubit}")'


class X(Rxy):
    """
    A single qubit rotation of 180 degrees around the X-axis.


    This operation can be represented by the following unitary:

    .. math::

        X = \\sigma_x = \\begin{bmatrix}
             0 & 1 \\\\
             1 & 0 \\ \\end{bmatrix}

    """

    def __init__(self, qubit: str):
        """
        Parameters
        ----------
        qubit : str
            the target qubit
        """
        super().__init__(theta=180, phi=0, qubit=qubit)
        self.qubit = qubit
        self.data["name"] = f"X {qubit}"
        self.data["gate_info"]["tex"] = r"$X_{\pi}$"

    def __repr__(self):
        return f'{self.__class__.__name__}("{self.qubit}")'


class X90(Rxy):
    """
    A single qubit rotation of 90 degrees around the X-axis.
    """

    def __init__(self, qubit: str):
        """
        Parameters
        ----------
        qubit : str
            the target qubit
        """
        super().__init__(theta=90, phi=0, qubit=qubit)
        self.qubit = qubit
        self.data["name"] = f"X_90 {qubit}"
        self.data["gate_info"]["tex"] = r"$X_{\pi/2}$"

    def __repr__(self):
        return f'{self.__class__.__name__}("{self.qubit}")'


class Y(Rxy):
    """
    A single qubit rotation of 180 degrees around the Y-axis.


    .. math::

        \\mathsf Y = \\sigma_y = \\begin{bmatrix}
             0 & -i \\\\
             i & 0 \\end{bmatrix}

    """

    def __init__(self, qubit: str):
        """
        Parameters
        ----------
        qubit : str
            the target qubit
        """
        super().__init__(theta=180, phi=90, qubit=qubit)
        self.data["name"] = f"Y {qubit}"
        self.data["gate_info"]["tex"] = r"$Y_{\pi/2}$"

    def __repr__(self):
        return f'{self.__class__.__name__}("{self.qubit}")'


class Y90(Rxy):
    """
    A single qubit rotation of 90 degrees around the Y-axis.
    """

    def __init__(self, qubit: str):
        """
        Parameters
        ----------
        qubit : str
            the target qubit
        """
        super().__init__(theta=90, phi=90, qubit=qubit)
        self.data["name"] = f"Y_90 {qubit}"
        self.data["gate_info"]["tex"] = r"$Y_{\pi/2}$"

    def __repr__(self):
        return f'{self.__class__.__name__}("{self.qubit}")'


class CNOT(Operation):
    """
    Conditional-NOT gate, a common entangling gate.

    Performs an X gate on the target qubit qT conditional on the state
    of the control qubit qC.

    This operation can be represented by the following unitary:

    .. math::

        \\mathrm{CNOT}  = \\begin{bmatrix}
            1 & 0 & 0 & 0 \\\\
            0 & 1 & 0 & 0 \\\\
            0 & 0 & 0 & 1 \\\\
            0 & 0 & 1 & 0 \\ \\end{bmatrix}

    """

    def __init__(self, qC: str, qT: str):
        self.qC = qC
        self.qT = qT

        data = {
            "gate_info": {
                "unitary": np.array(
                    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]
                ),
                "tex": r"CNOT",
                "plot_func": "quantify.scheduler.visualization.circuit_diagram.cnot",
                "qubits": [qC, qT],
                "operation_type": "CNOT",
            }
        }
        super().__init__(f"CNOT ({qC}, {qT})", data=data)

    def __repr__(self):
        return f'{self.__class__.__name__}("{self.qC}", "{self.qT}")'


class CZ(Operation):
    """
    Conditional-phase gate, a common entangling gate.

    Performs a Z gate on the target qubit qT conditional on the state
    of the control qubit qC.

    This operation can be represented by the following unitary:

    .. math::

        \\mathrm{CZ}  = \\begin{bmatrix}
            1 & 0 & 0 & 0 \\\\
            0 & 1 & 0 & 0 \\\\
            0 & 0 & 1 & 0 \\\\
            0 & 0 & 0 & -1 \\ \\end{bmatrix}

    """

    def __init__(self, qC: str, qT: str):
        self.qC = qC
        self.qT = qT

        data = {
            "gate_info": {
                "unitary": np.array(
                    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]]
                ),
                "tex": r"CZ",
                "plot_func": "quantify.scheduler.visualization.circuit_diagram.cz",
                "qubits": [qC, qT],
                "operation_type": "CZ",
            }
        }
        super().__init__(f"CZ ({qC}, {qT})", data=data)

    def __repr__(self):
        return f'{self.__class__.__name__}("{self.qC}", "{self.qT}")'


class Reset(Operation):
    """
    Reset a qubit to the :math:`|0\\rangle` state.

    .. note::
        strictly speaking this is not a gate as it can not
        be described by a unitary.

    """

    def __init__(self, *qubits: str):
        self._qubits = qubits
        data = {
            "gate_info": {
                "unitary": None,
                "tex": r"$|0\rangle$",
                "plot_func": "quantify.scheduler.visualization.circuit_diagram.reset",
                "qubits": list(qubits),
                "operation_type": "reset",
            }
        }
        super().__init__(f"Reset {qubits}", data=data)

    def __repr__(self):
        return f"{self.__class__.__name__}(*{self._qubits})"


class Measure(Operation):
    """
    A projective measurement in the Z-basis.

    .. note::
        strictly speaking this is not a gate as it can not
        be described by a unitary.
    """

    def __init__(
        self,
        *qubits: str,
        acq_channel: Union[Tuple[int, ...], int] = None,
        acq_index: Union[Tuple[int, ...], int] = None,
    ):
        """
        Gate level description for a measurement.

        The measurement is compiled according to what is specified in the config.

        Parameters
        ----------
        qubits
            The qubits you want to measure
        acq_channel
            Acquisition channel on which the measurement is performed
        acq_index
            Index of the register where the measurement is stored.
        """
        self._qubits = qubits
        self._acq_channel = acq_channel
        self._acq_index = acq_index

        if isinstance(acq_index, int):
            acq_index = (acq_index,)
        elif acq_index is None:
            acq_index = tuple(i for i in range(len(qubits)))

        if isinstance(acq_channel, int):
            acq_channel = (acq_channel,)
        elif acq_channel is None:
            acq_channel = tuple(i for i in range(len(qubits)))

        data = {
            "gate_info": {
                "unitary": None,
                "plot_func": "quantify.scheduler.visualization.circuit_diagram.meter",
                "tex": r"$\langle0|$",
                "qubits": list(qubits),
                "acq_channel": acq_channel,
                "acq_index": acq_index,
                "operation_type": "measure",
            }
        }
        super().__init__(f"Measure {qubits}", data=data)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(*{self._qubits}, "
            f"acq_channel={self._acq_channel}, acq_index={self._acq_index})"
        )
