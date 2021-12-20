# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the master branch
# pylint: disable=invalid-name
"""Standard gateset for use with the quantify_scheduler."""
from typing import Optional, Tuple, Union

import numpy as np

from quantify_scheduler.enums import BinMode

from .operation import Operation


# pylint: disable=too-many-ancestors
class Rxy(Operation):
    # pylint: disable=line-too-long
    r"""
    A single qubit rotation around an axis in the equator of the Bloch sphere.


    This operation can be represented by the following unitary:

    .. math::

        \mathsf {R}_{xy} \left(\theta, \varphi\right) = \begin{bmatrix}
        \textrm {cos}(\theta /2) & -ie^{-i\varphi }\textrm {sin}(\theta /2)
        \\ -ie^{i\varphi }\textrm {sin}(\theta /2) & \textrm {cos}(\theta /2) \end{bmatrix}

    """

    def __init__(
        self, theta: float, phi: float, qubit: str, data: Optional[dict] = None
    ):
        """
        A single qubit rotation around an axis in the equator of the Bloch sphere.

        Parameters
        ----------
        theta
            rotation angle in degrees
        phi
            phase of the rotation axis
        qubit
            the target qubit
        data
            The operation's dictionary, by default None
            Note: if the data parameter is not None all other parameters are
            overwritten using the contents of data.
        """
        if not isinstance(theta, float):
            theta = float(theta)
        if not isinstance(phi, float):
            phi = float(phi)

        # this solves an issue where different rotations with the same rotation angle
        # modulo a full period are treated as distinct operations in the OperationDict.
        theta = theta % 360
        phi = phi % 360

        if data is None:
            tex = r"$R_{xy}^{" + f"{theta:.0f}, {phi:.0f}" + r"}$"
            plot_func = "quantify_scheduler.visualization.circuit_diagram.gate_box"
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

            data = {
                "name": f"Rxy({theta:.8g}, {phi:.8g}, '{qubit}')",
                "gate_info": {
                    "unitary": unitary,
                    "tex": tex,
                    "plot_func": plot_func,
                    "qubits": [qubit],
                    "operation_type": "Rxy",
                    "theta": theta,
                    "phi": phi,
                },
            }

        super().__init__(data["name"], data=data)

    def __str__(self) -> str:
        gate_info = self.data["gate_info"]
        theta = gate_info["theta"]
        phi = gate_info["phi"]
        qubit = gate_info["qubits"][0]
        return f"{self.__class__.__name__}(theta={theta:.8g}, phi={phi:.8g}, qubit='{qubit}')"


class X(Rxy):
    r"""
    A single qubit rotation of 180 degrees around the X-axis.


    This operation can be represented by the following unitary:

    .. math::

        X = \sigma_x = \begin{bmatrix}
             0 & 1 \\
             1 & 0 \\ \end{bmatrix}

    """

    def __init__(self, qubit: str, data: Optional[dict] = None):
        """
        Parameters
        ----------
        qubit
            the target qubit
        data
            The operation's dictionary, by default None
            Note: if the data parameter is not None all other parameters are
            overwritten using the contents of data.
        """
        super().__init__(theta=180, phi=0, qubit=qubit, data=data)
        self.data["name"] = f"X {qubit}"
        self.data["gate_info"]["tex"] = r"$X_{\pi}$"

    def __str__(self) -> str:
        qubit = self.data["gate_info"]["qubits"][0]
        return f"{self.__class__.__name__}(qubit='{qubit}')"


class X90(Rxy):
    """
    A single qubit rotation of 90 degrees around the X-axis.
    """

    def __init__(self, qubit: str, data: Optional[dict] = None):
        """
        Create a new instance of X90.

        Parameters
        ----------
        qubit
            The target qubit.
        data
            The operation's dictionary, by default None
            Note: if the data parameter is not None all other parameters are
            overwritten using the contents of data.
        """
        super().__init__(theta=90.0, phi=0.0, qubit=qubit, data=data)
        self.qubit = qubit
        self.data["name"] = f"X_90 {qubit}"
        self.data["gate_info"]["tex"] = r"$X_{\pi/2}$"

    def __str__(self) -> str:
        qubit = self.data["gate_info"]["qubits"][0]
        return f"{self.__class__.__name__}(qubit='{qubit}')"


class Y(Rxy):
    r"""
    A single qubit rotation of 180 degrees around the Y-axis.


    .. math::

        \mathsf Y = \sigma_y = \begin{bmatrix}
             0 & -i \\
             i & 0 \end{bmatrix}

    """

    def __init__(self, qubit: str, data: Optional[dict] = None):
        """
        Create a new instance of Y.

        The Y gate corresponds to a rotation of 180 degrees around the y-axis in the
        single-qubit Bloch sphere.

        Parameters
        ----------
        qubit
            The target qubit.
        data
            The operation's dictionary, by default None
            Note: if the data parameter is not None all other parameters are
            overwritten using the contents of data.
        """
        super().__init__(theta=180.0, phi=90.0, qubit=qubit, data=data)
        self.data["name"] = f"Y {qubit}"
        self.data["gate_info"]["tex"] = r"$Y_{\pi}$"

    def __str__(self) -> str:
        qubit = self.data["gate_info"]["qubits"][0]
        return f"{self.__class__.__name__}(qubit='{qubit}')"


class Y90(Rxy):
    """
    A single qubit rotation of 90 degrees around the Y-axis.
    """

    def __init__(self, qubit: str, data: Optional[dict] = None):
        """
        Create a new instance of Y90.

        The Y gate corresponds to a rotation of 90 degrees around the y-axis in the
        single-qubit Bloch sphere.

        Parameters
        ----------
        qubit
            The target qubit.
        data
            The operation's dictionary, by default None
            Note: if the data parameter is not None all other parameters are
            overwritten using the contents of data.
        """
        super().__init__(theta=90.0, phi=90.0, qubit=qubit, data=data)
        self.data["name"] = f"Y_90 {qubit}"
        self.data["gate_info"]["tex"] = r"$Y_{\pi/2}$"

    def __str__(self) -> str:
        """
        Returns a concise string representation
        which can be evaluated into a new instance
        using `eval(str(operation))` only when the
        data dictionary has not been modified.

        This representation is guaranteed to be
        unique.
        """
        qubit = self.data["gate_info"]["qubits"][0]
        return f"{self.__class__.__name__}(qubit='{qubit}')"


class CNOT(Operation):
    r"""
    Conditional-NOT gate, a common entangling gate.

    Performs an X gate on the target qubit qT conditional on the state
    of the control qubit qC.

    This operation can be represented by the following unitary:

    .. math::

        \mathrm{CNOT}  = \begin{bmatrix}
            1 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0 \\
            0 & 0 & 0 & 1 \\
            0 & 0 & 1 & 0 \\ \end{bmatrix}

    """

    def __init__(self, qC: str, qT: str, data: Optional[dict] = None):
        """
        Create a new instance of the two-qubit CNOT or Controlled-NOT gate.

        The CNOT gate performs an X gate on the target qubit(qT) conditional on the
        state of the control qubit(qC).

        Parameters
        ----------
        qC
            The control qubit.
        qT
            The target qubit
        data
            The operation's dictionary, by default None
            Note: if the data parameter is not None all other parameters are
            overwritten using the contents of data.
        """
        if data is None:
            plot_func = "quantify_scheduler.visualization.circuit_diagram.cnot"
            data = {
                "name": f"CNOT ({qC}, {qT})",
                "gate_info": {
                    "unitary": np.array(
                        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]
                    ),
                    "tex": r"CNOT",
                    "plot_func": plot_func,
                    "qubits": [qC, qT],
                    "operation_type": "CNOT",
                },
            }
        super().__init__(data["name"], data=data)

    def __str__(self) -> str:
        gate_info = self.data["gate_info"]
        qC = gate_info["qubits"][0]
        qT = gate_info["qubits"][1]
        return f"{self.__class__.__name__}(qC='{qC}',qT='{qT}')"


class CZ(Operation):
    r"""
    Conditional-phase gate, a common entangling gate.

    Performs a Z gate on the target qubit qT conditional on the state
    of the control qubit qC.

    This operation can be represented by the following unitary:

    .. math::

        \mathrm{CZ}  = \begin{bmatrix}
            1 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0 \\
            0 & 0 & 1 & 0 \\
            0 & 0 & 0 & -1 \\ \end{bmatrix}

    """

    def __init__(self, qC: str, qT: str, data: Optional[dict] = None):
        """
        Create a new instance of the two-qubit CZ or conditional-phase gate.

        The CZ gate performs an Z gate on the target qubit(qT) conditional on the
        state of the control qubit(qC).

        Parameters
        ----------
        qC
            The control qubit.
        qT
            The target qubit
        data
            The operation's dictionary, by default None
            Note: if the data parameter is not None all other parameters are
            overwritten using the contents of data.
        """
        if data is None:
            plot_func = "quantify_scheduler.visualization.circuit_diagram.cz"
            data = {
                "name": f"CZ ({qC}, {qT})",
                "gate_info": {
                    "unitary": np.array(
                        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]]
                    ),
                    "tex": r"CZ",
                    "plot_func": plot_func,
                    "qubits": [qC, qT],
                    "operation_type": "CZ",
                },
            }
        super().__init__(data["name"], data=data)

    def __str__(self) -> str:
        gate_info = self.data["gate_info"]
        qC = gate_info["qubits"][0]
        qT = gate_info["qubits"][1]

        return f"{self.__class__.__name__}(qC='{qC}',qT='{qT}')"


class Reset(Operation):
    r"""
    Reset a qubit to the :math:`|0\rangle` state.

    The Reset gate is an idle operation that is used to initialize one or more qubits.

    .. note::

        Strictly speaking this is not a gate as it can not
        be described by a unitary.

    .. admonition:: Examples
        :class: tip

        The operation can be used in several ways:

        .. jupyter-execute::

            from quantify_scheduler.operations.gate_library import Reset

            reset_1 = Reset("q0")
            reset_2 = Reset("q1", "q2")
            reset_3 = Reset(*[f"q{i}" for i in range(3, 6)])
    """

    def __init__(self, *qubits: str, data: Optional[dict] = None):
        """
        Create a new instance of Reset operation that is used to initialize one or
        more qubits.


        Parameters
        ----------
        qubits
            The qubit(s) to reset. NB one or more qubits can be specified, e.g.,
            :code:`Reset("q0")`, :code:`Reset("q0", "q1", "q2")`, etc..
        data
            The operation's dictionary, by default :code:`None`.
            Note: if the data parameter is not :code:`None` all other parameters are
            overwritten using the contents of data.
        """
        if data is None:
            plot_func = "quantify_scheduler.visualization.circuit_diagram.reset"
            data = {
                "name": f"Reset {', '.join(qubits)}",
                "gate_info": {
                    "unitary": None,
                    "tex": r"$|0\rangle$",
                    "plot_func": plot_func,
                    "qubits": list(qubits),
                    "operation_type": "reset",
                },
            }
        super().__init__(data["name"], data=data)

    def __str__(self) -> str:
        qubits = map(lambda x: f"'{x}'", self.data["gate_info"]["qubits"])
        return f'{self.__class__.__name__}({",".join(qubits)})'


class Measure(Operation):
    """
    A projective measurement in the Z-basis.

    .. note::

        Strictly speaking this is not a gate as it can not
        be described by a unitary.

    """

    def __init__(
        self,
        *qubits: str,
        acq_channel: Union[Tuple[int, ...], int] = None,
        acq_index: Union[Tuple[int, ...], int] = None,
        bin_mode: Union[BinMode, None] = None,
        data: Optional[dict] = None,
    ):
        """
        Gate level description for a measurement.

        The measurement is compiled according to the type of acquisition specified
        in the device configuration.

        Parameters
        ----------
        qubits
            The qubits you want to measure
        acq_channel
            Acquisition channel on which the measurement is performed
        acq_index
            Index of the register where the measurement is stored.
        bin_mode
            The binning mode that is to be used. If not None, it will overwrite
            the binning mode used for Measurements in the quantum-circuit to
            quantum-device compilation step.
        data
            The operation's dictionary, by default None
            Note: if the data parameter is not None all other parameters are
            overwritten using the contents of data.
        """

        if isinstance(acq_index, int):
            acq_index = (acq_index,)
        elif acq_index is None:
            acq_index = tuple(i for i in range(len(qubits)))

        if isinstance(acq_channel, int):
            acq_channel = (acq_channel,)
        elif acq_channel is None:
            acq_channel = tuple(i for i in range(len(qubits)))

        if data is None:
            plot_func = "quantify_scheduler.visualization.circuit_diagram.meter"
            data = {
                "name": f"Measure {', '.join(qubits)}",
                "gate_info": {
                    "unitary": None,
                    "plot_func": plot_func,
                    "tex": r"$\langle0|$",
                    "qubits": list(qubits),
                    "acq_channel": acq_channel,
                    "acq_index": acq_index,
                    "bin_mode": bin_mode,
                    "operation_type": "measure",
                },
            }
        super().__init__(data["name"], data=data)

    def __str__(self) -> str:
        qubits = map(lambda x: f"'{x}'", self.data["gate_info"]["qubits"])
        acq_channel = self.data["gate_info"]["acq_channel"]
        acq_index = self.data["gate_info"]["acq_index"]
        return (
            f'{self.__class__.__name__}({",".join(qubits)},'
            + f"acq_channel={acq_channel},acq_index={acq_index})"
        )
