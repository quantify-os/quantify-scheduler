# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
# pylint: disable=invalid-name
"""Standard gateset for use with the quantify_scheduler."""
from typing import Literal, Optional, Tuple, Union
import warnings

import numpy as np

from .operation import Operation
from ..enums import BinMode


# pylint: disable=too-many-ancestors
class Rxy(Operation):
    # pylint: disable=line-too-long
    r"""
    A single qubit rotation around an axis in the equator of the Bloch sphere.


    This operation can be represented by the following unitary as defined in https://doi.org/10.1109/TQE.2020.2965810:

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
            rotation angle in degrees, will be casted to the [-180, 180) domain.
        phi
            phase of the rotation axis, will be casted to the [0, 360) domain.
        qubit
            the target qubit
        data
            The operation's dictionary, by default None\n
            Note: if the data parameter is not None all other parameters are
            overwritten using the contents of data.\n
            Deprecated: support for the data argument will be dropped in
            quantify-scheduler >= 0.13.0. Please consider updating the data
            dictionary after initialization.
        """
        if not isinstance(theta, float):
            theta = float(theta)
        if not isinstance(phi, float):
            phi = float(phi)

        # this solves an issue where different rotations with the same rotation angle
        # modulo a full period are treated as distinct operations in the OperationDict.
        theta = (theta + 180) % 360 - 180

        phi = phi % 360
        if data is None:
            tex = r"$R_{xy}^{" + f"{theta:.0f}, {phi:.0f}" + r"}$"
            plot_func = (
                "quantify_scheduler.schedules._visualization.circuit_diagram.gate_box"
            )
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
                        -1j * np.exp(1j * phi_r) * np.sin(theta_r / 2),
                        np.cos(theta_r / 2),
                    ],
                ]
            )
            super().__init__(f"Rxy({theta:.8g}, {phi:.8g}, '{qubit}')")
            self.data.update(
                {
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
            )
            self._update()
        else:
            warnings.warn(
                "Support for the data argument will be dropped in"
                "quantify-scheduler >= 0.13.0.\n"
                "Please consider updating the data "
                "dictionary after initialization.",
                FutureWarning,
            )
            super().__init__(name=data["name"], data=data)

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

        X180 = R_{X180} = \begin{bmatrix}
             0 & -i \\
             -i & 0 \\ \end{bmatrix}

    """

    def __init__(self, qubit: str, data: Optional[dict] = None):
        """
        Parameters
        ----------
        qubit
            the target qubit
        data
            The operation's dictionary, by default None\n
            Note: if the data parameter is not None all other parameters are
            overwritten using the contents of data.\n
            Deprecated: support for the data argument will be dropped in
            quantify-scheduler >= 0.13.0. Please consider updating the data
            dictionary after initialization.
        """
        if data is not None:
            warnings.warn(
                "Support for the data argument will be dropped in"
                "quantify-scheduler >= 0.13.0.\n"
                "Please consider updating the data "
                "dictionary after initialization.",
                FutureWarning,
            )
        super().__init__(theta=180, phi=0, qubit=qubit, data=data)
        self.data["name"] = f"X {qubit}"
        self.data["gate_info"]["tex"] = r"$X_{\pi}$"

    def __str__(self) -> str:
        qubit = self.data["gate_info"]["qubits"][0]
        return f"{self.__class__.__name__}(qubit='{qubit}')"


class X90(Rxy):
    r"""
    A single qubit rotation of 90 degrees around the X-axis.

    It is identical to the Rxy gate with theta=90 and phi=0

    Defined by the unitary:

    .. math::
        X90 = R_{X90} = \frac{1}{\sqrt{2}}\begin{bmatrix}
                1 & -i \\
                -i & 1 \\ \end{bmatrix}

    """

    def __init__(self, qubit: str, data: Optional[dict] = None):
        """
        Create a new instance of X90.

        Parameters
        ----------
        qubit
            The target qubit.
        data
            The operation's dictionary, by default None\n
            Note: if the data parameter is not None all other parameters are
            overwritten using the contents of data.\n
            Deprecated: support for the data argument will be dropped in
            quantify-scheduler >= 0.13.0. Please consider updating the data
            dictionary after initialization.
        """
        if data is not None:
            warnings.warn(
                "Support for the data argument will be dropped in"
                "quantify-scheduler >= 0.13.0.\n"
                "Please consider updating the data "
                "dictionary after initialization.",
                FutureWarning,
            )
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

    It is identical to the Rxy gate with theta=180 and phi=90

    Defined by the unitary: 

    .. math::
        Y180 = R_{Y180} = \begin{bmatrix}
             0 & -1 \\
             1 & 0 \\ \end{bmatrix}

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
            The operation's dictionary, by default None\n
            Note: if the data parameter is not None all other parameters are
            overwritten using the contents of data.\n
            Deprecated: support for the data argument will be dropped in
            quantify-scheduler >= 0.13.0. Please consider updating the data
            dictionary after initialization.
        """
        if data is not None:
            warnings.warn(
                "Support for the data argument will be dropped in"
                "quantify-scheduler >= 0.13.0.\n"
                "Please consider updating the data "
                "dictionary after initialization.",
                FutureWarning,
            )
        super().__init__(theta=180.0, phi=90.0, qubit=qubit, data=data)
        self.data["name"] = f"Y {qubit}"
        self.data["gate_info"]["tex"] = r"$Y_{\pi}$"

    def __str__(self) -> str:
        qubit = self.data["gate_info"]["qubits"][0]
        return f"{self.__class__.__name__}(qubit='{qubit}')"


class Y90(Rxy):
    r"""
    A single qubit rotation of 90 degrees around the Y-axis.

    It is identical to the Rxy gate with theta=90 and phi=90

    Defined by the unitary: 

    .. math::

        Y90 = R_{Y90} = \frac{1}{\sqrt{2}}\begin{bmatrix}
                1 & -1 \\
                1 & 1 \\ \end{bmatrix}

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
            The operation's dictionary, by default None\n
            Note: if the data parameter is not None all other parameters are
            overwritten using the contents of data.\n
            Deprecated: support for the data argument will be dropped in
            quantify-scheduler >= 0.13.0. Please consider updating the data
            dictionary after initialization.
        """
        if data is not None:
            warnings.warn(
                "Support for the data argument will be dropped in"
                "quantify-scheduler >= 0.13.0.\n"
                "Please consider updating the data "
                "dictionary after initialization.",
                FutureWarning,
            )
        super().__init__(theta=90.0, phi=90.0, qubit=qubit, data=data)
        self.data["name"] = f"Y_90 {qubit}"
        self.data["gate_info"]["tex"] = r"$Y_{\pi/2}$"

    def __str__(self) -> str:
        """
        Returns a concise string representation
        which can be evaluated into a new instance
        using :code:`eval(str(operation))` only when the
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
            The operation's dictionary, by default None\n
            Note: if the data parameter is not None all other parameters are
            overwritten using the contents of data.\n
            Deprecated: support for the data argument will be dropped in
            quantify-scheduler >= 0.13.0. Please consider updating the data
            dictionary after initialization.
        """
        if data is None:
            plot_func = (
                "quantify_scheduler.schedules._visualization.circuit_diagram.cnot"
            )
            super().__init__(f"CNOT ({qC}, {qT})")
            self.data.update(
                {
                    "name": f"CNOT ({qC}, {qT})",
                    "gate_info": {
                        "unitary": np.array(
                            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]
                        ),
                        "tex": r"CNOT",
                        "plot_func": plot_func,
                        "qubits": [qC, qT],
                        "symmetric": False,
                        "operation_type": "CNOT",
                    },
                }
            )
            self._update()
        else:
            warnings.warn(
                "Support for the data argument will be dropped in"
                "quantify-scheduler >= 0.13.0.\n"
                "Please consider updating the data "
                "dictionary after initialization.",
                FutureWarning,
            )
            super().__init__(name=data["name"], data=data)

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
            The operation's dictionary, by default None\n
            Note: if the data parameter is not None all other parameters are
            overwritten using the contents of data.\n
            Deprecated: support for the data argument will be dropped in
            quantify-scheduler >= 0.13.0. Please consider updating the data
            dictionary after initialization.
        """
        if data is None:
            plot_func = "quantify_scheduler.schedules._visualization.circuit_diagram.cz"
            super().__init__(f"CZ ({qC}, {qT})")
            self.data.update(
                {
                    "name": f"CZ ({qC}, {qT})",
                    "gate_info": {
                        "unitary": np.array(
                            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]]
                        ),
                        "tex": r"CZ",
                        "plot_func": plot_func,
                        "qubits": [qC, qT],
                        "symmetric": True,
                        "operation_type": "CZ",
                    },
                }
            )
            self._update()
        else:
            warnings.warn(
                "Support for the data argument will be dropped in"
                "quantify-scheduler >= 0.13.0.\n"
                "Please consider updating the data "
                "dictionary after initialization.",
                FutureWarning,
            )
            super().__init__(name=data["name"], data=data)

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
            The operation's dictionary, by default None\n
            Note: if the data parameter is not None all other parameters are
            overwritten using the contents of data.\n
            Deprecated: support for the data argument will be dropped in
            quantify-scheduler >= 0.13.0. Please consider updating the data
            dictionary after initialization.
        """
        if data is None:
            super().__init__(f"Reset {', '.join(qubits)}")
            plot_func = (
                "quantify_scheduler.schedules._visualization.circuit_diagram.reset"
            )
            self.data.update(
                {
                    "name": f"Reset {', '.join(qubits)}",
                    "gate_info": {
                        "unitary": None,
                        "tex": r"$|0\rangle$",
                        "plot_func": plot_func,
                        "qubits": list(qubits),
                        "operation_type": "reset",
                    },
                }
            )
            self._update()
        else:
            warnings.warn(
                "Support for the data argument will be dropped in"
                "quantify-scheduler >= 0.13.0.\n"
                "Please consider updating the data "
                "dictionary after initialization.",
                FutureWarning,
            )
            super().__init__(name=data["name"], data=data)

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
        # These are the currently supported acquisition protocols.
        acq_protocol: Literal[
            "SSBIntegrationComplex",
            "Trace",
            "TriggerCount",
            None,
        ] = None,
        bin_mode: BinMode = None,
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
            If None specified, it will default if a tuple(0
        acq_protocol
            Acquisition protocol (currently ``"SSBIntegrationComplex"`` and ``"Trace"``)
            are supported. If ``None`` is specified, the default protocol is chosen
            based on the device and backend configuration.
        bin_mode
            The binning mode that is to be used. If not None, it will overwrite
            the binning mode used for Measurements in the quantum-circuit to
            quantum-device compilation step.
        data
            The operation's dictionary, by default None\n
            Note: if the data parameter is not None all other parameters are
            overwritten using the contents of data.\n
            Deprecated: support for the data argument will be dropped in
            quantify-scheduler >= 0.13.0. Please consider updating the data
            dictionary after initialization.
        """

        # this if else statement a workaround to support multiplexed measurements (#262)

        # this snippet has some automatic behaviour that is error prone.
        # see #262
        if len(qubits) == 1:
            if acq_channel is None:
                acq_channel = 0
            if acq_index is None:
                acq_index = 0
        else:
            if isinstance(acq_index, int):
                acq_index = [
                    acq_index,
                ] * len(qubits)
            elif acq_index is None:
                # defaults to writing the result of all qubits to acq_index 0.
                # note that this will result in averaging data together if multiple
                # measurements are present in the same schedule (#262)
                acq_index = list(0 for i in range(len(qubits)))

            # defaults to mapping qubits to channels dependent on the order of the
            # arguments. note that this will result in mislabeling data if not all
            # measurements in an experiment contain the same order of qubits (#262)
            if acq_channel is None:
                acq_channel = list(i for i in range(len(qubits)))
            else:
                warnings.warn(
                    "`acq_channel` keyword argument does not have any effect if specified here"
                    "and should be set in the device layer. See `BasicTransmonElement.measure.acq_channel`"
                    "for more info on how to set it. This keyword argument will be removed in "
                    "quantify-scheduler >= 0.12.0.",
                    FutureWarning,
                )
        if data is None:
            plot_func = (
                "quantify_scheduler.schedules._visualization.circuit_diagram.meter"
            )
            super().__init__(f"Measure {', '.join(qubits)}")
            self.data.update(
                {
                    "name": f"Measure {', '.join(qubits)}",
                    "gate_info": {
                        "unitary": None,
                        "plot_func": plot_func,
                        "tex": r"$\langle0|$",
                        "qubits": list(qubits),
                        "acq_channel": acq_channel,
                        "acq_index": acq_index,
                        "acq_protocol": acq_protocol,
                        "bin_mode": bin_mode,
                        "operation_type": "measure",
                    },
                }
            )
            self._update()
        else:
            warnings.warn(
                "Support for the data argument will be dropped in"
                "quantify-scheduler >= 0.13.0.\n"
                "Please consider updating the data "
                "dictionary after initialization.",
                FutureWarning,
            )
            super().__init__(name=data["name"], data=data)

    def __str__(self) -> str:
        gate_info = self.data["gate_info"]
        qubits = map(lambda x: f"'{x}'", gate_info["qubits"])
        acq_index = gate_info["acq_index"]
        acq_protocol = gate_info["acq_protocol"]
        bin_mode = gate_info["bin_mode"]
        return (
            f'{self.__class__.__name__}({",".join(qubits)}, '
            f'acq_index={acq_index}, acq_protocol="{acq_protocol}", '
            f"bin_mode={str(bin_mode)})"
        )
