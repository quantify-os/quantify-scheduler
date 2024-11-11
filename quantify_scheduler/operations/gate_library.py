# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Standard gateset for use with the quantify_scheduler."""
from __future__ import annotations

from typing import TYPE_CHECKING, Hashable, Literal

import numpy as np

from .operation import Operation

if TYPE_CHECKING:
    from quantify_scheduler.enums import BinMode


class Rxy(Operation):
    r"""
    A single qubit rotation around an axis in the equator of the Bloch sphere.

    This operation can be represented by the following unitary as defined in
    https://doi.org/10.1109/TQE.2020.2965810:

    .. math::

        \mathsf {R}_{xy} \left(\theta, \varphi\right) = \begin{bmatrix}
        \textrm {cos}(\theta /2) & -ie^{-i\varphi }\textrm {sin}(\theta /2)
        \\ -ie^{i\varphi }\textrm {sin}(\theta /2) & \textrm {cos}(\theta /2)
        \end{bmatrix}


    Parameters
    ----------
    theta
        Rotation angle in degrees, will be casted to the [-180, 180) domain.
    phi
        Phase of the rotation axis, will be casted to the [0, 360) domain.
    qubit
        The target qubit.
    device_overrides
        Device level parameters that override device configuration values
        when compiling from circuit to device level.

    """

    def __init__(
        self,
        theta: float,
        phi: float,
        qubit: str,
        **device_overrides,
    ) -> None:
        if not isinstance(theta, float):
            theta = float(theta)
        if not isinstance(phi, float):
            phi = float(phi)

        # this solves an issue where different rotations with the same rotation angle
        # modulo a full period are treated as distinct operations in the OperationDict
        # Here we map [0,360[ onto ]-180,180] so that X180 has positive amplitude
        theta = round(_modulo_360_with_mapping(theta), 8)

        phi = round(phi % 360, 8)

        tex = r"$R_{xy}^{" + f"{theta:.0f}, {phi:.0f}" + r"}$"
        plot_func = "quantify_scheduler.schedules._visualization.circuit_diagram.gate_box"
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
        self.data["gate_info"] = {
            "unitary": unitary,
            "tex": tex,
            "plot_func": plot_func,
            "qubits": [qubit],
            "operation_type": "Rxy",
            "theta": theta,
            "phi": phi,
            "device_overrides": device_overrides,
        }
        self._update()

    def __str__(self) -> str:
        gate_info = self.data["gate_info"]
        theta = gate_info["theta"]
        phi = gate_info["phi"]
        qubit = gate_info["qubits"][0]
        return f"{self.__class__.__name__}({theta=:.8g}, {phi=:.8g}, qubit='{qubit}')"


class X(Rxy):
    r"""
    A single qubit rotation of 180 degrees around the X-axis.

    This operation can be represented by the following unitary:

    .. math::

        X180 = R_{X180} = \begin{bmatrix}
             0 & -i \\
             -i & 0 \\ \end{bmatrix}

    Parameters
    ----------
    qubit
        The target qubit.
    device_overrides
        Device level parameters that override device configuration values
        when compiling from circuit to device level.

    """

    def __init__(self, qubit: str, **device_overrides) -> None:
        super().__init__(theta=180.0, phi=0, qubit=qubit, **device_overrides)
        self.data["name"] = f"X {qubit}"
        self.data["gate_info"]["tex"] = r"$X_{\pi}$"
        self._update()

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

    Parameters
    ----------
    qubit
        The target qubit.
    device_overrides
        Device level parameters that override device configuration values
        when compiling from circuit to device level.

    """

    def __init__(
        self,
        qubit: str,
        **device_overrides,
    ) -> None:
        super().__init__(theta=90.0, phi=0.0, qubit=qubit, **device_overrides)
        self.qubit = qubit
        self.data["name"] = f"X_90 {qubit}"
        self.data["gate_info"]["tex"] = r"$X_{\pi/2}$"
        self._update()

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

    Parameters
    ----------
    qubit
        The target qubit.
    device_overrides
        Device level parameters that override device configuration values
        when compiling from circuit to device level.

    """

    def __init__(
        self,
        qubit: str,
        **device_overrides,
    ) -> None:
        super().__init__(theta=180.0, phi=90.0, qubit=qubit, **device_overrides)
        self.data["name"] = f"Y {qubit}"
        self.data["gate_info"]["tex"] = r"$Y_{\pi}$"
        self._update()

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

    Parameters
    ----------
    qubit
        The target qubit.
    device_overrides
        Device level parameters that override device configuration values
        when compiling from circuit to device level.

    """

    def __init__(self, qubit: str, **device_overrides) -> None:
        super().__init__(theta=90.0, phi=90.0, qubit=qubit, **device_overrides)
        self.data["name"] = f"Y_90 {qubit}"
        self.data["gate_info"]["tex"] = r"$Y_{\pi/2}$"
        self._update()

    def __str__(self) -> str:
        """
        Returns a unique, evaluable string for unchanged data.

        Returns a concise string representation
        which can be evaluated into a new instance
        using :code:`eval(str(operation))` only when the
        data dictionary has not been modified.

        This representation is guaranteed to be
        unique.
        """
        qubit = self.data["gate_info"]["qubits"][0]
        return f"{self.__class__.__name__}(qubit='{qubit}')"


class Rz(Operation):
    r"""
    A single qubit rotation about the Z-axis of the Bloch sphere.

    This operation can be represented by the following unitary as defined in
    https://www.quantum-inspire.com/kbase/rz-gate/:

    .. math::

        \mathsf {R}_{z} \left(\theta\right) = \begin{bmatrix}
        e^{-i\theta/2} & 0
        \\ 0 & e^{i\theta/2} \end{bmatrix}

    Parameters
    ----------
    theta
        Rotation angle in degrees, will be cast to the [-180, 180) domain.
    qubit
        The target qubit.
    device_overrides
        Device level parameters that override device configuration values
        when compiling from circuit to device level.

    """

    def __init__(self, theta: float, qubit: str, **device_overrides) -> None:
        if not isinstance(theta, float):
            theta = float(theta)

        # this solves an issue where different rotations with the same rotation angle
        # modulo a full period are treated as distinct operations in the OperationDict
        # Here we map [0,360[ onto ]-180,180] so that X180 has positive amplitude
        theta = _modulo_360_with_mapping(theta)

        tex = r"$R_{z}^{" + f"{theta:.0f}" + r"}$"
        plot_func = "quantify_scheduler.schedules._visualization.circuit_diagram.gate_box"
        theta_r = np.deg2rad(theta)

        # not all operations have a valid unitary description
        # (e.g., measure and init)
        unitary = np.array(
            [
                [np.exp(-1j * theta_r / 2), 0],
                [0, np.exp(1j * theta_r / 2)],
            ]
        )
        super().__init__(f"Rz({theta:.8g}, '{qubit}')")
        self.data["gate_info"] = {
            "unitary": unitary,
            "tex": tex,
            "plot_func": plot_func,
            "qubits": [qubit],
            "operation_type": "Rz",
            "theta": theta,
            "device_overrides": device_overrides,
        }
        self._update()

    def __str__(self) -> str:
        gate_info = self.data["gate_info"]
        theta = gate_info["theta"]
        qubit = gate_info["qubits"][0]
        return f"{self.__class__.__name__}({theta=:.8g}, qubit='{qubit}')"


class Z(Rz):
    r"""
    A single qubit rotation of 180 degrees around the Z-axis.

    Note that the gate implements :math:`R_z(\pi) = -iZ`, adding a global phase of :math:`-\pi/2`.
    This operation can be represented by the following unitary:

    .. math::

        Z180 = R_{Z180} = -iZ = e^{-\frac{\pi}{2}}Z = \begin{bmatrix}
             -i & 0 \\
             0 & i \\ \end{bmatrix}

    Parameters
    ----------
    qubit
        The target qubit.
    device_overrides
        Device level parameters that override device configuration values
        when compiling from circuit to device level.

    """

    def __init__(self, qubit: str, **device_overrides) -> None:
        super().__init__(theta=180.0, qubit=qubit, **device_overrides)
        self.data["name"] = f"Z {qubit}"
        self.data["gate_info"]["tex"] = r"$Z_{\pi}$"
        self._update()

    def __str__(self) -> str:
        qubit = self.data["gate_info"]["qubits"][0]
        return f"{self.__class__.__name__}(qubit='{qubit}')"


class Z90(Rz):
    r"""
    A single qubit rotation of 90 degrees around the Z-axis.

    This operation can be represented by the following unitary:

    .. math::

        Z90 =
        R_{Z90} =
        e^{-\frac{\pi/2}{2}}S =
        e^{-\frac{\pi/2}{2}}\sqrt{Z} = \frac{1}{\sqrt{2}}\begin{bmatrix}
             1-i & 0 \\
             0 & 1+i \\ \end{bmatrix}

    Parameters
    ----------
    qubit
        The target qubit.
    device_overrides
        Device level parameters that override device configuration values
        when compiling from circuit to device level.

    """

    def __init__(self, qubit: str, **device_overrides) -> None:
        super().__init__(theta=90.0, qubit=qubit, **device_overrides)
        self.data["name"] = f"Z_90 {qubit}"
        self.data["gate_info"]["tex"] = r"$Z_{\pi/2}$"
        self._update()

    def __str__(self) -> str:
        qubit = self.data["gate_info"]["qubits"][0]
        return f"{self.__class__.__name__}(qubit='{qubit}')"


class H(Operation):
    r"""
    A single qubit Hadamard gate.

    Note that the gate uses :math:`R_z(\pi) = -iZ`, adding a global phase of :math:`-\pi/2`.
    This operation can be represented by the following unitary:

    .. math::

        H = Y90 \cdot Z = \frac{-i}{\sqrt{2}}\begin{bmatrix}
             1 & 1 \\
             1 & -1 \\ \end{bmatrix}

    Parameters
    ----------
    qubit
        The target qubit.
    device_overrides
        Device level parameters that override device configuration values
        when compiling from circuit to device level.

    """

    def __init__(self, *qubits: str, **device_overrides) -> None:
        tex = r"$H$"
        plot_func = "quantify_scheduler.schedules._visualization.circuit_diagram.gate_box"

        unitary = -1j / np.sqrt(2) * np.array([[1, 1], [1, -1]], dtype=complex)
        super().__init__(f"H, '{qubits}')")
        self.data["gate_info"] = {
            "unitary": unitary,
            "tex": tex,
            "plot_func": plot_func,
            "qubits": list(qubits),
            "operation_type": "H",
            "device_overrides": device_overrides,
        }
        self._update()

    def __str__(self) -> str:
        qubits = map(lambda x: f"'{x}'", self.data["gate_info"]["qubits"])
        return f'{self.__class__.__name__}({",".join(qubits)})'


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

    Parameters
    ----------
    qC
        The control qubit.
    qT
        The target qubit
    device_overrides
        Device level parameters that override device configuration values
        when compiling from circuit to device level.

    """

    def __init__(self, qC: str, qT: str, **device_overrides) -> None:
        plot_func = "quantify_scheduler.schedules._visualization.circuit_diagram.cnot"
        super().__init__(f"CNOT ({qC}, {qT})")
        self.data.update(
            {
                "name": f"CNOT ({qC}, {qT})",
                "gate_info": {
                    "unitary": np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]),
                    "tex": r"CNOT",
                    "plot_func": plot_func,
                    "qubits": [qC, qT],
                    "symmetric": False,
                    "operation_type": "CNOT",
                    "device_overrides": device_overrides,
                },
            }
        )
        self._update()

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

    Parameters
    ----------
    qC
        The control qubit.
    qT
        The target qubit
    device_overrides
        Device level parameters that override device configuration values
        when compiling from circuit to device level.

    """

    def __init__(self, qC: str, qT: str, **device_overrides) -> None:
        plot_func = "quantify_scheduler.schedules._visualization.circuit_diagram.cz"
        super().__init__(f"CZ ({qC}, {qT})")
        self.data.update(
            {
                "name": f"CZ ({qC}, {qT})",
                "gate_info": {
                    "unitary": np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]]),
                    "tex": r"CZ",
                    "plot_func": plot_func,
                    "qubits": [qC, qT],
                    "symmetric": True,
                    "operation_type": "CZ",
                    "device_overrides": device_overrides,
                },
            }
        )
        self._update()

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

    Parameters
    ----------
    qubits
        The qubit(s) to reset. NB one or more qubits can be specified, e.g.,
        :code:`Reset("q0")`, :code:`Reset("q0", "q1", "q2")`, etc..
    device_overrides
        Device level parameters that override device configuration values
        when compiling from circuit to device level.

    """

    def __init__(self, *qubits: str, **device_overrides) -> None:
        super().__init__(f"Reset {', '.join(qubits)}")
        plot_func = "quantify_scheduler.schedules._visualization.circuit_diagram.reset"
        self.data.update(
            {
                "name": f"Reset {', '.join(qubits)}",
                "gate_info": {
                    "unitary": None,
                    "tex": r"$|0\rangle$",
                    "plot_func": plot_func,
                    "qubits": list(qubits),
                    "operation_type": "reset",
                    "device_overrides": device_overrides,
                },
            }
        )
        self._update()

    def __str__(self) -> str:
        qubits = map(lambda x: f"'{x}'", self.data["gate_info"]["qubits"])
        return f'{self.__class__.__name__}({",".join(qubits)})'


class Measure(Operation):
    """
    A projective measurement in the Z-basis.

    The measurement is compiled according to the type of acquisition specified
    in the device configuration.

    .. note::

        Strictly speaking this is not a gate as it can not
        be described by a unitary.

    Parameters
    ----------
    qubits
        The qubits you want to measure.
    acq_channel
        Only for special use cases.
        By default (if None): the acquisition channel specified in the device element is used.
        If set, this acquisition channel is used for this measurement.
    acq_index
        Index of the register where the measurement is stored.  If None specified,
        this defaults to writing the result of all qubits to acq_index 0. By default
        None.
    acq_protocol : "SSBIntegrationComplex" | "Trace" | "TriggerCount" | \
            "NumericalSeparatedWeightedIntegration" | \
            "NumericalWeightedIntegration" | None, optional
        Acquisition protocols that are supported. If ``None`` is specified, the
        default protocol is chosen based on the device and backend configuration. By
        default None.
    bin_mode
        The binning mode that is to be used. If not None, it will overwrite the
        binning mode used for Measurements in the circuit-to-device compilation
        step. By default None.
    feedback_trigger_label : str
        The label corresponding to the feedback trigger, which is mapped by the
        compiler to a feedback trigger address on hardware, by default None.
    device_overrides
        Device level parameters that override device configuration values
        when compiling from circuit to device level.

    """

    def __init__(
        self,
        *qubits: str,
        acq_channel: Hashable | None = None,
        acq_index: tuple[int, ...] | int | None = None,
        # These are the currently supported acquisition protocols.
        acq_protocol: (
            Literal[
                "SSBIntegrationComplex",
                "Timetag",
                "TimetagTrace",
                "Trace",
                "TriggerCount",
                "NumericalSeparatedWeightedIntegration",
                "NumericalWeightedIntegration",
                "ThresholdedAcquisition",
            ]
            | None
        ) = None,
        bin_mode: BinMode | str | None = None,
        feedback_trigger_label: str | None = None,
        **device_overrides,
    ) -> None:
        # this if else statement a workaround to support multiplexed measurements (#262)

        # this snippet has some automatic behaviour that is error prone.
        # see #262
        if len(qubits) == 1:
            if acq_index is None:
                acq_index = 0
        elif isinstance(acq_index, int):
            acq_index = [
                acq_index,
            ] * len(qubits)
        elif acq_index is None:
            # defaults to writing the result of all qubits to acq_index 0.
            # note that this will result in averaging data together if multiple
            # measurements are present in the same schedule (#262)
            acq_index = list(0 for i in range(len(qubits)))

        plot_func = "quantify_scheduler.schedules._visualization.circuit_diagram.meter"
        super().__init__(f"Measure {', '.join(qubits)}")
        self.data.update(
            {
                "name": f"Measure {', '.join(qubits)}",
                "gate_info": {
                    "unitary": None,
                    "plot_func": plot_func,
                    "tex": r"$\langle0|$",
                    "qubits": list(qubits),
                    "acq_channel_override": acq_channel,
                    "acq_index": acq_index,
                    "acq_protocol": acq_protocol,
                    "bin_mode": bin_mode,
                    "operation_type": "measure",
                    "feedback_trigger_label": feedback_trigger_label,
                    "device_overrides": device_overrides,
                },
            }
        )
        self._update()

    def __str__(self) -> str:
        gate_info = self.data["gate_info"]
        qubits = map(lambda x: f"'{x}'", gate_info["qubits"])
        acq_channel = gate_info["acq_channel_override"]
        acq_index = gate_info["acq_index"]
        acq_protocol = gate_info["acq_protocol"]
        bin_mode = gate_info["bin_mode"]
        feedback_trigger_label = gate_info["feedback_trigger_label"]
        return (
            f'{self.__class__.__name__}({",".join(qubits)}, '
            f"acq_channel={acq_channel}, "
            f"acq_index={acq_index}, "
            f'acq_protocol="{acq_protocol}", '
            f"bin_mode={str(bin_mode)}, "
            f"feedback_trigger_label={feedback_trigger_label})"
        )


def _modulo_360_with_mapping(theta: float) -> float:
    """
    Maps an input angle ``theta`` (in degrees) onto the range ``]-180, 180]``.

    By mapping the input angle to the range ``]-180, 180]`` (where -180 is
    excluded), it ensures that the output amplitude is always minimized on the
    hardware. This mapping should not have an effect on the qubit in general.

    -180 degrees is excluded to ensure positive amplitudes in the gates like
    X180 and Z180.

    Note that an input of -180 degrees is remapped to 180 degrees to maintain
    the positive amplitude constraint.

    Parameters
    ----------
    theta : float
        The rotation angle in degrees. This angle will be mapped to the interval
        ``]-180, 180]``.

    Returns
    -------
    float
        The mapped angle in degrees, which will be in the range ``]-180, 180]``.
        This mapping ensures the output amplitude is always minimized for
        transmon operations.

    Example
    -------
    ```
    >>> _modulo_360_with_mapping(360)
    0.0
    >>> _modulo_360_with_mapping(-180)
    180.0
    >>> _modulo_360_with_mapping(270)
    -90.0
    ```

    """
    mapped_theta = -((-theta - 180) % 360) + 180
    return mapped_theta
