"""
Compilation backend for quantum-circuit to quantum-device layer.
"""
from copy import deepcopy
from quantify_scheduler.schedules.schedule import Schedule
from quantify_core.utilities.general import import_python_object_from_string


def compile_circuit_to_device(schedule: Schedule, device_cfg: dict) -> Schedule:
    """
    Adds pulse_info and acquisition_info to all operations that have gate_info
    specified
    """

    # validate_config(device_cfg, scheme_fn=new_config_format)

    for op in schedule.operations.values():
        if op.valid_pulse:
            _verify_pulse_clock_present(operation=op, schedule=schedule)
            continue
        if op.valid_acquisition:
            # no verification at this point.
            continue

        # if operation is a valid pulse or acquisition it will not attempt to add
        # pulse/acquisition info in the lines below.

        qubits = op.data["gate_info"]["qubits"]
        operation_type = op.data["gate_info"]["operation_type"]

        if len(qubits) == 1:
            qubit = qubits[0]
            # deepcopy because operation_type can occur multiple times
            # (e.g., parametrized operations)
            operation_cfg = deepcopy(device_cfg["qubits"][qubit][operation_type])
            # TODO: add proper exception for when qubit key does not exist
            # TODO: add proper exception when operation type is not in the gateset (including
            # what operations are in the gateset).

            generator_func = operation_cfg.pop("generator_func")
            # if specified as an importable string, import the function.
            if isinstance(generator_func, str):
                generator_func = import_python_object_from_string(generator_func)

            generator_kwargs = {}
            # retrieve keyword args for parametrized operations from the gate info
            if "gate_info_generator_kwargs" in operation_cfg:
                for key in operation_cfg.pop("gate_info_generator_kwargs"):
                    generator_kwargs[key] = op.data["gate_info"][key]

            # add all other keyword args from the device configuration file.
            # the pop of generator_func and _kwargs should ensure the arguments match.
            generator_kwargs.update(operation_cfg)

            # FIXME: create the generator functions
            device_op = generator_func(**generator_kwargs)
            # FIXME: check if this is how the operations should be added.
            if isinstance(device_op, AcquisitionOperation):
                schedule.add_acquisition(device_op)
            else:
                schedule.add_pulse(device_op)

        elif len(qubits) == 2:
            edge = f"{qubits[0]}-{qubits[1]}"
            operation_cfg = device_cfg["edges"][edge][gate_type]
        else:
            raise ValueError("Operations on more than 2 qubits are not supported")


def _verify_pulse_clock_present(operation, schedule):
    for pulse in operation["pulse_info"]:
        if "clock" in pulse:
            if pulse["clock"] not in schedule.resources:
                raise ValueError(
                    "Operation '{}' contains an unknown clock '{}'; ensure "
                    "this resource has been added to the schedule.".format(
                        str(operation), pulse["clock"]
                    )
                )


def add_pulse_information_transmon(schedule: Schedule, device_cfg: dict) -> Schedule:
    """
    Adds pulse information specified in the device config to the schedule.

    Parameters
    ------------
    schedule
        The schedule for which to add pulse information.

    device_cfg
        A dictionary specifying the required pulse information.


    Returns
    ----------
    :
        a new schedule object where the pulse information has been added.


    .. rubric:: Supported operations


    The following gate type operations are supported by this compilation step.

    - :class:`~quantify_scheduler.operations.gate_library.Rxy`
    - :class:`~quantify_scheduler.operations.gate_library.Reset`
    - :class:`~quantify_scheduler.operations.gate_library.Measure`
    - :class:`~quantify_scheduler.operations.gate_library.CZ`


    .. rubric:: Configuration specification

    .. jsonschema:: schemas/transmon_cfg.json

    """
    validate_config(device_cfg, scheme_fn="transmon_cfg.json")

    for op in schedule.operations.values():
        if op.valid_pulse:
            for p in op["pulse_info"]:
                if "clock" in p:
                    if p["clock"] not in schedule.resources:
                        raise ValueError(
                            "Operation '{}' contains an unknown clock '{}'; ensure "
                            "this resource has been added to the schedule.".format(
                                str(op), p["clock"]
                            )
                        )
            continue

        if op.valid_acquisition:
            continue

        if op["gate_info"]["operation_type"] == "measure":
            for idx, q in enumerate(op["gate_info"]["qubits"]):
                q_cfg = device_cfg["qubits"][q]

                # If the user specifies bin-mode use that otherwise use a default
                # better would be to get it from the config file in the "or"
                bin_mode = op["gate_info"]["bin_mode"] or BinMode.AVERAGE

                if q_cfg["params"]["acquisition"] == "SSBIntegrationComplex":
                    # readout pulse
                    op.add_pulse(
                        SquarePulse(
                            amp=q_cfg["params"]["ro_pulse_amp"],
                            duration=q_cfg["params"]["ro_pulse_duration"],
                            port=q_cfg["resources"]["port_ro"],
                            clock=q_cfg["resources"]["clock_ro"],
                        )
                    )
                    op.add_acquisition(
                        SSBIntegrationComplex(
                            duration=q_cfg["params"]["ro_acq_integration_time"],
                            t0=q_cfg["params"]["ro_acq_delay"],
                            acq_channel=op["gate_info"]["acq_channel"][idx],
                            acq_index=op["gate_info"]["acq_index"][idx],
                            port=q_cfg["resources"]["port_ro"],
                            clock=q_cfg["resources"]["clock_ro"],
                            bin_mode=bin_mode,
                        )
                    )

                    if q_cfg["resources"]["clock_ro"] not in schedule.resources.keys():
                        schedule.add_resources(
                            [
                                ClockResource(
                                    q_cfg["resources"]["clock_ro"],
                                    freq=q_cfg["params"]["ro_freq"],
                                )
                            ]
                        )

                if q_cfg["params"]["acquisition"] == "Trace":
                    # readout pulse
                    op.add_pulse(
                        SquarePulse(
                            amp=q_cfg["params"]["ro_pulse_amp"],
                            duration=q_cfg["params"]["ro_pulse_duration"],
                            port=q_cfg["resources"]["port_ro"],
                            clock=q_cfg["resources"]["clock_ro"],
                        )
                    )
                    # pylint: disable=fixme
                    op.add_acquisition(  # TODO protocol hardcoded
                        Trace(
                            clock=q_cfg["resources"]["clock_ro"],
                            duration=q_cfg["params"]["ro_acq_integration_time"],
                            t0=q_cfg["params"]["ro_acq_delay"],
                            acq_channel=op["gate_info"]["acq_channel"][idx],
                            acq_index=op["gate_info"]["acq_index"][idx],
                            port=q_cfg["resources"]["port_ro"],
                        )
                    )

                    if q_cfg["resources"]["clock_ro"] not in schedule.resources.keys():
                        schedule.add_resources(
                            [
                                ClockResource(
                                    q_cfg["resources"]["clock_ro"],
                                    freq=q_cfg["params"]["ro_freq"],
                                )
                            ]
                        )

        elif op["gate_info"]["operation_type"] == "Rxy":
            q = op["gate_info"]["qubits"][0]
            # read info from config
            q_cfg = device_cfg["qubits"][q]

            # G_amp is the gaussian amplitude introduced in
            # https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.103.110501
            # 180 refers to the normalization, theta is in degrees, and
            # mw_amp180 is the amplitude necessary to get the
            # maximum 180 degree theta (experimentally)
            G_amp = q_cfg["params"]["mw_amp180"] * op["gate_info"]["theta"] / 180
            D_amp = G_amp * q_cfg["params"]["mw_motzoi"]

            pulse = DRAGPulse(
                G_amp=G_amp,
                D_amp=D_amp,
                phase=op["gate_info"]["phi"],
                port=q_cfg["resources"]["port_mw"],
                duration=q_cfg["params"]["mw_duration"],
                clock=q_cfg["resources"]["clock_01"],
            )
            op.add_pulse(pulse)

            # add clock to resources
            if q_cfg["resources"]["clock_01"] not in schedule.resources.keys():
                schedule.add_resources(
                    [
                        ClockResource(
                            q_cfg["resources"]["clock_01"],
                            freq=q_cfg["params"]["mw_freq"],
                        )
                    ]
                )

        elif op["gate_info"]["operation_type"] == "CNOT":
            # These methods don't raise exceptions as they will be implemented shortly
            logger.warning("Not Implemented yet")
            logger.warning(
                'Operation type "{}" not supported by backend'.format(
                    op["gate_info"]["operation_type"]
                )
            )

        elif op["gate_info"]["operation_type"] == "CZ":
            # pylint: disable=fixme
            # todo mock implementation, needs a proper version before release
            q0 = op["gate_info"]["qubits"][0]
            q1 = op["gate_info"]["qubits"][1]

            # this reflective edge is a unique property of the CZ gate
            try:
                edge_cfg = _find_edge(device_cfg, q0, q1, "CZ")
            except ValueError:
                try:
                    edge_cfg = _find_edge(device_cfg, q1, q0, "CZ")
                except ValueError as e:
                    raise e

            amp = edge_cfg["params"]["flux_amp_control"]

            # pylint: disable=fixme
            # FIXME: placeholder. currently puts a soft square pulse on the designated
            # port of both qubits
            pulse = SoftSquarePulse(
                amp=amp,
                duration=edge_cfg["params"]["flux_duration"],
                port=edge_cfg["resource_map"][q0],
                clock=BasebandClockResource.IDENTITY,
            )
            op.add_pulse(pulse)

            pulse = SoftSquarePulse(
                amp=amp,
                duration=edge_cfg["params"]["flux_duration"],
                port=edge_cfg["resource_map"][q1],
                clock=BasebandClockResource.IDENTITY,
            )

            op.add_pulse(pulse)
        elif op["gate_info"]["operation_type"] == "reset":
            # Initialization through relaxation
            qubits = op["gate_info"]["qubits"]
            init_times = []
            for q in qubits:
                init_times.append(device_cfg["qubits"][q]["params"]["init_duration"])
            op.add_pulse(IdlePulse(max(init_times)))

        else:
            raise NotImplementedError(
                'Operation type "{}" not supported by backend'.format(
                    op["gate_info"]["operation_type"]
                )
            )

    return schedule
