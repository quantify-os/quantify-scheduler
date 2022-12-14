# Repository: https://gitlab.com/quantify-os/quantify-scheduler
# Licensed according to the LICENCE file on the main branch
"""Compiler for the quantify_scheduler."""
import logging
from copy import deepcopy
from typing import Literal, Optional, Union

from quantify_core.utilities import deprecated
from quantify_scheduler.backends.circuit_to_device import DeviceCompilationConfig
from quantify_scheduler.backends.graph_compilation import (
    SimpleNodeConfig,
    SerialCompilationConfig,
)
from quantify_scheduler.helpers.importers import import_python_object_from_string
from quantify_scheduler.json_utils import load_json_schema, validate_json
from quantify_scheduler.schedules.schedule import CompiledSchedule, Schedule

logger = logging.getLogger(__name__)


def determine_absolute_timing(
    schedule: Schedule, time_unit: Literal["physical", "ideal", None] = "physical"
) -> Schedule:
    """
    Determines the absolute timing of a schedule based on the timing constraints.

    This function determines absolute timings for every operation in the
    :attr:`~.ScheduleBase.schedulables`. It does this by:

        1. iterating over all and elements in the :attr:`~.ScheduleBase.schedulables`.
        2. determining the absolute time of the reference operation.
        3. determining the start of the operation based on the `rel_time` and `duration`
           of operations.

    Parameters
    ----------
    schedule
        The schedule for which to determine timings.
    time_unit
        Whether to use physical units to determine the absolute time or ideal time.
        When :code:`time_unit == "physical"` the duration attribute is used.
        When :code:`time_unit == "ideal"` the duration attribute is ignored and treated
        as if it is :code:`1`.
        When :code:`time_unit == None` it will revert to :code:`"physical"`.

    Returns
    -------
    :
        a new schedule object where the absolute time for each operation has been
        determined.
    """

    if len(schedule.schedulables) == 0:
        raise ValueError(f"schedule '{schedule.name}' contains no schedulables.")

    if time_unit is None:
        time_unit = "physical"
    valid_time_units = ("physical", "ideal")
    if time_unit not in valid_time_units:
        raise ValueError(
            f"Undefined time_unit '{time_unit}'! Must be one of {valid_time_units}"
        )

    # iterate over the objects in the schedule.
    last_schedulable = next(iter(schedule.schedulables.values()))
    last_op = schedule.operations[last_schedulable["operation_repr"]]

    last_schedulable["abs_time"] = 0

    for schedulable in list(schedule.data["schedulables"].values())[1:]:
        curr_op = schedule.operations[schedulable["operation_repr"]]
        if len(schedulable.data["timing_constraints"]) == 0:
            schedulable.add_timing_constraint(ref_schedulable=last_schedulable)
        for t_constr in schedulable.data["timing_constraints"]:
            if t_constr["ref_schedulable"] is None:
                ref_schedulable = last_schedulable
                ref_op = last_op
            else:
                # this assumes the reference op exists. This is ensured in schedule.add
                ref_schedulable = schedule.schedulables[
                    str(t_constr["ref_schedulable"])
                ]
                ref_op = schedule.operations[ref_schedulable["operation_repr"]]

            # duration = 1 is useful when e.g., drawing a circuit diagram.
            duration_ref_op = ref_op.duration if time_unit == "physical" else 1

            if t_constr["ref_pt"] == "start":
                t0 = ref_schedulable["abs_time"]
            elif t_constr["ref_pt"] == "center":
                t0 = ref_schedulable["abs_time"] + duration_ref_op / 2
            elif t_constr["ref_pt"] == "end":
                t0 = ref_schedulable["abs_time"] + duration_ref_op
            else:
                raise NotImplementedError(
                    f'Timing "{ref_schedulable["abs_time"]}" not supported by backend.'
                )

            duration_new_op = curr_op.duration if time_unit == "physical" else 1

            if t_constr["ref_pt_new"] == "start":
                abs_time = t0 + t_constr["rel_time"]
            elif t_constr["ref_pt_new"] == "center":
                abs_time = t0 + t_constr["rel_time"] - duration_new_op / 2
            elif t_constr["ref_pt_new"] == "end":
                abs_time = t0 + t_constr["rel_time"] - duration_new_op
            if "abs_time" not in schedulable or abs_time > schedulable["abs_time"]:
                schedulable["abs_time"] = abs_time

        # update last_constraint and operation for next iteration of the loop
        last_schedulable = schedulable
        last_op = curr_op

    return schedule


def _find_edge(device_cfg, parent_element_name, child_element_name, op_name):
    try:
        edge_cfg = device_cfg["edges"][f"{parent_element_name}_{child_element_name}"]
    except KeyError as e:
        raise ValueError(
            f"Attempting operation '{op_name}' on qubits {parent_element_name} "
            f"and {child_element_name} which lack a connective edge."
        ) from e
    return edge_cfg


def validate_config(config: dict, scheme_fn: str) -> bool:
    """
    Validate a configuration using a schema.

    Parameters
    ----------
    config
        The configuration to validate
    scheme_fn
        The name of a json schema in the quantify_scheduler.schemas folder.

    Returns
    ----------
    :
        True if valid
    """
    scheme = load_json_schema(__file__, scheme_fn)
    validate_json(config, scheme)
    return True


@deprecated(
    "0.9.0",
    "Use the `QuantifyCompiler.compile` method instead. "
    "See the user guide section on compilers for details.",
)
def qcompile(
    schedule: Schedule,
    device_cfg: Optional[Union[DeviceCompilationConfig, dict]] = None,
    hardware_cfg: Optional[dict] = None,
) -> CompiledSchedule:
    # pylint: disable=line-too-long
    """
    Compile and assemble a :class:`~.Schedule` into a
    :class:`~.CompiledSchedule` ready for execution using the
    :class:`~.InstrumentCoordinator`.

    Parameters
    ----------
    schedule
        The schedule to be compiled.
    device_cfg
        Device specific configuration, defines the compilation step from
        the quantum-circuit layer to the quantum-device layer description.
    hardware_cfg
        Hardware configuration, defines the compilation step from
        the quantum-device to a hardware layer.

    Returns
    -------
    :
        The prepared schedule if no backend is provided, otherwise whatever object
        returned by the backend


    .. rubric:: Configuration specification

    .. jsonschema:: https://gitlab.com/quantify-os/quantify-scheduler/-/raw/main/quantify_scheduler/schemas/transmon_cfg.json

    .. todo::

        Add a schema for the hardware config.
    """

    def _construct_compilation_config_from_dev_hw_cfg(device_config, hardware_config):

        compilation_passes = []

        # this is to support compiling when no device config is supplied
        if device_config is not None:
            compilation_passes.append(
                SimpleNodeConfig(
                    name="circuit_to_device",
                    compilation_func=device_config.backend,
                    compilation_options=device_config,
                )
            )

        compilation_passes.append(
            SimpleNodeConfig(
                name="determine_absolute_timing",
                compilation_func="quantify_scheduler.compilation.determine_absolute_timing",
            )
        )

        # If statements to support the different (currently unstructured) hardware
        # configs.
        if hardware_config is None:
            backend_name = "Device compilation"
        elif (
            hardware_config["backend"]
            == "quantify_scheduler.backends.qblox_backend.hardware_compile"
        ):
            backend_name = "Qblox backend"
            compilation_passes.append(
                SimpleNodeConfig(
                    name="qblox_hardware_compile",
                    compilation_func=hardware_config["backend"],
                    compilation_options=hardware_config,
                )
            )
        elif (
            hardware_config["backend"]
            == "quantify_scheduler.backends.zhinst_backend.compile_backend"
        ):
            backend_name = "Zhinst backend"
            compilation_passes.append(
                SimpleNodeConfig(
                    name="zhinst_hardware_compile",
                    compilation_func=hardware_config["backend"],
                    compilation_options=hardware_config,
                )
            )

        else:
            raise NotImplementedError(
                f"Hardware backend {hardware_config['backend']} not recognized"
            )

        compilation_config = SerialCompilationConfig(
            name=backend_name, compilation_passes=compilation_passes
        )
        return compilation_config

    compilation_config = _construct_compilation_config_from_dev_hw_cfg(
        device_cfg, hardware_cfg
    )

    # to prevent the original input schedule from being modified.
    schedule = deepcopy(schedule)

    backend_class = import_python_object_from_string(compilation_config.backend)
    backend = backend_class(name=compilation_config.name)
    compiled_schedule = backend.compile(schedule=schedule, config=compilation_config)

    return compiled_schedule


@deprecated(
    "0.9.0",
    "Use the `QuantifyCompiler.compile` method instead. "
    "See the user guide section on compilers for details.",
)
def device_compile(schedule: Schedule, device_cfg: DeviceCompilationConfig) -> Schedule:
    """
    Add pulse information to operations based on device config file.

    Parameters
    ----------
    schedule
        To be compiled.
    device_cfg
        Device specific configuration, defines the compilation step from
        the gate-level to the pulse level description.

    Returns
    -------
    :
        The updated schedule.
    """

    device_compilation_bck = import_python_object_from_string(device_cfg.backend)

    schedule = device_compilation_bck(schedule=schedule, device_cfg=device_cfg)
    schedule = determine_absolute_timing(schedule=schedule, time_unit="physical")

    return schedule


@deprecated(
    "0.9.0",
    "Use the `QuantifyCompiler.compile` method instead. "
    "See the user guide section on compilers for details.",
)
def hardware_compile(schedule: Schedule, hardware_cfg: dict) -> CompiledSchedule:
    """
    Add compiled instructions to the schedule based on the hardware config file.

    Parameters
    ----------
    schedule
        To be compiled.
    hardware_cfg
        Hardware specific configuration, defines the compilation step from
        the quantum-device layer to the control-hardware layer.

    Returns
    -------
    :
        The compiled schedule.
    """

    hw_compile = import_python_object_from_string(hardware_cfg["backend"])
    compiled_schedule = hw_compile(schedule, hardware_cfg=hardware_cfg)
    return compiled_schedule
