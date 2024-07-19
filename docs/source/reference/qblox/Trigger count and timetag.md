---
file_format: mystnb
kernelspec:
    name: python3

---
(sec-qblox-trigger-count)=

# Trigger count and Timetag acquisition

This page describes important Qblox-specific behaviour of the {class}`~quantify_scheduler.operations.acquisition_library.TriggerCount` and {class}`~quantify_scheduler.operations.acquisition_library.Timetag` acquisition protocols. Explanations of the protocols themselves can be found in {ref}`sec-acquisition-protocols` and detailed usage examples can be found in the {ref}`sec-acquisitions` tutorials.

## Duration

On all Qblox modules, the actual duration of the trigger count and timetag acquisitions is **4 ns shorter** than the duration specified upon creating the operation. For example, a `TriggerCount(port="q0:res", clock="q0.ro", duration=1e-6)` will acquire data for 996 ns, but the operation will occupy 1 µs of schedule time. The start time of the actual acquisition is the same as the start time of the operation.

## Bin modes and module support

Not all acquisitions work with all bin modes or module types. This section lists exactly what is supported for the {class}`~quantify_scheduler.operations.acquisition_library.TriggerCount` and {class}`~quantify_scheduler.operations.acquisition_library.Timetag` acquisitions.

For the {class}`~quantify_scheduler.operations.acquisition_library.TriggerCount` acquisition, QRM modules support both {data}`BinMode.APPEND <quantify_scheduler.enums.BinMode.APPEND>` and {data}`BinMode.AVERAGE <quantify_scheduler.enums.BinMode.AVERAGE>` modes. In contrast, QTM modules only support {data}`BinMode.APPEND <quantify_scheduler.enums.BinMode.APPEND>`, with the same behaviour as in QRM modules.

The {class}`~quantify_scheduler.operations.acquisition_library.Timetag` acquisition is only possible on the QTM and supports both bin modes.

For more information about the bin modes, please see the {ref}`tutorials <sec-acquisitions>` and {ref}`reference guide <sec-acquisition-protocols>`.
