---
file_format: mystnb
kernelspec:
    name: python3

---
(sec-qblox-trigger-count)=

# Trigger count acquisition

This page describes important Qblox-specific behaviour of the {class}`~quantify_scheduler.operations.acquisition_library.TriggerCount` acquisition protocol. An explanation of the protocol itself can be found in {ref}`sec-acquisition-protocols-trigger-count` and detailed usage examples can be found in the {ref}`sec-acquisitions-trigger-count` tutorial.

## Duration

On all Qblox modules, the actual duration of the acquisition is **4 ns shorter** than the duration specified upon creating the operation. For example, a `TriggerCount(port="q0:res", clock="q0.ro", duration=1e-6)` will acquire data for 996 ns, but the operation will occupy 1 µs of schedule time. The start time of the actual acquisition is the same as the start time of the operation.

## Bin modes

QRM modules implement both {data}`BinMode.APPEND <quantify_scheduler.enums.BinMode.APPEND>` and {data}`BinMode.AVERAGE <quantify_scheduler.enums.BinMode.AVERAGE>` modes, while QTM modules only implement {data}`BinMode.APPEND <quantify_scheduler.enums.BinMode.APPEND>` mode. The {data}`BinMode.APPEND <quantify_scheduler.enums.BinMode.APPEND>` behaviour is the same for both module types.
