---
file_format: mystnb
kernelspec:
    name: python3

---
(sec-qblox-acquisition-details)=

# Acquisition details

This page describes important Qblox-specific behaviour of the {class}`~quantify_scheduler.operations.acquisition_library.TriggerCount`, {class}`~quantify_scheduler.operations.acquisition_library.Timetag`, {class}`~quantify_scheduler.operations.acquisition_library.TimetagTrace` and {class}`~quantify_scheduler.operations.acquisition_library.Trace` acquisition protocols. Explanations of the protocols themselves can be found in {ref}`sec-acquisition-protocols` and detailed usage examples can be found in the {ref}`sec-acquisitions` tutorials.

## Duration

On all Qblox modules, the actual duration of the trigger count, timetag and timetag-trace acquisitions is **4 ns shorter** than the duration specified upon creating the operation. For example, a `TriggerCount(port="q0:res", clock="q0.ro", duration=1e-6)` will acquire data for 996 ns, but the operation will occupy 1 µs of schedule time. The start time of the actual acquisition is the same as the start time of the operation.

## Bin modes and module support

Not all acquisitions work with all {class}`bin modes <quantify_scheduler.enums.BinMode>` or module types. The table below lists exactly what is supported for the {class}`~quantify_scheduler.operations.acquisition_library.TriggerCount`, {class}`~quantify_scheduler.operations.acquisition_library.Timetag`, {class}`~quantify_scheduler.operations.acquisition_library.TimetagTrace` and {class}`~quantify_scheduler.operations.acquisition_library.Trace` acquisitions.

For more information about the bin modes, please see the {ref}`tutorials <sec-acquisitions>` and {ref}`reference guide <sec-acquisition-protocols>`.

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-0pky{border-color:inherit;text-align:left;vertical-align:top}
</style>
<table class="tg"><thead>
  <tr>
    <th class="tg-0pky"><span style="font-weight:bold">Protocol</span></th>
    <th class="tg-0pky"><span style="font-weight:bold">Support modules</span></th>
    <th class="tg-0pky"><span style="font-weight:bold">Supported bin modes</span></th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-0pky" rowspan="2">TriggerCount</td>
    <td class="tg-0pky">QRM</td>
    <td class="tg-0pky"><code>BinMode.APPEND</code>, <code>BinMode.AVERAGE</code></td>
  </tr>
  <tr>
    <td class="tg-0pky">QTM</td>
    <td class="tg-0pky"><code>BinMode.APPEND</code></td>
  </tr>
  <tr>
    <td class="tg-0pky">Timetag</td>
    <td class="tg-0pky">QTM</td>
    <td class="tg-0pky"><code>BinMode.APPEND</code>, <code>BinMode.AVERAGE</code></td>
  </tr>
  <tr>
    <td class="tg-0pky">TimetagTrace</td>
    <td class="tg-0pky">QTM</td>
    <td class="tg-0pky"><code>BinMode.APPEND</code></td>
  </tr>
  <tr>
    <td class="tg-0pky" rowspan="2">Trace</td>
    <td class="tg-0pky">QRM, QRM-RF</td>
    <td class="tg-0pky"><code>BinMode.AVERAGE</code></td>
  </tr>
  <tr>
    <td class="tg-0pky">QTM</td>
    <td class="tg-0pky"><code>BinMode.FIRST</code></td>
  </tr>
</tbody>
</table>
