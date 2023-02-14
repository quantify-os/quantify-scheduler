---
file_format: mystnb
kernelspec:
    name: python3

mystnb:
  execution_timeout: 120
---
(sec-tutorial-ops-qubits)=

# Tutorial: Operations and Qubits

```{seealso}
The complete source code of this tutorial can be found in

{nb-download}`Operations and Qubits.ipynb`
```

## Gates, measurements and qubits

In the previous tutorials, experiments were created on the {ref}`quantum-device level<sec-user-guide-quantum-device>`. On this level,
operations are defined in terms of explicit signals and locations on chip, rather than the qubit and the intended operation.
To work at a greater level of abstraction, `quantify_scheduler` allows creating operations on the
{ref}`quantum-circuit level<sec-user-guide-quantum-circuit>`.
Instead of signals, clocks, and ports, operations are defined by the the effect they have on specific qubits. This representation of the schedules can be compiled to the quantum-device level to create the pulse schemes.

In this tutorial we show how to define operations on the {ref}`quantum-circuit level<sec-user-guide-quantum-circuit>`, combine them into schedules, and show their circuit-level visualization.
We go through the configuration file needed to compile the schedule to the quantum-device level and show how these configuration files can be created automatically and dynamically.
Finally, we showcase the hybrid nature of `quantify_scheduler`, allowing the scheduling circuit-level and device-level operations side by side in the same schedule.

Many of the gates used in the circuit layer description are defined in
{class}`~quantify_scheduler.operations.gate_library` such as {class}`~quantify_scheduler.operations.gate_library.Reset`, {class}`~quantify_scheduler.operations.gate_library.X90` and
{class}`~quantify_scheduler.operations.gate_library.Measure`.
Operations are instantiated by providing them with the name of the qubit(s) on which
they operate:


```python
from quantify_scheduler.operations.gate_library import CZ, Measure, Reset, X90

q0, q1 = ("q0", "q1")
X90(q0)
Measure(q1)
CZ(q0, q1)
Reset(q0)


```




    {'name': 'Reset q0', 'gate_info': {'unitary': None, 'tex': '$|0\\rangle$', 'plot_func': 'quantify_scheduler.visualization.circuit_diagram.reset', 'qubits': ['q0'], 'operation_type': 'reset'}, 'pulse_info': [], 'acquisition_info': [], 'logic_info': {}}



Let's investigate the different components present in the circuit-level description of
the operation. As an example, we create a 45 degree rotation operation over the
x-axis.


```python
from pprint import pprint
from quantify_scheduler.operations.gate_library import Rxy

rxy45 = Rxy(theta=45.0, phi=0.0, qubit=q0)
pprint(rxy45.data)


```

    {'acquisition_info': [],
     'gate_info': {'operation_type': 'Rxy',
                   'phi': 0.0,
                   'plot_func': 'quantify_scheduler.visualization.circuit_diagram.gate_box',
                   'qubits': ['q0'],
                   'tex': '$R_{xy}^{45, 0}$',
                   'theta': 45.0,
                   'unitary': array([[0.92387953+0.j        , 0.        -0.38268343j],
           [0.        -0.38268343j, 0.92387953+0.j        ]])},
     'logic_info': {},
     'name': "Rxy(45, 0, 'q0')",
     'pulse_info': []}
    

As we can see, the structure of a circuit-level operation is similar to a pulse-level
operation. However, the information is contained inside the {code}`gate_info` entry rather
than the {code}`pulse_info` entry of the data dictionary.
Importantly, there is no device-specific information coupled to the operation such that
it represents the abstract notion of this qubit rotation, rather than how to perform it
on any physical qubit implementation.

The entries present above are documented in the `operation` schema.
Generally, these schemas are only important when defining custom operations, which is
not part of this tutorial. This schema can be inspected via:


```python
import importlib.resources
import json
from quantify_scheduler import schemas

operation_schema = json.loads(importlib.resources.read_text(schemas, "operation.json"))
pprint(operation_schema["properties"]["gate_info"]["properties"])


```

    {'operation_type': {'description': 'Defines what class of operations this gate '
                                       'refers to (e.g. Rxy, CZ etc.).',
                        'type': 'string'},
     'plot_func': {'description': 'reference to a function for plotting this '
                                  'operation. If not specified, defaults to using '
                                  ':func:`~quantify_scheduler.visualization.circuit_diagram.gate_box`.',
                   'type': ['string', 'null']},
     'qubits': {'description': 'A list of strings indicating the qubits the gate '
                               'acts on. Valid qubits are strings that appear in '
                               'the device_config.json file.',
                'type': 'array'},
     'symmetric': {'description': 'A boolean to indicate whether a two qubit gate '
                                  'is symmetric. This is used in the device config '
                                  'compilation stage. By default, it is set as '
                                  'False',
                   'type': 'boolean'},
     'tex': {'description': 'latex snippet for plotting', 'type': 'string'},
     'unitary': {'description': 'A unitary matrix describing the operation.'}}
    

## Schedule creation from the circuit layer (Bell)

The circuit-level operations can be used to create a `schedule` within
`quantify_scheduler` using the same method as for the pulse-level operations.
This enables creating schedules on a more abstract level.
Here, we demonstrate this extra layer of abstraction by creating a `schedule` for measuring
`Bell violations`.

```{note}
Within a single `schedule`, high-level circuit layer operations can be mixed with quantum-device level operations. This mixed representation is useful for experiments where some pulses cannot easily be represented as qubit gates. An example of this is given by the `Chevron` experiment given in {ref}`Mixing pulse and circuit layer operations (Chevron)`.
```

As the first example, we want to create a schedule for performing the
[Bell experiment](https://en.wikipedia.org/wiki/Bell%27s_theorem).
The goal of the Bell experiment is to create a Bell state
{math}`|\Phi ^+\rangle=\frac{1}{2}(|00\rangle+|11\rangle)` which is a perfectly entangled state, followed by a measurement.
By rotating the measurement basis, or equivalently one of the qubits, it is possible
to observe violations of the CSHS inequality.

We create this experiment using the
{ref}`quantum-circuit level<sec-user-guide-quantum-circuit>` description.
This allows defining the Bell schedule as:


```python
import numpy as np
from quantify_scheduler import Schedule
from quantify_scheduler.operations.gate_library import CZ, Measure, Reset, Rxy, X90

sched = Schedule("Bell experiment")

for acq_idx, theta in enumerate(np.linspace(0, 360, 21)):
    sched.add(Reset(q0, q1))
    sched.add(X90(q0))
    sched.add(X90(q1), ref_pt="start")  # Start at the same time as the other X90
    sched.add(CZ(q0, q1))
    sched.add(Rxy(theta=theta, phi=0, qubit=q0))

    sched.add(Measure(q0, acq_index=acq_idx), label="M q0 {:.2f} deg".format(theta))
    sched.add(  # Start at the same time as the other measure
        Measure(q1, acq_index=acq_idx),
        label="M q1 {:.2f} deg".format(theta),
        ref_pt="start",
    )

sched


```




    Schedule "Bell experiment" containing (66) 147  (unique) operations.



By scheduling 7 operations for 21 different values for {code}`theta` we indeed get a schedule containing 7\*21=147 operations. To minimize the size of the schedule, identical operations are stored only once. For example, the {class}`~quantify_scheduler.operations.gate_library.CZ` operation is stored only once but used 21 times, which leaves only 66 unique operations in the schedule.

```{note}
The acquisitions are different for every iteration due to their different {code}`acq_index`. The {class}`~quantify_scheduler.operations.gate_library.Rxy`-gate rotates over a different angle every iteration and must therefore also be different for every iteration (except for the last since {math}`R^{360}=R^0`). Hence the number of unique operations is 3\*21-1+4=66.
```

## Visualizing the quantum circuit

We can directly visualize the created schedule on the
{ref}`quantum-circuit level<sec-user-guide-quantum-circuit>`.
This visualization shows every operation on a line representing the different qubits.


```python
%matplotlib inline
import matplotlib.pyplot as plt

_, ax = sched.plot_circuit_diagram()
# all gates are plotted, but it doesn't all fit in a matplotlib figure.
# Therefore we use :code:`set_xlim` to limit the number of gates shown.
ax.set_xlim(-0.5, 9.5)
plt.show()


```


    
![png](Operations%20and%20Qubits%20%286%29_files/Operations%20and%20Qubits%20%286%29_9_0.png)
    


In previous tutorials, we visualized the `schedules` on the pulse level using {meth}`~quantify_scheduler.schedules.schedule.ScheduleBase.plot_pulse_diagram` .
Up until now, however, all gates have been defined on the
{ref}`quantum-circuit level<sec-user-guide-quantum-circuit>` without defining the
corresponding pulse shapes.
Therefore, trying to run {meth}`~quantify_scheduler.schedules.schedule.ScheduleBase.plot_pulse_diagram` will raise an error which
signifies no {code}`pulse_info` is present in the schedule:


```python
sched.plot_pulse_diagram()


```


    ---------------------------------------------------------------------------

    RuntimeError                              Traceback (most recent call last)

    Cell In [6], line 1
    ----> 1 sched.plot_pulse_diagram()
    

    File c:\Users\David Quinn\anaconda3\envs\quantify-env\lib\site-packages\quantify_scheduler\schedules\schedule.py:293, in ScheduleBase.plot_pulse_diagram(self, port_list, sampling_rate, modulation, modulation_if, plot_backend, plot_kwargs)
        286 if plot_backend == "mpl":
        287     # NB imported here to avoid circular import
        288     # pylint: disable=import-outside-toplevel
        289     from quantify_scheduler.visualization.pulse_diagram import (
        290         pulse_diagram_matplotlib,
        291     )
    --> 293     return pulse_diagram_matplotlib(
        294         schedule=self,
        295         sampling_rate=sampling_rate,
        296         port_list=port_list,
        297         modulation=modulation,
        298         modulation_if=modulation_if,
        299         **plot_kwargs,
        300     )
        301 if plot_backend == "plotly":
        302     # NB imported here to avoid circular import
        303     # pylint: disable=import-outside-toplevel
        304     from quantify_scheduler.visualization.pulse_diagram import (
        305         pulse_diagram_plotly,
        306     )
    

    File c:\Users\David Quinn\anaconda3\envs\quantify-env\lib\site-packages\quantify_scheduler\visualization\pulse_diagram.py:484, in pulse_diagram_matplotlib(schedule, port_list, sampling_rate, modulation, modulation_if, ax)
        450 def pulse_diagram_matplotlib(
        451     schedule: Union[Schedule, CompiledSchedule],
        452     port_list: Optional[List[str]] = None,
       (...)
        456     ax: Optional[matplotlib.axes.Axes] = None,
        457 ) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
        458     """
        459     Plots a schedule using matplotlib.
        460 
       (...)
        482         The matplotlib ax.
        483     """
    --> 484     times, pulses = sample_schedule(
        485         schedule,
        486         sampling_rate=sampling_rate,
        487         port_list=port_list,
        488         modulation=modulation,
        489         modulation_if=modulation_if,
        490     )
        491     if ax is None:
        492         _, ax = plt.subplots()
    

    File c:\Users\David Quinn\anaconda3\envs\quantify-env\lib\site-packages\quantify_scheduler\visualization\pulse_diagram.py:386, in sample_schedule(schedule, port_list, modulation, modulation_if, sampling_rate)
        383 logger.debug(f"time_window {time_window}, port_map {port_map}")
        385 if time_window is None:
    --> 386     raise RuntimeError(
        387         f"Attempting to sample schedule {schedule.name}, "
        388         "but the schedule does not contain any `pulse_info`. "
        389         "Please verify that the schedule has been populated and "
        390         "device compilation has been performed."
        391     )
        393 timestamps = np.arange(time_window[0], time_window[1], 1 / sampling_rate)
        394 waveforms = {key: np.zeros_like(timestamps) for key in port_map}
    

    RuntimeError: Attempting to sample schedule Bell experiment, but the schedule does not contain any `pulse_info`. Please verify that the schedule has been populated and device compilation has been performed.


And similarly for the {code}`timing_table`:


```python
sched.timing_table

```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    Cell In [7], line 1
    ----> 1 sched.timing_table
    

    File c:\Users\David Quinn\anaconda3\envs\quantify-env\lib\site-packages\quantify_scheduler\schedules\schedule.py:368, in ScheduleBase.timing_table(self)
        365 for schedulable in self.schedulables.values():
        366     if "abs_time" not in schedulable:
        367         # when this exception is encountered
    --> 368         raise ValueError("Absolute time has not been determined yet.")
        369     operation = self.operations[schedulable["operation_repr"]]
        371     # iterate over pulse information
    

    ValueError: Absolute time has not been determined yet.


## Device configuration

Up until now the schedule is not specific to any qubit implementation.
The aim of this section is to add device specific information to the schedule.
This knowledge is contained in the {ref}`device configuration<sec-device-config>`, which we introduce in this section.
By compiling the schedule to the quantum-device layer, we incorporate the device configuration into the schedule (for example by adding pulse information to every gate) and thereby enable it to run on a specific qubit implementation.

To start this section, we will unpack the structure of the device configuration.
Here we will use an example device configuration for a transmon-based system that is used in the
`quantify-scheduler` test suite.


```python
from quantify_scheduler.backends.circuit_to_device import DeviceCompilationConfig
from quantify_scheduler.schemas.examples.circuit_to_device_example_cfgs import (
    example_transmon_cfg,
)

device_cfg = DeviceCompilationConfig.parse_obj(example_transmon_cfg)

list(device_cfg.dict())

```




    ['backend', 'clocks', 'elements', 'edges']



Before explaining how this can be used to compile schedules, let us first investigate
the contents of the device configuration.


```python
device_cfg.backend


```




    'quantify_scheduler.backends.circuit_to_device.compile_circuit_to_device'



The backend of the device configuration specifies what function will be used to add
pulse information to the gates. In other words, it specifies how to interpret the
qubit parameters present in the device configuration and achieve the required gates.
Let us briefly investigate the backend function:


```python
from quantify_scheduler.helpers.importers import import_python_object_from_string

help(import_python_object_from_string(device_cfg.backend))


```

    Help on function compile_circuit_to_device in module quantify_scheduler.backends.circuit_to_device:
    
    compile_circuit_to_device(schedule: quantify_scheduler.schedules.schedule.Schedule, device_cfg: Union[quantify_scheduler.backends.circuit_to_device.DeviceCompilationConfig, dict, NoneType] = None) -> quantify_scheduler.schedules.schedule.Schedule
        Adds the information required to represent operations on the quantum-device
        abstraction layer to operations that contain information on how to be represented
        on the quantum-circuit layer.
        
        Parameters
        ----------
        schedule
            The schedule to be compiled.
        device_cfg
            Device specific configuration, defines the compilation step from
            the quantum-circuit layer to the quantum-device layer description.
            Note, if a dictionary is passed, it will be parsed to a
            :class:`~DeviceCompilationConfig`.
    
    

The {ref}`device configuration <sec-device-config>` also contains the
parameters required by the backend for all qubits and edges.


```python
print(list(device_cfg.elements))
print(list(device_cfg.edges))
print(list(device_cfg.clocks))


```

    ['q0', 'q1']
    ['q0_q1']
    ['q0.01', 'q0.ro', 'q1.01', 'q1.ro']
    

For every qubit and edge we can investigate the contained parameters.


```python
print(device_cfg.elements["q0"])
print(device_cfg.elements["q0"]["Rxy"].factory_kwargs)


```

    {'reset': OperationCompilationConfig(factory_func='quantify_scheduler.operations.pulse_library.IdlePulse', factory_kwargs={'duration': 0.0002}, gate_info_factory_kwargs=None), 'Rxy': OperationCompilationConfig(factory_func='quantify_scheduler.operations.pulse_factories.rxy_drag_pulse', factory_kwargs={'amp180': 0.32, 'motzoi': 0.45, 'port': 'q0:mw', 'clock': 'q0.01', 'duration': 2e-08}, gate_info_factory_kwargs=['theta', 'phi']), 'Z': OperationCompilationConfig(factory_func='quantify_scheduler.operations.pulse_library.SoftSquarePulse', factory_kwargs={'amp': 0.23, 'duration': 4e-09, 'port': 'q0:fl', 'clock': 'cl0.baseband'}, gate_info_factory_kwargs=None), 'measure': OperationCompilationConfig(factory_func='quantify_scheduler.operations.measurement_factories.dispersive_measurement', factory_kwargs={'port': 'q0:res', 'clock': 'q0.ro', 'pulse_type': 'SquarePulse', 'pulse_amp': 0.25, 'pulse_duration': 1.6e-07, 'acq_delay': 1.2e-07, 'acq_duration': 3e-07, 'acq_channel': 0}, gate_info_factory_kwargs=['acq_index', 'bin_mode', 'acq_protocol'])}
    {'amp180': 0.32, 'motzoi': 0.45, 'port': 'q0:mw', 'clock': 'q0.01', 'duration': 2e-08}
    


```python
print(device_cfg.edges)


```

    {'q0_q1': {'CZ': OperationCompilationConfig(factory_func='quantify_scheduler.operations.pulse_factories.composite_square_pulse', factory_kwargs={'square_port': 'q0:fl', 'square_clock': 'cl0.baseband', 'square_amp': 0.5, 'square_duration': 2e-08, 'virt_z_parent_qubit_phase': 44, 'virt_z_parent_qubit_clock': 'q0.01', 'virt_z_child_qubit_phase': 63, 'virt_z_child_qubit_clock': 'q1.01'}, gate_info_factory_kwargs=None)}}
    


```python
print(device_cfg.clocks)



```

    {'q0.01': 6020000000.0, 'q0.ro': 7040000000.0, 'q1.01': 5020000000.0, 'q1.ro': 6900000000.0}
    

Lastly, the complete example device configuration (also see {class}`~quantify_scheduler.backends.circuit_to_device.DeviceCompilationConfig`):


```python
pprint(example_transmon_cfg)


```

    {'backend': 'quantify_scheduler.backends.circuit_to_device.compile_circuit_to_device',
     'clocks': {'q0.01': 6020000000.0,
                'q0.ro': 7040000000.0,
                'q1.01': 5020000000.0,
                'q1.ro': 6900000000.0},
     'edges': {'q0_q1': {'CZ': {'factory_func': 'quantify_scheduler.operations.pulse_factories.composite_square_pulse',
                                'factory_kwargs': {'square_amp': 0.5,
                                                   'square_clock': 'cl0.baseband',
                                                   'square_duration': 2e-08,
                                                   'square_port': 'q0:fl',
                                                   'virt_z_child_qubit_clock': 'q1.01',
                                                   'virt_z_child_qubit_phase': 63,
                                                   'virt_z_parent_qubit_clock': 'q0.01',
                                                   'virt_z_parent_qubit_phase': 44}}}},
     'elements': {'q0': {'Rxy': {'factory_func': 'quantify_scheduler.operations.pulse_factories.rxy_drag_pulse',
                                 'factory_kwargs': {'amp180': 0.32,
                                                    'clock': 'q0.01',
                                                    'duration': 2e-08,
                                                    'motzoi': 0.45,
                                                    'port': 'q0:mw'},
                                 'gate_info_factory_kwargs': ['theta', 'phi']},
                         'Z': {'factory_func': 'quantify_scheduler.operations.pulse_library.SoftSquarePulse',
                               'factory_kwargs': {'amp': 0.23,
                                                  'clock': 'cl0.baseband',
                                                  'duration': 4e-09,
                                                  'port': 'q0:fl'}},
                         'measure': {'factory_func': 'quantify_scheduler.operations.measurement_factories.dispersive_measurement',
                                     'factory_kwargs': {'acq_channel': 0,
                                                        'acq_delay': 1.2e-07,
                                                        'acq_duration': 3e-07,
                                                        'clock': 'q0.ro',
                                                        'port': 'q0:res',
                                                        'pulse_amp': 0.25,
                                                        'pulse_duration': 1.6e-07,
                                                        'pulse_type': 'SquarePulse'},
                                     'gate_info_factory_kwargs': ['acq_index',
                                                                  'bin_mode',
                                                                  'acq_protocol']},
                         'reset': {'factory_func': 'quantify_scheduler.operations.pulse_library.IdlePulse',
                                   'factory_kwargs': {'duration': 0.0002}}},
                  'q1': {'Rxy': {'factory_func': 'quantify_scheduler.operations.pulse_factories.rxy_drag_pulse',
                                 'factory_kwargs': {'amp180': 0.4,
                                                    'clock': 'q1.01',
                                                    'duration': 2e-08,
                                                    'motzoi': 0.25,
                                                    'port': 'q1:mw'},
                                 'gate_info_factory_kwargs': ['theta', 'phi']},
                         'measure': {'factory_func': 'quantify_scheduler.operations.measurement_factories.dispersive_measurement',
                                     'factory_kwargs': {'acq_channel': 1,
                                                        'acq_delay': 1.2e-07,
                                                        'acq_duration': 3e-07,
                                                        'clock': 'q1.ro',
                                                        'port': 'q1:res',
                                                        'pulse_amp': 0.21,
                                                        'pulse_duration': 1.6e-07,
                                                        'pulse_type': 'SquarePulse'},
                                     'gate_info_factory_kwargs': ['acq_index',
                                                                  'bin_mode',
                                                                  'acq_protocol']},
                         'reset': {'factory_func': 'quantify_scheduler.operations.pulse_library.IdlePulse',
                                   'factory_kwargs': {'duration': 0.0002}}}}}
    

## Quantum Devices and Elements

The {ref}`device configuration<sec-device-config>` contains all knowledge
of the physical device under test (DUT).
To generate these device configurations on the fly, `quantify_scheduler` provides the
{class}`~quantify_scheduler.device_under_test.quantum_device.QuantumDevice` and
{class}`~quantify_scheduler.device_under_test.device_element.DeviceElement` classes.

These classes contain the information necessary to generate the device configs and allow
changing their parameters on-the-fly.
The {class}`~quantify_scheduler.device_under_test.quantum_device.QuantumDevice` class
represents the DUT containing different {class}`~quantify_scheduler.device_under_test.device_element.DeviceElement` s.
Currently, `quantify_scheduler` contains the
{class}`~quantify_scheduler.device_under_test.transmon_element.BasicTransmonElement` class
to represent a fixed-frequency transmon qubit connected to a feedline. We show their interaction below:


```python
from quantify_scheduler.device_under_test.quantum_device import QuantumDevice
from quantify_scheduler.device_under_test.transmon_element import BasicTransmonElement

# First create a device under test
dut = QuantumDevice("DUT")

# Then create a transmon element
qubit = BasicTransmonElement("qubit")

# Finally, add the transmon element to the QuantumDevice
dut.add_element(qubit)
dut, dut.elements()


```




    (<QuantumDevice: DUT>, ['qubit'])



The different transmon properties can be set through attributes of the {class}`~quantify_scheduler.device_under_test.transmon_element.BasicTransmonElement` class instanc, e.g.:


```python
qubit.clock_freqs.f01(6e9)

print(list(qubit.submodules.keys()))
print()
for submodule_name, submodule in qubit.submodules.items():
    print(f"{qubit.name}.{submodule_name}: {list(submodule.parameters.keys())}")


```

    ['reset', 'rxy', 'measure', 'ports', 'clock_freqs']
    
    qubit.reset: ['duration']
    qubit.rxy: ['amp180', 'motzoi', 'duration']
    qubit.measure: ['pulse_type', 'pulse_amp', 'pulse_duration', 'acq_channel', 'acq_delay', 'integration_time', 'reset_clock_phase', 'acq_weight_type']
    qubit.ports: ['microwave', 'flux', 'readout']
    qubit.clock_freqs: ['f01', 'f12', 'readout']
    

The device configuration is now simply obtained using {code}`dut.generate_device_config()`.
In order for this command to provide a correct device configuration, the different
parameters need to be specified in the {class}`~quantify_scheduler.device_under_test.transmon_element.BasicTransmonElement` and {class}`~quantify_scheduler.device_under_test.quantum_device.QuantumDevice` objects.


```python
pprint(dut.generate_device_config())


```

    DeviceCompilationConfig(backend='quantify_scheduler.backends.circuit_to_device.compile_circuit_to_device', clocks={'qubit.01': 6000000000.0, 'qubit.12': nan, 'qubit.ro': nan}, elements={'qubit': {'reset': OperationCompilationConfig(factory_func='quantify_scheduler.operations.pulse_library.IdlePulse', factory_kwargs={'duration': 0.0002}, gate_info_factory_kwargs=None), 'Rxy': OperationCompilationConfig(factory_func='quantify_scheduler.operations.pulse_factories.rxy_drag_pulse', factory_kwargs={'amp180': nan, 'motzoi': 0, 'port': 'qubit:mw', 'clock': 'qubit.01', 'duration': 2e-08}, gate_info_factory_kwargs=['theta', 'phi']), 'measure': OperationCompilationConfig(factory_func='quantify_scheduler.operations.measurement_factories.dispersive_measurement', factory_kwargs={'port': 'qubit:res', 'clock': 'qubit.ro', 'pulse_type': 'SquarePulse', 'pulse_amp': 0.25, 'pulse_duration': 3e-07, 'acq_delay': 0, 'acq_duration': 1e-06, 'acq_channel': 0, 'acq_protocol_default': 'SSBIntegrationComplex', 'reset_clock_phase': True}, gate_info_factory_kwargs=['acq_index', 'bin_mode', 'acq_protocol'])}}, edges={})
    

The device configuration is now simply obtained using {code}`dut.generate_device_config()`.
In order for this command to provide a correct device configuration, the different
parameters need to be specified in the {class}`~quantify_scheduler.device_under_test.transmon_element.BasicTransmonElement` and {class}`~quantify_scheduler.device_under_test.quantum_device.QuantumDevice` objects.

## Mixing pulse and circuit layer operations (Chevron)

As well as defining our schedules in terms of gates, we can also mix the circuit layer
representation with pulse-level operations.
This can be useful for experiments involving pulses not easily represented by Gates,
such as the Chevron experiment.
In this experiment, we want to vary the length and amplitude of a square pulse between
X gates on a pair of qubits.


```python
from quantify_scheduler import Schedule
from quantify_scheduler.operations.gate_library import Measure, Reset, X, X90
from quantify_scheduler.operations.pulse_library import SquarePulse
from quantify_scheduler.resources import ClockResource

sched = Schedule("Chevron Experiment")
acq_idx = 0

# Multiples of 4 ns need to be used due to sampling rate of the Qblox modules
for duration in np.linspace(start=20e-9, stop=60e-9, num=6):
    for amp in np.linspace(start=0.1, stop=1.0, num=10):
        reset = sched.add(Reset("q0", "q1"))
        sched.add(X("q0"), ref_op=reset, ref_pt="end")  # Start at the end of the reset
        # We specify a clock for tutorial purposes, Chevron experiments do not necessarily use modulated square pulses
        square = sched.add(SquarePulse(amp, duration, "q0:mw", clock="q0.01"))
        sched.add(X90("q0"), ref_op=square)  # Start at the end of the square pulse
        sched.add(X90("q1"), ref_op=square)
        sched.add(Measure(q0, acq_index=acq_idx), label=f"M q0 {acq_idx}")
        sched.add(  # Start at the same time as the other measure
            Measure(q1, acq_index=acq_idx),
            label=f"M q1 {acq_idx}",
            ref_pt="start",
        )

        acq_idx += 1


# We add each clock to the schedule
sched.add_resources([ClockResource("q0.01", 6.02e9),ClockResource("q1.01", 6.02e9),ClockResource("q0.ro", 5.02e9),ClockResource("q1.ro", 5.02e9)]) 

```


```python
fig, ax = sched.plot_circuit_diagram()
ax.set_xlim(-0.5, 9.5)
for t in ax.texts:
    if t.get_position()[0] > 9.5:
        t.set_visible(False)


```


    
![png](Operations%20and%20Qubits%20%286%29_files/Operations%20and%20Qubits%20%286%29_37_0.png)
    


This example shows that we add gates using the same interface as pulses. Gates are Operations, and
as such support the same timing and reference operators as Pulses.

```{warning}
When adding a Pulse to a schedule, the clock is not automatically added to the
resources of the schedule. It may be necessary to add this clock manually, as in
the final line of the example above.
```

### Device and Hardware compilation combined : Serial Compiler
{class}`~quantify_scheduler.backends.SerialCompiler` is used when a {class}`quantify_scheduler.device_under_test.quantum_device.QuantumDevice` is used to define your device configuration. In this tutorial we do not attach any hardware configuration to the `QuantumDevice`. The [compiling to hardware](https://quantify-quantify-scheduler.readthedocs-hosted.com/en/latest/tutorials/Compiling%20to%20Hardware.html) tutorial demonstrates how to correctly use a hardware configuration with a `QuantumDevice`.


```python
from quantify_scheduler.backends import SerialCompiler
from quantify_scheduler.device_under_test.quantum_device import QuantumDevice
from quantify_scheduler.device_under_test.transmon_element import BasicTransmonElement

dut.close()
# BasicTransmonElement.close_all()
dut = QuantumDevice("DUT")
q0 = BasicTransmonElement("q0")
q1 = BasicTransmonElement("q1")
# dut.hardware_config({}) # This is the default, showing for completeness
dut.add_element(q0)
dut.add_element(q1)
dut.get_element("q0").rxy.amp180(0.6)
dut.get_element("q1").rxy.amp180(0.6)

compiler = SerialCompiler(name='compiler')
compiled_sched = compiler.compile(schedule=sched,config=dut.generate_compilation_config())
```

So, finally, we can show the timing table associated to the chevron schedule and plot
its pulse diagram:


```python
compiled_sched.timing_table.hide(slice(11, None), axis="index").hide(
    "waveform_op_id", axis="columns"
)


```




<style type="text/css">
</style>
<table id="T_194a0">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_194a0_level0_col1" class="col_heading level0 col1" >port</th>
      <th id="T_194a0_level0_col2" class="col_heading level0 col2" >clock</th>
      <th id="T_194a0_level0_col3" class="col_heading level0 col3" >is_acquisition</th>
      <th id="T_194a0_level0_col4" class="col_heading level0 col4" >abs_time</th>
      <th id="T_194a0_level0_col5" class="col_heading level0 col5" >duration</th>
      <th id="T_194a0_level0_col6" class="col_heading level0 col6" >operation</th>
      <th id="T_194a0_level0_col7" class="col_heading level0 col7" >wf_idx</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_194a0_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_194a0_row0_col1" class="data row0 col1" >None</td>
      <td id="T_194a0_row0_col2" class="data row0 col2" >cl0.baseband</td>
      <td id="T_194a0_row0_col3" class="data row0 col3" >False</td>
      <td id="T_194a0_row0_col4" class="data row0 col4" >0.0 ns</td>
      <td id="T_194a0_row0_col5" class="data row0 col5" >200,000.0 ns</td>
      <td id="T_194a0_row0_col6" class="data row0 col6" >Reset('q0','q1')</td>
      <td id="T_194a0_row0_col7" class="data row0 col7" >0</td>
    </tr>
    <tr>
      <th id="T_194a0_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_194a0_row1_col1" class="data row1 col1" >None</td>
      <td id="T_194a0_row1_col2" class="data row1 col2" >cl0.baseband</td>
      <td id="T_194a0_row1_col3" class="data row1 col3" >False</td>
      <td id="T_194a0_row1_col4" class="data row1 col4" >0.0 ns</td>
      <td id="T_194a0_row1_col5" class="data row1 col5" >200,000.0 ns</td>
      <td id="T_194a0_row1_col6" class="data row1 col6" >Reset('q0','q1')</td>
      <td id="T_194a0_row1_col7" class="data row1 col7" >1</td>
    </tr>
    <tr>
      <th id="T_194a0_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_194a0_row2_col1" class="data row2 col1" >q0:mw</td>
      <td id="T_194a0_row2_col2" class="data row2 col2" >q0.01</td>
      <td id="T_194a0_row2_col3" class="data row2 col3" >False</td>
      <td id="T_194a0_row2_col4" class="data row2 col4" >200,000.0 ns</td>
      <td id="T_194a0_row2_col5" class="data row2 col5" >20.0 ns</td>
      <td id="T_194a0_row2_col6" class="data row2 col6" >X(qubit='q0')</td>
      <td id="T_194a0_row2_col7" class="data row2 col7" >0</td>
    </tr>
    <tr>
      <th id="T_194a0_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_194a0_row3_col1" class="data row3 col1" >q0:mw</td>
      <td id="T_194a0_row3_col2" class="data row3 col2" >q0.01</td>
      <td id="T_194a0_row3_col3" class="data row3 col3" >False</td>
      <td id="T_194a0_row3_col4" class="data row3 col4" >200,020.0 ns</td>
      <td id="T_194a0_row3_col5" class="data row3 col5" >20.0 ns</td>
      <td id="T_194a0_row3_col6" class="data row3 col6" >SquarePulse(amp=0.1,duration=2e-08,port='q0:mw',clock='q0.01',phase=0,t0=0)</td>
      <td id="T_194a0_row3_col7" class="data row3 col7" >0</td>
    </tr>
    <tr>
      <th id="T_194a0_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_194a0_row4_col1" class="data row4 col1" >q0:mw</td>
      <td id="T_194a0_row4_col2" class="data row4 col2" >q0.01</td>
      <td id="T_194a0_row4_col3" class="data row4 col3" >False</td>
      <td id="T_194a0_row4_col4" class="data row4 col4" >200,040.0 ns</td>
      <td id="T_194a0_row4_col5" class="data row4 col5" >20.0 ns</td>
      <td id="T_194a0_row4_col6" class="data row4 col6" >X90(qubit='q0')</td>
      <td id="T_194a0_row4_col7" class="data row4 col7" >0</td>
    </tr>
    <tr>
      <th id="T_194a0_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_194a0_row5_col1" class="data row5 col1" >q1:mw</td>
      <td id="T_194a0_row5_col2" class="data row5 col2" >q1.01</td>
      <td id="T_194a0_row5_col3" class="data row5 col3" >False</td>
      <td id="T_194a0_row5_col4" class="data row5 col4" >200,040.0 ns</td>
      <td id="T_194a0_row5_col5" class="data row5 col5" >20.0 ns</td>
      <td id="T_194a0_row5_col6" class="data row5 col6" >X90(qubit='q1')</td>
      <td id="T_194a0_row5_col7" class="data row5 col7" >0</td>
    </tr>
    <tr>
      <th id="T_194a0_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_194a0_row6_col1" class="data row6 col1" >None</td>
      <td id="T_194a0_row6_col2" class="data row6 col2" >q0.ro</td>
      <td id="T_194a0_row6_col3" class="data row6 col3" >False</td>
      <td id="T_194a0_row6_col4" class="data row6 col4" >200,060.0 ns</td>
      <td id="T_194a0_row6_col5" class="data row6 col5" >0.0 ns</td>
      <td id="T_194a0_row6_col6" class="data row6 col6" >Measure('q0', acq_index=0, acq_protocol="None", bin_mode=None)</td>
      <td id="T_194a0_row6_col7" class="data row6 col7" >0</td>
    </tr>
    <tr>
      <th id="T_194a0_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_194a0_row7_col1" class="data row7 col1" >q0:res</td>
      <td id="T_194a0_row7_col2" class="data row7 col2" >q0.ro</td>
      <td id="T_194a0_row7_col3" class="data row7 col3" >False</td>
      <td id="T_194a0_row7_col4" class="data row7 col4" >200,060.0 ns</td>
      <td id="T_194a0_row7_col5" class="data row7 col5" >300.0 ns</td>
      <td id="T_194a0_row7_col6" class="data row7 col6" >Measure('q0', acq_index=0, acq_protocol="None", bin_mode=None)</td>
      <td id="T_194a0_row7_col7" class="data row7 col7" >1</td>
    </tr>
    <tr>
      <th id="T_194a0_level0_row8" class="row_heading level0 row8" >8</th>
      <td id="T_194a0_row8_col1" class="data row8 col1" >q0:res</td>
      <td id="T_194a0_row8_col2" class="data row8 col2" >q0.ro</td>
      <td id="T_194a0_row8_col3" class="data row8 col3" >True</td>
      <td id="T_194a0_row8_col4" class="data row8 col4" >200,060.0 ns</td>
      <td id="T_194a0_row8_col5" class="data row8 col5" >1,000.0 ns</td>
      <td id="T_194a0_row8_col6" class="data row8 col6" >Measure('q0', acq_index=0, acq_protocol="None", bin_mode=None)</td>
      <td id="T_194a0_row8_col7" class="data row8 col7" >0</td>
    </tr>
    <tr>
      <th id="T_194a0_level0_row9" class="row_heading level0 row9" >9</th>
      <td id="T_194a0_row9_col1" class="data row9 col1" >None</td>
      <td id="T_194a0_row9_col2" class="data row9 col2" >q1.ro</td>
      <td id="T_194a0_row9_col3" class="data row9 col3" >False</td>
      <td id="T_194a0_row9_col4" class="data row9 col4" >200,060.0 ns</td>
      <td id="T_194a0_row9_col5" class="data row9 col5" >0.0 ns</td>
      <td id="T_194a0_row9_col6" class="data row9 col6" >Measure('q1', acq_index=0, acq_protocol="None", bin_mode=None)</td>
      <td id="T_194a0_row9_col7" class="data row9 col7" >0</td>
    </tr>
    <tr>
      <th id="T_194a0_level0_row10" class="row_heading level0 row10" >10</th>
      <td id="T_194a0_row10_col1" class="data row10 col1" >q1:res</td>
      <td id="T_194a0_row10_col2" class="data row10 col2" >q1.ro</td>
      <td id="T_194a0_row10_col3" class="data row10 col3" >False</td>
      <td id="T_194a0_row10_col4" class="data row10 col4" >200,060.0 ns</td>
      <td id="T_194a0_row10_col5" class="data row10 col5" >300.0 ns</td>
      <td id="T_194a0_row10_col6" class="data row10 col6" >Measure('q1', acq_index=0, acq_protocol="None", bin_mode=None)</td>
      <td id="T_194a0_row10_col7" class="data row10 col7" >1</td>
    </tr>
  </tbody>
</table>





```python
f, ax = compiled_sched.plot_pulse_diagram()
ax.set_xlim(200e-6, 200.4e-6)
```




    (0.0002, 0.0002004)




    
![png](Operations%20and%20Qubits%20%286%29_files/Operations%20and%20Qubits%20%286%29_42_1.png)
    