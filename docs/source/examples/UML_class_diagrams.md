---
file_format: mystnb
kernelspec:
    name: python3

---
```{seealso}
This notebook can be downloaded {nb-download}`here <UML_class_diagrams.ipynb>`
```

# UML class diagram generator

- This notebook generates UML diagrams of class hierarchies
- Dependencies: `pylint`, `ipykernel` (install these two in a python env), [graphviz](https://graphviz.org/download/)

```{code-cell} ipython3
from IPython.display import Image, display
from quantify_scheduler.helpers.inspect import make_uml_diagram
```

## General notes
- Yellow labels indicate addition as a submodule
- Different colors indicate different packages
- Generated figures are saved in png format
- Options must be given in `list[str]` format
- For more information, visit the [Wikipedia page](https://en.wikipedia.org/wiki/Class_diagram#Relationships) on relationships in class diagrams

## Plotting all classes in a module
- Be aware that this option will only plot classes that are contained within `module_to_plot`, and not related classes defined outside the module.
- Extra options:
    - Show ancestors (aka parent classes): `-A`
    - Ignore specific submodules: `--ignore <file[,file...]>` (e.g. `["--ignore", "circuit_to_device.py,corrections.py,zhinst"]`)  


```{code-cell} ipython3

from quantify_scheduler.backends import qblox

module_to_plot = qblox
options = ["-A"]

diagram_name = make_uml_diagram(module_to_plot, options)
if diagram_name:
    display(Image(diagram_name))
```

## Plotting ancestors and submodules of a class
- Remove `--only-classnames` option to show all class attributes


```{code-cell} ipython3

from quantify_scheduler.device_under_test.transmon_element import BasicTransmonElement

class_to_plot = BasicTransmonElement
options = ["--only-classnames"]

diagram_name = make_uml_diagram(class_to_plot, options)
if diagram_name:
    display(Image(diagram_name))
```


