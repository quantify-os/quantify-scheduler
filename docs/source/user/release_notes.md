---
file_format: mystnb
kernelspec:
    name: python3
---
<!-- mystnb doesn't support nested executable cells, so we move it out of the {include} directive -->
```{code-cell} ipython3
:tags: [remove-cell]

import xarray as xr
trigger_data = [1, 2, 3, 2]
counts = [5, 4, 2, 1]
dataset = xr.Dataset(
    {0: xr.DataArray([trigger_data],
            dims=["repetition", "counts"],
            coords={"repetition": [0], "counts": counts},
        )
    }
)
from myst_nb import glue
glue("trigger_dataset", dataset)
```

```{include} ../../../RELEASE_NOTES.md
```
