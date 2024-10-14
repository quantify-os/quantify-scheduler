# Release Notes

## Release 0.21.2

### 🐛 Bug Fixes

This release comes with a temporary fix when long schedules cannot compile due to accumulating floating point errors. 

If a calculated timing exceeds a certain tolerance, the compiler will raise a `ValueError` such as

```
ValueError: Attempting to use a time value of 168345600000.003 ns. Please ensure that the durations of operations and wait times between operations are multiples of 4 ns (tolerance: 1e-03 ns). If you think this is a mistake, try increasing the tolerance by setting e.g. `quantify_scheduler.backends.qblox.constants.GRID_TIME_TOLERANCE_TIME = 1e-2` at the top of your script.
```

If you are certain that this is due to floating point errors, and not due to incorrect timings you can change the tolerance:

```python
import quantify_scheduler.backends.qblox.constants as constants

constants.GRID_TIME_TOLERANCE_TIME = 0.1e-3
```
