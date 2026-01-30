# GPU & Device Utilities

Utilities for checking GPU compatibility and querying CUDA device details.

## `check_gpu_compatibility(min_compute_cap: float = 5.0) -> dict` {#check-gpu-compatibility}

Check whether the current GPU meets llcuda’s minimum compute capability.

```python
import llcuda

compat = llcuda.check_gpu_compatibility()
print(compat)
```

The returned dictionary includes fields like:
- `compatible` (bool)
- `compute_capability` (float | None)
- `gpu_name` (str | None)
- `reason` (str | None)
- `platform` (`local`, `colab`, or `kaggle`)

## `core.get_device_count() -> int` {#get-device-count}

Return the number of CUDA devices.

```python
import llcuda.core as core

count = core.get_device_count()
print(count)
```

## `core.get_device_properties(device_id: int = 0)` {#get-device-properties}

Return device properties for a specific CUDA device.

```python
import llcuda.core as core

props = core.get_device_properties(0)
print(props)
```
