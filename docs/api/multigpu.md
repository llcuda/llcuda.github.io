# MultiGPU API

Multi-GPU configuration and utilities for dual T4 setup.

## Overview

The `multigpu` module provides GPU detection and configuration for Kaggle dual T4.

## Basic Usage

```python
from llcuda.api.multigpu import detect_gpus, kaggle_t4_dual_config

# Detect GPUs
gpus = detect_gpus()
for gpu in gpus:
    print(f"GPU {gpu.index}: {gpu.name}")

# Get Kaggle dual T4 configuration
config = kaggle_t4_dual_config(model_path="model.gguf")
```

## Functions

### `detect_gpus()`

```python
def detect_gpus() -> List[GPUInfo]:
    """Detect available GPUs.
    
    Returns:
        List of GPUInfo objects
    """
```

### `kaggle_t4_dual_config()`

```python
def kaggle_t4_dual_config(
    model_path: str,
    tensor_split: str = "0.5,0.5"
) -> ServerConfig:
    """Get optimized config for Kaggle dual T4.
    
    Args:
        model_path: Path to GGUF model
        tensor_split: GPU split ratio
        
    Returns:
        ServerConfig for dual T4
    """
```

### `estimate_model_vram()`

```python
def estimate_model_vram(
    model_size_b: float,
    quant_type: str = "Q4_K_M"
) -> float:
    """Estimate VRAM usage in GB.
    
    Args:
        model_size_b: Model size in billions
        quant_type: Quantization type
        
    Returns:
        Estimated VRAM in GB
    """
```

## Classes

### `GPUInfo`

```python
class GPUInfo:
    index: int
    name: str
    compute_capability: Tuple[int, int]
    memory_total: int  # bytes
    memory_free: int   # bytes
```

## Examples

See [Multi-GPU Tutorial](../tutorials/03-multi-gpu.md)
