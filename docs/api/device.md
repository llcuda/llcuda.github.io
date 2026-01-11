# GPU Device Management

Complete API for GPU device detection, compatibility checking, and CUDA management in llcuda v2.0.6.

## Overview

llcuda provides comprehensive GPU device management functions to help you:

- Detect CUDA-capable GPUs
- Check compatibility with llcuda binaries
- Get device properties and VRAM information
- Configure optimal inference settings
- Handle multi-GPU environments

## Core Functions

### `check_gpu_compatibility()`

Check if your GPU is compatible with llcuda binaries.

**Signature:**
```python
def check_gpu_compatibility(min_compute_cap: float = 5.0) -> Dict[str, Any]
```

**Parameters:**
- `min_compute_cap` (float, optional): Minimum compute capability required. Default: 5.0

**Returns:**
```python
{
    'compatible': bool,              # Whether GPU is compatible
    'compute_capability': float,     # GPU compute capability (e.g., 7.5)
    'gpu_name': str,                 # GPU name (e.g., "Tesla T4")
    'reason': str,                   # Explanation if not compatible
    'platform': str                  # Detected platform (local/colab/kaggle)
}
```

**Example:**
```python
import llcuda

# Check GPU compatibility
compat = llcuda.check_gpu_compatibility()

if compat['compatible']:
    print(f"✅ {compat['gpu_name']} is compatible!")
    print(f"   Compute Capability: {compat['compute_capability']}")
    print(f"   Platform: {compat['platform']}")
else:
    print(f"⚠️ {compat['gpu_name']} is not compatible")
    print(f"   Reason: {compat['reason']}")
```

**Output on Tesla T4:**
```
✅ Tesla T4 is compatible!
   Compute Capability: 7.5
   Platform: colab
```

---

### `detect_cuda()`

Detect CUDA installation and get detailed GPU information.

**Signature:**
```python
def detect_cuda() -> Dict[str, Any]
```

**Returns:**
```python
{
    'available': bool,     # Whether CUDA is available
    'version': str,        # CUDA version (e.g., "12.2")
    'gpus': [              # List of GPU information
        {
            'name': str,                  # GPU name
            'memory': str,                # Total VRAM (e.g., "15360 MiB")
            'driver_version': str,        # NVIDIA driver version
            'compute_capability': str     # Compute capability
        }
    ]
}
```

**Example:**
```python
import llcuda

cuda_info = llcuda.detect_cuda()

if cuda_info['available']:
    print(f"CUDA Version: {cuda_info['version']}")
    print(f"Number of GPUs: {len(cuda_info['gpus'])}")

    for i, gpu in enumerate(cuda_info['gpus']):
        print(f"\nGPU {i}:")
        print(f"  Name: {gpu['name']}")
        print(f"  VRAM: {gpu['memory']}")
        print(f"  Driver: {gpu['driver_version']}")
        print(f"  Compute Capability: {gpu['compute_capability']}")
else:
    print("CUDA is not available")
```

**Output:**
```
CUDA Version: 12.2
Number of GPUs: 1

GPU 0:
  Name: Tesla T4
  VRAM: 15360 MiB
  Driver: 535.104.05
  Compute Capability: 7.5
```

---

### `get_cuda_device_info()`

Get simplified CUDA device information.

**Signature:**
```python
def get_cuda_device_info() -> Optional[Dict[str, Any]]
```

**Returns:**
```python
{
    'cuda_version': str,   # CUDA version
    'gpus': list          # List of GPU dictionaries
}
# Returns None if CUDA is not available
```

**Example:**
```python
import llcuda

device_info = llcuda.get_cuda_device_info()

if device_info:
    print(f"CUDA: {device_info['cuda_version']}")
    print(f"GPUs detected: {len(device_info['gpus'])}")
else:
    print("No CUDA devices found")
```

---

### `check_cuda_available()`

Quick check if CUDA is available.

**Signature:**
```python
def check_cuda_available() -> bool
```

**Returns:**
- `True` if CUDA is available, `False` otherwise

**Example:**
```python
import llcuda

if llcuda.check_cuda_available():
    print("✅ CUDA is available")
    # Proceed with GPU inference
else:
    print("❌ CUDA not available - CPU mode only")
```

---

## Supported GPUs

llcuda binaries support NVIDIA GPUs with compute capability 5.0 and higher:

### Architecture Support

| Architecture | Compute Cap | Examples | Status |
|--------------|-------------|----------|---------|
| **Maxwell** | 5.0 - 5.3 | GTX 900, Tesla M40 | ✅ Supported |
| **Pascal** | 6.0 - 6.2 | GTX 10xx, Tesla P100 | ✅ Supported |
| **Volta** | 7.0 | Tesla V100 | ✅ Supported |
| **Turing** | 7.5 | RTX 20xx, **Tesla T4**, GTX 16xx | ✅ Verified |
| **Ampere** | 8.0 - 8.6 | RTX 30xx, A100 | ✅ Supported |
| **Ada Lovelace** | 8.9 | RTX 40xx | ✅ Supported |
| **Hopper** | 9.0 | H100 | ✅ Supported |

### Popular GPUs

| GPU Model | VRAM | Compute Cap | Recommended Model Size |
|-----------|------|-------------|------------------------|
| **Tesla T4** | 15 GB | 7.5 | Up to 7B (Q4_K_M) |
| RTX 3060 | 12 GB | 8.6 | Up to 7B (Q4_K_M) |
| RTX 3070 | 8 GB | 8.6 | Up to 3B (Q4_K_M) |
| RTX 3080 | 10 GB | 8.6 | Up to 7B (Q4_K_M) |
| RTX 3090 | 24 GB | 8.6 | Up to 13B (Q4_K_M) |
| RTX 4070 | 12 GB | 8.9 | Up to 7B (Q4_K_M) |
| RTX 4090 | 24 GB | 8.9 | Up to 13B (Q4_K_M) |
| A100 | 40 GB | 8.0 | Up to 30B (Q4_K_M) |
| A100 | 80 GB | 8.0 | Up to 70B (Q4_K_M) |

---

## VRAM Management

### Get Available VRAM

```python
import llcuda

cuda_info = llcuda.detect_cuda()

if cuda_info['available'] and cuda_info['gpus']:
    gpu = cuda_info['gpus'][0]
    vram_str = gpu['memory']

    # Parse VRAM
    if 'GiB' in vram_str:
        vram_gb = float(vram_str.split()[0])
    elif 'MiB' in vram_str:
        vram_mb = float(vram_str.split()[0])
        vram_gb = vram_mb / 1024

    print(f"Available VRAM: {vram_gb:.1f} GB")
```

### VRAM Recommendations

Get recommended settings based on available VRAM:

```python
from llcuda.utils import get_recommended_gpu_layers

# For a 1.2 GB model with 15 GB VRAM
gpu_layers = get_recommended_gpu_layers(
    model_size_gb=1.2,
    vram_gb=15.0
)

print(f"Recommended GPU layers: {gpu_layers}")
# Output: 99 (full GPU offload)
```

**VRAM to GPU Layers Mapping:**

| Available VRAM | Model Size | Recommended Layers |
|----------------|------------|-------------------|
| >= 1.2x model | Any | 99 (full offload) |
| >= 0.8x model | Any | 40 (most layers) |
| >= 0.6x model | Any | 30 (many layers) |
| >= 0.4x model | Any | 20 (some layers) |
| >= 0.2x model | Any | 10 (few layers) |
| < 0.2x model | Any | 0 (CPU only) |

---

## Auto-Configuration

### `auto_configure_for_model()`

Automatically configure optimal settings for your GPU and model.

**Signature:**
```python
from llcuda.utils import auto_configure_for_model
from pathlib import Path

def auto_configure_for_model(
    model_path: Path,
    vram_gb: Optional[float] = None
) -> Dict[str, Any]
```

**Parameters:**
- `model_path` (Path): Path to GGUF model file
- `vram_gb` (float, optional): VRAM in GB (auto-detected if not provided)

**Returns:**
```python
{
    'gpu_layers': int,      # Number of layers to offload
    'ctx_size': int,        # Context window size
    'batch_size': int,      # Batch size for processing
    'ubatch_size': int,     # Micro-batch size
    'n_parallel': int       # Parallel sequences
}
```

**Example:**
```python
from llcuda.utils import auto_configure_for_model
from pathlib import Path

# Auto-configure for model
model_path = Path("/path/to/model.gguf")
settings = auto_configure_for_model(model_path)

print("Recommended settings:")
print(f"  GPU Layers: {settings['gpu_layers']}")
print(f"  Context Size: {settings['ctx_size']}")
print(f"  Batch Size: {settings['batch_size']}")
print(f"  Micro-batch Size: {settings['ubatch_size']}")

# Use settings with InferenceEngine
engine.load_model(
    str(model_path),
    gpu_layers=settings['gpu_layers'],
    ctx_size=settings['ctx_size'],
    batch_size=settings['batch_size'],
    ubatch_size=settings['ubatch_size']
)
```

**Output on Tesla T4 (15 GB):**
```
✓ Auto-configured for 15.0 GB VRAM
  GPU Layers: 99
  Context Size: 4096
  Batch Size: 2048
  Micro-batch Size: 512
```

---

## Platform Detection

llcuda automatically detects the execution environment:

```python
import llcuda

compat = llcuda.check_gpu_compatibility()
platform = compat['platform']

if platform == 'colab':
    print("Running on Google Colab")
    print("Expected GPU: Tesla T4")
elif platform == 'kaggle':
    print("Running on Kaggle")
    print("Expected GPU: Tesla P100 or T4")
else:
    print("Running on local machine")
```

**Detected Platforms:**

- `colab` - Google Colab
- `kaggle` - Kaggle Notebooks
- `local` - Local machine or other cloud

---

## Multi-GPU Support

### Selecting Specific GPU

```python
import os

# Use GPU 0 only
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Use GPU 1 only
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# Use GPUs 0 and 2
os.environ['CUDA_VISIBLE_DEVICES'] = '0,2'

# Then import llcuda
import llcuda
```

### Checking Multiple GPUs

```python
import llcuda

cuda_info = llcuda.detect_cuda()

if cuda_info['available']:
    num_gpus = len(cuda_info['gpus'])
    print(f"Detected {num_gpus} GPU(s)")

    for i, gpu in enumerate(cuda_info['gpus']):
        print(f"\nGPU {i}: {gpu['name']}")
        print(f"  VRAM: {gpu['memory']}")
        print(f"  Compute: {gpu['compute_capability']}")
```

---

## Environment Setup

### `setup_environment()`

Automatically configure environment variables for optimal performance.

```python
from llcuda.utils import setup_environment

# Setup environment
env_vars = setup_environment()

print("Environment configured:")
for key, value in env_vars.items():
    print(f"  {key}: {value}")
```

**Configured Variables:**

- `LD_LIBRARY_PATH` - Shared library path (Linux)
- `CUDA_VISIBLE_DEVICES` - Visible GPUs
- `LLAMA_CPP_DIR` - llama.cpp installation directory

---

## System Information

### `print_system_info()`

Print comprehensive system information for debugging.

```python
from llcuda.utils import print_system_info

print_system_info()
```

**Output:**
```
============================================================
llcuda System Information
============================================================

Python:
  Version: 3.10.12 (main, Nov 20 2023, 15:14:05)
  Executable: /usr/bin/python3

Operating System:
  System: Linux
  Release: 5.15.0-91-generic
  Machine: x86_64

CUDA:
  Available: True
  Version: 12.2
  GPUs: 1
    GPU 0: Tesla T4
      Memory: 15360 MiB
      Driver: 535.104.05
      Compute: 7.5

GGUF Models Found: 3
  - gemma-3-1b-it-Q4_K_M.gguf (872.5 MB)
  - llama-3.2-3b-Q4_K_M.gguf (1856.2 MB)
  - qwen2.5-7b-Q4_K_M.gguf (4368.7 MB)

============================================================
```

---

## Common Patterns

### Complete GPU Verification

```python
import llcuda

def verify_gpu_setup():
    """Verify GPU setup before running inference."""

    # Check CUDA availability
    if not llcuda.check_cuda_available():
        print("❌ CUDA not available")
        return False

    # Get detailed info
    cuda_info = llcuda.detect_cuda()
    print(f"✅ CUDA {cuda_info['version']} detected")

    # Check compatibility
    compat = llcuda.check_gpu_compatibility()

    if not compat['compatible']:
        print(f"❌ {compat['gpu_name']} is not compatible")
        print(f"   {compat['reason']}")
        return False

    print(f"✅ {compat['gpu_name']} is compatible")
    print(f"   Compute Capability: {compat['compute_capability']}")
    print(f"   Platform: {compat['platform']}")

    # Get VRAM info
    gpu = cuda_info['gpus'][0]
    print(f"   VRAM: {gpu['memory']}")

    return True

# Use it
if verify_gpu_setup():
    print("\n🚀 Ready for inference!")
    # Proceed with model loading...
else:
    print("\n⚠️ GPU setup incomplete")
```

### Auto-Configure and Load

```python
import llcuda
from llcuda.utils import auto_configure_for_model
from pathlib import Path

# Verify GPU
compat = llcuda.check_gpu_compatibility()
if not compat['compatible']:
    raise RuntimeError(f"GPU not compatible: {compat['reason']}")

# Auto-configure
model_path = Path("model.gguf")
settings = auto_configure_for_model(model_path)

# Create engine with optimal settings
engine = llcuda.InferenceEngine()
engine.load_model(
    str(model_path),
    **settings,  # Use all auto-configured settings
    silent=True
)

print("✅ Model loaded with optimal settings!")
```

---

## Error Handling

### Handle No GPU

```python
import llcuda

try:
    compat = llcuda.check_gpu_compatibility()

    if not compat['compatible']:
        raise RuntimeError(compat['reason'])

    # Proceed with GPU inference
    engine = llcuda.InferenceEngine()
    engine.load_model("model.gguf", gpu_layers=99)

except RuntimeError as e:
    print(f"GPU Error: {e}")
    print("Falling back to CPU mode...")

    # Load with CPU
    engine = llcuda.InferenceEngine()
    engine.load_model("model.gguf", gpu_layers=0)
```

### Handle Insufficient VRAM

```python
import llcuda
from llcuda.utils import auto_configure_for_model
from pathlib import Path

model_path = Path("large-model.gguf")

try:
    # Try auto-configuration
    settings = auto_configure_for_model(model_path)

    if settings['gpu_layers'] == 0:
        print("⚠️ Insufficient VRAM for GPU offload")
        print("   Using CPU mode")
    elif settings['gpu_layers'] < 99:
        print(f"⚠️ Partial GPU offload: {settings['gpu_layers']} layers")
        print("   Consider using a smaller model for better performance")

    # Load with recommended settings
    engine = llcuda.InferenceEngine()
    engine.load_model(str(model_path), **settings)

except Exception as e:
    print(f"Error: {e}")
    print("Try a smaller model or more aggressive quantization")
```

---

## See Also

- [API Overview](overview.md) - Complete API reference
- [InferenceEngine](inference-engine.md) - Inference API
- [Models & GGUF](models.md) - Model management
- [Performance Guide](../performance/optimization.md) - Optimization techniques
- [Troubleshooting](../guides/troubleshooting.md) - Common issues
