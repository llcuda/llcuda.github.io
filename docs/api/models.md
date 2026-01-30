# Models API Reference

API documentation for model management, discovery, and downloading in llcuda.

## Overview

The `llcuda.models` module provides utilities for:

- Loading models from registry or HuggingFace
- Downloading and caching GGUF models
- Getting model metadata and information
- Recommending optimal inference settings

## Functions

### `load_model_smart()`

Smart model loading with automatic download and path resolution.

```python
def load_model_smart(
    model_name_or_path: str,
    interactive: bool = True
) -> Path
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name_or_path` | `str` | Required | Model name, HF repo, or local path |
| `interactive` | `bool` | `True` | Ask for confirmation before downloading |

**Returns:**

- `Path` - Path to model file

**Example:**

```python
from llcuda.models import load_model_smart

# Load from registry
model_path = load_model_smart("gemma-3-1b-Q4_K_M")

# Load from HuggingFace
model_path = load_model_smart(
    "unsloth/gemma-3-1b-it-GGUF:gemma-3-1b-it-Q4_K_M.gguf"
)

# Load local file
model_path = load_model_smart("/path/to/model.gguf")
```

### `download_model()`

Download a model from HuggingFace.

```python
def download_model(
    repo_id: str,
    filename: str,
    cache_dir: Optional[str] = None
) -> Path
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `repo_id` | `str` | Required | HuggingFace repository ID |
| `filename` | `str` | Required | Model filename to download |
| `cache_dir` | `Optional[str]` | `None` | Custom cache directory |

**Returns:**

- `Path` - Path to downloaded model

**Example:**

```python
from llcuda.models import download_model

# Download from HuggingFace
model_path = download_model(
    repo_id="unsloth/gemma-3-1b-it-GGUF",
    filename="gemma-3-1b-it-Q4_K_M.gguf"
)
```

### `list_registry_models()`

List available models in the registry.

```python
def list_registry_models() -> List[Dict[str, Any]]
```

**Returns:**

- `List[Dict]` - List of model information dictionaries

**Example:**

```python
from llcuda.models import list_registry_models

models = list_registry_models()
for model in models:
    print(f"{model['name']}: {model['description']}")
```

## Classes

### `ModelInfo`

Extract metadata from GGUF models.

```python
class ModelInfo:
    def __init__(self, filepath: str)
```

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `filepath` | `Path` | Path to GGUF file |
| `architecture` | `Optional[str]` | Model architecture (e.g., "llama", "gemma") |
| `parameter_count` | `Optional[int]` | Estimated parameter count |
| `context_length` | `Optional[int]` | Maximum context length |
| `quantization` | `Optional[str]` | Quantization type |
| `file_size_mb` | `float` | File size in MB |

**Methods:**

#### `get_recommended_settings()`

Get recommended inference settings based on model and hardware.

```python
def get_recommended_settings(
    vram_gb: float = 8.0
) -> Dict[str, Any]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vram_gb` | `float` | `8.0` | Available VRAM in GB |

**Returns:**

```python
{
    'gpu_layers': int,
    'ctx_size': int,
    'batch_size': int,
    'ubatch_size': int
}
```

**Example:**

```python
from llcuda.models import ModelInfo

# Load model info
info = ModelInfo("gemma-3-1b-Q4_K_M.gguf")

print(f"Architecture: {info.architecture}")
print(f"Parameters: {info.parameter_count}B")
print(f"Context: {info.context_length}")
print(f"Size: {info.file_size_mb:.1f} MB")

# Get recommended settings for T4 (15GB)
settings = info.get_recommended_settings(vram_gb=15.0)
print(f"Recommended gpu_layers: {settings['gpu_layers']}")
print(f"Recommended ctx_size: {settings['ctx_size']}")
```

## Model Registry

llcuda includes a built-in model registry with popular models:

```python
REGISTRY = {
    "gemma-3-1b-Q4_K_M": {
        "repo": "unsloth/gemma-3-1b-it-GGUF",
        "file": "gemma-3-1b-it-Q4_K_M.gguf",
        "size": "700 MB",
        "description": "Gemma 3 1B Instruct, Q4_K_M quantized"
    },
    "llama-3.2-3b-Q4_K_M": {
        "repo": "unsloth/Llama-3.2-3B-Instruct-GGUF",
        "file": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "size": "1.9 GB",
        "description": "Llama 3.2 3B Instruct, Q4_K_M"
    }
}
```

## See Also

- [InferenceEngine API](inference-engine.md)
- [Model Selection Guide](../guides/model-selection.md)
- [GGUF Format](../gguf/overview.md)
