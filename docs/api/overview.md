# API Reference Overview

Complete API documentation for llcuda v2.2.0.

## :material-puzzle: Main Components

llcuda provides a simple, PyTorch-style API for GPU-accelerated LLM inference.

### Core Classes

| Class | Purpose | Documentation |
|-------|---------|---------------|
| `InferenceEngine` | Main interface for model loading and inference | [Details](inference-engine.md) |
| `InferenceResult` | Container for inference results with metrics | [Details](inference-engine.md#inference-result) |

### Utility Functions

| Function | Purpose | Documentation |
|----------|---------|---------------|
| `check_gpu_compatibility()` | Verify GPU support | [Details](device.md#check-gpu-compatibility) |
| `get_device_properties()` | Get GPU device information | [Details](device.md#device-properties) |

## :rocket: Quick API Reference

### Basic Usage

```python
import llcuda

# Create engine
engine = llcuda.InferenceEngine()

# Load model
engine.load_model(
    "unsloth/gemma-3-1b-it-GGUF:gemma-3-1b-it-Q4_K_M.gguf",
    silent=True
)

# Run inference
result = engine.infer("What is AI?", max_tokens=100)

# Access results
print(result.text)                    # Generated text
print(result.tokens_per_sec)          # Speed in tokens/sec
print(result.latency_ms)              # Latency in milliseconds
print(result.tokens_generated)        # Number of tokens generated
```

### InferenceEngine Methods

#### `__init__(server_url=None)`

Create a new inference engine instance.

**Parameters:**
- `server_url` (str, optional): Custom llama-server URL. Default: `http://127.0.0.1:8090`

#### `load_model(model_path, silent=False, auto_start=True, **kwargs)`

Load a GGUF model for inference.

**Parameters:**
- `model_path` (str): Model identifier or path
  - HuggingFace: `"unsloth/repo-name:filename.gguf"`
  - Registry: `"gemma-3-1b-Q4_K_M"`
  - Local: `"/path/to/model.gguf"`
- `silent` (bool): Suppress llama-server output. Default: `False`
- `auto_start` (bool): Start server automatically. Default: `True`
- `**kwargs`: Additional options (context_size, gpu_layers, etc.)

#### `infer(prompt, max_tokens=512, temperature=0.7, **kwargs)`

Run inference on a single prompt.

**Parameters:**
- `prompt` (str): Input text
- `max_tokens` (int): Maximum tokens to generate. Default: 512
- `temperature` (float): Sampling temperature. Default: 0.7
- `top_p` (float): Nucleus sampling threshold. Default: 0.9
- `top_k` (int): Top-k sampling. Default: 40
- `stop_sequences` (list): Stop generation at these sequences

**Returns:**
- `InferenceResult`: Result object with text and metrics

#### `batch_infer(prompts, max_tokens=512, **kwargs)`

Run inference on multiple prompts.

**Parameters:**
- `prompts` (list[str]): List of input texts
- `max_tokens` (int): Maximum tokens per prompt
- `**kwargs`: Same as `infer()`

**Returns:**
- `list[InferenceResult]`: List of results

#### `get_metrics()`

Get aggregated performance metrics.

**Returns:**
- `dict`: Metrics dictionary with throughput and latency stats

### InferenceResult Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `text` | str | Generated text |
| `tokens_per_sec` | float | Generation speed |
| `latency_ms` | float | Total latency in ms |
| `tokens_generated` | int | Number of tokens |

### Utility Functions

#### `check_gpu_compatibility()`

Check if current GPU is compatible with llcuda.

**Returns:**
```python
{
    'gpu_name': str,          # e.g., "Tesla T4"
    'compute_capability': str, # e.g., "7.5"
    'compatible': bool,       # True if supported
    'platform': str          # e.g., "kaggle", "local"
}
```

**Example:**
```python
compat = llcuda.check_gpu_compatibility()
if compat['compatible']:
    print(f"✅ {compat['gpu_name']} is compatible!")
else:
    print(f"⚠️ {compat['gpu_name']} may not work")
```

## :books: Detailed Documentation

- [InferenceEngine](inference-engine.md) - Complete InferenceEngine documentation
- [Models & GGUF](models.md) - Model loading and GGUF format
- [GPU & Device](device.md) - GPU management and compatibility
- [Examples](examples.md) - Code examples and use cases

## :link: See Also

- [Quick Start Guide](../guides/quickstart.md)
- [Tutorials](../tutorials/index.md)
- [Performance Benchmarks](../performance/benchmarks.md)
