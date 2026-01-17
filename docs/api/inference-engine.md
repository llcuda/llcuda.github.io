# InferenceEngine API Reference

Complete API documentation for the `InferenceEngine` class, the main interface for llcuda inference.

## Class Overview

```python
class InferenceEngine:
    """
    High-level Python interface for LLM inference with CUDA acceleration.

    Provides automatic server management, model loading, and inference APIs.
    """
```

## Constructor

### `__init__(server_url="http://127.0.0.1:8090")`

Initialize a new inference engine.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `server_url` | `str` | `"http://127.0.0.1:8090"` | URL of llama-server backend |

**Example:**

```python
import llcuda

# Default URL
engine = llcuda.InferenceEngine()

# Custom port
engine = llcuda.InferenceEngine(server_url="http://127.0.0.1:8091")

# Remote server
engine = llcuda.InferenceEngine(server_url="http://192.168.1.100:8090")
```

## Methods

### `load_model()`

Load a GGUF model for inference with automatic configuration.

```python
def load_model(
    model_name_or_path: str,
    gpu_layers: Optional[int] = None,
    ctx_size: Optional[int] = None,
    auto_start: bool = True,
    auto_configure: bool = True,
    n_parallel: int = 1,
    verbose: bool = True,
    interactive_download: bool = True,
    silent: bool = False,
    **kwargs
) -> bool
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name_or_path` | `str` | Required | Model registry name, local path, or HF repo |
| `gpu_layers` | `Optional[int]` | `None` | Number of layers to offload to GPU (None=auto) |
| `ctx_size` | `Optional[int]` | `None` | Context window size (None=auto) |
| `auto_start` | `bool` | `True` | Automatically start llama-server if not running |
| `auto_configure` | `bool` | `True` | Auto-detect optimal settings |
| `n_parallel` | `int` | `1` | Number of parallel sequences |
| `verbose` | `bool` | `True` | Print status messages |
| `interactive_download` | `bool` | `True` | Ask before downloading models |
| `silent` | `bool` | `False` | Suppress llama-server output |
| `**kwargs` | `dict` | `{}` | Additional server parameters |

**Returns:**

- `bool` - True if model loaded successfully

**Raises:**

- `FileNotFoundError` - Model file not found
- `ValueError` - Model download cancelled
- `ConnectionError` - Server not running and auto_start=False
- `RuntimeError` - Server failed to start

**Loading Methods:**

=== "Registry Name"

    ```python
    # Auto-download from HuggingFace registry
    engine.load_model("gemma-3-1b-Q4_K_M")
    ```

=== "HuggingFace Syntax"

    ```python
    # Direct HF download
    engine.load_model(
        "unsloth/gemma-3-1b-it-GGUF:gemma-3-1b-it-Q4_K_M.gguf"
    )
    ```

=== "Local Path"

    ```python
    # Load local GGUF file
    engine.load_model("/path/to/model.gguf")
    ```

**Example:**

```python
# Auto-configuration (recommended)
engine.load_model(
    "gemma-3-1b-Q4_K_M",
    auto_start=True,
    auto_configure=True,
    verbose=True
)

# Manual configuration
engine.load_model(
    "gemma-3-1b-Q4_K_M",
    gpu_layers=35,
    ctx_size=4096,
    batch_size=512,
    ubatch_size=128,
    auto_configure=False
)

# Silent mode
engine.load_model(
    "gemma-3-1b-Q4_K_M",
    silent=True,  # No llama-server output
    verbose=False # No status messages
)
```

### `infer()`

Run inference on a single prompt.

```python
def infer(
    prompt: str,
    max_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 40,
    seed: int = 0,
    stop_sequences: Optional[List[str]] = None
) -> InferResult
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | `str` | Required | Input prompt text |
| `max_tokens` | `int` | `128` | Maximum tokens to generate |
| `temperature` | `float` | `0.7` | Sampling temperature (0.0-2.0) |
| `top_p` | `float` | `0.9` | Nucleus sampling threshold |
| `top_k` | `int` | `40` | Top-k sampling limit |
| `seed` | `int` | `0` | Random seed (0=random) |
| `stop_sequences` | `Optional[List[str]]` | `None` | Stop generation at these sequences |

**Returns:**

- `InferResult` - Result object with text and metrics

**Example:**

```python
# Basic inference
result = engine.infer(
    prompt="What is AI?",
    max_tokens=100
)
print(result.text)

# Advanced parameters
result = engine.infer(
    prompt="Write a poem about",
    max_tokens=200,
    temperature=0.9,  # More creative
    top_p=0.95,
    top_k=50,
    seed=42,
    stop_sequences=["\n\n", "###"]
)

# Check results
if result.success:
    print(f"Generated: {result.text}")
    print(f"Speed: {result.tokens_per_sec:.1f} tok/s")
    print(f"Latency: {result.latency_ms:.0f} ms")
else:
    print(f"Error: {result.error_message}")
```

### `batch_infer()`

Run batch inference on multiple prompts.

```python
def batch_infer(
    prompts: List[str],
    max_tokens: int = 128,
    **kwargs
) -> List[InferResult]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompts` | `List[str]` | Required | List of input prompts |
| `max_tokens` | `int` | `128` | Maximum tokens per prompt |
| `**kwargs` | `dict` | `{}` | Additional parameters (temperature, top_p, etc.) |

**Returns:**

- `List[InferResult]` - List of result objects

**Example:**

```python
prompts = [
    "What is machine learning?",
    "Explain neural networks.",
    "What is deep learning?"
]

results = engine.batch_infer(
    prompts,
    max_tokens=100,
    temperature=0.7
)

for i, result in enumerate(results):
    print(f"Prompt {i+1}: {result.text}")
    print(f"Speed: {result.tokens_per_sec:.1f} tok/s\n")
```

### `infer_stream()`

Run streaming inference with real-time callbacks.

```python
def infer_stream(
    prompt: str,
    callback: Callable[[str], None],
    max_tokens: int = 128,
    temperature: float = 0.7,
    **kwargs
) -> InferResult
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | `str` | Required | Input prompt text |
| `callback` | `Callable` | Required | Function called for each chunk |
| `max_tokens` | `int` | `128` | Maximum tokens to generate |
| `temperature` | `float` | `0.7` | Sampling temperature |
| `**kwargs` | `dict` | `{}` | Additional parameters |

**Returns:**

- `InferResult` - Complete result after streaming

**Example:**

```python
def print_chunk(text):
    print(text, end='', flush=True)

result = engine.infer_stream(
    prompt="Write a story about AI:",
    callback=print_chunk,
    max_tokens=200,
    temperature=0.8
)

print(f"\n\nTotal speed: {result.tokens_per_sec:.1f} tok/s")
```

### `check_server()`

Check if llama-server is running and accessible.

```python
def check_server() -> bool
```

**Returns:**

- `bool` - True if server is accessible, False otherwise

**Example:**

```python
if engine.check_server():
    print("Server is running")
else:
    print("Server is not running")
    # Optionally start it
    engine.load_model("model.gguf", auto_start=True)
```

### `get_metrics()`

Get current performance metrics.

```python
def get_metrics() -> Dict[str, Any]
```

**Returns:**

- `dict` - Dictionary with latency, throughput, and GPU metrics

**Return Structure:**

```python
{
    'latency': {
        'mean_ms': float,
        'p50_ms': float,
        'p95_ms': float,
        'p99_ms': float,
        'min_ms': float,
        'max_ms': float,
        'sample_count': int
    },
    'throughput': {
        'total_tokens': int,
        'total_requests': int,
        'tokens_per_sec': float,
        'requests_per_sec': float
    }
}
```

**Example:**

```python
metrics = engine.get_metrics()

print(f"Average speed: {metrics['throughput']['tokens_per_sec']:.1f} tok/s")
print(f"P95 latency: {metrics['latency']['p95_ms']:.0f} ms")
print(f"Total requests: {metrics['throughput']['total_requests']}")
```

### `reset_metrics()`

Reset performance metrics counters.

```python
def reset_metrics() -> None
```

**Example:**

```python
# Reset before benchmark
engine.reset_metrics()

# Run tests
for i in range(100):
    engine.infer(f"Test {i}", max_tokens=50)

# Get clean metrics
metrics = engine.get_metrics()
print(f"Benchmark: {metrics['throughput']['tokens_per_sec']:.1f} tok/s")
```

### `unload_model()`

Unload the current model and stop server.

```python
def unload_model() -> None
```

**Example:**

```python
# Unload when done
engine.unload_model()

# Load different model
engine.load_model("other-model.gguf", auto_start=True)
```

## Properties

### `is_loaded`

Check if a model is currently loaded.

```python
@property
def is_loaded() -> bool
```

**Example:**

```python
if engine.is_loaded:
    print("Model is loaded")
    result = engine.infer("Test", max_tokens=10)
else:
    print("No model loaded")
    engine.load_model("model.gguf")
```

## Context Manager Support

`InferenceEngine` supports context manager protocol for automatic cleanup.

**Example:**

```python
with llcuda.InferenceEngine() as engine:
    engine.load_model("gemma-3-1b-Q4_K_M", auto_start=True)

    result = engine.infer("Hello!", max_tokens=50)
    print(result.text)

# Server automatically stops when exiting context
```

## InferResult Class

Result object returned by inference methods.

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `success` | `bool` | Whether inference succeeded |
| `text` | `str` | Generated text |
| `tokens_generated` | `int` | Number of tokens generated |
| `latency_ms` | `float` | Inference latency in milliseconds |
| `tokens_per_sec` | `float` | Generation throughput |
| `error_message` | `str` | Error message if failed |

### Example:

```python
result = engine.infer("Test", max_tokens=50)

# Access properties
print(f"Text: {result.text}")
print(f"Success: {result.success}")
print(f"Tokens: {result.tokens_generated}")
print(f"Latency: {result.latency_ms:.0f} ms")
print(f"Speed: {result.tokens_per_sec:.1f} tok/s")

# String representation
print(str(result))  # Returns result.text

# Repr
print(repr(result))
# Output: InferResult(tokens=50, latency=745.2ms, throughput=134.2 tok/s)
```

## Complete Example

```python
import llcuda

# Create engine
engine = llcuda.InferenceEngine()

# Load model with auto-configuration
engine.load_model(
    "gemma-3-1b-Q4_K_M",
    auto_start=True,
    verbose=True
)

# Single inference
result = engine.infer(
    prompt="What is machine learning?",
    max_tokens=100,
    temperature=0.7
)

print(f"Response: {result.text}")
print(f"Speed: {result.tokens_per_sec:.1f} tok/s")

# Batch inference
prompts = [
    "Explain AI",
    "What are neural networks?",
    "Define deep learning"
]

results = engine.batch_infer(prompts, max_tokens=80)
for i, r in enumerate(results):
    print(f"\nPrompt {i+1}: {r.text}")

# Get metrics
metrics = engine.get_metrics()
print(f"\nTotal speed: {metrics['throughput']['tokens_per_sec']:.1f} tok/s")

# Cleanup
engine.unload_model()
```

## See Also

- [Models API](models.md) - Model management
- [Device API](device.md) - GPU device management
- [Examples](examples.md) - More code examples
- [Performance Guide](../tutorials/performance.md) - Optimization tips
