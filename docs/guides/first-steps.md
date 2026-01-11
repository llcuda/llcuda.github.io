# First Steps with llcuda

After installing llcuda, this guide will help you get started with your first inference tasks and understand the core concepts.

## Prerequisites

Before proceeding, ensure you have:

- [x] Installed llcuda v2.0.6 from GitHub
- [x] Tesla T4 GPU (Google Colab or compatible)
- [x] CUDA 12.x runtime (pre-installed in Colab)
- [x] Python 3.11+

## Verify Installation

First, verify that llcuda is installed correctly:

```python
import llcuda

print(f"llcuda version: {llcuda.__version__}")
# Output: llcuda version: 2.0.6
```

## Check GPU Availability

Verify your GPU is compatible:

```python
import llcuda
from llcuda.core import get_device_properties

# Check CUDA availability
if llcuda.check_cuda_available():
    print("✓ CUDA is available")
else:
    print("✗ CUDA not available")

# Get device information
props = get_device_properties(0)
print(f"GPU: {props.name}")
print(f"Compute: SM {props.compute_capability_major}.{props.compute_capability_minor}")
print(f"Memory: {props.total_global_mem / (1024**3):.1f} GB")
```

**Expected Output (Google Colab T4):**
```
✓ CUDA is available
GPU: Tesla T4
Compute: SM 7.5
Memory: 14.8 GB
```

!!! success "Compatibility Check"
    llcuda v2.0.6 is optimized exclusively for Tesla T4 GPUs (SM 7.5). If you see a different GPU, consider using llcuda v1.2.2 for broader compatibility.

## Your First Inference

Let's run a simple inference using the `InferenceEngine` class:

### Step 1: Create the Engine

```python
import llcuda

# Create inference engine
engine = llcuda.InferenceEngine()
```

### Step 2: Load a Model

llcuda supports loading models from a registry, local paths, or HuggingFace:

=== "From Registry"

    ```python
    # Load Gemma 3-1B Q4_K_M from registry (auto-downloads)
    engine.load_model(
        "gemma-3-1b-Q4_K_M",
        auto_start=True,
        verbose=True
    )
    ```

=== "From HuggingFace"

    ```python
    # Load directly from HuggingFace
    engine.load_model(
        "unsloth/gemma-3-1b-it-GGUF:gemma-3-1b-it-Q4_K_M.gguf",
        auto_start=True,
        verbose=True
    )
    ```

=== "From Local Path"

    ```python
    # Load from local GGUF file
    engine.load_model(
        "/path/to/model.gguf",
        gpu_layers=99,
        ctx_size=2048,
        auto_start=True
    )
    ```

**What happens during loading:**

1. Model file is downloaded/validated
2. Optimal settings are auto-configured based on your GPU
3. llama-server starts automatically
4. Model loads into GPU memory

### Step 3: Run Inference

```python
# Simple inference
result = engine.infer(
    prompt="What is artificial intelligence?",
    max_tokens=100,
    temperature=0.7
)

print(result.text)
print(f"\nSpeed: {result.tokens_per_sec:.1f} tokens/sec")
print(f"Latency: {result.latency_ms:.0f} ms")
```

**Expected Output:**
```
Artificial intelligence (AI) is a branch of computer science that focuses on creating
intelligent machines that can perform tasks that typically require human intelligence,
such as visual perception, speech recognition, decision-making, and language translation...

Speed: 134.2 tokens/sec
Latency: 745 ms
```

## Understanding Key Concepts

### InferenceEngine

The `InferenceEngine` class is the main interface for llcuda:

```python
engine = llcuda.InferenceEngine(server_url="http://127.0.0.1:8090")
```

- Manages llama-server lifecycle
- Handles model loading and configuration
- Provides high-level inference API
- Tracks performance metrics

### Model Loading Options

llcuda provides flexible model loading:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `gpu_layers` | Number of layers to offload to GPU | Auto-configured |
| `ctx_size` | Context window size | Auto-configured |
| `auto_start` | Automatically start server | `True` |
| `auto_configure` | Auto-detect optimal settings | `True` |
| `verbose` | Print status messages | `True` |
| `silent` | Suppress llama-server output | `False` |

### Inference Parameters

Control generation with these parameters:

```python
result = engine.infer(
    prompt="Your prompt here",
    max_tokens=128,        # Maximum tokens to generate
    temperature=0.7,       # Sampling temperature (0.0-2.0)
    top_p=0.9,            # Nucleus sampling threshold
    top_k=40,             # Top-k sampling limit
    seed=42,              # Random seed (0=random)
    stop_sequences=["\n"] # Stop generation at these sequences
)
```

## Working with Results

The `InferResult` object contains generation output and metrics:

```python
result = engine.infer("Write a haiku about AI")

# Access generated text
print(result.text)

# Check if inference succeeded
if result.success:
    print(f"✓ Generated {result.tokens_generated} tokens")
    print(f"✓ Speed: {result.tokens_per_sec:.1f} tok/s")
    print(f"✓ Latency: {result.latency_ms:.0f} ms")
else:
    print(f"✗ Error: {result.error_message}")
```

## Batch Processing

Process multiple prompts efficiently:

```python
prompts = [
    "What is machine learning?",
    "Explain neural networks.",
    "What is deep learning?"
]

results = engine.batch_infer(prompts, max_tokens=100)

for i, result in enumerate(results):
    print(f"\n--- Prompt {i+1} ---")
    print(result.text)
    print(f"Speed: {result.tokens_per_sec:.1f} tok/s")
```

## Performance Metrics

Track inference performance:

```python
# Get current metrics
metrics = engine.get_metrics()

print(f"Total requests: {metrics['throughput']['total_requests']}")
print(f"Total tokens: {metrics['throughput']['total_tokens']}")
print(f"Average speed: {metrics['throughput']['tokens_per_sec']:.1f} tok/s")
print(f"Mean latency: {metrics['latency']['mean_ms']:.0f} ms")
print(f"P95 latency: {metrics['latency']['p95_ms']:.0f} ms")

# Reset metrics
engine.reset_metrics()
```

## Context Manager Pattern

Use context managers for automatic cleanup:

```python
with llcuda.InferenceEngine() as engine:
    engine.load_model("gemma-3-1b-Q4_K_M", auto_start=True)

    result = engine.infer("Hello, AI!", max_tokens=50)
    print(result.text)

# Server stops automatically when exiting context
```

## Common Patterns

### Quick Inference

For one-off inferences:

```python
from llcuda import quick_infer

text = quick_infer(
    prompt="What is Python?",
    model_path="gemma-3-1b-Q4_K_M",
    max_tokens=100,
    auto_start=True
)
print(text)
```

### Streaming Generation

For real-time output:

```python
def print_chunk(text):
    print(text, end='', flush=True)

result = engine.infer_stream(
    prompt="Write a story about AI",
    callback=print_chunk,
    max_tokens=200
)
```

### Model Switching

Switch between models:

```python
# Unload current model
engine.unload_model()

# Load different model
engine.load_model("llama-3.2-3b-Q4_K_M", auto_start=True)

# Run inference with new model
result = engine.infer("Test prompt", max_tokens=50)
```

## Auto-Configuration

llcuda automatically configures optimal settings:

```python
# Auto-configuration enabled (default)
engine.load_model(
    "gemma-3-1b-Q4_K_M",
    auto_configure=True  # Detects optimal gpu_layers, ctx_size, batch_size
)

# Manual configuration
engine.load_model(
    "gemma-3-1b-Q4_K_M",
    gpu_layers=35,
    ctx_size=4096,
    auto_configure=False
)
```

Auto-configuration analyzes:

- Model size and architecture
- Available GPU memory
- GPU compute capability
- Optimal batch sizes

## Troubleshooting First Steps

### Model Not Loading

```python
# Check model file exists
from pathlib import Path
model_path = Path("~/.cache/llcuda/models/gemma-3-1b-it-Q4_K_M.gguf").expanduser()
print(f"Model exists: {model_path.exists()}")

# Try manual download
from llcuda.models import download_model
download_model(
    "unsloth/gemma-3-1b-it-GGUF",
    "gemma-3-1b-it-Q4_K_M.gguf"
)
```

### Server Not Starting

```python
# Check if server is already running
if engine.check_server():
    print("Server is running")
else:
    print("Server is not running")

# Force restart
engine.unload_model()  # Stop current server
engine.load_model("gemma-3-1b-Q4_K_M", auto_start=True)
```

### Out of Memory

```python
# Reduce GPU layers
engine.load_model(
    "gemma-3-1b-Q4_K_M",
    gpu_layers=20,  # Reduce from 99
    ctx_size=1024,  # Reduce context
    auto_configure=False
)

# Check GPU memory
props = get_device_properties(0)
print(f"Total memory: {props.total_global_mem / (1024**3):.1f} GB")
```

## Next Steps

Now that you've completed your first inference, explore:

- [Model Selection Guide](model-selection.md) - Choose the right model for your task
- [Performance Optimization](../performance/optimization.md) - Maximize throughput
- [API Reference](../api/inference-engine.md) - Detailed API documentation
- [Unsloth Integration](../tutorials/unsloth-integration.md) - Fine-tune and deploy models
- [Google Colab Tutorial](../tutorials/gemma-3-1b-colab.md) - Complete hands-on tutorial

## Quick Reference

```python
# Basic workflow
import llcuda

# 1. Create engine
engine = llcuda.InferenceEngine()

# 2. Load model
engine.load_model("gemma-3-1b-Q4_K_M", auto_start=True)

# 3. Run inference
result = engine.infer("Your prompt", max_tokens=100)

# 4. Use result
print(result.text)
print(f"{result.tokens_per_sec:.1f} tok/s")

# 5. Cleanup
engine.unload_model()
```

!!! tip "Performance Tip"
    For best performance on Tesla T4, use Q4_K_M quantization with full GPU offload (gpu_layers=99). This achieves 130+ tokens/sec on Gemma 3-1B.

## Resources

- [Installation Guide](installation.md)
- [Quick Start](quickstart.md)
- [FAQ](faq.md)
- [Troubleshooting](troubleshooting.md)
- [GitHub Issues](https://github.com/waqasm86/llcuda/issues)
