# Performance Optimization

Advanced techniques to maximize inference performance with llcuda v2.0.6.

## Overview

This guide covers optimization strategies for:

- Quantization selection
- Context length tuning
- Batch size optimization
- Server configuration
- Memory optimization
- Multi-GPU setups

---

## Quantization Selection

### Understanding Quantization

Quantization reduces model precision to improve speed and reduce VRAM usage.

| Quantization | Bits/Weight | Speed | Quality | VRAM | Recommendation |
|--------------|-------------|-------|---------|------|----------------|
| **Q2_K** | 2.5 | Fastest | Poor | Lowest | Prototyping only |
| **Q3_K_M** | 3.5 | Very fast | Fair | Low | Low VRAM only |
| **Q4_0** | 4.0 | Fast | Good | Medium | Speed priority |
| **Q4_K_M** | 4.5 | Fast | Excellent | Medium | ✅ **Best balance** |
| **Q5_K_M** | 5.5 | Moderate | Near-perfect | High | Quality priority |
| **Q6_K** | 6.5 | Slow | Minimal loss | Higher | Rarely needed |
| **Q8_0** | 8.0 | Slower | Negligible loss | Highest | Development only |
| **F16** | 16.0 | Slowest | Perfect | Maximum | Not recommended |

### Choosing the Right Quantization

**For Tesla T4 (15 GB VRAM):**

```python
import llcuda

# Q4_K_M: Best overall choice
# - 134 tok/s
# - 1.2 GB VRAM
# - 99% quality
engine = llcuda.InferenceEngine()
engine.load_model(
    "unsloth/gemma-3-1b-it-GGUF:gemma-3-1b-it-Q4_K_M.gguf",
    silent=True
)
```

**For limited VRAM (< 8 GB):**

```python
# Q4_0: Faster, less VRAM
engine.load_model(
    "unsloth/gemma-3-1b-it-GGUF:gemma-3-1b-it-Q4_0.gguf",
    silent=True
)
```

**For quality-critical applications:**

```python
# Q5_K_M: Better quality, slower
engine.load_model(
    "unsloth/gemma-3-1b-it-GGUF:gemma-3-1b-it-Q5_K_M.gguf",
    silent=True
)
```

### Quantization Performance Comparison

On Tesla T4, Gemma 3-1B:

| Quantization | Tokens/sec | Speedup | Quality | Best For |
|--------------|------------|---------|---------|----------|
| Q4_K_M | 134.3 | 1.00x | 99% | General use ✅ |
| Q4_0 | 138.7 | 1.03x | 97% | Speed-critical |
| Q5_K_M | 110.2 | 0.82x | 99.5% | High quality |
| Q3_K_M | 142.3 | 1.06x | 92% | Low VRAM |

---

## Context Length Optimization

### Impact of Context Size

| Context Size | Use Case | Tokens/sec | VRAM | Latency/100tok |
|--------------|----------|------------|------|----------------|
| 512 | Quick Q&A | 142.5 | 0.9 GB | 702 ms |
| 1024 | Short chat | 138.7 | 1.0 GB | 721 ms |
| **2048** | **Standard** | **134.3** | **1.2 GB** | **745 ms** |
| 4096 | Long chat | 125.1 | 2.0 GB | 799 ms |
| 8192 | Documents | 105.3 | 3.5 GB | 950 ms |

**Rule of thumb:** Use the smallest context size that fits your use case.

### Optimal Context Sizes

**For interactive chat:**
```python
engine.load_model(
    model_path,
    ctx_size=2048,  # Optimal for most conversations
    silent=True
)
```

**For document Q&A:**
```python
engine.load_model(
    model_path,
    ctx_size=4096,  # Long context support
    silent=True
)
```

**For quick responses:**
```python
engine.load_model(
    model_path,
    ctx_size=1024,  # Faster, lower VRAM
    silent=True
)
```

### Memory vs Context Trade-off

VRAM usage grows quadratically with context (KV cache):

```python
# Calculate VRAM for context
def estimate_vram_gb(model_size_gb, ctx_size):
    base_vram = model_size_gb
    kv_cache_gb = (ctx_size / 2048) ** 1.2 * 0.3  # Approximate
    return base_vram + kv_cache_gb

# Example: Gemma 3-1B Q4_K_M
print(estimate_vram_gb(1.2, 2048))  # ~1.5 GB
print(estimate_vram_gb(1.2, 8192))  # ~3.2 GB
```

---

## Batch Size Tuning

### Understanding Batch Parameters

- `batch_size`: Maximum tokens processed in parallel (prompt processing)
- `ubatch_size`: Micro-batch size for generation (critical for low VRAM)

### Optimal Settings for Tesla T4

```python
engine.load_model(
    model_path,
    batch_size=512,     # Optimal for T4
    ubatch_size=128,    # Good balance
    silent=True
)
```

### Batch Size Recommendations

| VRAM | Model Size | batch_size | ubatch_size | Performance |
|------|------------|------------|-------------|-------------|
| 4 GB | 1B | 256 | 64 | Good |
| 8 GB | 1-3B | 512 | 128 | Optimal |
| 15 GB | 1-7B | 512 | 128 | Optimal |
| 24 GB | 1-13B | 1024 | 256 | Excellent |
| 40+ GB | Any | 2048 | 512 | Maximum |

### Impact Analysis

On Tesla T4, Gemma 3-1B Q4_K_M:

| batch_size | ubatch_size | Tokens/sec | VRAM | Notes |
|------------|-------------|------------|------|-------|
| 128 | 32 | 128.5 | 1.12 GB | Too small |
| 256 | 64 | 131.2 | 1.15 GB | Suboptimal |
| **512** | **128** | **134.3** | **1.18 GB** | **Optimal** |
| 1024 | 256 | 133.8 | 1.25 GB | Diminishing returns |
| 2048 | 512 | 132.1 | 1.42 GB | Slower |

---

## GPU Layer Offload

### Full vs Partial Offload

Always use full GPU offload when possible:

```python
# RECOMMENDED: Full GPU offload
engine.load_model(
    model_path,
    gpu_layers=99,  # Offload all layers
    silent=True
)
```

### Partial Offload (Limited VRAM)

If VRAM is insufficient:

```python
from llcuda.utils import get_recommended_gpu_layers

# Calculate optimal layers
gpu_layers = get_recommended_gpu_layers(
    model_size_gb=1.2,
    vram_gb=4.0
)

engine.load_model(
    model_path,
    gpu_layers=gpu_layers,  # Partial offload
    silent=True
)
```

### Layer Offload Performance

Tesla T4, Gemma 3-1B:

| GPU Layers | Tokens/sec | VRAM | Speedup |
|------------|------------|------|---------|
| 0 | 8.2 | 0 GB | 1.0x |
| 10 | 45.3 | 0.4 GB | 5.5x |
| 20 | 92.1 | 0.9 GB | 11.2x |
| 35 (full) | 134.3 | 1.2 GB | **16.4x** |

**Rule:** Each layer adds ~3.8 tok/s and ~34 MB VRAM.

---

## Server Configuration

### Optimal Server Settings

```python
engine.load_model(
    model_path,
    # Core settings
    gpu_layers=99,
    ctx_size=2048,
    batch_size=512,
    ubatch_size=128,

    # Server optimization
    n_parallel=1,           # Single request (interactive)
    threads=4,              # CPU threads for processing
    flash_attn=True,        # Enable FlashAttention

    # Advanced
    mlock=False,            # Don't lock memory (Colab)
    numa=False,             # Single NUMA node
    silent=True
)
```

### Parallel Request Handling

For server applications:

```python
# Handle 4 concurrent requests
engine.load_model(
    model_path,
    gpu_layers=99,
    ctx_size=2048,
    batch_size=512,
    ubatch_size=128,
    n_parallel=4,  # 4 parallel sequences
    silent=True
)
```

**Performance with n_parallel:**

| n_parallel | Total tok/s | Latency/request | VRAM | Best For |
|------------|-------------|-----------------|------|----------|
| 1 | 134 | 690 ms | 1.2 GB | Interactive chat |
| 2 | 250 | 755 ms | 1.5 GB | Small server |
| 4 | 460 | 830 ms | 2.3 GB | Production server |
| 8 | 790 | 980 ms | 3.9 GB | High throughput |

---

## FlashAttention Optimization

### When to Use FlashAttention

Enable for contexts > 4096:

```python
engine.load_model(
    model_path,
    ctx_size=8192,
    flash_attn=True,  # Enable FlashAttention
    silent=True
)
```

### FlashAttention Benefits

| Context Size | Without FA | With FA | Speedup | VRAM Saved |
|--------------|------------|---------|---------|------------|
| 2048 | 134.3 | 135.2 | 1.01x | 0.12 GB |
| 4096 | 95.5 | 125.1 | **1.31x** | 0.35 GB |
| 8192 | 55.2 | 105.3 | **1.91x** | 0.98 GB |
| 16384 | 28.5 | 78.5 | **2.75x** | 2.52 GB |

**Key Finding:** Significant benefits (1.3-2.8x) for long contexts.

---

## Memory Optimization

### Reduce VRAM Usage

**1. Use Aggressive Quantization:**
```python
# Q4_0 instead of Q4_K_M saves ~100 MB
engine.load_model(
    "model-Q4_0.gguf",
    silent=True
)
```

**2. Reduce Context Size:**
```python
# 1024 instead of 2048 saves ~200 MB
engine.load_model(
    model_path,
    ctx_size=1024,
    silent=True
)
```

**3. Lower Batch Sizes:**
```python
# Smaller batches use less VRAM
engine.load_model(
    model_path,
    batch_size=256,
    ubatch_size=64,
    silent=True
)
```

### Memory-Constrained Configuration

For GPUs with < 6 GB VRAM:

```python
engine.load_model(
    "model-Q4_0.gguf",
    gpu_layers=99,
    ctx_size=1024,
    batch_size=256,
    ubatch_size=64,
    n_parallel=1,
    silent=True
)
```

---

## Auto-Configuration

### Let llcuda Optimize

```python
from llcuda.utils import auto_configure_for_model
from pathlib import Path

# Auto-detect optimal settings
settings = auto_configure_for_model(Path("model.gguf"))

# Apply settings
engine.load_model(
    "model.gguf",
    **settings,  # Use all auto-configured values
    silent=True
)
```

### Manual Override

```python
# Start with auto-config
settings = auto_configure_for_model(Path("model.gguf"))

# Override specific settings
settings['ctx_size'] = 4096  # Use longer context
settings['n_parallel'] = 4   # Handle more requests

engine.load_model("model.gguf", **settings, silent=True)
```

---

## Model Selection for Performance

### Size vs Speed Trade-off

| Model Size | Tokens/sec (T4) | VRAM | Best For |
|------------|-----------------|------|----------|
| 1B | 134 | 1.2 GB | ✅ Interactive, production |
| 3B | 48 | 2.0 GB | Balanced quality/speed |
| 7B | 21 | 5.0 GB | Quality-focused |
| 13B | 12 | 9.0 GB | Maximum quality |

**Recommendation:** For T4, use 1B models for best performance.

### Popular High-Performance Models

```python
# Fastest: Gemma 3-1B (134 tok/s)
engine.load_model(
    "unsloth/gemma-3-1b-it-GGUF:gemma-3-1b-it-Q4_K_M.gguf",
    silent=True
)

# Balanced: Llama 3.2-3B (48 tok/s)
engine.load_model(
    "unsloth/Llama-3.2-3B-Instruct-Q4_K_M-GGUF",
    silent=True
)

# Quality: Qwen 2.5-7B (21 tok/s)
engine.load_model(
    "Qwen/Qwen2.5-7B-Instruct-GGUF:Q4_K_M",
    silent=True
)
```

---

## Generation Parameter Tuning

### Speed vs Quality

**Fastest (Deterministic):**
```python
result = engine.infer(
    prompt,
    max_tokens=100,
    temperature=0.1,
    top_k=10,
    top_p=0.9
)
# ~140 tok/s
```

**Balanced:**
```python
result = engine.infer(
    prompt,
    max_tokens=100,
    temperature=0.7,
    top_k=40,
    top_p=0.9
)
# ~134 tok/s
```

**Creative (Slower):**
```python
result = engine.infer(
    prompt,
    max_tokens=100,
    temperature=1.5,
    top_k=200,
    top_p=0.95
)
# ~118 tok/s
```

### Parameter Impact

| Parameter | Low Value | High Value | Speed Impact |
|-----------|-----------|------------|--------------|
| temperature | Faster | Slower | 5-12% |
| top_k | Faster | Slower | 2-5% |
| top_p | Minimal | Minimal | < 1% |

---

## Multi-GPU Configuration

### Select Specific GPU

```python
import os

# Use GPU 1
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import llcuda
engine = llcuda.InferenceEngine()
```

### Load Balance Across GPUs

```python
# Use GPUs 0 and 1
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

# llama.cpp will distribute layers automatically
import llcuda
engine = llcuda.InferenceEngine()
engine.load_model(model_path, gpu_layers=99, silent=True)
```

---

## Benchmarking Your Setup

### Quick Benchmark

```python
import llcuda
import time

# Setup
engine = llcuda.InferenceEngine()
engine.load_model("model.gguf", silent=True)

# Warmup
for _ in range(3):
    engine.infer("Warmup", max_tokens=10)

# Benchmark
engine.reset_metrics()
start = time.time()

for _ in range(10):
    engine.infer("Test prompt", max_tokens=100)

elapsed = time.time() - start
metrics = engine.get_metrics()

print(f"Throughput: {metrics['throughput']['tokens_per_sec']:.1f} tok/s")
print(f"Latency: {metrics['latency']['mean_ms']:.0f}ms")
print(f"Total time: {elapsed:.2f}s")
```

### Compare Configurations

```python
configs = [
    {"ctx_size": 1024, "batch_size": 256},
    {"ctx_size": 2048, "batch_size": 512},
    {"ctx_size": 4096, "batch_size": 1024},
]

for config in configs:
    engine = llcuda.InferenceEngine()
    engine.load_model("model.gguf", **config, silent=True)

    # Warmup
    for _ in range(3):
        engine.infer("Warmup", max_tokens=10)

    # Benchmark
    engine.reset_metrics()
    for _ in range(10):
        engine.infer("Test", max_tokens=100)

    metrics = engine.get_metrics()
    print(f"Config {config}: {metrics['throughput']['tokens_per_sec']:.1f} tok/s")

    engine.unload_model()
```

---

## Optimization Checklist

### Pre-Deployment Checklist

- [ ] Use Q4_K_M quantization (best balance)
- [ ] Enable full GPU offload (`gpu_layers=99`)
- [ ] Set optimal context size (2048 for most cases)
- [ ] Configure batch sizes (512/128 for T4)
- [ ] Enable FlashAttention for long contexts
- [ ] Test with warmup runs
- [ ] Benchmark your specific use case
- [ ] Monitor VRAM usage
- [ ] Profile latency distribution (P50, P95, P99)

### Production Settings

```python
import llcuda
from llcuda.utils import auto_configure_for_model
from pathlib import Path

# Step 1: Verify GPU
compat = llcuda.check_gpu_compatibility()
assert compat['compatible'], "GPU not compatible"

# Step 2: Auto-configure
model_path = Path("model.gguf")
settings = auto_configure_for_model(model_path)

# Step 3: Override for production
settings['n_parallel'] = 4      # Handle concurrent requests
settings['silent'] = True       # Clean logs

# Step 4: Load
engine = llcuda.InferenceEngine()
engine.load_model(str(model_path), **settings)

# Step 5: Warmup
for _ in range(5):
    engine.infer("Warmup", max_tokens=10)

print("✅ Production deployment ready!")
```

---

## Common Optimization Pitfalls

### ❌ Don't Do This

```python
# TOO SMALL: Limits performance
engine.load_model(model_path, batch_size=64, ubatch_size=16)

# TOO LARGE: Wastes VRAM
engine.load_model(model_path, ctx_size=32768)

# PARTIAL OFFLOAD: When full offload possible
engine.load_model(model_path, gpu_layers=20)  # Use 99 instead

# NO WARMUP: First inference is slow
result = engine.infer(prompt)  # Add warmup first
```

### ✅ Do This Instead

```python
# Optimal settings
engine.load_model(
    model_path,
    gpu_layers=99,
    ctx_size=2048,
    batch_size=512,
    ubatch_size=128,
    silent=True
)

# Warmup
for _ in range(3):
    engine.infer("Warmup", max_tokens=10)

# Production inference
result = engine.infer(prompt, max_tokens=200)
```

---

## See Also

- [Benchmarks](benchmarks.md) - Performance data
- [T4 Results](t4-results.md) - Detailed T4 analysis
- [Model Selection](../guides/model-selection.md) - Choosing models
- [Device API](../api/device.md) - GPU management
- [Troubleshooting](../guides/troubleshooting.md) - Common issues
