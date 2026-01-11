# Performance Optimization Tutorial

Learn how to optimize llcuda for maximum throughput and minimum latency on Tesla T4 GPUs.

!!! tip "Quick Win"
    For immediate performance gains, use Q4_K_M quantization with full GPU offload (gpu_layers=99). This achieves 130+ tok/s on Gemma 3-1B.

## Performance Overview

llcuda v2.0.6 achieves exceptional performance on Tesla T4:

- **Gemma 3-1B:** 134 tok/s (verified)
- **Latency:** < 700ms median
- **Memory:** 1.2 GB for 1B models
- **Throughput:** Consistent across batch sizes

## Key Performance Factors

### 1. Quantization Method

Choose the right quantization for your use case:

=== "Q4_K_M (Recommended)"

    ```python
    engine.load_model("model-Q4_K_M.gguf", gpu_layers=99)
    ```

    **Performance:** 134 tok/s (Gemma 3-1B)
    **Memory:** 1.2 GB
    **Quality:** Excellent (< 1% degradation)
    **Use case:** Production inference

=== "Q5_K_M (Better Quality)"

    ```python
    engine.load_model("model-Q5_K_M.gguf", gpu_layers=99)
    ```

    **Performance:** ~110 tok/s
    **Memory:** 1.5 GB
    **Quality:** Near-perfect
    **Use case:** Quality-critical applications

=== "Q8_0 (Highest Quality)"

    ```python
    engine.load_model("model-Q8_0.gguf", gpu_layers=99)
    ```

    **Performance:** ~75 tok/s
    **Memory:** 2.5 GB
    **Quality:** Minimal loss
    **Use case:** Accuracy-first scenarios

**Recommendation:** Use Q4_K_M for best performance/quality balance.

### 2. GPU Layer Offloading

Control how many layers run on GPU:

```python
# Full GPU offload (fastest)
engine.load_model("model.gguf", gpu_layers=99)  # 134 tok/s

# Partial offload (if VRAM limited)
engine.load_model("model.gguf", gpu_layers=20)  # ~80 tok/s

# CPU only (very slow)
engine.load_model("model.gguf", gpu_layers=0)   # ~8 tok/s
```

**Rule of thumb:** Always use gpu_layers=99 unless you have VRAM constraints.

### 3. Context Window Size

Balance between functionality and speed:

```python
# Small context (fastest)
engine.load_model("model.gguf", ctx_size=1024)  # +10% speed

# Medium context (balanced)
engine.load_model("model.gguf", ctx_size=2048)  # Baseline

# Large context (slower)
engine.load_model("model.gguf", ctx_size=8192)  # -20% speed
```

**Memory impact:**
- 1024 ctx: +0.5 GB
- 2048 ctx: +1.0 GB
- 4096 ctx: +2.0 GB
- 8192 ctx: +4.0 GB

### 4. Batch Processing

Use batch sizes for throughput:

```python
# Configure batch parameters
engine.load_model(
    "model.gguf",
    batch_size=512,    # Logical batch size
    ubatch_size=128,   # Physical batch size
    gpu_layers=99
)
```

**Batch size guidelines:**

| Model Size | batch_size | ubatch_size | Throughput |
|------------|------------|-------------|------------|
| 1B params  | 512        | 128         | 134 tok/s  |
| 3B params  | 256        | 64          | ~100 tok/s |
| 7B params  | 128        | 32          | ~50 tok/s  |

### 5. Flash Attention

llcuda v2.0.6 includes FlashAttention by default:

```python
# FlashAttention is automatically enabled for:
# - Compute capability 7.5+ (T4, RTX 20xx+)
# - Context sizes > 2048
# - All quantization types

# Benefit: 2-3x faster for long contexts
```

**Performance with FlashAttention:**

| Context Size | Without FA | With FA | Speedup |
|--------------|------------|---------|---------|
| 512          | 140 tok/s  | 142 tok/s | 1.01x |
| 2048         | 134 tok/s  | 135 tok/s | 1.01x |
| 4096         | 95 tok/s   | 125 tok/s | **1.32x** |
| 8192         | 55 tok/s   | 105 tok/s | **1.91x** |

## Optimization Workflow

### Step 1: Baseline Measurement

```python
import llcuda

engine = llcuda.InferenceEngine()
engine.load_model(
    "gemma-3-1b-Q4_K_M",
    auto_start=True,
    verbose=True
)

# Run baseline test
prompts = ["Test prompt"] * 10
results = engine.batch_infer(prompts, max_tokens=100)

# Check metrics
metrics = engine.get_metrics()
print(f"Baseline speed: {metrics['throughput']['tokens_per_sec']:.1f} tok/s")
print(f"Baseline latency: {metrics['latency']['mean_ms']:.0f} ms")
```

### Step 2: Optimize GPU Offload

```python
# Test different GPU layer counts
for gpu_layers in [10, 20, 35, 99]:
    engine.unload_model()
    engine.load_model(
        "model.gguf",
        gpu_layers=gpu_layers,
        auto_start=True,
        verbose=False
    )

    result = engine.infer("Test", max_tokens=50)
    print(f"gpu_layers={gpu_layers}: {result.tokens_per_sec:.1f} tok/s")

# Expected output:
# gpu_layers=10: 65.2 tok/s
# gpu_layers=20: 92.1 tok/s
# gpu_layers=35: 127.5 tok/s
# gpu_layers=99: 134.2 tok/s ← Best
```

### Step 3: Optimize Context Size

```python
# Test different context sizes
for ctx_size in [512, 1024, 2048, 4096]:
    engine.unload_model()
    engine.load_model(
        "model.gguf",
        ctx_size=ctx_size,
        gpu_layers=99,
        auto_start=True,
        verbose=False
    )

    result = engine.infer("Test", max_tokens=50)
    print(f"ctx_size={ctx_size}: {result.tokens_per_sec:.1f} tok/s")

# Choose smallest ctx_size that meets your needs
```

### Step 4: Optimize Batch Parameters

```python
# Test batch configurations
configs = [
    (256, 64),   # Small
    (512, 128),  # Medium (default)
    (1024, 256), # Large
]

for batch_size, ubatch_size in configs:
    engine.unload_model()
    engine.load_model(
        "model.gguf",
        batch_size=batch_size,
        ubatch_size=ubatch_size,
        gpu_layers=99,
        auto_start=True,
        verbose=False
    )

    result = engine.infer("Test", max_tokens=50)
    print(f"batch={batch_size}, ubatch={ubatch_size}: {result.tokens_per_sec:.1f} tok/s")
```

## Advanced Optimizations

### Parallel Sequences

Process multiple sequences in parallel:

```python
engine.load_model(
    "model.gguf",
    n_parallel=4,  # Process 4 sequences simultaneously
    gpu_layers=99,
    auto_start=True
)

# Submit multiple requests
import concurrent.futures

def infer_async(prompt):
    return engine.infer(prompt, max_tokens=50)

prompts = ["Prompt 1", "Prompt 2", "Prompt 3", "Prompt 4"]

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(infer_async, prompts))

# Total throughput: ~500 tok/s with n_parallel=4
```

### Continuous Batching

For serving applications:

```python
# Enable continuous batching
engine.load_model(
    "model.gguf",
    n_parallel=8,
    batch_size=512,
    ubatch_size=128,
    gpu_layers=99,
    auto_start=True
)

# Handles variable-length sequences efficiently
# Throughput increases with concurrent requests
```

### Temperature Tuning

Balance quality and speed:

```python
# Faster (less sampling)
result = engine.infer(
    "Prompt",
    temperature=0.1,  # Greedy-like
    top_k=10,         # Limit sampling
    max_tokens=100
)
# Speed: ~140 tok/s

# Balanced
result = engine.infer(
    "Prompt",
    temperature=0.7,  # Default
    top_k=40,
    max_tokens=100
)
# Speed: ~134 tok/s

# Creative (more sampling)
result = engine.infer(
    "Prompt",
    temperature=1.0,
    top_k=100,
    max_tokens=100
)
# Speed: ~125 tok/s
```

## Memory Optimization

### Model Caching

Cache models to avoid reloading:

```python
# Keep model in memory between sessions
engine = llcuda.InferenceEngine()
engine.load_model("model.gguf", auto_start=True)

# Reuse engine for multiple inferences
for i in range(1000):
    result = engine.infer(f"Prompt {i}", max_tokens=50)

# Don't unload until done
engine.unload_model()
```

### KV Cache Management

Control key-value cache:

```python
# Allocate more VRAM for KV cache
engine.load_model(
    "model.gguf",
    ctx_size=4096,       # Context window
    cache_size=None,     # Auto-calculate
    gpu_layers=99
)

# Manual cache control (advanced)
engine.load_model(
    "model.gguf",
    ctx_size=4096,
    cache_size=8192,     # 2x context for better caching
    gpu_layers=99
)
```

## Profiling and Monitoring

### Built-in Metrics

```python
# Get detailed metrics
metrics = engine.get_metrics()

print("Latency Stats:")
print(f"  Mean: {metrics['latency']['mean_ms']:.0f} ms")
print(f"  P50:  {metrics['latency']['p50_ms']:.0f} ms")
print(f"  P95:  {metrics['latency']['p95_ms']:.0f} ms")
print(f"  P99:  {metrics['latency']['p99_ms']:.0f} ms")

print("\nThroughput Stats:")
print(f"  Tokens/sec: {metrics['throughput']['tokens_per_sec']:.1f}")
print(f"  Requests/sec: {metrics['throughput']['requests_per_sec']:.2f}")
print(f"  Total tokens: {metrics['throughput']['total_tokens']}")
```

### GPU Monitoring

```python
import subprocess

def monitor_gpu():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"],
        capture_output=True,
        text=True
    )
    print(f"GPU: {result.stdout.strip()}")

# Monitor during inference
monitor_gpu()
result = engine.infer("Long prompt...", max_tokens=200)
monitor_gpu()
```

## Performance Checklist

Use this checklist to ensure optimal performance:

- [x] **Quantization:** Using Q4_K_M or Q5_K_M
- [x] **GPU Offload:** gpu_layers=99 (full offload)
- [x] **Context Size:** Smallest that meets requirements
- [x] **Batch Size:** 512/128 for 1B models
- [x] **FlashAttention:** Enabled (automatic on T4)
- [x] **CUDA Version:** 12.0+
- [x] **Driver:** Latest NVIDIA driver
- [x] **Model Choice:** Appropriate size for T4 (1B-3B)

## Common Performance Issues

### Issue: Slow Inference (<50 tok/s)

**Diagnosis:**
```python
metrics = engine.get_metrics()
print(f"Speed: {metrics['throughput']['tokens_per_sec']:.1f} tok/s")

# Check GPU usage
!nvidia-smi
```

**Solutions:**
1. Increase GPU layers: `gpu_layers=99`
2. Use Q4_K_M quantization
3. Reduce context size: `ctx_size=2048`
4. Check GPU utilization (should be >80%)

### Issue: High Latency (>2000ms)

**Diagnosis:**
```python
metrics = engine.get_metrics()
print(f"P95 latency: {metrics['latency']['p95_ms']:.0f} ms")
```

**Solutions:**
1. Reduce max_tokens
2. Use smaller context size
3. Check for CPU bottleneck
4. Verify T4 GPU (not CPU-only)

### Issue: Out of Memory

**Diagnosis:**
```bash
nvidia-smi  # Check memory usage
```

**Solutions:**
```python
# Reduce GPU layers
gpu_layers = 20  # Instead of 99

# Reduce context
ctx_size = 1024  # Instead of 4096

# Reduce batch size
batch_size = 256  # Instead of 512
```

## Best Configurations

### Configuration 1: Maximum Speed

```python
engine.load_model(
    "gemma-3-1b-Q4_K_M.gguf",
    gpu_layers=99,
    ctx_size=1024,
    batch_size=512,
    ubatch_size=128,
    n_parallel=1,
    auto_start=True
)

# Expected: 140+ tok/s
```

### Configuration 2: Balanced

```python
engine.load_model(
    "gemma-3-1b-Q4_K_M.gguf",
    gpu_layers=99,
    ctx_size=2048,
    batch_size=512,
    ubatch_size=128,
    n_parallel=1,
    auto_start=True
)

# Expected: 134 tok/s (default)
```

### Configuration 3: Long Context

```python
engine.load_model(
    "gemma-3-1b-Q4_K_M.gguf",
    gpu_layers=99,
    ctx_size=8192,
    batch_size=256,
    ubatch_size=64,
    n_parallel=1,
    auto_start=True
)

# Expected: 105 tok/s with FlashAttention
```

### Configuration 4: Multi-Request

```python
engine.load_model(
    "gemma-3-1b-Q4_K_M.gguf",
    gpu_layers=99,
    ctx_size=2048,
    batch_size=1024,
    ubatch_size=256,
    n_parallel=8,
    auto_start=True
)

# Expected: 400+ tok/s total throughput
```

## Next Steps

- [Benchmarks](../performance/benchmarks.md) - Compare model performance
- [T4 Results](../performance/t4-results.md) - Detailed T4 benchmarks
- [Optimization Guide](../performance/optimization.md) - Advanced tuning
- [Troubleshooting](../guides/troubleshooting.md) - Fix issues

!!! success "Performance Achieved"
    Following these optimizations, you should achieve 130+ tok/s on Gemma 3-1B with Tesla T4, matching our verified benchmarks.
