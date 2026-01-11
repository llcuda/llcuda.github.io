# Frequently Asked Questions

Common questions and answers about llcuda v2.0.6.

## General Questions

### What is llcuda?

llcuda is a Python library for fast LLM inference on NVIDIA GPUs, specifically optimized for Tesla T4. It provides:

- Pre-built CUDA binaries with FlashAttention
- One-step installation from GitHub
- 134 tokens/sec on Gemma 3-1B (verified)
- Simple Python API for inference
- Auto-downloading of models and binaries

### Why Tesla T4 only?

llcuda v2.0.6 is optimized exclusively for Tesla T4 (compute capability 7.5) to maximize performance:

- Tensor Core optimizations for SM 7.5
- FlashAttention tuned for Turing architecture
- Binary size reduction (266 MB vs 500+ MB for multi-GPU)
- Guaranteed compatibility

For other GPUs, use llcuda v1.2.2 which supports SM 5.0-8.9.

### How does llcuda compare to other solutions?

| Solution | Speed (Gemma 3-1B) | Setup | Ease of Use |
|----------|---------|-------|-------------|
| **llcuda v2.0.6** | **134 tok/s** | 1 min | Excellent |
| transformers | 45 tok/s | 5 min | Good |
| vLLM | 85 tok/s | 10 min | Moderate |
| llama.cpp CLI | 128 tok/s | 15 min | Moderate |

llcuda is 3x faster than PyTorch and easiest to set up.

## Installation

### How do I install llcuda?

```bash
pip install git+https://github.com/waqasm86/llcuda.git
```

Binaries auto-download on first import (~266 MB).

### Do I need to install CUDA Toolkit?

No! llcuda includes all necessary CUDA binaries. You only need:

- NVIDIA driver (pre-installed in Google Colab)
- CUDA runtime (pre-installed in Colab)
- Python 3.11+

### Can I install from PyPI?

llcuda v2.0.6 is GitHub-only for now. Use:
```bash
pip install git+https://github.com/waqasm86/llcuda.git
```

### Why do binaries download on first import?

To keep the pip package small (~62 KB), CUDA binaries (266 MB) download automatically on first import from GitHub Releases. This is a one-time download, then cached locally.

## Compatibility

### Which GPUs are supported?

llcuda v2.0.6: **Tesla T4 only** (SM 7.5)

llcuda v1.2.2: All GPUs with SM 5.0+ (Maxwell through Ada Lovelace)

### Can I use llcuda on CPU?

Yes, but not recommended. Set `gpu_layers=0` for CPU mode. Performance drops from 134 tok/s to ~8 tok/s.

### Does llcuda work on Windows?

llcuda v2.0.6 is Linux-only (Google Colab, Ubuntu). For Windows, compile from source or use WSL2.

### What Python versions are supported?

Python 3.11+ is required. Tested on Python 3.10, 3.11, and 3.12.

### What CUDA versions are supported?

CUDA 12.0+ required. Tested with CUDA 12.2, 12.4.

## Models

### Which models can I use?

Any GGUF model compatible with llama.cpp:

- Gemma (1B, 2B, 3B, 7B)
- Llama (3.1, 3.2, 3.3)
- Qwen (1.5B, 7B, 14B)
- Mistral (7B, 8x7B)
- Phi (2, 3)

### What quantization should I use?

**Q4_K_M** for best performance/quality balance on T4:

- Speed: 134 tok/s
- VRAM: 1.2 GB (Gemma 3-1B)
- Quality: < 1% degradation

Other options:
- **Q5_K_M:** Better quality, 18% slower
- **Q8_0:** Best quality, 44% slower

### How do I load a model from HuggingFace?

```python
engine.load_model(
    "unsloth/gemma-3-1b-it-GGUF:gemma-3-1b-it-Q4_K_M.gguf"
)
```

### Can I use my fine-tuned models?

Yes! Export to GGUF using Unsloth:

```python
# After fine-tuning with Unsloth
model.save_pretrained_gguf(
    "my-model",
    tokenizer,
    quantization_method="q4_k_m"
)

# Load with llcuda
engine.load_model("my-model-Q4_K_M.gguf")
```

See [Unsloth Integration](../tutorials/unsloth-integration.md) for details.

## Performance

### What performance can I expect?

On Tesla T4 with Q4_K_M quantization:

- **Gemma 3-1B:** 134 tok/s (verified)
- **Llama 3.2-3B:** ~48 tok/s (estimated)
- **Qwen 2.5-7B:** ~21 tok/s (estimated)
- **Llama 3.1-8B:** ~19 tok/s (estimated)

### Why is my inference slow?

Common causes:

1. **Not using T4:** Other GPUs need v1.2.2
2. **Low GPU offload:** Set `gpu_layers=99`
3. **Wrong quantization:** Use Q4_K_M
4. **Large context:** Reduce `ctx_size` to 2048
5. **CPU mode:** Check `nvidia-smi` shows GPU usage

See [Troubleshooting](troubleshooting.md) for solutions.

### How can I optimize performance?

```python
# Optimal configuration for T4
engine.load_model(
    "gemma-3-1b-Q4_K_M",
    gpu_layers=99,        # Full GPU offload
    ctx_size=2048,        # Balanced context
    batch_size=512,       # Optimal batch
    ubatch_size=128,
    auto_configure=True   # Let llcuda optimize
)
```

See [Performance Tutorial](../tutorials/performance.md) for details.

### Does llcuda support batching?

Yes:
```python
prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
results = engine.batch_infer(prompts, max_tokens=100)
```

For concurrent requests, use `n_parallel`:
```python
engine.load_model("model.gguf", n_parallel=4)
```

## Memory

### How much VRAM do I need?

Depends on model size and quantization:

| Model | Q4_K_M | Q5_K_M | Q8_0 |
|-------|--------|--------|------|
| 1B | 1.2 GB | 1.5 GB | 2.5 GB |
| 3B | 2.0 GB | 2.4 GB | 4.2 GB |
| 7B | 5.0 GB | 6.0 GB | 10 GB |
| 8B | 5.5 GB | 6.5 GB | 11 GB |

Tesla T4 has 15 GB, sufficient for models up to 7-8B.

### Can I run multiple models simultaneously?

Yes, on different ports:

```python
# Model 1
engine1 = llcuda.InferenceEngine(server_url="http://127.0.0.1:8090")
engine1.load_model("gemma-3-1b-Q4_K_M")

# Model 2
engine2 = llcuda.InferenceEngine(server_url="http://127.0.0.1:8091")
engine2.load_model("llama-3.2-3b-Q4_K_M")
```

Watch total VRAM usage with `nvidia-smi`.

### What if I run out of VRAM?

1. Use smaller model (1B instead of 3B)
2. Use Q4_K_M instead of Q8_0
3. Reduce `gpu_layers` (e.g., 20 instead of 99)
4. Reduce `ctx_size` (e.g., 1024 instead of 4096)
5. Close other GPU applications

## Usage

### How do I run inference?

```python
import llcuda

engine = llcuda.InferenceEngine()
engine.load_model("gemma-3-1b-Q4_K_M", auto_start=True)

result = engine.infer("What is AI?", max_tokens=100)
print(result.text)
print(f"Speed: {result.tokens_per_sec:.1f} tok/s")
```

### Can I stream outputs?

Yes:
```python
def print_chunk(text):
    print(text, end='', flush=True)

result = engine.infer_stream(
    "Write a story:",
    callback=print_chunk,
    max_tokens=200
)
```

### How do I stop generation early?

Use `stop_sequences`:
```python
result = engine.infer(
    "List items:",
    max_tokens=200,
    stop_sequences=["\n\n", "###"]
)
```

### Can I control randomness?

Yes, with `temperature` and `seed`:
```python
# Deterministic
result = engine.infer(
    "Prompt",
    temperature=0.1,
    seed=42
)

# Creative
result = engine.infer(
    "Prompt",
    temperature=1.0,
    top_k=100
)
```

## Google Colab

### Does llcuda work in Google Colab?

Yes! llcuda is optimized for Colab T4:

```python
# In Colab
!pip install git+https://github.com/waqasm86/llcuda.git

import llcuda
engine = llcuda.InferenceEngine()
engine.load_model("gemma-3-1b-Q4_K_M", auto_start=True)
```

### How do I get T4 in Colab?

Runtime > Change runtime type > Hardware accelerator > GPU > GPU type > T4

### Do I need Colab Pro?

No, but Colab Pro provides:

- Guaranteed T4 access
- Longer runtime (24h vs 12h)
- More RAM
- Priority execution

Free tier works but T4 availability varies.

### Can I save models between sessions?

Models cache to `~/.cache/llcuda/`. In Colab, this resets. Use:

```python
# Save to Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Copy model
!cp ~/.cache/llcuda/models/gemma-3-1b*.gguf /content/drive/MyDrive/

# Next session: load from Drive
engine.load_model("/content/drive/MyDrive/gemma-3-1b-Q4_K_M.gguf")
```

## Troubleshooting

### Import fails with "No module named llcuda"

```bash
# Reinstall
pip uninstall llcuda -y
pip install git+https://github.com/waqasm86/llcuda.git
```

### Binary download fails

See [Troubleshooting Guide](troubleshooting.md#binary-download-fails)

### Server won't start

Check port 8090 availability or use different port:
```python
engine = llcuda.InferenceEngine(server_url="http://127.0.0.1:8091")
```

### Performance is slow

See [Performance Troubleshooting](troubleshooting.md#performance-issues)

## Contributing

### Can I contribute to llcuda?

Yes! Contributions welcome:

- Bug reports: [GitHub Issues](https://github.com/waqasm86/llcuda/issues)
- Feature requests: Open an issue
- Code: Fork and submit PR
- Documentation: Help improve docs

### How do I build binaries?

See [Build Binaries Tutorial](../tutorials/build-binaries.md)

### How do I report bugs?

Open a [GitHub Issue](https://github.com/waqasm86/llcuda/issues/new) with:

- llcuda version
- GPU model
- CUDA version
- Python version
- Error message
- Minimal reproducible code

## Next Steps

- [Quick Start](quickstart.md)
- [First Steps](first-steps.md)
- [Troubleshooting](troubleshooting.md)
- [Performance Optimization](../tutorials/performance.md)
- [API Reference](../api/inference-engine.md)

## Still have questions?

Ask on [GitHub Discussions](https://github.com/waqasm86/llcuda/discussions) or open an [issue](https://github.com/waqasm86/llcuda/issues).
