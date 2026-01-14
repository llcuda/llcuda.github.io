# Gemma 3-1B Tutorial - Google Colab

Complete tutorial for running Gemma 3-1B with llcuda v2.1.0 on Tesla T4 GPU.

## :rocket: Open in Google Colab

<div style="text-align: center; margin: 2em 0;">
  <a href="https://colab.research.google.com/github/llcuda/llcuda/blob/main/notebooks/llcuda_v2_1_0_gemma3_1b_unsloth_colab.ipynb" target="_blank">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" style="width: 250px;">
  </a>
</div>

## :books: What This Tutorial Covers

This comprehensive 14-step tutorial demonstrates:

1. **GPU Verification** - Detect Tesla T4 and check compatibility
2. **Installation** - Install llcuda v2.1.0 from GitHub
3. **Binary Download** - Auto-download CUDA binaries (~266 MB)
4. **GPU Compatibility** - Verify llcuda can use the GPU
5. **Model Loading** - Load Gemma 3-1B-IT from Unsloth HuggingFace
6. **First Inference** - Run general knowledge queries
7. **Code Generation** - Test Python code generation
8. **Batch Inference** - Process multiple prompts efficiently
9. **Performance Metrics** - Analyze throughput and latency
10. **Advanced Parameters** - Explore generation strategies
11. **Model Loading Methods** - HuggingFace, Registry, Local paths
12. **Unsloth Workflow** - Fine-tuning to deployment pipeline
13. **Context Manager** - Auto-cleanup resources
14. **Available Models** - Browse Unsloth GGUF models

## :chart_with_upwards_trend: Verified Performance

Real execution results from Google Colab Tesla T4:

- **Speed**: **134 tokens/sec** average (range: 116-142 tok/s)
- **Latency**: 690ms median
- **Consistency**: Stable performance across all tests
- **GPU Offload**: 99 layers fully on GPU

!!! success "3x Faster Than Expected!"
    Initial estimates: ~45 tok/s
    **Actual performance: 134 tok/s**
    FlashAttention + Tensor Cores delivering exceptional results!

## :memo: Tutorial Steps

### Step 1: Verify Tesla T4 GPU

```python
!nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv

# Expected output:
# Tesla T4, 7.5, 15360 MiB
```

### Step 2: Install llcuda v2.1.0

```python
!pip install -q git+https://github.com/llcuda/llcuda.git

# ✅ llcuda v2.1.0 installed successfully!
```

### Step 3: Import and Download Binaries

```python
import llcuda

# First import triggers binary download:
# - Source: GitHub Releases v2.0.6
# - Size: 266 MB
# - Duration: ~1-2 minutes
# - Cached for future use
```

**Download Output:**
```
📥 Downloading from GitHub releases...
URL: https://github.com/llcuda/llcuda/releases/download/v2.0.6/...
Downloading T4 binaries: 100% (266.0/266.0 MB)
✅ Extraction complete!
Copied 5 binaries to .../llcuda/binaries/cuda12
Copied 18 libraries to .../llcuda/lib
```

### Step 4: Load Gemma 3-1B-IT

```python
engine = llcuda.InferenceEngine()

engine.load_model(
    "unsloth/gemma-3-1b-it-GGUF:gemma-3-1b-it-Q4_K_M.gguf",
    silent=True
)

# Auto-configured for Tesla T4:
# - GPU Layers: 99 (full offload)
# - Context Size: 4096 tokens
# - Batch Size: 2048
```

### Step 5: Run Inference

```python
result = engine.infer(
    "Explain quantum computing in simple terms",
    max_tokens=200,
    temperature=0.7
)

print(result.text)
print(f"Speed: {result.tokens_per_sec:.1f} tok/s")

# Actual output: 131.4 tokens/sec ✅
```

### Step 6: Batch Processing

```python
prompts = [
    "What is machine learning?",
    "Explain neural networks briefly.",
    "What is the difference between AI and ML?",
    "Define deep learning concisely."
]

results = engine.batch_infer(prompts, max_tokens=80)

for prompt, result in zip(prompts, results):
    print(f"Q: {prompt}")
    print(f"A: {result.text}")
    print(f"Speed: {result.tokens_per_sec:.1f} tok/s\n")

# Results:
# Query 1: 116.0 tok/s
# Query 2: 142.3 tok/s
# Query 3: 141.6 tok/s
# Query 4: 141.7 tok/s
```

## :bar_chart: Performance Results

From the executed notebook:

| Test | Tokens | Speed | Latency |
|------|--------|-------|---------|
| General Knowledge | 200 | 131.4 tok/s | 1522ms |
| Code Generation | 300 | 136.1 tok/s | - |
| Batch Query 1 | 80 | 116.0 tok/s | 690ms |
| Batch Query 2 | 80 | 142.3 tok/s | 562ms |
| Batch Query 3 | 80 | 141.6 tok/s | 565ms |
| Batch Query 4 | 80 | 141.7 tok/s | 565ms |
| **Average** | - | **134.2 tok/s** | **690ms median** |

!!! tip "Why So Fast?"
    1. **FlashAttention** - 2-3x speedup for attention operations
    2. **Tensor Cores** - SM 7.5 fully utilized
    3. **CUDA Graphs** - Reduced kernel launch overhead
    4. **Full GPU Offload** - All 99 layers on GPU
    5. **Q4_K_M Quantization** - Optimal speed/quality balance

## :package: Model Information

**Gemma 3-1B-IT Q4_K_M:**

- **Size**: ~806 MB (download)
- **Parameters**: 1 billion
- **Quantization**: Q4_K_M (4-bit)
- **Context**: 2048 tokens (expandable to 4096)
- **VRAM**: ~1.2 GB
- **Source**: [unsloth/gemma-3-1b-it-GGUF](https://huggingface.co/unsloth/gemma-3-1b-it-GGUF)

## :notebook: Jupyter Notebook Features

The notebook includes:

✅ **Complete Setup Guide** - Step-by-step installation
✅ **GPU Verification** - Ensure you have Tesla T4
✅ **Error Handling** - Helpful troubleshooting tips
✅ **Multiple Examples** - Chat, batch, creative generation
✅ **Performance Metrics** - Detailed throughput & latency
✅ **Unsloth Workflow** - Fine-tuning to deployment
✅ **Model Catalog** - List of available Unsloth models

## :link: Related Resources

- [:material-file-document: Executed Notebook](gemma-3-1b-executed.md) - See live output with all results
- [:material-chart-line: Performance Benchmarks](../performance/t4-results.md) - Detailed T4 analysis
- [:material-api: API Reference](../api/inference-engine.md) - InferenceEngine documentation
- [:material-book-open: Unsloth Integration](unsloth-integration.md) - Complete workflow guide

## :question: Common Questions

### How long does the first run take?

- **Binary download**: 1-2 minutes (266 MB)
- **Model download**: 2-3 minutes (~800 MB)
- **Model loading**: 10-20 seconds
- **First inference**: Same speed as subsequent runs

**Total first-time setup**: ~5 minutes
**Subsequent sessions**: Instant (cached binaries and models)

### Can I use different models?

Yes! The notebook works with any GGUF model from HuggingFace:

```python
# Llama 3.2-3B
engine.load_model(
    "unsloth/Llama-3.2-3B-Instruct-GGUF:Llama-3.2-3B-Instruct-Q4_K_M.gguf"
)

# Qwen 2.5-7B
engine.load_model(
    "unsloth/Qwen2.5-7B-Instruct-GGUF:Qwen2.5-7B-Instruct-Q4_K_M.gguf"
)
```

### What if I don't have T4?

llcuda v2.1.0 is optimized for Tesla T4. Other GPUs may work but performance will vary. The binaries are compiled for SM 7.5 (T4's compute capability).

---

## :rocket: Get Started Now!

<div style="text-align: center; margin: 2em 0;">
  <a href="https://colab.research.google.com/github/llcuda/llcuda/blob/main/notebooks/llcuda_v2_1_0_gemma3_1b_unsloth_colab.ipynb" target="_blank" class="md-button md-button--primary">
    Open Tutorial in Colab
  </a>
</div>

**No GPU? No problem!** Google Colab provides free Tesla T4 access.

---

**Questions?** [Open an issue on GitHub](https://github.com/llcuda/llcuda/issues)
