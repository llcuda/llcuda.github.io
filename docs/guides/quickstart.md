# Quick Start

Get started with llcuda v2.1.0 in 5 minutes!

## :rocket: 5-Minute Quickstart

### Step 1: Install llcuda

```bash
pip install git+https://github.com/waqasm86/llcuda.git
```

!!! tip "Google Colab Users"
    Add `!` before the command: `!pip install -q git+https://github.com/waqasm86/llcuda.git`

### Step 2: Import and Verify

```python
import llcuda

# Check version
print(f"llcuda version: {llcuda.__version__}")
# Output: 2.1.0

# Verify GPU
compat = llcuda.check_gpu_compatibility()
print(f"GPU: {compat['gpu_name']}")
print(f"Compatible: {compat['compatible']}")
```

!!! info "First Import"
    First `import llcuda` downloads CUDA binaries (266 MB) from GitHub Releases.
    This takes 1-2 minutes. Subsequent imports are instant!

### Step 3: Load a Model

```python
# Create inference engine
engine = llcuda.InferenceEngine()

# Load Gemma 3-1B from Unsloth
engine.load_model(
    "unsloth/gemma-3-1b-it-GGUF:gemma-3-1b-it-Q4_K_M.gguf",
    silent=True  # Suppress llama-server output
)

print("✅ Model loaded!")
```

### Step 4: Run Inference

```python
# Ask a question
result = engine.infer(
    "Explain quantum computing in simple terms",
    max_tokens=200,
    temperature=0.7
)

# Print results
print(f"Response: {result.text}")
print(f"Speed: {result.tokens_per_sec:.1f} tokens/sec")
print(f"Latency: {result.latency_ms:.0f}ms")
```

**Expected output on Tesla T4:**
```
Speed: 134.2 tokens/sec
Latency: 690ms
```

---

## :zap: Complete Example

Here's a complete, copy-paste ready example:

```python
import llcuda

# Initialize engine
engine = llcuda.InferenceEngine()

# Load model (downloads ~800 MB on first run)
print("Loading model...")
engine.load_model(
    "unsloth/gemma-3-1b-it-GGUF:gemma-3-1b-it-Q4_K_M.gguf",
    silent=True
)

# Single inference
result = engine.infer(
    "What is machine learning?",
    max_tokens=150
)

print(f"Response: {result.text}")
print(f"Performance: {result.tokens_per_sec:.1f} tok/s")
```

---

## :books: Common Use Cases

### Interactive Chat

```python
import llcuda

engine = llcuda.InferenceEngine()
engine.load_model(
    "unsloth/gemma-3-1b-it-GGUF:gemma-3-1b-it-Q4_K_M.gguf",
    silent=True
)

print("Chat with Gemma 3-1B (type 'exit' to quit)")

while True:
    user_input = input("\nYou: ")
    if user_input.lower() == "exit":
        break

    result = engine.infer(user_input, max_tokens=300)
    print(f"AI: {result.text}")
    print(f"({result.tokens_per_sec:.1f} tok/s)")
```

### Batch Processing

```python
import llcuda

engine = llcuda.InferenceEngine()
engine.load_model(
    "unsloth/gemma-3-1b-it-GGUF:gemma-3-1b-it-Q4_K_M.gguf",
    silent=True
)

# Multiple prompts
prompts = [
    "What is AI?",
    "Explain neural networks.",
    "Define deep learning."
]

# Process in batch
results = engine.batch_infer(prompts, max_tokens=80)

for prompt, result in zip(prompts, results):
    print(f"\nQ: {prompt}")
    print(f"A: {result.text}")
    print(f"Speed: {result.tokens_per_sec:.1f} tok/s")
```

---

## :tada: Try on Google Colab

Click the button below to try llcuda in your browser:

<div style="text-align: center; margin: 2em 0;">
  <a href="https://colab.research.google.com/github/waqasm86/llcuda/blob/main/notebooks/llcuda_v2_1_0_gemma3_1b_unsloth_colab.ipynb" target="_blank">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" style="width: 200px;">
  </a>
</div>

**What this notebook includes:**

- ✅ Complete Tesla T4 setup guide
- ✅ GPU verification steps
- ✅ Binary download walkthrough
- ✅ Multiple inference examples
- ✅ Performance benchmarking
- ✅ Batch processing demo

---

## :chart_with_upwards_trend: Expected Performance

On Google Colab Tesla T4:

| Task | Speed | Latency |
|------|-------|---------|
| Simple query | 134 tok/s | ~690ms |
| Code generation | 136 tok/s | ~1.5s |
| Batch (4 prompts) | 135 tok/s avg | ~2.4s total |

**These are verified real-world results!** See the [executed notebook](../tutorials/gemma-3-1b-executed.md) for proof.

---

## :bulb: Pro Tips

### Silent Mode

Suppress llama-server output for cleaner logs:

```python
engine.load_model(
    "unsloth/gemma-3-1b-it-GGUF:gemma-3-1b-it-Q4_K_M.gguf",
    silent=True  # ← Add this!
)
```

### Context Manager

Auto-cleanup resources:

```python
with llcuda.InferenceEngine() as engine:
    engine.load_model(
        "unsloth/gemma-3-1b-it-GGUF:gemma-3-1b-it-Q4_K_M.gguf",
        silent=True
    )
    result = engine.infer("Test prompt", max_tokens=50)
    print(result.text)
# Server automatically stopped here
```

### Check GPU Before Loading

```python
# Verify GPU compatibility first
compat = llcuda.check_gpu_compatibility()

if not compat['compatible']:
    print(f"⚠️ GPU {compat['gpu_name']} may not be compatible")
    print(f"   llcuda is optimized for Tesla T4")
else:
    print(f"✅ {compat['gpu_name']} is compatible!")
    # Proceed with loading model...
```

---

## :question: Troubleshooting

### Model Download Slow?

HuggingFace downloads can be slow. First download is cached:

```python
# First run: Downloads ~800 MB (2-3 minutes)
engine.load_model("unsloth/gemma-3-1b-it-GGUF:gemma-3-1b-it-Q4_K_M.gguf")

# Subsequent runs: Uses cached model (instant)
engine.load_model("unsloth/gemma-3-1b-it-GGUF:gemma-3-1b-it-Q4_K_M.gguf")
```

### Out of Memory?

Try a smaller model or reduce context:

```python
# Smaller model
engine.load_model(
    "unsloth/gemma-3-1b-it-GGUF:gemma-3-1b-it-Q2_K.gguf",  # Q2_K instead of Q4_K_M
    silent=True
)

# Or reduce context size
engine.load_model(
    "unsloth/gemma-3-1b-it-GGUF:gemma-3-1b-it-Q4_K_M.gguf",
    context_size=2048,  # Default is 4096
    silent=True
)
```

### Binary Download Failed?

Manual installation:

```bash
wget https://github.com/waqasm86/llcuda/releases/download/v2.0.6/llcuda-binaries-cuda12-t4-v2.0.6.tar.gz
mkdir -p ~/.cache/llcuda
tar -xzf llcuda-binaries-cuda12-t4-v2.0.6.tar.gz -C ~/.cache/llcuda/
```

---

## :link: Next Steps

<div class="grid cards" markdown>

- [:material-google: **Google Colab Tutorial**](../tutorials/gemma-3-1b-colab.md)

    Complete walkthrough with Tesla T4 examples

- [:material-code-braces: **API Reference**](../api/overview.md)

    Full API documentation and advanced features

- [:material-chart-line: **Performance Guide**](../performance/benchmarks.md)

    Benchmarks and optimization tips

- [:material-book-open: **More Examples**](../examples/chat.md)

    Additional use cases and code samples

</div>

---

**Questions?** Check the [FAQ](faq.md) or [open an issue](https://github.com/waqasm86/llcuda/issues)!
