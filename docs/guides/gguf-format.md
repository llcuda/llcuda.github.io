# GGUF Format Guide

Complete guide to the GGUF (GPT-Generated Unified Format) model format used by llcuda v2.1.0.

## What is GGUF?

**GGUF** (GPT-Generated Unified Format) is a binary format for storing large language models developed by the [llama.cpp](https://github.com/ggerganov/llama.cpp) project.

### Key Features

✅ **Single-file distribution** - Everything in one portable file
✅ **Efficient storage** - Compact binary format with compression
✅ **Memory mapping** - Fast loading without full RAM allocation
✅ **Quantization support** - Multiple precision levels (INT4, INT8, FP16)
✅ **Metadata included** - Model architecture, tokenizer, and configuration
✅ **Cross-platform** - Works on Linux, macOS, Windows
✅ **GPU acceleration** - Full CUDA support for inference

### Why GGUF?

GGUF replaced the older GGML format and offers significant improvements:

| Feature | GGML (Old) | GGUF (Current) |
|---------|------------|----------------|
| Metadata | External files | Embedded |
| Versioning | Limited | Full versioning |
| Tokenizer | Separate file | Included |
| Architecture | Hard-coded | Dynamic |
| Compatibility | Breaking changes | Forward compatible |

## GGUF File Structure

A GGUF file contains:

```
GGUF File (.gguf)
├── Header (magic number, version)
├── Metadata (KV pairs)
│   ├── Architecture (llama, gemma, qwen, etc.)
│   ├── Model parameters (layers, heads, etc.)
│   ├── Tokenizer (vocabulary, special tokens)
│   ├── Quantization method
│   └── Author, license, source
├── Tensor info (names, shapes, offsets)
└── Tensor data (model weights)
```

### Example GGUF Metadata

```python
import llcuda
from llcuda.gguf_parser import GGUFReader

# Read GGUF metadata
reader = GGUFReader("gemma-3-1b-it-Q4_K_M.gguf")

print(f"Architecture: {reader.architecture}")
print(f"Parameter count: {reader.parameter_count}")
print(f"Quantization: {reader.quantization}")
print(f"Context length: {reader.context_length}")
```

## Quantization Types

GGUF supports multiple quantization methods that trade off quality for size and speed.

### Quantization Comparison

| Type | Bits | Size Multiplier | Quality | Speed | Use Case |
|------|------|-----------------|---------|-------|----------|
| **F16** | 16 | 1.0x (largest) | Best | Slowest | Reference quality |
| **Q8_0** | 8 | 0.5x | Excellent | Slow | High quality needed |
| **Q6_K** | 6 | 0.4x | Very good | Medium | Balanced |
| **Q5_K_M** | 5 | 0.35x | Good | Medium-fast | Good balance |
| **Q4_K_M** | 4 | 0.25x | Good | **Fast** | **Recommended** |
| **Q4_K_S** | 4 | 0.25x | Acceptable | Fast | Smaller variant |
| **Q3_K_M** | 3 | 0.2x | Fair | Very fast | Experimental |
| **Q2_K** | 2 | 0.15x (smallest) | Poor | Fastest | Testing only |

### Recommended: Q4_K_M

For Tesla T4 GPUs, **Q4_K_M** provides the best balance:

✅ **Good quality** - Minimal accuracy loss vs FP16
✅ **Fast inference** - 134 tok/s on Gemma 3-1B
✅ **Small size** - 4 bits per parameter
✅ **Low VRAM** - Fits larger models in 16 GB

**Example sizes for Gemma 3-1B:**
- F16: ~2.6 GB
- Q8_0: ~1.4 GB
- **Q4_K_M: ~650 MB** ← Recommended
- Q2_K: ~400 MB

## Using GGUF Models with llcuda

### Method 1: From HuggingFace (Recommended)

Load directly from Unsloth or other HuggingFace repositories:

```python
import llcuda

engine = llcuda.InferenceEngine()

# Load from Unsloth repository
engine.load_model(
    "unsloth/gemma-3-1b-it-GGUF:gemma-3-1b-it-Q4_K_M.gguf"
)

# Format: repo_id:filename
```

**Popular Unsloth GGUF models:**
- `unsloth/gemma-3-1b-it-GGUF:gemma-3-1b-it-Q4_K_M.gguf`
- `unsloth/Llama-3.2-3B-Instruct-GGUF:Llama-3.2-3B-Instruct-Q4_K_M.gguf`
- `unsloth/Qwen2.5-7B-Instruct-GGUF:Qwen2.5-7B-Instruct-Q4_K_M.gguf`

### Method 2: From Local File

Use a downloaded GGUF file:

```python
engine.load_model("/path/to/model.gguf")
```

### Method 3: From URL

Direct download from any URL:

```python
engine.load_model(
    "https://huggingface.co/user/repo/resolve/main/model.gguf"
)
```

## Converting Models to GGUF

### From PyTorch/HuggingFace

Use the `convert_hf_to_gguf.py` script from llama.cpp:

```bash
# Clone llama.cpp
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# Install dependencies
pip install -r requirements.txt

# Convert model
python convert_hf_to_gguf.py \
    /path/to/huggingface/model \
    --outfile model-f16.gguf \
    --outtype f16
```

### From Unsloth Fine-Tuned Models

Export directly from Unsloth:

```python
from unsloth import FastLanguageModel

# After fine-tuning
model.save_pretrained_gguf(
    "my_model",
    tokenizer,
    quantization_method="q4_k_m"  # Creates Q4_K_M GGUF
)

# Output: my_model/unsloth.Q4_K_M.gguf
```

**Supported quantization methods:**
- `"f16"` - Full precision
- `"q8_0"` - 8-bit quantization
- `"q6_k"` - 6-bit K-quant
- `"q5_k_m"` - 5-bit K-quant medium
- `"q4_k_m"` - 4-bit K-quant medium (recommended)
- `"q4_k_s"` - 4-bit K-quant small
- `"q3_k_m"` - 3-bit K-quant medium
- `"q2_k"` - 2-bit K-quant

## Quantizing Existing GGUF

Convert between quantization levels:

```bash
# Using llama-quantize (included with llcuda binaries)
~/.cache/llcuda/bin/llama-quantize \
    model-f16.gguf \
    model-q4_k_m.gguf \
    Q4_K_M
```

**Available quantization types:**
```
Q4_0, Q4_1, Q5_0, Q5_1, Q8_0
Q4_K_S, Q4_K_M, Q5_K_S, Q5_K_M, Q6_K
IQ1_S, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S
```

## GGUF Inspection Tools

### Using llcuda

```python
from llcuda.gguf_parser import GGUFReader

reader = GGUFReader("model.gguf")

print(f"Architecture: {reader.architecture}")
print(f"Quantization: {reader.quantization}")
print(f"Parameter count: {reader.parameter_count:,}")
print(f"Context length: {reader.context_length}")
print(f"Embedding size: {reader.embedding_size}")
print(f"Layers: {reader.num_layers}")
print(f"Heads: {reader.num_heads}")
print(f"File size: {reader.file_size / 1024**3:.2f} GB")
```

### Using llama.cpp Tools

```bash
# Check GGUF metadata
~/.cache/llcuda/bin/llama-cli \
    --model model.gguf \
    --verbose
```

## Model Compatibility

### Supported Architectures

llcuda v2.1.0 supports these model architectures via GGUF:

✅ **LLaMA** (LLaMA, LLaMA-2, LLaMA-3, LLaMA-3.1, LLaMA-3.2)
✅ **Gemma** (Gemma, Gemma-2, Gemma-3)
✅ **Qwen** (Qwen, Qwen-2, Qwen-2.5)
✅ **Mistral** (Mistral, Mistral-7B)
✅ **Mixtral** (Mixtral 8x7B, 8x22B)
✅ **Phi** (Phi-2, Phi-3)
✅ **Yi** (Yi-6B, Yi-34B)
✅ **StableLM** (StableLM-2, StableLM-3)

### Checking Compatibility

```python
import llcuda

# Check if model is compatible
compat = llcuda.check_model_compatibility("model.gguf")

print(f"Compatible: {compat['compatible']}")
print(f"Architecture: {compat['architecture']}")
print(f"Warnings: {compat.get('warnings', [])}")
```

## GGUF Best Practices

### 1. Choose Right Quantization

**For Tesla T4:**
- **Small models (1-3B):** Q4_K_M or Q5_K_M
- **Medium models (7-8B):** Q4_K_M (fits in VRAM)
- **Large models (13B+):** Q4_K_M or Q3_K_M (if needed)

### 2. Verify GGUF Integrity

```python
from llcuda.gguf_parser import GGUFReader

try:
    reader = GGUFReader("model.gguf")
    print("✅ Valid GGUF file")
except Exception as e:
    print(f"❌ Invalid GGUF: {e}")
```

### 3. Test Before Production

```python
# Quick test
engine = llcuda.InferenceEngine()
engine.load_model("model.gguf", silent=True)

result = engine.infer("Test prompt", max_tokens=20)
print(f"Output: {result.text}")
print(f"Speed: {result.tokens_per_sec:.1f} tok/s")
```

### 4. Optimize Storage

**Use Q4_K_M for distribution:**
- Smaller download size
- Faster loading
- Good quality
- Better inference speed

## GGUF vs Other Formats

| Format | Size | Speed | Compatibility | Ease of Use |
|--------|------|-------|---------------|-------------|
| **GGUF** | Small | Fast | llama.cpp | ✅ Easy |
| **SafeTensors** | Large | Medium | PyTorch | Medium |
| **PyTorch (.pt)** | Large | Medium | PyTorch only | Medium |
| **ONNX** | Large | Fast | ONNX Runtime | Complex |
| **TensorRT** | Custom | Fastest | NVIDIA only | Complex |

**Why GGUF for llcuda:**
- ✅ Smallest file size (with quantization)
- ✅ Fast inference on CPU and GPU
- ✅ Single-file distribution
- ✅ Works with llama.cpp ecosystem
- ✅ Easy to share and deploy

## Finding GGUF Models

### Unsloth HuggingFace

Most popular source for GGUF models:

[https://huggingface.co/unsloth](https://huggingface.co/unsloth)

**Example repositories:**
- `unsloth/gemma-3-1b-it-GGUF`
- `unsloth/Llama-3.2-3B-Instruct-GGUF`
- `unsloth/Qwen2.5-7B-Instruct-GGUF`
- `unsloth/Meta-Llama-3.1-8B-Instruct-GGUF`

### TheBloke (Legacy)

Older GGUF models (pre-Unsloth era):

[https://huggingface.co/TheBloke](https://huggingface.co/TheBloke)

### Bartowski

Recent high-quality quantizations:

[https://huggingface.co/bartowski](https://huggingface.co/bartowski)

## Troubleshooting GGUF Issues

### Issue: Invalid GGUF Magic Number

**Error:** `Invalid GGUF file: wrong magic number`

**Solution:**
- File is corrupted or incomplete
- Re-download the GGUF file
- Verify SHA256 checksum

### Issue: Unsupported Quantization

**Error:** `Quantization type not supported`

**Solution:**
- Use Q4_K_M, Q5_K_M, or Q8_0
- Avoid experimental quantizations (IQ types)
- Re-quantize with llama-quantize

### Issue: Model Too Large

**Error:** `CUDA out of memory`

**Solution:**
- Use lower quantization (Q4_K_M instead of Q8_0)
- Use smaller model variant
- Clear GPU cache before loading

## Advanced GGUF Topics

### Custom Metadata

Add custom metadata to GGUF:

```python
from llcuda.gguf_parser import GGUFWriter

writer = GGUFWriter("output.gguf")
writer.add_metadata("author", "Your Name")
writer.add_metadata("description", "Fine-tuned for specific task")
writer.add_metadata("license", "MIT")
writer.finalize()
```

### Merging GGUF Models

Combine multiple LoRA adapters (experimental):

```bash
# Using llama.cpp tools
llama-export-lora \
    base-model.gguf \
    lora-adapter.gguf \
    merged-model.gguf
```

## References

- **GGUF Specification:** [github.com/ggerganov/ggml/blob/master/docs/gguf.md](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
- **llama.cpp:** [github.com/ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp)
- **Unsloth GGUF Export:** [docs.unsloth.ai/basics/saving-to-gguf](https://docs.unsloth.ai/basics/running-and-saving-models/saving-to-gguf)

## Next Steps

- [:material-select: Model Selection Guide](model-selection.md) - Choose the right model
- [:material-rocket: Quick Start](quickstart.md) - Start using GGUF models
- [:material-speedometer: Performance](../performance/benchmarks.md) - Benchmark GGUF models
- [:material-book-open: Unsloth Integration](../tutorials/unsloth-integration.md) - Create GGUF from fine-tuned models

---

**GGUF makes LLM deployment simple and efficient!** 🚀
