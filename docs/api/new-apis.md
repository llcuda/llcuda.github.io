# New APIs (v2.1+)

llcuda v2.1+ introduces four comprehensive API modules for advanced LLM inference optimization.

## Overview

The new APIs provide:

1. **Quantization** - NF4, GGUF conversion, dynamic quantization
2. **Unsloth Integration** - Seamless fine-tuning to deployment
3. **CUDA Optimization** - Tensor Cores, CUDA Graphs, Triton kernels
4. **Advanced Inference** - FlashAttention, KV-cache, batch optimization

## Quick Examples

### Quantization

```python
from llcuda.quantization import DynamicQuantizer

# Auto-select optimal quantization
quantizer = DynamicQuantizer(target_vram_gb=12.0)
config = quantizer.recommend_config(model_size_gb=3.0)

print(f"Use: {config['quant_type']}")  # Q4_K_M
```

### Unsloth Integration

```python
from llcuda.unsloth import export_to_llcuda

# Export fine-tuned model
export_to_llcuda(
    model=model,
    tokenizer=tokenizer,
    output_path="model.gguf",
    quant_type="Q4_K_M"
)
```

### CUDA Optimization

```python
from llcuda.cuda import enable_tensor_cores

# Enable Tensor Cores (2-4x speedup)
enable_tensor_cores(dtype=torch.float16)
```

### Advanced Inference

```python
from llcuda.inference import get_optimal_context_length

# Get optimal context for your VRAM
ctx_len = get_optimal_context_length(
    model_size_b=3.0,
    available_vram_gb=12.0,
    use_flash_attention=True
)
```

## Detailed Documentation

For complete API reference, see:

- [Quantization API](quantization.md)
- [Unsloth Integration](unsloth.md)
- [CUDA Optimization](cuda.md)
- [Advanced Inference](inference.md)

## Performance Impact

| Optimization | Benefit |
|--------------|---------|
| Tensor Cores | 2-4x speedup |
| CUDA Graphs | 20-40% latency ↓ |
| FlashAttention | 2-3x for long ctx |
| Q4_K_M Quant | 8.5x compression |

## Migration from v2.0

No breaking changes! All v2.0 code still works.

**Before (v2.0)**:
```python
import llcuda
engine = llcuda.InferenceEngine()
engine.load_model("model.gguf")
```

**After (v2.1+)** - Same code + optional optimizations:
```python
import llcuda
from llcuda.cuda import enable_tensor_cores

enable_tensor_cores()  # NEW: 2-4x faster!
engine = llcuda.InferenceEngine()
engine.load_model("model.gguf")
```

## Complete Workflow

```python
from unsloth import FastLanguageModel
from llcuda.unsloth import export_to_llcuda
from llcuda.cuda import enable_tensor_cores
import llcuda

# 1. Train with Unsloth
model, tokenizer = FastLanguageModel.from_pretrained("base")
# ... training ...

# 2. Export to GGUF
export_to_llcuda(model, tokenizer, "model.gguf")

# 3. Deploy with optimizations
enable_tensor_cores()
engine = llcuda.InferenceEngine()
engine.load_model("model.gguf")

# 4. Infer
result = engine.infer("Hello!")
print(f"{result.text} ({result.tokens_per_sec:.1f} tok/s)")
```

## Next Steps

- Try the [Quick Start Guide](../guides/quickstart.md)
- Read the [Complete API Reference](../../API_REFERENCE.md)
- Explore [Examples](https://github.com/waqasm86/llcuda/tree/main/examples)
