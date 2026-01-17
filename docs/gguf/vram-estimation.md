# VRAM Estimation

Estimate VRAM requirements for GGUF models.

## Formula

```
VRAM (GB) ≈ Parameters (B) × Bits / 8
```

## Common Models

| Model | Quant | VRAM | Fits Single T4? | Fits Dual T4? |
|-------|-------|------|-----------------|---------------|
| Gemma 2-2B | Q4_K_M | ~3 GB | ✅ Yes | ✅ Yes |
| Llama-3.2-3B | Q4_K_M | ~4 GB | ✅ Yes | ✅ Yes |
| Qwen-2.5-7B | Q4_K_M | ~7 GB | ✅ Yes | ✅ Yes |
| Llama-70B | Q4_K_M | ~40 GB | ❌ No | ❌ No |
| Llama-70B | IQ3_XS | ~28 GB | ❌ No | ✅ Yes |

## Python Estimation

```python
from llcuda.api.multigpu import estimate_model_vram

vram = estimate_model_vram(model_size_b=7, quant_type="Q4_K_M")
print(f"Est. VRAM: {vram:.1f} GB")
```

## See Also

- [GGUF Tutorial](../tutorials/04-gguf-quantization.md)
- [Multi-GPU](../tutorials/03-multi-gpu.md)
