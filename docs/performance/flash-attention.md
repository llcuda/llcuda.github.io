# FlashAttention

FlashAttention v2 optimization in llcuda.

## What is FlashAttention?

Memory-efficient attention algorithm:
- 2-3x faster than standard attention
- Lower memory usage
- Exact (not approximate)

## Enable in llcuda

```python
config = ServerConfig(
    flash_attn=True,  # Enable FlashAttention
)
```

## Supported

- ✅ All quantization types
- ✅ All context sizes
- ✅ Both GPUs (tensor-split)

## Performance Impact

| Model | Without FA | With FA | Speedup |
|-------|------------|---------|---------|
| 7B | ~15 tok/s | ~35 tok/s | 2.3x |
| 13B | ~8 tok/s | ~18 tok/s | 2.3x |
| 70B | ~5 tok/s | ~12 tok/s | 2.4x |

## Requirements

- SM 7.5+ (Tesla T4 ✅)
- CUDA 12.x
- Built with `-DGGML_CUDA_FA_ALL_QUANTS=ON`
