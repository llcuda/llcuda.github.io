# GGUF Format Selection Guide

Choose the right quantization for your use case.

## Quick Reference

| Model Size | VRAM | Recommended Quant | Speed |
|------------|------|-------------------|-------|
| 1-3B | 3-5 GB | Q4_K_M | ~60 tok/s |
| 7-8B | 6-8 GB | Q4_K_M | ~25 tok/s |
| 70B (dual T4) | 28 GB | IQ3_XS | ~12 tok/s |

## K-Quants (Quality Priority)

- **Q4_K_M**: Best balance (recommended)
- **Q5_K_M**: Higher quality
- **Q6_K**: Excellent quality
- **Q8_0**: Near-FP16

## I-Quants (Size Priority)

- **IQ3_XS**: For 70B on dual T4
- **IQ2_XXS**: Ultra-compressed

## Decision Tree

```
Model < 8B?
├─ Yes → Q4_K_M (single T4)
└─ No → Is it 70B?
    ├─ Yes → IQ3_XS (dual T4)
    └─ No → Q4_K_M (dual T4)
```

## See Also

- [GGUF Tutorial](../tutorials/04-gguf-quantization.md)
- [VRAM Estimation](vram-estimation.md)
