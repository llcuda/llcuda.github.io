# I-Quants (Importance Quantization)

Ultra-compressed quantization for 70B models on 30GB VRAM.

## Overview

I-quants use importance-based quantization to achieve extreme compression while maintaining quality.

## I-Quant Types

| Type | Bits | 70B VRAM | Quality | Use Case |
|------|------|----------|---------|----------|
| IQ3_XS | ~3-bit | ~28 GB | Good | 70B on dual T4 |
| IQ2_XXS | ~2-bit | ~21 GB | Fair | Ultra-compressed |

## When to Use I-Quants

- Running 70B models on 30GB VRAM
- Dual T4 Kaggle setup
- Prioritize model size over quality

## Example

```python
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="unsloth/Llama-3.1-70B-Instruct-GGUF",
    filename="Llama-3.1-70B-Instruct-IQ3_XS.gguf"
)
```

## Performance

- **Llama-70B IQ3_XS**: ~12 tokens/sec on dual T4
- **VRAM**: ~28-29 GB total

## See Also

- [Large Models Tutorial](../tutorials/09-large-models.md)
- [K-Quants](../gguf/k-quants.md)
