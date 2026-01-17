# GGUF Quantization

Understanding GGUF format, K-quants, I-quants, and VRAM estimation.

**Level**: Intermediate | **Time**: 20 minutes | **VRAM Required**: Varies

---

## GGUF Formats

**K-Quants** (recommended for quality):
- `Q4_K_M` - Best balance (4-bit)
- `Q5_K_M` - Higher quality (5-bit)
- `Q6_K` - Excellent quality (6-bit)
- `Q8_0` - Near-FP16 quality (8-bit)

**I-Quants** (for 70B models):
- `IQ3_XS` - 3-bit, fits 70B on 30GB
- `IQ2_XXS` - 2-bit, ultra-compressed

## VRAM Estimation

```python
from llcuda.api.gguf import estimate_vram

vram_gb = estimate_vram(
    model_size_b=7,  # 7B parameters
    quant_type="Q4_K_M"
)
print(f"Est. VRAM: {vram_gb:.1f} GB")
```

## Parse GGUF Files

```python
from llcuda.utils import GGUFParser

parser = GGUFParser(model_path="model.gguf")
print(f"Parameters: {parser.get_parameter_count() / 1e9:.1f}B")
print(f"Quantization: {parser.get_quantization()}")
print(f"Context: {parser.get_context_length()}")
```

## Open in Kaggle

[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/waqasm86/04-gguf-quantization-llcuda-v2-2-0)
