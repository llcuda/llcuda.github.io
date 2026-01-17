# GGUF Format Overview

Understanding GGUF quantization in llcuda.

## What is GGUF?

GGUF (GPT-Generated Unified Format):
- Binary model format
- Efficient quantization
- Fast loading
- llama.cpp native format

## Quantization Types

### K-Quants (Recommended)
- **Q4_K_M**: 4.8 bpw, best for most models
- **Q5_K_M**: 5.7 bpw, higher quality
- **Q6_K**: 6.6 bpw, near FP16
- **Q8_0**: 8.5 bpw, very high quality

### I-Quants (Compression)
- **IQ3_XS**: 3.3 bpw, for 70B models
- **IQ4_XS**: 4.3 bpw, better quality
- **IQ2_XS**: 2.3 bpw, extreme compression

### Legacy
- **Q4_0**: 4.5 bpw
- **Q5_0**: 5.5 bpw

## Selection Guide

| VRAM | Model Size | Recommended Quant |
|------|------------|-------------------|
| 5GB | 1-3B | Q4_K_M |
| 10GB | 7-8B | Q4_K_M |
| 15GB | 13B | Q4_K_M |
| 30GB | 70B | IQ3_XS |

See:
- [K-Quants Guide](k-quants.md)
- [I-Quants Guide](i-quants.md)
- [Selection Guide](selection.md)
