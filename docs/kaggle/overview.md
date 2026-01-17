# Kaggle Dual T4 Overview

llcuda v2.2.0 is optimized for Kaggle's dual Tesla T4 GPU environment.

## Hardware Specs

- **2× NVIDIA Tesla T4**
- **30GB total VRAM** (15GB each)
- **SM 7.5** (Turing architecture)
- **FlashAttention support**

## What You Can Run

| Model Size | Quantization | Strategy |
|------------|--------------|----------|
| 1-13B | Q4_K_M | Single T4 |
| 32-34B | Q4_K_M | Dual T4 tensor-split |
| 70B | IQ3_XS | Dual T4 tensor-split |

See: [Multi-GPU Inference](multi-gpu-inference.md)
