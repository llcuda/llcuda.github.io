# K-Quants Guide

Understanding K-quantization formats.

## What are K-Quants?

K-quants use "double quantization":
1. Quantize weights
2. Quantize the quantization parameters

Result: Better quality-to-size ratio

## Available K-Quants

### Q4_K_M (Recommended)
- **Bits per weight**: 4.8
- **Size**: ~40% of FP16
- **Quality**: Excellent for most uses
- **Use case**: Default choice

### Q5_K_M
- **Bits per weight**: 5.7
- **Size**: ~48% of FP16
- **Quality**: Higher than Q4_K_M
- **Use case**: Quality-sensitive

### Q6_K
- **Bits per weight**: 6.6
- **Size**: ~55% of FP16
- **Quality**: Near FP16
- **Use case**: Maximum quality

### Q8_0
- **Bits per weight**: 8.5
- **Size**: ~70% of FP16
- **Quality**: Virtually lossless
- **Use case**: Research, validation

## Size Comparison

| Model | FP16 | Q4_K_M | Q5_K_M | Q6_K | Q8_0 |
|-------|------|--------|--------|------|------|
| 1B | 2GB | 0.8GB | 1GB | 1.1GB | 1.4GB |
| 7B | 14GB | 5.6GB | 6.7GB | 7.7GB | 9.8GB |
| 13B | 26GB | 10.4GB | 12.5GB | 14.3GB | 18.2GB |
| 70B | 140GB | 56GB | 67GB | 77GB | 98GB |
