# Performance Benchmarks

Real-world performance on Kaggle dual T4.

## Single GPU Results

| Model | Quant | GPU | Tokens/sec | VRAM |
|-------|-------|-----|------------|------|
| Gemma 3-1B | Q4_K_M | 1× T4 | ~45 tok/s | 3 GB |
| Qwen2.5-1.5B | Q4_K_M | 1× T4 | ~50 tok/s | 2.5 GB |
| Llama-3.2-3B | Q4_K_M | 1× T4 | ~30 tok/s | 4 GB |

## Dual GPU Results

| Model | Quant | GPUs | Tokens/sec | VRAM |
|-------|-------|------|------------|------|
| Gemma 2-2B | Q4_K_M | 2× T4 | ~60 tok/s | 4 GB |
| Qwen2.5-7B | Q4_K_M | 2× T4 | ~35 tok/s | 10 GB |
| Llama-70B | IQ3_XS | 2× T4 | ~12 tok/s | 27 GB |

## Optimization Impact

| Optimization | Speedup |
|--------------|---------|
| FlashAttention | 2-3x |
| Tensor Cores | 1.5x |
| CUDA Graphs | 1.2x |

