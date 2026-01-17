# Dual T4 Performance

Detailed benchmarks for Kaggle dual T4 setup.

## Configuration

- **GPUs**: 2× Tesla T4 (15GB each)
- **CUDA**: 12.5
- **Driver**: 535.104.05
- **FlashAttention**: Enabled

## Measured Performance

### Gemma 2-2B (Q4_K_M)
- **Tokens/sec**: 58-62
- **Latency**: ~16ms/token
- **VRAM**: 4.2 GB total
- **Strategy**: tensor-split 0.5,0.5

### Qwen2.5-7B (Q4_K_M)
- **Tokens/sec**: 33-37
- **Latency**: ~28ms/token  
- **VRAM**: 10.1 GB total
- **Strategy**: tensor-split 0.5,0.5

### Llama-70B (IQ3_XS)
- **Tokens/sec**: 10-14
- **Latency**: ~80ms/token
- **VRAM**: 26.8 GB total
- **Strategy**: tensor-split 0.48,0.48

## Tuning Tips

1. Enable FlashAttention
2. Use optimal batch size
3. Adjust tensor-split ratios
4. Monitor VRAM usage
