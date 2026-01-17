# Optimization Guide

Optimize llcuda performance on Kaggle.

## 1. Enable FlashAttention

```python
config = ServerConfig(
    flash_attn=True,  # 2-3x speedup
)
```

## 2. Optimize Batch Size

```python
config = ServerConfig(
    batch_size=2048,   # Larger for throughput
    ubatch_size=512,   # Smaller for latency
)
```

## 3. Tune Context Size

```python
# Smaller context = faster
config = ServerConfig(
    context_size=2048,  # vs 8192
)
```

## 4. Use K-Quants

- **Q4_K_M**: Best balance
- **Q5_K_M**: Higher quality
- **IQ3_XS**: For 70B models

## 5. Monitor VRAM

```python
from llcuda.api.multigpu import detect_gpus

gpus = detect_gpus()
for gpu in gpus:
    print(f"GPU {gpu.id}: {gpu.memory_used_gb:.1f} / {gpu.memory_total_gb:.1f} GB")
```
