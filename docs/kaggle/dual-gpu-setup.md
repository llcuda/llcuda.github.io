# Dual GPU Setup

Configure both T4 GPUs on Kaggle for llcuda.

## Enable Dual T4

1. Settings → Accelerator → **GPU T4 × 2**
2. Settings → Internet → **On**

## Verify Setup

```python
from llcuda.api.multigpu import detect_gpus

gpus = detect_gpus()
assert len(gpus) == 2, "Need 2 GPUs!"
print(f"✓ {len(gpus)} T4 GPUs detected")
```

## GPU Assignment

- **GPU 0**: Primary for LLM inference
- **GPU 1**: Secondary for tensor-split OR Graphistry

See: [Split-GPU Architecture](../architecture/split-gpu.md)
