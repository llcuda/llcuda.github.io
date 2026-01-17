# Multi-GPU Inference

Use both Kaggle T4 GPUs with tensor-split for larger models.

**Level**: Beginner | **Time**: 20 minutes | **VRAM Required**: 15-25 GB (dual T4)

---

## GPU Detection

```python
from llcuda.api.multigpu import detect_gpus

gpus = detect_gpus()
for gpu in gpus:
    print(f"GPU {gpu.index}: {gpu.name}, {gpu.memory_total / 1024**3:.1f} GB")
```

## Tensor-Split Configuration

```python
from llcuda.server import ServerConfig

# Equal split across 2 GPUs
config = ServerConfig(
    model_path="model.gguf",
    tensor_split="0.5,0.5",
    split_mode="layer",
    n_gpu_layers=99,
    flash_attn=True
)
```

## Kaggle Dual T4 Preset

```python
from llcuda.api.multigpu import kaggle_t4_dual_config

config = kaggle_t4_dual_config(model_path="model.gguf")
```

## Open in Kaggle

[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/waqasm86/03-multi-gpu-inference-llcuda-v2-2-0)
