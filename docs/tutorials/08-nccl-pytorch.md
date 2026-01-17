# NCCL and PyTorch

Understanding NCCL vs tensor-split for distributed workloads.

**Level**: Advanced | **Time**: 25 minutes | **VRAM Required**: 15-25 GB

---

## Key Differences

**llama-server tensor-split**:
- Native CUDA layer distribution
- NO NCCL required
- For LLM inference

**PyTorch DDP with NCCL**:
- Distributed training
- Requires NCCL
- For fine-tuning

## llama-server (NO NCCL)

```python
from llcuda.server import ServerConfig

config = ServerConfig(
    model_path="model.gguf",
    tensor_split="0.5,0.5",  # Native CUDA split
    n_gpu_layers=99
)
```

## PyTorch DDP (Uses NCCL)

```python
import torch.distributed as dist

dist.init_process_group(backend="nccl")

# DDP training code here
```

## Open in Kaggle

[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/waqasm86/08-nccl-pytorch-llcuda-v2-2-0)
