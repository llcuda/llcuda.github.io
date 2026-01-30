# NCCL Integration

NCCL vs tensor-split for distributed workloads.

## Overview

llcuda uses **native CUDA tensor-split** (NOT NCCL) for multi-GPU inference.

## Key Differences

**llama-server (llcuda)**:
- Native CUDA layer distribution
- NO NCCL required
- For LLM inference

**PyTorch DDP**:
- NCCL for distributed training
- For fine-tuning

## llcuda tensor-split

```python
from llcuda.server import ServerConfig

config = ServerConfig(
    model_path="model.gguf",
    tensor_split="0.5,0.5",  # Native CUDA
    n_gpu_layers=99
)
```

## PyTorch with NCCL

```python
import torch.distributed as dist

dist.init_process_group(backend="nccl")
# Training code here
```

## When to Use Each

- **llcuda tensor-split**: Multi-GPU inference
- **PyTorch NCCL**: Multi-GPU training

## Examples

See [Tensor Split vs NCCL](../architecture/tensor-split-vs-nccl.md)
