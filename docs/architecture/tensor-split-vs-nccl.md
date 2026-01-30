# Tensor Split vs NCCL

Understanding multi-GPU approaches.

## Tensor Split (llama.cpp)

**What**: Native CUDA layer distribution
**Used by**: llama.cpp server
**Purpose**: Inference parallelism

```python
# llama.cpp uses tensor-split
config = ServerConfig(
    tensor_split="0.5,0.5",
    split_mode="layer",
)
```

## NCCL (PyTorch)

**What**: Multi-GPU communication primitives
**Used by**: PyTorch distributed training
**Purpose**: Training parallelism

```python
# PyTorch uses NCCL
import torch.distributed as dist
dist.init_process_group(backend="nccl")
```

## Key Differences

| Feature | Tensor Split | NCCL |
|---------|--------------|------|
| Purpose | Inference | Training |
| Backend | llama.cpp | PyTorch |
| Communication | Direct CUDA | Collectives |
| Use Case | Model too large | Distributed training |

## When to Use Each

- **Tensor Split**: Run 70B model on dual T4
- **NCCL**: Train model with DDP

See: [Tensor-Split Guide](../kaggle/tensor-split.md) and [NCCL Integration API](../api/nccl.md)
