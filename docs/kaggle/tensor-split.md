# Tensor Split Configuration

Understand tensor-split for dual T4 inference.

## What is Tensor Split?

Native CUDA mechanism to split model layers across GPUs.

**NOT NCCL** - llama.cpp uses native CUDA, not NCCL.

## Configuration

```python
config = ServerConfig(
    tensor_split="0.5,0.5",  # 50% GPU 0, 50% GPU 1
    split_mode="layer",       # Split by layers
)
```

## Split Modes

- **layer**: Split layers across GPUs (recommended)
- **row**: Split tensor rows (requires special support)

## When to Use

- Models > 15GB (won't fit single T4)
- 32B+ models with Q4_K_M
- 70B models with IQ3_XS
