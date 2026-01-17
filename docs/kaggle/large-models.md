# Running 70B Models

Run 70B parameter models on Kaggle dual T4 (30GB VRAM).

## Requirements

- **Quantization**: IQ3_XS (3.3 bpw)
- **VRAM**: ~25-27 GB
- **Strategy**: Dual T4 tensor-split

## Configuration

```python
from llcuda.server import ServerConfig

config = ServerConfig(
    model_path="llama-70b-IQ3_XS.gguf",
    n_gpu_layers=99,
    tensor_split="0.48,0.48",  # Leave headroom
    context_size=2048,          # Smaller context
    batch_size=128,             # Smaller batch
    flash_attn=True,
)
```

## Performance

- **Speed**: ~8-12 tokens/sec
- **Quality**: Good with IQ3_XS
- **VRAM**: ~27 GB used

See: [Tutorial 09 - Large Models](../tutorials/09-large-models.md)
