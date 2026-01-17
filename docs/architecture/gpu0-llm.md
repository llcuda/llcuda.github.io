# GPU 0 - LLM Inference

Configure GPU 0 for llama.cpp server.

## Setup

```python
from llcuda.server import ServerManager, ServerConfig

config = ServerConfig(
    model_path="model.gguf",
    n_gpu_layers=99,        # All layers on GPU 0
    flash_attn=True,
)

# llama-server uses GPU 0 by default
server = ServerManager()
server.start_with_config(config)
```

## VRAM Usage

| Model | Quant | VRAM on GPU 0 |
|-------|-------|---------------|
| 1-3B | Q4_K_M | 2-4 GB |
| 7B | Q4_K_M | 5-6 GB |
| 13B | Q4_K_M | 8-9 GB |

## Performance

- **FlashAttention**: 2-3x speedup
- **Tensor Cores**: FP16/TF32 acceleration
- **Context**: Up to 8192 tokens
