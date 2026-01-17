# Multi-GPU Inference

Run models across both T4 GPUs with tensor-split.

## Basic Multi-GPU

```python
from llcuda.server import ServerManager, ServerConfig

config = ServerConfig(
    model_path="model.gguf",
    n_gpu_layers=99,
    tensor_split="0.5,0.5",  # Equal split
    split_mode="layer",
    flash_attn=True,
)

server = ServerManager()
server.start_with_config(config)
```

## Kaggle Preset

```python
from llcuda.api.multigpu import kaggle_t4_dual_config

config = kaggle_t4_dual_config(model_size_gb=25)
print(config.to_cli_args())
```

## Performance

| Model | Tokens/sec |
|-------|------------|
| Gemma 2-2B Q4_K_M | ~60 tok/s |
| Qwen2.5-7B Q4_K_M | ~35 tok/s |
| Llama-70B IQ3_XS | ~12 tok/s |
