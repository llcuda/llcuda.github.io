# Unsloth Integration

llcuda as CUDA12 inference backend for Unsloth.

## Workflow

```
1. Fine-Tune (Unsloth)
   ↓
2. Export GGUF (Unsloth)
   ↓
3. Deploy (llcuda)
```

## Why Unsloth + llcuda?

- **Unsloth**: 2x faster training, 70% less VRAM
- **llcuda**: Fast CUDA12 inference on Kaggle
- **Seamless**: Direct GGUF export

## Quick Example

```python
# 1. Fine-tune with Unsloth
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/Qwen2.5-1.5B-Instruct",
    load_in_4bit=True,
)
# ... train ...

# 2. Export to GGUF
model.save_pretrained_gguf(
    "my_model",
    tokenizer,
    quantization_method="q4_k_m"
)

# 3. Deploy with llcuda
from llcuda.server import ServerManager, ServerConfig

server = ServerManager()
server.start_with_config(ServerConfig(
    model_path="my_model-Q4_K_M.gguf",
))
```

See: [Fine-Tuning Workflow](fine-tuning.md)
