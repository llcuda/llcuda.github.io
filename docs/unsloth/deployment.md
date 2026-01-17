# Deployment Pipeline

Deploy Unsloth models with llcuda.

## Complete Pipeline

### 1. Fine-Tune (Unsloth)
```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(...)
# ... training ...
```

### 2. Export (Unsloth)
```python
model.save_pretrained_gguf(
    "my_model",
    tokenizer,
    quantization_method="q4_k_m"
)
```

### 3. Deploy (llcuda)
```python
from llcuda.server import ServerManager, ServerConfig

config = ServerConfig(
    model_path="my_model-Q4_K_M.gguf",
    n_gpu_layers=99,
    tensor_split="0.5,0.5",  # Dual T4
    flash_attn=True,
)

server = ServerManager()
server.start_with_config(config)
```

### 4. Serve (OpenAI API)
```python
from llcuda.api import LlamaCppClient

client = LlamaCppClient()
response = client.chat.completions.create(
    messages=[{"role": "user", "content": "Hello!"}],
)
```

## Production Checklist

- [ ] Model exported to GGUF
- [ ] VRAM requirements verified
- [ ] FlashAttention enabled
- [ ] Server health checked
- [ ] API tested
