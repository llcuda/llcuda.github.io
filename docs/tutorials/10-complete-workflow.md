# Complete Workflow

End-to-end production workflow: Unsloth → GGUF → Multi-GPU → Deployment.

**Level**: Advanced | **Time**: 45 minutes | **VRAM Required**: Varies

---

## Full Workflow

```
1. Environment Setup
2. Model Selection
3. Unsloth Fine-tuning
4. GGUF Export
5. Multi-GPU Deployment
6. OpenAI API Client
7. Production Monitoring
```

## 1. Environment Setup

```bash
pip install llcuda unsloth graphistry
```

## 2. Model Selection

```python
# Choose model based on VRAM
# Gemma 2-2B: ~3-4 GB
# Llama-3.2-3B: ~4-5 GB
# Qwen-2.5-7B: ~6-7 GB
# Llama-70B IQ3_XS: ~28-29 GB (dual T4)
```

## 3. Fine-tune (Optional)

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/gemma-2-2b-it",
    load_in_4bit=True
)

# Add LoRA and train
```

## 4. Export to GGUF

```python
model.save_pretrained_gguf("output", tokenizer, quantization_method="q4_k_m")
```

## 5. Deploy Multi-GPU

```python
from llcuda.server import ServerManager, ServerConfig
from llcuda.api.multigpu import kaggle_t4_dual_config

config = kaggle_t4_dual_config(model_path="model.gguf")
server = ServerManager()
server.start_with_config(config)
```

## 6. Use OpenAI Client

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="none")
response = client.chat.completions.create(
    model="local",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## 7. Monitor Performance

```python
# Check logs
logs = server.get_logs()

# Monitor VRAM
import torch
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.memory_allocated(i) / 1024**3:.1f} GB")
```

## Production Best Practices

- Use FlashAttention for 2-3x speedup
- Monitor VRAM usage
- Configure appropriate context size
- Use tensor-split for larger models
- Enable health checking
- Log all requests

## Open in Kaggle

[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/waqasm86/10-complete-workflow-llcuda-v2-2-0)
