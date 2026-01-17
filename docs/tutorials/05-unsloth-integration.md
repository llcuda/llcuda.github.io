# Unsloth Integration

Complete workflow: Fine-tune with Unsloth → Export GGUF → Deploy with llcuda.

**Level**: Intermediate | **Time**: 30 minutes | **VRAM Required**: 10-15 GB

---

## Workflow Overview

```
Unsloth (Fine-tune) → GGUF Export → llcuda Deployment
```

## Step 1: Fine-tune with Unsloth

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gemma-2-2b-it",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

# Fine-tune (add your training code here)
```

## Step 2: Export to GGUF

```python
# Save as GGUF Q4_K_M
model.save_pretrained_gguf(
    "output_model",
    tokenizer,
    quantization_method="q4_k_m"
)
```

## Step 3: Deploy with llcuda

```python
from llcuda.server import ServerManager, ServerConfig

config = ServerConfig(
    model_path="output_model/model-Q4_K_M.gguf",
    n_gpu_layers=99,
    flash_attn=True
)

server = ServerManager()
server.start_with_config(config)
```

## Open in Kaggle

[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/waqasm86/05-unsloth-integration-llcuda-v2-2-0)
