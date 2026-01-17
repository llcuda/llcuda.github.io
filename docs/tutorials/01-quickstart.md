# Quick Start

Get started with llcuda v2.2.0 in 5 minutes on Kaggle dual T4 GPUs.

**Level**: Beginner | **Time**: 5 minutes | **VRAM Required**: 3-5 GB (single T4)

---

## Overview

This tutorial covers the essentials:

- Installing llcuda v2.2.0
- Downloading a GGUF model
- Starting the llama-server
- Making your first chat completion
- Cleaning up resources

## Step 1: Install llcuda

```bash
pip install llcuda
```

## Step 2: Check GPUs

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
print(f"GPU 0: {torch.cuda.get_device_name(0)}")
if torch.cuda.device_count() > 1:
    print(f"GPU 1: {torch.cuda.get_device_name(1)}")
```

## Step 3: Download Model

```python
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="unsloth/gemma-2-2b-it-GGUF",
    filename="gemma-2-2b-it-Q4_K_M.gguf"
)
```

## Step 4: Start Server

```python
from llcuda.server import ServerManager, ServerConfig

config = ServerConfig(
    model_path=model_path,
    n_gpu_layers=99,
    flash_attn=True
)

server = ServerManager()
server.start_with_config(config)
```

## Step 5: Make Request

```python
from llcuda.api.client import LlamaCppClient

client = LlamaCppClient(base_url="http://localhost:8080")
response = client.create_chat_completion(
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=200
)
print(response["choices"][0]["message"]["content"])
```

## Step 6: Cleanup

```python
server.stop()
```

## Expected Performance

- **Speed**: ~60 tokens/sec (Gemma 2-2B Q4_K_M)
- **Latency**: ~500ms
- **VRAM**: ~3-4 GB

## Next Steps

- [02 - Server Setup](02-server-setup.md)
- [03 - Multi-GPU Inference](03-multi-gpu.md)

## Open in Kaggle

[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/waqasm86/01-quickstart-llcuda-v2-2-0)
