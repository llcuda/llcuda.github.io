# Large Models on Kaggle

Run 70B models on Kaggle's 30GB dual T4 setup with I-quants.

**Level**: Advanced | **Time**: 30 minutes | **VRAM Required**: 25-30 GB (dual T4)

---

## 70B Model Strategy

Use **IQ3_XS** quantization to fit 70B models in 30GB VRAM.

## Download 70B Model

```python
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="unsloth/Llama-3.1-70B-Instruct-GGUF",
    filename="Llama-3.1-70B-Instruct-IQ3_XS.gguf"
)
```

## Configure for 70B

```python
from llcuda.server import ServerConfig

config = ServerConfig(
    model_path=model_path,
    tensor_split="0.48,0.48",  # Leave 2GB each for overhead
    split_mode="layer",
    n_gpu_layers=80,  # Adjust as needed
    context_size=2048,  # Smaller context
    n_batch=128,       # Smaller batch
    flash_attn=True
)
```

## Expected Performance

- **Speed**: ~12 tokens/sec (Llama-70B IQ3_XS)
- **VRAM**: ~28-29 GB total
- **Context**: 2048 tokens (can increase if VRAM allows)

## VRAM Monitoring

```python
import torch

for i in range(torch.cuda.device_count()):
    mem_alloc = torch.cuda.memory_allocated(i) / 1024**3
    mem_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
    print(f"GPU {i}: {mem_alloc:.1f} / {mem_total:.1f} GB")
```

## Open in Kaggle

[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/waqasm86/09-large-models-kaggle-llcuda-v2-2-0)
