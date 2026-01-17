# Quick Start Guide

Get llcuda v2.2.0 running on Kaggle in 5 minutes!

---

## Step 1: Install (1 minute)

```bash
pip install git+https://github.com/llcuda/llcuda.git@v2.2.0
```

---

## Step 2: Verify Dual T4 (30 seconds)

```python
from llcuda.api.multigpu import detect_gpus

gpus = detect_gpus()
print(f"✓ Detected {len(gpus)} GPUs")
for gpu in gpus:
    print(f"  GPU {gpu.id}: {gpu.name} ({gpu.memory_total_gb:.1f} GB)")
```

---

## Step 3: Start Server (2 minutes)

```python
from llcuda.server import ServerManager, ServerConfig

config = ServerConfig(
    model_path="model.gguf",
    n_gpu_layers=99,
    flash_attn=True,
)

server = ServerManager()
server.start_with_config(config)
server.wait_until_ready()

print("✓ Server running at http://localhost:8080")
```

---

## Step 4: Run Inference (1 minute)

```python
from llcuda.api import LlamaCppClient

client = LlamaCppClient("http://localhost:8080")
response = client.chat.completions.create(
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=100
)

print(response.choices[0].message.content)
```

---

## Next Steps

- [Tutorial Notebooks](../tutorials/index.md)
- [Multi-GPU Guide](../kaggle/multi-gpu-inference.md)
- [API Reference](../api/overview.md)
