# First Steps

Your first steps with llcuda v2.2.0 on Kaggle.

---

## 1. Load a Model

```python
from llcuda.server import ServerManager, ServerConfig

# Basic configuration
config = ServerConfig(
    model_path="/path/to/model.gguf",
    n_gpu_layers=99,  # Offload all to GPU
)

server = ServerManager()
server.start_with_config(config)
```

---

## 2. Make Your First Request

```python
from llcuda.api import LlamaCppClient

client = LlamaCppClient()
response = client.chat.completions.create(
    messages=[
        {"role": "user", "content": "What is machine learning?"}
    ],
    max_tokens=200
)

print(response.choices[0].message.content)
```

---

## 3. Explore Notebooks

Try the tutorial notebooks:
- [01 - Quick Start](../tutorials/01-quickstart.md)
- [02 - Server Setup](../tutorials/02-server-setup.md)
- [03 - Multi-GPU](../tutorials/03-multi-gpu.md)
